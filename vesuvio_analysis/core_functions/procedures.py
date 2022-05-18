
from vesuvio_analysis.core_functions.bootstrap import runIndependentBootstrap, runJointBootstrap
from vesuvio_analysis.core_functions.fit_in_yspace import fitInYSpaceProcedure
from .analysis_functions import iterativeFitForDataReduction, switchFirstTwoAxis
from mantid.api import AnalysisDataService, mtd
from ..ICHelpers import completeICFromInputs
import numpy as np


def runScript(userCtr, scriptName, wsBackIC, wsFrontIC, bckwdIC, fwdIC, yFitIC, bootIC):

    # Set extra attributes from user attributes
    completeICFromInputs(fwdIC, scriptName, wsFrontIC)
    completeICFromInputs(bckwdIC, scriptName, wsBackIC)


    if userCtr.procedure == "BACKWARD":
        assert userCtr.procedure == userCtr.fitInYSpace, "For isolated forward and backward, procedure needs to match fitInYSpace."
        def runProcedure():
            return runIndependentIterativeProcedure(bckwdIC)

    elif userCtr.procedure == "FORWARD":
        assert userCtr.procedure == userCtr.fitInYSpace, "For isolated forward and backward, procedure needs to match fitInYSpace."
        def runProcedure():
            return runIndependentIterativeProcedure(fwdIC)
    
    elif userCtr.procedure == "JOINT":
        def runProcedure():
            return runJointBackAndForwardProcedure(bckwdIC, fwdIC)
    else:
        raise ValueError("Procedure not recognized.")


    if userCtr.fitInYSpace == "BACKWARD":
        wsNames = buildFinalWSNames(scriptName, ["BACKWARD"], [bckwdIC])
        ICs = [bckwdIC]

    elif userCtr.fitInYSpace == "FORWARD":
        wsNames = buildFinalWSNames(scriptName, ["FORWARD"], [fwdIC])
        ICs = [fwdIC]

    elif userCtr.fitInYSpace == "JOINT":
        wsNames = buildFinalWSNames(scriptName, ["BACKWARD", "FORWARD"], [bckwdIC, fwdIC])
        ICs = [bckwdIC, fwdIC]
    else:
        raise ValueError("fitInYSpace not recognized.")


    if userCtr.bootstrap == None:
        pass

    elif userCtr.bootstrap == "BACKWARD":
        runIndependentBootstrap(bckwdIC, bootIC, yFitIC)
        return

    elif userCtr.bootstrap == "FORWARD":
        runIndependentBootstrap(fwdIC, bootIC, yFitIC)
        return

    elif userCtr.bootstrap == "JOINT":
        runJointBootstrap(bckwdIC, fwdIC, bootIC, yFitIC)
        return
    else:
        raise ValueError("Bootstrap option not recognized.")


    # Check if final ws are loaded:
    wsInMtd = [ws in mtd for ws in wsNames]   # List of bool

    # If final ws are already loaded
    if all(wsInMtd):
        for wsName, IC in zip(wsNames, ICs):
            fitInYSpaceProcedure(yFitIC, IC, mtd[wsName])
        return
    
    runProcedure()

    for wsName, IC in zip(wsNames, ICs):
        fitInYSpaceProcedure(yFitIC, IC, mtd[wsName])
    return


def buildFinalWSNames(scriptName: str, procedures: list, inputIC: list):
    wsNames = []
    for proc, IC in zip(procedures, inputIC):
        name = scriptName + "_" + proc + "_" + str(IC.noOfMSIterations-1)
        wsNames.append(name)
    return wsNames


def runIndependentIterativeProcedure(IC, clearWS=True):
    """
    Runs the iterative fitting of NCP, cleaning any previously stored workspaces.
    input: Backward or Forward scattering initial conditions object
    output: Final workspace that was fitted, object with results arrays
    """

    # Clear worksapces before running one of the procedures below
    if clearWS:
        AnalysisDataService.clear()
        
    wsFinal, ncpFitResultsObject = iterativeFitForDataReduction(IC)
    return wsFinal, ncpFitResultsObject


def runJointBackAndForwardProcedure(bckwdIC, fwdIC, clearWS=True):
    assert bckwdIC.modeRunning == "BACKWARD", "Missing backward IC, args usage: (bckwdIC, fwdIC)"
    assert fwdIC.modeRunning == "FORWARD", "Missing forward IC, args usage: (bckwdIC, fwdIC)"

    # Clear worksapces before running one of the procedures below
    if clearWS:
        AnalysisDataService.clear()

    if isHPresent(fwdIC.masses) and (bckwdIC.HToMass0Ratio==None):
        wsFinal, bckwdScatResults, fwdScatResults = runHPresentAndHRatioNotKnown(bckwdIC, fwdIC)

    else:
        assert (isHPresent(fwdIC.masses) != (bckwdIC.HToMass0Ratio==None)), "When H is not present, HToMass0Ratio has to be set to None"
        
        wsFinal, bckwdScatResults, fwdScatResults = runJoint(bckwdIC, fwdIC)

    return wsFinal, bckwdScatResults, fwdScatResults


def runHPresentAndHRatioNotKnown(bckwdIC, fwdIC):
    """
    Used when H is present and H to first mass ratio is not known.
    Preliminary forward scattering is run to get rough estimate of H to first mass ratio.
    Runs iterative procedure with alternating back and forward scattering.
    """

    assert bckwdIC.runningSampleWS == False, "Procedure not suitable for Bootstrap."

    nIter = askUserNoOfIterations()
 
    # Run preliminary forward with a good guess for the widths of non-H masses
    wsFinal, fwdScatResults = iterativeFitForDataReduction(fwdIC)
    for i in range(int(nIter)):    # Loop until convergence is achieved

        AnalysisDataService.clear()    # Clears all Workspaces

        bckwdIC.HToMass0Ratio = calculateHToMass0Ratio(fwdScatResults)
        wsFinal, bckwdScatResults, fwdScatResults = runJoint(bckwdIC, fwdIC)

    print(f"\n\nFinal estimate for HToMass0Ratio: {calculateHToMass0Ratio(fwdScatResults):.3f}\n")
    return wsFinal, bckwdScatResults, fwdScatResults


def askUserNoOfIterations():
    print("\nH was detected but HToMass0Ratio was not provided.")
    print("\nSugested preliminary procedure:\n\nrun_forward\nfor n:\n    estimate_HToMass0Ratio\n    run_backward\n    run_forward")
    userInput = input("\n\nDo you wish to run preliminary procedure to estimate HToMass0Ratio? (y/n)") 
    if not((userInput=="y") or (userInput=="Y")): raise KeyboardInterrupt("Preliminary procedure interrupted.")
    
    nIter = int(input("\nHow many iterations do you wish to run? n="))
    return nIter
 

def calculateHToMass0Ratio(fwdScatResults):
    fwdMeanIntensityRatios = fwdScatResults.all_mean_intensities[-1] 
    return fwdMeanIntensityRatios[0] / fwdMeanIntensityRatios[1]


def runJoint(bckwdIC, fwdIC):
    wsFinal, bckwdScatResults = iterativeFitForDataReduction(bckwdIC)
    setInitFwdParsFromBackResults(bckwdScatResults, bckwdIC.HToMass0Ratio, fwdIC)
    wsFinal, fwdScatResults = iterativeFitForDataReduction(fwdIC)
    return wsFinal, bckwdScatResults, fwdScatResults   


def setInitFwdParsFromBackResults(bckwdScatResults, HToMass0Ratio, fwdIC):
    """
    Used to pass mean widths and intensities from back scattering onto intial conditions of forward scattering.
    Checks if H is present and adjust the passing accordingly:
    If H present, use HToMass0Ratio to recalculate intensities and fix only non-H widths.
    If H not present, widths and intensities are directly mapped and all widhts except first are fixed. 
    """

    # Get widts and intensity ratios from backscattering results
    backMeanWidths = bckwdScatResults.all_mean_widths[-1]
    backMeanIntensityRatios = bckwdScatResults.all_mean_intensities[-1] 

    if isHPresent(fwdIC.masses):

        assert len(backMeanWidths) == fwdIC.noOfMasses-1, "H Mass present, no of masses in front needs to be bigger than back by 1."

        # Use H ratio to calculate intensity ratios 
        HIntensity = HToMass0Ratio * backMeanIntensityRatios[0]
        initialFwdIntensityRatios = np.append([HIntensity], backMeanIntensityRatios)
        initialFwdIntensityRatios /= np.sum(initialFwdIntensityRatios)

        # Set calculated intensity ratios to forward scattering 
        fwdIC.initPars[0::3] = initialFwdIntensityRatios
        # Set forward widths from backscattering
        fwdIC.initPars[4::3] = backMeanWidths
        # Fix all widths except for H, i.e. the first one
        fwdIC.bounds[4::3] = backMeanWidths[:, np.newaxis] * np.ones((1,2))

    else:   # H mass not present anywhere

        assert len(backMeanWidths) == fwdIC.noOfMasses, "H Mass not present, no of masses needs to be the same for front and back scattering."

        # Set widths and intensity ratios
        fwdIC.initPars[1::3] = backMeanWidths       
        fwdIC.initPars[0::3] = backMeanIntensityRatios  

        if len(backMeanWidths) > 1:           # In the case of single mass, width is not fixed
            # Fix all widhts except first
            fwdIC.bounds[4::3] = backMeanWidths[1:][:, np.newaxis] * np.ones((1,2))   

    print("\nChanged initial conditions of forward scattering according to mean widhts and intensity ratios from backscattering.\n")
    return


def isHPresent(masses) -> bool:

    Hmask = np.abs(masses-1)/1 < 0.1        # H mass whithin 10% of 1 au

    if np.any(Hmask):    # H present

        print("\nH mass detected.\n")
        assert len(Hmask) > 1, "When H is only mass present, run independent forward procedure, not joint."
        assert Hmask[0], "H mass needs to be the first mass in masses and initPars."
        assert sum(Hmask) == 1, "More than one mass very close to H were detected."
        return True
    else:
        return False



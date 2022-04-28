from .analysis_functions import iterativeFitForDataReduction, switchFirstTwoAxis
from mantid.api import AnalysisDataService, mtd
import numpy as np

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

    
    Hmask = np.abs(fwdIC.masses-1)/1 < 0.1
    if np.any(Hmask):  # Check if H present
        print("\nH mass detected.\n")
        assert Hmask[0], "H mass needs to be the first mass in masses and initPars."

        if (bckwdIC.HToMass0Ratio==None) or (bckwdIC.HToMass0Ratio==0):
            wsFinal, bckwdScatResults, fwdScatResults = runHPresentAndHRatioNotKnown(bckwdIC, fwdIC)
        else:
            wsFinal, bckwdScatResults, fwdScatResults = runHPresentAndKnownHRatio(bckwdIC, fwdIC)
    else:
        wsFinal, bckwdScatResults, fwdScatResults = runHNotPresent(bckwdIC, fwdIC)

    return wsFinal, bckwdScatResults, fwdScatResults


def runHNotPresent(bckwdIC, fwdIC):
    """
    Used when H is not present and for one or several masses.
    Runs backward scattering and uses the resulting widhts and intensity ratios
    to set all initial forward scattering widths and intensities.
    First width is not fixed.
    In the case of more than one mass present, all other widths are fixed.
    """


    # Run backward scattering
    wsFinal, bckwdScatResults = iterativeFitForDataReduction(bckwdIC)

    # Extract mean widhts and intensity ratios from backscattering results
    backMeanWidths = bckwdScatResults.all_mean_widths[-1]
    backMeanIntensityRatios = bckwdScatResults.all_mean_intensities[-1] 

    # Set widths and intensity ratios
    fwdIC.initPars[1::3] = backMeanWidths       
    fwdIC.initPars[0::3] = backMeanIntensityRatios  

    if len(backMeanWidths) > 1:
        # Fix all widhts except first
        fwdIC.bounds[4::3] = backMeanWidths[1:][:, np.newaxis] * np.ones((1,2))   

    print("\nStarting forward scattering with mean widhts and intensity ratios from backscattering:")
    print("Assigned all widths and intensities and fixed all widths excet first.\n")

    # Run forward scattering
    wsFinal, fwdScatResults = iterativeFitForDataReduction(fwdIC)  

    return wsFinal, bckwdScatResults, fwdScatResults


def runHPresentAndKnownHRatio(bckwdIC, fwdIC):
    """
    Used when H is present and H to first mass ratio is known. 
    Assumes more than one mass.
    Assumes H is the first mass.
    Runs backscattering and uses results and H to mass ratio to set up initial forward parameters.
    """

    # If H to first mass ratio is known, can run MS correction for backscattering
    # Back scattering produces precise results for widhts and intensity ratios for non-H masses
    wsFinal, bckwdScatResults = iterativeFitForDataReduction(bckwdIC)
    setInitFwdParsFromBackResultsAndHRatio(bckwdScatResults, bckwdIC.HToMass0Ratio, fwdIC)
    wsFinal, fwdScatResults = iterativeFitForDataReduction(fwdIC)
    return wsFinal, bckwdScatResults, fwdScatResults


def runHPresentAndHRatioNotKnown(bckwdIC, fwdIC):
    """
    Used when H is present and H to first mass ratio is not known.
    Assumes more than one mass.
    Assumes H is the first mass.
    Preliminary forward scattering is run to get rough estimate of H to first mass ratio.
    Runs iterative procedure with alternating back and forward scattering.
    """
    #TODO: Have not tried this procedure with Bootstrap, but I believe it is not suitable

    # Run preliminary forward with a good guess for the widths of non-H masses
    wsFinal, fwdScatResults = iterativeFitForDataReduction(fwdIC)
    for i in range(2):    # Loop until convergence is achieved

        AnalysisDataService.clear()    # Clears all Workspaces

        # Get estimate of H to mass0 ratio
        fwdMeanIntensityRatios = fwdScatResults.all_mean_intensities[-1] 
        bckwdIC.HToMass0Ratio = fwdMeanIntensityRatios[0] / fwdMeanIntensityRatios[1]

        # Run backward procedure with this estimate
        wsFinal, bckwdScatResults = iterativeFitForDataReduction(bckwdIC)
        # Set forward scatterign initial widths and intensity ratios
        setInitFwdParsFromBackResultsAndHRatio(bckwdScatResults, bckwdIC.HToMass0Ratio, fwdIC)
        # Run forward procedure with altered withs and intensity ratios
        wsFinal, fwdScatResults = iterativeFitForDataReduction(fwdIC)

    return wsFinal, bckwdScatResults, fwdScatResults
 

def setInitFwdParsFromBackResultsAndHRatio(bckwdScatResults, HToMass0Ratio, fwdIC):
    """
    Used in the case of H present and H ratio to first mass known.
    Assumes more than one mass present.
    Takes the backscattering results and H ratio to set forward scattering widths and intensity ratios.
    """

    # Get widts and intensity ratios from backscattering results
    backMeanWidths = bckwdScatResults.all_mean_widths[-1]
    backMeanIntensityRatios = bckwdScatResults.all_mean_intensities[-1] 

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

    print("\nChanged initial conditions of forward scattering according to mean widhts and intensity ratios from backscattering.\n")
    return    # Changes were implemented of fwdIC object

 

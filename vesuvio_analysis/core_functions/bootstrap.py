from scipy import stats
import numpy as np
from .analysis_functions import arraysFromWS, histToPointData, prepareFitArgs, fitNcpToArray
from .analysis_functions import iterativeFitForDataReduction
from .fit_in_yspace import fitInYSpaceProcedure
from .procedures import runJointBackAndForwardProcedure, runIndependentIterativeProcedure
from vesuvio_analysis.core_functions.ICHelpers import noOfHistsFromTOFBinning
from mantid.api import AnalysisDataService, mtd
from mantid.simpleapi import CloneWorkspace, SaveNexus, Load, SumSpectra
from pathlib import Path
import time
import matplotlib.pyplot as plt
plt.style.use("ggplot")
currentPath = Path(__file__).parent.absolute()

# TODO: Warn user to only use one of these procedures isolated and not one after the other


def runBootstrap(inputIC, bootIC, yFitIC):

    checkOutputDirExists(inputIC, bootIC)            # Checks to see if those directories exits already
    askUserConfirmation(inputIC, bootIC)
    AnalysisDataService.clear()

    if bootIC.runningJackknife and (len(inputIC)==2):
        runOriginalBeforeBootstrap(bootIC, inputIC, yFitIC)  # Just to alter initial conditions fwdIC
        bckwdIC, fwdIC = inputIC
        bckwdJackRes = bootstrapProcedure(bootIC, [bckwdIC], yFitIC)
        fwdJackRes = bootstrapProcedure(bootIC, [fwdIC], yFitIC)
        return [bckwdJackRes[0], fwdJackRes[0]]    # For consistency
    
    else:
        assert (len(inputIC)==1) or (len(inputIC)==2), "Wrong length of inputIC"
        return bootstrapProcedure(bootIC, inputIC, yFitIC)


def checkOutputDirExists(inputIC, bootIC):
    for IC in inputIC:                # Check files already exist
        if bootIC.runningTest:
            continue

        if IC.bootSavePath.is_file() or IC.bootYFitSavePath.is_file():
            print(f"\nOutput data files were detected:" \
                f"\n{IC.bootSavePath.name}\n{IC.bootYFitSavePath.name}" \
                f"\nAborting Run of Bootstrap to prevent overwriting data." \
                f"\nTo avoid this issue you can change the number of samples to run.")
            raise ValueError("Output data directories already exist. Aborted Bootstrap.")
    return 


def bootstrapProcedure(bootIC, inputIC: list, yFitIC):
    """
    Main algorithm for the Bootstrap.
    Allows for Jackknife or Bootstrap depending on bool flag set in bootIC.
    Chooses fast or slow (correct) version of bootstrap depending on flag set in bootIC.
    Performs either independent or joint procedure depending of len(inputIC).
    """
    AnalysisDataService.clear()
 
    parentResults, parentWSnNCPs = runOriginalBeforeBootstrap(bootIC, inputIC, yFitIC)
    corrCoefs = autoCorrResiduals(parentWSnNCPs)   # TODO: Introduce some check here for the autocorrelation of residuals

    nSamples = chooseNSamples(bootIC, parentWSnNCPs)

    bootResults = initializeResults(parentResults, nSamples, corrCoefs)
    parentWSNCPSavePaths = convertWSToSavePaths(parentWSnNCPs)

    iStart, iEnd = chooseLoopRange(bootIC, nSamples)

    # Form each bootstrap workspace and run ncp fit with MS corrections
    for i in range(iStart, iEnd):
        AnalysisDataService.clear()

        sampleInputWS, parentWS = createSampleWS(bootIC.runningJackknife, parentWSNCPSavePaths, i)   # Creates ith sample
        formSampleIC(inputIC, bootIC, sampleInputWS, parentWS)  

        # try:
        iterResults = runMainProcedure(inputIC, yFitIC, runYFit=not(bootIC.runningJackknife))   # Conversion to YSpace with masked column
        # except RuntimeError:    # TODO: Think about the errors to except
        #     continue     # If due to a very unlikely random sample the procedure fails, skip to next iteration
        
        storeBootIter(bootResults, i, iterResults)   # Stores results for each iteration
        saveBootstrapResults(bootResults, inputIC)      
    return bootResults


def askUserConfirmation(inputIC: list, bootIC):
    """Estimates running time for all samples and asks the user to confirm the run."""
    
    if not(bootIC.userConfirmation):   # Skip user confirmation 
        return

    totalTimeOriginal = 0
    totalTimeBootstrap = 0
    for IC in inputIC:
        if IC.modeRunning == "FORWARD":
            timeNoMS = 0.13
            timePerMS = 1.2

        elif IC.modeRunning == "BACKWARD":
            timeNoMS = 0.27
            timePerMS = 2.6
        
        else:
            raise ValueError("Mode running not recognized.")
        
        timeOriginalIC = timeNoMS + (IC.noOfMSIterations) * (timeNoMS+timePerMS)
        totalTimeOriginal += timeOriginalIC      

        # nSamples for either Bootstap or Jackknife
        nSamples = bootIC.nSamples 
        if bootIC.runningJackknife:
            nSamples = 3 if bootIC.runningTest else noOfHistsFromTOFBinning(IC)

        # Either fast or slow bootstrap
        if bootIC.skipMSIterations:
            totalTimeBootstrap += nSamples * timeNoMS
        else:
            totalTimeBootstrap += nSamples * timeOriginalIC

    print(f"\n\nTime estimates are based on a personal laptop with 4 cores, likely oversestimated.")
    print(f"\nEstimated time for original procedure: {totalTimeOriginal:.2f} minutes.")
    userInput = input(f"\nEstimated time for {nSamples} Bootstrap samples: {totalTimeBootstrap/60:.3f} hours.\nProceed? (y/n): ")
    if (userInput == "y") or (userInput == "Y"):
        return
    else:
        raise KeyboardInterrupt ("Bootstrap procedure interrupted.")


def chooseLoopRange(bootIC, nSamples):
    iStart = 0
    iEnd = nSamples
    if bootIC.runningJackknife and bootIC.runningTest:
        iStart = int(nSamples/2)
        iEnd = iStart + 3   
    return iStart, iEnd


def runOriginalBeforeBootstrap(bootIC, inputIC: list, yFitIC):
    """Runs unaltered procedure to store parent results and select parent ws"""

    setICsToDefault(inputIC, yFitIC)
    parentResults = runMainProcedure(inputIC, yFitIC, runYFit=not(bootIC.runningJackknife))
    parentWSnNCPs = selectParentWorkspaces(inputIC, bootIC.skipMSIterations)

    return parentResults, parentWSnNCPs


def chooseNSamples(bootIC, parentWSnNCPs: list):
    """
    Returns number of samples to run.
    If Jackknife is running, no of samples is the number of bins in the workspace."""

    nSamples = bootIC.nSamples
    if bootIC.runningJackknife:
        assert len(parentWSnNCPs) == 1, "Running Jackknife, supports only one IC at a time."
        nSamples = parentWSnNCPs[0][0].blocksize()-1   # -1 becuase last column is ignored during procedure
    return nSamples


def setICsToDefault(inputIC: list, yFitIC):
    """Disables some features of yspace fit, makes sure the default """

    # Disable global fit 
    yFitIC.globalFitFlag = False
    # Run automatic minos by default
    yFitIC.forceManualMinos = False
    # Hide plots
    yFitIC.showPlots = False

    for IC in inputIC:    # Default is not to run with bootstrap ws
        IC.runningSampleWS = False


def runMainProcedure(inputIC: list, yFitIC, runYFit: bool = True):
    """Decides main procedure to run based on the initial conditions offered as inputs."""

    if len(inputIC) == 2:
        bckwdIC, fwdIC = inputIC

        wsParent, bckwdScatResP, fwdScatResP = runJointBackAndForwardProcedure(bckwdIC, fwdIC, clearWS=False)
        parentResults = [bckwdScatResP, fwdScatResP]
        if runYFit:
            yFitResultsParent = fitInYSpaceProcedure(yFitIC, fwdIC, wsParent)
            parentResults = [bckwdScatResP, fwdScatResP, yFitResultsParent]
    
    elif len(inputIC) == 1:
        singleIC = inputIC[0]

        wsParent, singleScatResP = runIndependentIterativeProcedure(singleIC, clearWS=False)
        parentResults = [singleScatResP]
        if runYFit:
            yFitResultsParent = fitInYSpaceProcedure(yFitIC, singleIC, wsParent)
            parentResults = [singleScatResP, yFitResultsParent]  

    else:
        raise ValueError("len(inputIC) needs to be one or two initial conditions (forward and backward).")

    return parentResults


def selectParentWorkspaces(inputIC: list, fastBoot: bool):
    """
    Selects parent workspace from which the Bootstrap replicas will be created.
    If fast mode, the parent ws is the final ws after MS corrections.
    """
    
    parentWSnNCPs = []
    for IC in inputIC:

        if fastBoot:
            wsIter = str(IC.noOfMSIterations)    # Selects last ws after MS corrections

        else:
            wsIter = "0"

        parentWS = mtd[IC.name+wsIter]
        # Extract total ncp fit to the selected workspace
        parentNCP = mtd[parentWS.name()+"_TOF_Fitted_Profiles"]

        parentWSnNCPs.append([parentWS, parentNCP])

    return parentWSnNCPs


def autoCorrResiduals(parentWSnNCP: list):
    """
    Calculates the self-correlation of residuals for each spectrum.
    """
    corrCoefs = []
    for (parentWS, parentNCP) in parentWSnNCP:
        dataY = parentWS.extractY()[:, :-1]
        totNcp = parentNCP.extractY()[:, :]     
        residuals = dataY - totNcp 

        lag = 1     # For lag-plot of self-correlation
        corr = np.zeros((len(residuals), 2))
        for i, rowRes in enumerate(residuals):
            corr[i] = stats.pearsonr(rowRes[:-lag], rowRes[lag:])  
        corrCoefs.append(corr)
    return corrCoefs
    

def initializeResults(parentResults: list, nSamples, corrCoefs):
    """
    Initializes a list with objects to store output data.
    [BootBackResults, BootFrontResults, BootYSpaceResults]
    """
    bootResultObjs = []
    # Initialize results for scattering, no of iteratons = len(corrCoefs) 
    for pResults, corr in zip(parentResults, corrCoefs):
        bootResultObjs.append(BootScattResults(pResults, nSamples, corr))
    
    # Initialize result for y-space fit
    if len(parentResults) == len(corrCoefs) + 1:
        bootResultObjs.append(BootYFitResults(parentResults[-1], nSamples))
    return bootResultObjs


class BootScattResults:

    def __init__(self, parentResults, nSamples, corr):
        self.parentResult = parentResults.all_spec_best_par_chi_nit[-1]
        self.bootSamples = np.full((nSamples, *self.parentResult.shape), np.nan)
        self.corrResiduals = corr

    def storeBootIterResults(self, j, bootResult):
        self.bootSamples[j] = bootResult.all_spec_best_par_chi_nit[-1]
    
    def saveResults(self, IC):
        np.savez(IC.bootSavePath, boot_samples=self.bootSamples,
             parent_result=self.parentResult, corr_residuals=self.corrResiduals)


class BootYFitResults:

    def __init__(self, parentResults, nSamples):
        self.parentPopt = parentResults.popt
        self.parentPerr = parentResults.perr
        self.bootSamples = np.full((nSamples, *self.parentPopt.shape), np.nan)

    def storeBootIterResults(self, j, bootResult):
        self.bootSamples[j] = bootResult.popt
    
    def saveResults(self, IC):
        np.savez(IC.bootYFitSavePath, boot_samples=self.bootSamples,
             parent_popt=self.parentPopt, parent_perr=self.parentPerr)    


def storeBootIter(bootResultObjs: list, j: int, bootIterResults: list):
    for bootObj, iterRes in zip(bootResultObjs, bootIterResults):
        bootObj.storeBootIterResults(j, iterRes)


def saveBootstrapResults(bootResultObjs: list, inputIC: list):
    for bootObj, IC in zip(bootResultObjs, inputIC):    # len(inputIC) is at most 2
        bootObj.saveResults(IC)
    bootResultObjs[-1].saveResults(inputIC[-1])   # Account for YFit object


def convertWSToSavePaths(parentWSnNCPs: list):
    savePaths = []
    for wsList in parentWSnNCPs:
        savePaths.append(saveWorkspacesLocally(wsList))
    return savePaths


def saveWorkspacesLocally(args: list):
    wsSavePaths = []
    for ws in args:

        keys = ws.name().split("_")
        saveName = "Parent"

        if "FORWARD" in keys:
            saveName += "_Front"
        elif "BACKWARD" in keys:
            saveName += "_Back"

        if "Profiles" in keys:
            saveName += "_NCP"
        
        saveName += ".nxs"
        savePath = currentPath / "bootstrap_ws" / saveName
        SaveNexus(ws, str(savePath))
        wsSavePaths.append(savePath)

    return wsSavePaths



def createSampleWS(runningJackknife: bool, parentWSNCPSavePaths: list, j: int):

    if runningJackknife:
        return createJackknifeWS(parentWSNCPSavePaths, j)
    else:
        return createBootstrapWS(parentWSNCPSavePaths, j)



def createBootstrapWS(parentWSNCPSavePaths, j):
    # TODO: use j iteration somewhere?
    """
    Creates bootstrap ws replica.
    Inputs: Experimental (parent) workspace and corresponding NCP total fit
    """

    bootInputWS = []
    parentInputWS = []
    for (parentWSPath, totNcpWSPath) in parentWSNCPSavePaths:
        parentWS, totNcpWS = loadWorkspacesFromPath(parentWSPath, totNcpWSPath)

        dataY = parentWS.extractY()[:, :-1]
        totNcp = totNcpWS.extractY()[:, :]     

        residuals = dataY - totNcp

        bootRes = bootstrapResidualsSample(residuals)
        bootDataY = totNcp + bootRes

        wsBoot = CloneWorkspace(parentWS, OutputWorkspace=parentWS.name()+"_Bootstrap")
        for i, row in enumerate(bootDataY):
            wsBoot.dataY(i)[:-1] = row     # Last column will be ignored in ncp fit anyway

        assert np.all(wsBoot.extractY()[:, :-1] == bootDataY), "Bootstrap data not being correctly passed onto ws."

        bootInputWS.append(wsBoot)
        parentInputWS.append([parentWS, totNcpWS])

    return bootInputWS, parentInputWS


def bootstrapResidualsSample(residuals):
    """Randomly choose points from residuals of each spectra (same statistical weigth)"""

    bootRes = np.zeros(residuals.shape)
    for i, res in enumerate(residuals):
        rowIdxs = np.random.randint(0, len(res), len(res))    # [low, high)
        bootRes[i] = res[rowIdxs]
    return bootRes


def createJackknifeWS(parentWSNCPSavePaths: list, j: int):
    """
    Creates jackknife ws replicas.
    Inputs: Experimental (parent) workspace and corresponding NCP total fit
    """

    jackInputWS = []
    parentInputWS = []
    for (parentWSPath, totNcpWSPath) in parentWSNCPSavePaths:
        parentWS, totNcpWS = loadWorkspacesFromPath(parentWSPath, totNcpWSPath)

        dataY = parentWS.extractY()[:, :-1]
        dataE = parentWS.extractE()[:, :-1]

        jackDataY = dataY.copy()
        jackDataE = dataE.copy()

        jackDataY[:, j] = 0   # Masks j collumn with zeros
        jackDataE[:, j] = 0   # The fit fails if these errors are accidentally used

        wsJack = CloneWorkspace(parentWS, OutputWorkspace=parentWS.name()+"_Jackknife")
        for i, (yRow, eRow) in enumerate(zip(jackDataY, jackDataE)):
            wsJack.dataY(i)[:-1] = yRow     # Last column will be ignored in ncp fit anyway
            wsJack.dataE(i)[:-1] = eRow

        assert np.all(wsJack.extractY()[:, :-1] == jackDataY), "Bootstrap data not being correctly passed onto ws."
        assert np.all(wsJack.extractE()[:, :-1] == jackDataE), "Bootstrap data not being correctly passed onto ws."

        jackInputWS.append(wsJack)
        parentInputWS.append([parentWS, totNcpWS])
    return jackInputWS, parentInputWS


def loadWorkspacesFromPath(*savePaths):
    wsList = []
    for path in savePaths:
        saveName = path.name.split(".")[0]
        ws = Load(str(path), OutputWorkspace=saveName)
        SumSpectra(ws, OutputWorkspace=ws.name()+"_Sum")
        wsList.append(ws)

    return wsList


def formSampleIC(inputIC, bootIC, sampleInputWS, parentWS):
    """Adds atributes to initial conditions to start procedure with sample ws."""

    for IC, wsSample, parentWSnNCP in zip(inputIC, sampleInputWS, parentWS):
        IC.runningSampleWS = True
        IC.runningJackknife = bootIC.runningJackknife

        if bootIC.skipMSIterations:
            IC.noOfMSIterations = 0

        IC.sampleWS = wsSample
        IC.parentWS = parentWSnNCP[0]     # Select workspace with parent data
   


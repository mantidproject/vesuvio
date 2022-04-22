import numpy as np
from .analysis_functions import arraysFromWS, histToPointData, prepareFitArgs, fitNcpToArray
from .analysis_functions import iterativeFitForDataReduction
from .fit_in_yspace import fitInYSpaceProcedure
from .procedures import runJointBackAndForwardProcedure
from mantid.api import AnalysisDataService, mtd
from mantid.simpleapi import CloneWorkspace, SaveNexus, Load
from pathlib import Path
import time
currentPath = Path(__file__).parent.absolute()


def runBootstrap(bckwdIC, fwdIC, bootIC, yFitIC, checkUserIn=True):

    # Disable global fit 
    yFitIC.globalFitFlag = False
    # Run automatic minos by default
    yFitIC.forceManualMinos = False
    # Hide plots
    yFitIC.showPlots = False

    # if bootIC.speedQuick:
    #     quickBootstrap(ic, bootIC, yFitIC)
    # else:
    # slowBootstrap(ic, bootIC, yFitIC)

    bootBackSamples, bootFrontSamples, bootYFitVals = BootstrapJoint(bckwdIC, fwdIC, bootIC, yFitIC, checkUserIn)
    return bootBackSamples, bootFrontSamples, bootYFitVals

# def quickBootstrap(ic, bootIC):

#     AnalysisDataService.clear()
#     wsFinal, fittingResults = iterativeFitForDataReduction(ic)

#     np.random.seed(1)  # Comment this line later on

#     dataYws, dataXws, dataEws = arraysFromWS(wsFinal) 
#     dataY, dataX, dataE = histToPointData(dataYws, dataXws, dataEws)  
#     resolutionPars, instrPars, kinematicArrays, ySpacesForEachMass = prepareFitArgs(ic, dataX)

#     totNcp = fittingResults.all_tot_ncp[-1]
#     parentResult = fittingResults.all_spec_best_par_chi_nit[-1]

#     residuals = dataY - totNcp    # y = g(x) + res

#     bootSamples = np.zeros((bootIC.nSamples, len(dataY), len(ic.initPars)+3))
#     for j in range(bootIC.nSamples):
        
#         # Form the bootstrap residuals
#         bootRes = bootstrapResidualsSample(residuals)

#         bootDataY = totNcp + bootRes
        
#         print("\nFit Bootstrap Sample ...\n")
#         arrFitPars = fitNcpToArray(ic, bootDataY, dataE, resolutionPars, instrPars, kinematicArrays, ySpacesForEachMass)

#         bootSamples[j] = arrFitPars

#     np.savez(ic.bootQuickSavePath, boot_samples=bootSamples, parent_result=parentResult)


def BootstrapJoint(bckwdIC, fwdIC, bootIC, yFitIC, checkUserIn=True):

    bckwdIC.bootSample = False
    fwdIC.bootSample = False

    t0 = time.time()
    wsParent, bckwdScatResP, fwdScatResP = runJointBackAndForwardProcedure(bckwdIC, fwdIC)
    yFitResultsParent = fitInYSpaceProcedure(yFitIC, fwdIC, wsParent)
    t1 = time.time()

    if checkUserIn:
        userIn = checkUserInput(t1-t0, bootIC.nSamples)
        if (userIn != "y") and (userIn != "Y"): return

    parentBackWS, totNCPBack = selectParentWorkspace(bckwdIC)
    parentFrontWS, totNCPFront = selectParentWorkspace(fwdIC)
    savePathBackWS, savePathBackNCP, savePathFrontWS, savePathFrontNCP = saveWorkspacesLocally(parentBackWS, totNCPBack, parentFrontWS, totNCPFront)

    bootResults = BootstrapResults(fwdIC, bckwdIC, bootIC, yFitIC, parentBackWS.getNumberHistograms(), parentFrontWS.getNumberHistograms())
    bootResults.storeParentResults(bckwdScatResP, fwdScatResP, yFitResultsParent)

    # Form each bootstrap workspace and run ncp fit with MS corrections
    for j in range(bootIC.nSamples):
        AnalysisDataService.clear()

        wsBootB = createBootstrapWS(savePathBackWS, savePathBackNCP)
        wsBootF = createBootstrapWS(savePathFrontWS, savePathFrontNCP)

        bckwdIC.bootSample = True    # Tells script to use input bootstrap ws
        fwdIC.bootSample = True    

        bckwdIC.bootWS = wsBootB      # bootstrap ws to input
        fwdIC.bootWS = wsBootF     

        # Run procedure for bootstrap ws
        try:
            wsFinalBoot, bckwdScatResBoot, fwdScatResBoot = runJointBackAndForwardProcedure(bckwdIC, fwdIC, clearWS=False)
            yFitResultsBoot = fitInYSpaceProcedure(yFitIC, fwdIC, wsFinalBoot)
        except:
            continue

        bootResults.storeBootIterResults(j, bckwdScatResBoot, fwdScatResBoot, yFitResultsBoot)
        bootResults.saveResults(bckwdIC, fwdIC)
            
    return bootResults.bootBackSamples, bootResults.bootFrontSamples, bootResults.bootYFitVals


def checkUserInput(t, nSamples):
    print(f"\nRan original procedure, time: {t:.2f} s.")
    print(f"Estimated time for {nSamples} samples: {nSamples*t/60/60:.1f} hours.")
    userIn = input("Continue with Bootstrap? y/n: ")
    return userIn


def selectParentWorkspace(ic):
    """Select workspaces to draw the bootstrap replicas from."""

    # Currently the default is the inital workspace before MS iterations
    initialWS = mtd[ic.name+"0"]
    # Extract total ncp fit to the selected workspace
    initialWSNCP = mtd[initialWS.name()+"_TOF_Fitted_Profiles"]
    return initialWS, initialWSNCP


def saveWorkspacesLocally(*args):
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


def loadWorkspacesFromPath(*args):
    wsList = []
    for path in args:
        saveName = path.name.split(".")[0]
        ws = Load(str(path), OutputWorkspace=saveName)
        wsList.append(ws)

    return wsList


def createBootstrapWS(parentWSPath, totNcpWSPath):
    """
    Creates bootstrap ws replica.
    Inputs: Experimental (parent) workspace and corresponding NCP total fit"""

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
    return wsBoot


def bootstrapResidualsSample(residuals):
    """Randomly choose points from residuals of each spectra (same statistical weigth)"""

    bootRes = np.zeros(residuals.shape)
    for i, res in enumerate(residuals):
        rowIdxs = np.random.randint(0, len(res), len(res))    # [low, high)
        bootRes[i] = res[rowIdxs]

    return bootRes


class BootstrapResults:

    def __init__(self, fwdIC, bckwdIC, bootIC, yFitIC, backWS_len, frontWS_len):
        self.bootBackSamples = np.zeros((bootIC.nSamples, backWS_len, len(bckwdIC.initPars)+3))
        self.bootFrontSamples = np.zeros((bootIC.nSamples, frontWS_len, len(fwdIC.initPars)+3))
        
        yFitNPars = 5 if yFitIC.singleGaussFitToHProfile else 6
        self.bootYFitVals = np.zeros((bootIC.nSamples, 3, yFitNPars))


    def storeParentResults(self, bckwdScatRes, fwdScatRes, yFitRes):
        self.backParentPars = bckwdScatRes.all_spec_best_par_chi_nit[-1]
        self.frontParentPars = fwdScatRes.all_spec_best_par_chi_nit[-1]
        self.poptParent = yFitRes.popt


    def storeBootIterResults(self, j, bckwdScatResBoot, fwdScatResBoot, yFitResultsBoot):
        self.bootBackSamples[j] = bckwdScatResBoot.all_spec_best_par_chi_nit[-1]
        self.bootFrontSamples[j] = fwdScatResBoot.all_spec_best_par_chi_nit[-1]
        self.bootYFitVals[j] = yFitResultsBoot.popt     

    
    def saveResults(self, bckwdIC, fwdIC):

        np.savez(bckwdIC.bootSlowSavePath, boot_samples=self.bootBackSamples,
             parent_result=self.backParentPars)
        
        np.savez(fwdIC.bootSlowSavePath, boot_samples=self.bootFrontSamples,
             parent_result=self.frontParentPars)
        
        np.savez(fwdIC.bootSlowYFitSavePath, boot_vals=self.bootYFitVals,
                parent_popt=self.poptParent)     


# TODO: Figure out a way of switching between single independent and joint procedures

# Legacy function for single individual back or forward procedure
def slowBootstrap(ic, bootIC, yFitIC):
    """Runs bootstrap of full procedure (with MS corrections)"""

    ic.bootSample = False

    # Run procedure without modifications to get parent results
    AnalysisDataService.clear()
    t0 = time.time()
    wsParent, scatResultsParent = iterativeFitForDataReduction(ic)
    yFitResultsParent = fitInYSpaceProcedure(yFitIC, ic, wsParent)
    t1 = time.time()
    userIn = checkUserInput(t1-t0, bootIC.nSamples)
    if (userIn != "y") and (userIn != "Y"): return

    # Extract dataY and ncp from starting workspace
    initialWS = mtd[ic.name+"0"]
    dataY = initialWS.extractY()[:, :-1]
    totNcp = scatResultsParent.all_tot_ncp[0]     # Select fitted ncp corresponding to initialWS
    residuals = dataY - totNcp
    # Save initialWS to preserve dataX and dataE
    saveBootWSPath = currentPath / "bootstrap_ws" / "wsFinal.nxs"
    SaveNexus(initialWS, str(saveBootWSPath))

    # Initialize arrays
    bootSamples = np.zeros((bootIC.nSamples, len(dataY), len(ic.initPars)+3))
    yFitNPars = 5 if yFitIC.singleGaussFitToHProfile else 6
    bootYFitVals = np.zeros((bootIC.nSamples, 3, yFitNPars))

    AnalysisDataService.clear()
    # Form each bootstrap workspace and run ncp fit with MS corrections
    for j in range(bootIC.nSamples):

        bootRes = bootstrapResidualsSample(residuals)
        bootDataY = totNcp + bootRes

        # From workspace with bootstrap dataY
        wsBoot = Load(str(saveBootWSPath), OutputWorkspace="wsBoot")
        for i, row in enumerate(bootDataY):
            wsBoot.dataY(i)[:-1] = row     # Last column will be ignored in ncp fit

        ic.bootSample = True    # Tells script to use input bootstrap ws
        ic.bootWS = wsBoot      # bootstrap ws to input

        # Run procedure for bootstrap ws
        wsFinalBoot, scatResultsBoot = iterativeFitForDataReduction(ic)
        yFitResults = fitInYSpaceProcedure(yFitIC, ic, wsFinalBoot)

        bootSamples[j] = scatResultsBoot.all_spec_best_par_chi_nit[-1]
        bootYFitVals[j] = yFitResults.popt

        # Save result at each iteration in case of failure for long runs
        np.savez(ic.bootSlowSavePath, boot_samples=bootSamples,
             parent_result=scatResultsParent.all_spec_best_par_chi_nit[-1])
        np.savez(ic.bootSlowYFitSavePath, boot_vals=bootYFitVals,
                parent_popt=yFitResultsParent.popt, parent_perr=yFitResultsParent.perr)
        AnalysisDataService.clear()    # Clear all ws




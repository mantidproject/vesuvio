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


def runBootstrap(bckwdIC, fwdIC, bootIC, yFitIC):

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
    BootstrapJoint(bckwdIC, fwdIC, bootIC, yFitIC)


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


def bootstrapResidualsSample(residuals):
    """Randomly choose points from residuals of each spectra (same statistical weigth)"""

    bootRes = np.zeros(residuals.shape)
    for i, res in enumerate(residuals):
        rowIdxs = np.random.randint(0, len(res), len(res))    # [low, high)
        bootRes[i] = res[rowIdxs]

    return bootRes


def checkUserInput(t, nSamples):
    print(f"\nRan original procedure, time: {t:.2f} s.")
    print(f"Estimated time for {nSamples} samples: {nSamples*t/60/60:.1f} hours.")
    userIn = input("Continue with Bootstrap? y/n: ")
    return userIn


def BootstrapJoint(bckwdIC, fwdIC, bootIC, yFitIC):

    bckwdIC.bootSample = False
    fwdIC.bootSample = False

    t0 = time.time()
    wsParent, bckwdScatResP, fwdScatResP = runJointBackAndForwardProcedure(bckwdIC, fwdIC)
    yFitResultsParent = fitInYSpaceProcedure(yFitIC, fwdIC, wsParent)
    t1 = time.time()
    userIn = checkUserInput(t1-t0, bootIC.nSamples)
    if (userIn != "y") and (userIn != "Y"): return

    initialBackWS = mtd[bckwdIC.name+"0"]
    initialFrontWS = mtd[fwdIC.name+"0"]

    saveBootBackWSPath = currentPath / "bootstrap_ws" / "initialBackWS.nxs"
    saveBootFrontWSPath = currentPath / "bootstrap_ws" / "initialFrontWS.nxs"
    SaveNexus(initialBackWS, str(saveBootBackWSPath))
    SaveNexus(initialFrontWS, str(saveBootFrontWSPath))

    dataYB = initialBackWS.extractY()[:, :-1]
    dataYF = initialFrontWS.extractY()[:, :-1]

    totNcpB = bckwdScatResP.all_tot_ncp[0]     # Select fitted ncp corresponding to initialWS
    totNcpF = fwdScatResP.all_tot_ncp[0]     # Select fitted ncp corresponding to initialWS

    resiB = dataYB - totNcpB
    resiF = dataYF - totNcpF

    # Initialize arrays
    bootBackSamples = np.zeros((bootIC.nSamples, len(dataYB), len(bckwdIC.initPars)+3))
    bootFrontSamples = np.zeros((bootIC.nSamples, len(dataYF), len(fwdIC.initPars)+3))
    yFitNPars = 5 if yFitIC.singleGaussFitToHProfile else 6
    bootYFitVals = np.zeros((bootIC.nSamples, 3, yFitNPars))

    AnalysisDataService.clear()
    # Form each bootstrap workspace and run ncp fit with MS corrections
    for j in range(bootIC.nSamples):

        bootResB = bootstrapResidualsSample(resiB)
        bootDataYB = totNcpB + bootResB

        bootResF = bootstrapResidualsSample(resiF)
        bootDataYF = totNcpF + bootResF

        # From workspace with bootstrap dataY
        wsBootB = Load(str(saveBootBackWSPath), OutputWorkspace="wsBootB")
        wsBootF = Load(str(saveBootFrontWSPath), OutputWorkspace="wsBootF")
        for i, (rowB, rowF) in enumerate(zip(bootDataYB, bootDataYF)):
            wsBootB.dataY(i)[:-1] = rowB     # Last column will be ignored in ncp fit
            wsBootF.dataY(i)[:-1] = rowF     # Last column will be ignored in ncp fit

        bckwdIC.bootSample = True    # Tells script to use input bootstrap ws
        bckwdIC.bootWS = wsBootB      # bootstrap ws to input
       
        fwdIC.bootSample = True    # Tells script to use input bootstrap ws
        fwdIC.bootWS = wsBootF      # bootstrap ws to input

        # Run procedure for bootstrap ws
        wsFinalBoot, bckwdScatResBoot, fwdScatResBoot = runJointBackAndForwardProcedure(bckwdIC, fwdIC, clearWS=False)
        yFitResultsBoot = fitInYSpaceProcedure(yFitIC, fwdIC, wsFinalBoot)

        bootBackSamples[j] = bckwdScatResBoot.all_spec_best_par_chi_nit[-1]
        bootFrontSamples[j] = fwdScatResBoot.all_spec_best_par_chi_nit[-1]
        bootYFitVals[j] = yFitResultsBoot.popt

        # Save result at each iteration in case of failure for long runs
        np.savez(bckwdIC.bootSlowSavePath, boot_samples=bootBackSamples,
             parent_result=bckwdScatResP.all_spec_best_par_chi_nit[-1])
        
        np.savez(fwdIC.bootSlowSavePath, boot_samples=bootFrontSamples,
             parent_result=fwdScatResP.all_spec_best_par_chi_nit[-1])
        
        np.savez(fwdIC.bootSlowYFitSavePath, boot_vals=bootYFitVals,
                parent_popt=yFitResultsParent.popt, parent_perr=yFitResultsParent.perr)
        AnalysisDataService.clear()    # Clear all ws




# TODO: Figure out a way of switching between single independent and joint procedures

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




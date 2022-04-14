import numpy as np
from .analysis_functions import arraysFromWS, histToPointData, prepareFitArgs, fitNcpToArray
from .analysis_functions import iterativeFitForDataReduction
from .fit_in_yspace import fitInYSpaceProcedure
from mantid.api import AnalysisDataService, mtd
from mantid.simpleapi import CloneWorkspace, SaveNexus, Load
from pathlib import Path
currentPath = Path(__file__).parent.absolute()


def runBootstrap(ic, bootIC, yFitIC):

    # Disable global fit 
    yFitIC.globalFitFlag = False
    # Run automatic minos by default
    yFitIC.forceManualMinos = False
    # Hide plots
    yFitIC.showPlots = False

    # if bootIC.speedQuick:
    #     quickBootstrap(ic, bootIC, yFitIC)
    # else:
    slowBootstrap(ic, bootIC, yFitIC)


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
#         bootRes = bootstrapResiduals(residuals)

#         bootDataY = totNcp + bootRes
        
#         print("\nFit Bootstrap Sample ...\n")
#         arrFitPars = fitNcpToArray(ic, bootDataY, dataE, resolutionPars, instrPars, kinematicArrays, ySpacesForEachMass)

#         bootSamples[j] = arrFitPars

#     np.savez(ic.bootQuickSavePath, boot_samples=bootSamples, parent_result=parentResult)


def bootstrapResiduals(residuals):
    """Randomly choose points from residuals of each spectra (same statistical weigth)"""

    bootRes = np.zeros(residuals.shape)
    for i, res in enumerate(residuals):
        rowIdxs = np.random.randint(0, len(res), len(res))    # [low, high)
        bootRes[i] = res[rowIdxs]

    return bootRes


def slowBootstrap(ic, bootIC, yFitIC):
    """Runs bootstrap of full procedure (with MS corrections)"""

    ic.bootSample = False

    # Run procedure without modifications to get parent results
    AnalysisDataService.clear()
    wsParent, scatResultsParent = iterativeFitForDataReduction(ic)
    yFitResultsParent = fitInYSpaceProcedure(yFitIC, ic, wsParent)
    parentResult = scatResultsParent.all_spec_best_par_chi_nit[-1]

    oriNoMS = ic.noOfMSIterations   # Store value in seperate variable

    # Run ncp fit without MS corrections
    ic.noOfMSIterations = 1
    AnalysisDataService.clear()
    wsFinal, scatteringResults = iterativeFitForDataReduction(ic)

    # Save workspace to create copy at each iteration
    saveWSPath = currentPath/"bootstrap_ws"/"wsFinal.nxs"
    SaveNexus(wsFinal, str(saveWSPath))

    totNcp = scatteringResults.all_tot_ncp[-1]
    dataY = wsFinal.extractY()[:, :-1]  # Last column cut off
    # Calculate residuals to be basis of bootstrap sampling
    residuals = dataY - totNcp

    # Change no of MS corrections back to original value
    ic.noOfMSIterations = oriNoMS

    # Initialize arrays
    bootSamples = np.zeros((bootIC.nSamples, len(dataY), len(ic.initPars)+3))
    yFitNPars = 5 if yFitIC.singleGaussFitToHProfile else 6
    bootYFitVals = np.zeros((bootIC.nSamples, 3, yFitNPars))
    bootYFitErrs = np.zeros(bootYFitVals.shape)

    AnalysisDataService.clear()
    # Form each bootstrap workspace and run ncp fit with MS corrections
    for j in range(bootIC.nSamples):

        bootRes = bootstrapResiduals(residuals)

        bootDataY = totNcp + bootRes

        # From workspace with bootstrap dataY
        wsBoot = Load(str(saveWSPath), OutputWorkspace="wsBoot")
  
        for i, row in enumerate(bootDataY):
            wsBoot.dataY(i)[:-1] = row     # Last column will be ignored in ncp fit

        ic.bootSample = True    # Tells script to use input bootstrap ws
        ic.bootWS = wsBoot      # bootstrap ws to input

        # Run procedure for bootstrap ws
        wsFinalBoot, scatResultsBoot = iterativeFitForDataReduction(ic)
   
        yFitResults = fitInYSpaceProcedure(yFitIC, ic, wsFinalBoot)

        bootSamples[j] = scatResultsBoot.all_spec_best_par_chi_nit[-1]
        bootYFitVals[j] = yFitResults.popt
        bootYFitErrs[j] = yFitResults.perr

        # Save result at each iteration in case of failure for long runs
        np.savez(ic.bootSlowSavePath, boot_samples=bootSamples, parent_result=parentResult)
        np.savez(ic.bootSlowYFitSavePath, boot_vals=bootYFitVals, boot_errs=bootYFitErrs,
                parent_popt=yFitResultsParent.popt, parent_perr=yFitResultsParent.perr)
        AnalysisDataService.clear()    # Clear all ws




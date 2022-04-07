import numpy as np
from .analysis_functions import arraysFromWS, histToPointData, prepareFitArgs, fitNcpToArray
from .analysis_functions import iterativeFitForDataReduction
from mantid.api import AnalysisDataService, mtd
from mantid.simpleapi import CloneWorkspace, SaveNexus, Load
from pathlib import Path
currentPath = Path(__file__).parent.absolute()

def quickBootstrap(ic, nSamples, savePath):

    AnalysisDataService.clear()
    wsFinal, fittingResults = iterativeFitForDataReduction(ic)

    np.random.seed(1)  # Comment this line later on

    dataYws, dataXws, dataEws = arraysFromWS(wsFinal) 
    dataY, dataX, dataE = histToPointData(dataYws, dataXws, dataEws)  
    resolutionPars, instrPars, kinematicArrays, ySpacesForEachMass = prepareFitArgs(ic, dataX)

    totNcp = fittingResults.all_tot_ncp[-1]
    parentResult = fittingResults.all_spec_best_par_chi_nit[-1]

    residuals = dataY - totNcp    # y = g(x) + res

    bootSamples = np.zeros((nSamples, len(dataY), len(ic.initPars)+3))
    for j in range(nSamples):
        
        # Form the bootstrap residuals
        bootRes = bootstrapResiduals(residuals)

        bootDataY = totNcp + bootRes
        
        print("\nFit Bootstrap Sample ...\n")
        arrFitPars = fitNcpToArray(ic, bootDataY, dataE, resolutionPars, instrPars, kinematicArrays, ySpacesForEachMass)

        bootSamples[j] = arrFitPars

    np.savez(savePath, boot_samples=bootSamples, parent_result=parentResult)


def bootstrapResiduals(residuals):
    """Randomly choose points from residuals of each spectra (same statistical weigth)"""

    bootRes = np.zeros(residuals.shape)
    for i, res in enumerate(residuals):
        rowIdxs = np.random.randint(0, len(res), len(res))    # [low, high)
        bootRes[i] = res[rowIdxs]

    return bootRes


def slowBootstrap(ic, nSamples, savePath):
    """Runs bootstrap of full procedure (with MS corrections)"""

    ic.bootSample = False

    # Run procedure without modifications to get parent results
    AnalysisDataService.clear()
    wsParent, scatResultsParent = iterativeFitForDataReduction(ic)
    parentResult = scatResultsParent.all_spec_best_par_chi_nit[-1]

    oriNoMS = ic.noOfMSIterations   # Store value in seperate variable

    # Run ncp fit without MS corrections
    ic.noOfMSIterations = 1
    AnalysisDataService.clear()
    wsFinal, scatteringResults = iterativeFitForDataReduction(ic)

    # Save workspace to be used at each iteration
    saveWSPath = currentPath / "bootstrap_ws" / "wsFinal.nxs"
    SaveNexus(wsFinal, str(saveWSPath))

    totNcp = scatteringResults.all_tot_ncp[-1]
    dataY = wsFinal.extractY()[:, :-1]  # Last column cut off
    # Calculate residuals to be basis of bootstrap sampling
    residuals = dataY - totNcp

    # Change no of MS corrections back to original value
    ic.noOfMSIterations = oriNoMS

    # Form each bootstrap workspace and run ncp fit with MS corrections
    bootSamples = np.zeros((nSamples, len(dataY), len(ic.initPars)+3))
    AnalysisDataService.clear()
    for j in range(nSamples):

        bootRes = bootstrapResiduals(residuals)

        bootDataY = totNcp + bootRes

        # From workspace with bootstrap dataY
        wsBoot = Load(str(saveWSPath), OutputWorkspace="wsBoot")
        # wsBoot = CloneWorkspace(wsFinal)
        for i, row in enumerate(bootDataY):
            wsBoot.dataY(i)[:-1] = row

        ic.bootSample = True    # Tells script to use input bootstrap ws
        ic.bootWS = wsBoot      # bootstrap ws to input

        # Run procedure for bootstrap ws
        wsFinalBoot, scatResultsBoot = iterativeFitForDataReduction(ic)
   
        bootSamples[j] = scatResultsBoot.all_spec_best_par_chi_nit[-1]

        # Save result at each iteration in case of failure for long runs
        np.savez(savePath, boot_samples=bootSamples, parent_result=parentResult)
        AnalysisDataService.clear()    # Clear all ws




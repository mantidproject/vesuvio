import numpy as np
from .analysis_functions import arraysFromWS, histToPointData, prepareFitArgs, fitNcpToArray


def bootstrapAfterProcedure(ic, wsFinal, fittingResults, nSamples, savePath):
    np.random.seed(1)  # Comment this line later on

    dataYws, dataXws, dataEws = arraysFromWS(wsFinal) 
    dataY, dataX, dataE = histToPointData(dataYws, dataXws, dataEws)  
    resolutionPars, instrPars, kinematicArrays, ySpacesForEachMass = prepareFitArgs(ic, dataX)

    totNcp = fittingResults.all_tot_ncp[-1]

    residuals = dataY - totNcp    # y = g(x) + res

    bootSamples = np.zeros((nSamples, len(dataY), len(ic.initPars)+3))
    for j in range(nSamples):
        
        # Form the bootstrap residuals
        bootRes = bootstrapResiduals(residuals)

        bootDataY = totNcp + bootRes
        
        print("\nFit Bootstrap Sample ...\n")
        arrFitPars = fitNcpToArray(ic, bootDataY, dataE, resolutionPars, instrPars, kinematicArrays, ySpacesForEachMass)

        bootSamples[j] = arrFitPars

    np.savez(savePath, boot_samples=bootSamples)


def bootstrapResiduals(residuals):
    """Randomly choose points from residuals of each spectra (same statistical weigth)"""

    bootRes = np.zeros(residuals.shape)
    for i, res in enumerate(residuals):
        rowIdxs = np.random.randint(0, len(res), len(res))    # [low, high)
        bootRes[i] = res[rowIdxs]

    return bootRes
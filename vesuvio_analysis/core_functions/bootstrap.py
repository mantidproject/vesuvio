import numpy as np
from .analysis_functions import arraysFromWS, histToPointData, prepareFitArgs, fitNcpToArray


def bootstrapAfterProcedure(ic, wsFinal, fittingResults, nSamples, savePath):
    np.random.seed(1)

    dataY, dataX, dataE = arraysFromWS(wsFinal) 
    dataY, dataX, dataE = histToPointData(dataY, dataX, dataE)  
    totNcp = fittingResults.all_tot_ncp[-1]

    residuals  = dataY - totNcp    # y = g(x) + res

    bootSamples = np.zeros((nSamples, len(dataY), len(ic.initPars)+3))
    for j in range(nSamples):
        
        # Form the bootstrap residuals
        bootRes = np.zeros(residuals.shape)
        for i in range(len(residuals)):
            rowIdxs = np.random.randint(0, len(residuals[0]), len(residuals[0]))
            bootRes[i] = residuals[i, rowIdxs]

        bootDataY = totNcp + bootRes
        print("Bootstrap Procedure under development...")

        resolutionPars, instrPars, kinematicArrays, ySpacesForEachMass = prepareFitArgs(ic, dataX)
        
        # Fit Bootstrap sample
        arrFitPars = fitNcpToArray(ic, bootDataY, dataE, resolutionPars, instrPars, kinematicArrays, ySpacesForEachMass)

        bootSamples[j] = arrFitPars

    np.savez(savePath, boot_samples=bootSamples)
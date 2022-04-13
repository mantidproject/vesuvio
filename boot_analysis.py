
import numpy as np
import matplotlib .pyplot as plt
from pathlib import Path
from vesuvio_analysis.core_functions.analysis_functions import calculateMeansAndStds, filterWidthsAndIntensities
currentPath = Path(__file__).parent.absolute() 
experimentsPath = currentPath / "experiments"


def calcBootMeans(bestPars):
    """Performs the means and std on each bootstrap sample"""
    bootWidths = bestPars[:, :, 1::3]
    bootIntensities = bestPars[:, :, 0::3]

    bootMeanW = np.zeros((len(bootWidths[0,0,:]), len(bootWidths)))
    bootStdW = np.zeros(bootMeanW.shape)
    bootMeanI = np.zeros(bootMeanW.shape)
    bootStdI = np.zeros(bootMeanW.shape)

    for j, (widths, intensities) in enumerate(zip(bootWidths, bootIntensities)):

        meanW, stdW, meanI, stdI = calculateMeansAndStds(widths.T, intensities.T)

        bootMeanW[:, j] = meanW
        bootStdW[:, j] = stdW
        bootMeanI[:, j] = meanI
        bootStdI[:, j] = stdI

    return bootMeanW, bootMeanI, bootStdW, bootStdI


def histSampleMeans(meanWidths, meanIntensities, nBins):

    fig, axs = plt.subplots(1, 2, figsize=(15, 8))
    for mode, ax, means in zip(["Widhts", "Intensities"], axs, [meanWidths, meanIntensities]):

        ax.set_title(f"Histogram of Mean {mode}")
        print(f"\nBootstrap distribution of {mode}: \n")
        for i, bootHist in enumerate(means):

            leg = f"{mode} {i}: {np.mean(bootHist):>6.3f} \u00B1 {np.std(bootHist):<6.3f}"
            print(leg)
            ax.hist(bootHist, nBins, histtype="step", label=leg)

        ax.legend()
    plt.show()


def plot3DRows(rows):
    fig= plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(rows[0], rows[1], rows[2])
    ax.set_xlabel("0")
    ax.set_ylabel("1")
    ax.set_zlabel("2")
    plt.show()



# def calcBootWeightAvgMeans(bestPars):
#     """Calculates bootstrap means and std of pars for each spectra and weighted avgs over all spectra."""

#     bootMeans = np.mean(bestPars, axis=0)    # Mean of each fit parameter
#     bootStd = np.std(bestPars, axis=0)     # Error on each fit parameter

#     widthsM = bootMeans[:, 1::3].T
#     intensitiesM = bootMeans[:, 0::3].T

#     print(widthsM.shape)
#     betterWidhtsM, betterIntensitiesM = filterWidthsAndIntensities(widthsM, intensitiesM)

#     widthsE = bootStd[:, 1::3].T
#     intensitiesE = bootStd[:, 0::3].T

#     print(np.sum(np.isnan(betterWidhtsM)))

#     # Ignore results with zero error
#     widthsE[np.isnan(betterWidhtsM) | (widthsE==0)] = np.inf
#     intensitiesE[np.isnan(intensitiesM) | (intensitiesE==0)] = np.inf

#     avgMeansW, avgErrorsW = weightedAvg(betterWidhtsM, widthsE)
#     avgMeansI, avgErrorsI = weightedAvg(betterIntensitiesM, intensitiesE)
#     return avgMeansW, avgErrorsW, avgMeansI, avgErrorsI


# def weightedAvg(means, errors):
#     avgMeans = np.nansum(means/np.square(errors), axis=1) / np.nansum(1/np.square(errors), axis=1)
#     avgErrors = np.sqrt(1 / np.nansum(1/np.square(errors), axis=1))
#     return avgMeans, avgErrors


def printResults(arrM, arrE, mode):
    print(f"\n{mode}:\n")
    for i, (m, e) in enumerate(zip(arrM, arrE)):
        print(f"{mode} {i}: {m:>6.3f} \u00B1 {e:<6.3f}")

# def calculateMeanWidhtsAndIntensities(bestPars):
#     """Replicates means and intensities of original code but with numpy arrays"""

#     widths = bestPars[:, :, 1::3]
#     intensities = bestPars[:, :, 0::3]

#     maskSpec = np.all(widths==0, axis=2)
#     widths[maskSpec] = np.nan     # (100, 132, 3)
#     intensities[maskSpec] = np.nan

#     meanWidths = np.nanmean(widths, axis=1)[:, np.newaxis, :]      # (100, 1, 3)
#     stdWidths = np.std(widths, axis=1)[:, np.newaxis, :]

#     widthDev = np.abs(widths - meanWidths)
#     betterWidths = np.where(widthDev > stdWidths, np.nan, widths)
#     betterIntensities = np.where(widthDev > stdWidths, np.nan, intensities)

#     meanWidths = np.nanmean(betterWidths, axis=1)

#     normIntensities = np.sum(betterIntensities, axis=2)[:, :, np.newaxis]  # (100, 132, 1)
#     betterIntensities = betterIntensities / normIntensities
#     meanIntensities = np.nanmean(betterIntensities, axis=1)

#     stdWidths = np.nanstd(betterWidths, axis=1)
#     stdIntensities = np.nanstd(betterIntensities, axis=1)

#     return meanWidths.T, meanIntensities.T, stdWidths.T, stdIntensities.T


def extractData(sampleName, firstSpec, lastSpec, msIter, MS, GC, nSamples):
    # Build Filename based on ic
    corr = ""
    if MS & (msIter>1):
        corr+="_MS"
    if GC & (msIter>1):
        corr+="_GC"

    fileName = f"spec_{firstSpec}-{lastSpec}_iter_{msIter}{corr}"
    # fileNameYSpace = fileName + "_ySpaceFit"

    fileNameZ = fileName + ".npz"
    # fileNameYSpaceZ = fileNameYSpace + ".npz"

    bootOutPath = experimentsPath / sampleName / "bootstrap_data"
    

    bootName = fileName + f"_nsampl_{nSamples}"
    bootNameZ = bootName + ".npz"

    quickLoadPath = bootOutPath / "quick" / bootNameZ
    slowLoadPath = bootOutPath / "slow" / bootNameZ


    bootQuickData = np.load(quickLoadPath)
    bootSlowData = np.load(slowLoadPath)
    
    bootQuickPars = bootQuickData["boot_samples"][:, :, 1:-2]
    bootSlowPars = bootSlowData["boot_samples"][:, :, 1:-2]
    
    parentQuickPars = bootQuickData["parent_result"][:, 1:-2]
    parentSlowPars = bootSlowData["parent_result"][:, 1:-2]
    
    np.testing.assert_array_almost_equal(parentQuickPars, parentSlowPars)
    
    return bootQuickPars, bootSlowPars, parentQuickPars


sampleName = "D_HMT"
firstSpec = 3
lastSpec = 134
msIter = 4
MS = True
GC = False
nSamples = 1000

nBins = 30

bootQuickPars, bootSlowPars, parentSlowPars = extractData(sampleName, firstSpec, lastSpec, msIter, MS, GC, nSamples)

meanWp, meanIp, stdWp, stdIp = calcBootMeans(parentSlowPars[np.newaxis, :, :])
print(f"\nExperimental Sample results:\n")
printResults(meanWp.flatten(), stdWp.flatten(), "Widths Parent")
printResults(meanIp.flatten(), stdIp.flatten(), "Intensities Parent")

for bestPars, runSpeed in zip([bootQuickPars, bootSlowPars], ["QUICK", "SLOW"]):
    print(f"\n{runSpeed}\n")
    meanW0, meanI0, stdW0, stdI0 = calcBootMeans(bestPars)
    histSampleMeans(meanW0, meanI0, nBins)
    plot3DRows(meanW0)


# # Results of bootstrap error on each parameter and performing weighted avg
# weightedAvgFlag = False
# if weightedAvgFlag:
#     avgMW, avgEW, avgMI, avgEI = calcBootWeightAvgMeans(bestPars)
#     print(f"\nWeighted avg results:\n")
#     printResults(avgMW, avgEW, "Widths")
#     printResults(avgMI, avgEI, "Intensities")





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


def printResults(arrM, arrE, mode):
    print(f"\n{mode}:\n")
    for i, (m, e) in enumerate(zip(arrM, arrE)):
        print(f"{mode} {i}: {m:>6.3f} \u00B1 {e:<6.3f}")


def extractData(sampleName, firstSpec, lastSpec, msIter, MS, GC, nSamples, speed):
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
    bootNameYFit = fileName + "_ySpaceFit" + f"_nsampl_{nSamples}"

    bootNameZ = bootName + ".npz"
    bootNameYFitZ = bootNameYFit + ".npz"

    loadPath = bootOutPath / speed / bootNameZ
    bootData = np.load(loadPath)

    loadYFitPath = bootOutPath / speed / bootNameYFitZ
    # bootYFitData = np.load(loadYFitPath)

    bootPars = bootData["boot_samples"][:, :, 1:-2]
    parentPars = bootData["parent_result"][:, 1:-2]

    # bootYFitVals = bootYFitData["boot_vals"]
        
    return bootPars, parentPars#, bootYFitVals


# sampleName = "D_HMT"
# firstSpec = 3
# lastSpec = 134
# msIter = 4
# MS = True
# GC = False
# nSamples = 1000
# nBins = 30

sampleName = "starch_80_RD"
firstSpec = 3
lastSpec = 134
msIter = 1
MS = False
GC = False
nSamples = 40
nBins = 10
speed = "slow"


# bootQuickPars, parentQuickPars = extractData(sampleName, firstSpec, lastSpec, msIter, MS, GC, nSamples, speed)
bootPars, parentPars = extractData(sampleName, firstSpec, lastSpec, msIter, MS, GC, nSamples, speed)
# np.testing.assert_array_almost_equal(parentQuickPars, parentSlowPars)

# mFitVals = bootYFitVals[:, 0, :-1].T  # Last value is chi
# histSampleMeans(mFitVals, mFitVals, nBins)



meanWp, meanIp, stdWp, stdIp = calcBootMeans(parentPars[np.newaxis, :, :])
print(f"\nExperimental Sample results:\n")
printResults(meanWp.flatten(), stdWp.flatten(), "Widths Parent")
printResults(meanIp.flatten(), stdIp.flatten(), "Intensities Parent")

print(f"\n{speed}\n")
meanW0, meanI0, stdW0, stdI0 = calcBootMeans(bootPars)
histSampleMeans(meanW0, meanI0, nBins)
plot3DRows(meanW0)




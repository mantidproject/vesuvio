import numpy as np
import matplotlib .pyplot as plt
from pathlib import Path

currentPath = Path(__file__).parent.absolute() 

bootPath = currentPath / "back_bootstrap.npz"
bootData = np.load(bootPath)

bestPars = bootData["boot_samples"][:, :, 1:-2]
print(bestPars.shape)


def histSampleMeans(meanWidths, meanIntensities):
    print(meanWidths.shape)
    fig, axs = plt.subplots(1, 2)
    for mode, ax, means in zip(["Widhts", "Intensities"], axs, [meanWidths, meanIntensities]):
        ax.set_title(f"Histogram of Mean {mode}")
        for i, bootHist in enumerate(means):
            leg = f"{mode} {i}: {np.mean(bootHist):>6.3f} \u00B1 {np.std(bootHist):<6.3f}"
            ax.hist(bootHist, 20, histtype="step", label=leg)
        ax.legend()
    plt.show()


def calculateMeanWidhtsAndIntensities(bestPars):
    #TODO: Check that this is doing what is intended with simple example
    widths = bestPars[:, :, 1::3]
    intensities = bestPars[:, :, 0::3]

    maskSpec = np.all(widths==0, axis=2)
    widths[maskSpec] = np.nan     # (100, 132, 3)
    intensities[maskSpec] = np.nan

    meanWidths = np.nanmean(widths, axis=1)[:, np.newaxis, :]      # (100, 1, 3)
    stdWidths = np.std(widths, axis=1)[:, np.newaxis, :]

    widthDev = np.abs(widths - meanWidths)
    betterWidths = np.where(widthDev > stdWidths, np.nan, widths)
    betterIntensities = np.where(widthDev > stdWidths, np.nan, intensities)

    meanWidths = np.nanmean(betterWidths, axis=1)

    normIntensities = np.sum(betterIntensities, axis=2)[:, :, np.newaxis]  # (100, 132, 1)
    betterIntensities = betterIntensities / normIntensities
    meanIntensities = np.nanmean(betterIntensities, axis=1)

    stdWidths = np.nanstd(betterWidths, axis=1)
    stdIntensities = np.nanstd(betterIntensities, axis=1)

    return meanWidths.T, meanIntensities.T, stdWidths.T, stdIntensities.T


meanWidths, meanIntensities, stdWidths, stdIntensities = calculateMeanWidhtsAndIntensities(bestPars)
histSampleMeans(meanWidths, meanIntensities)
# histSampleMeans(stdWidths, stdIntensities)


def printResults(meanWidths, meanIntensities):
    for mode, means in zip(["Widths", "Intensities"], [meanWidths, meanIntensities]):
        print(f"\nBootstrap distribution of {mode}: \n")
        for i, sample in enumerate(means):
            print(f"{mode} {i}: {np.mean(sample):>8.3f} +/- {np.std(sample):<8.3f}")

printResults(meanWidths, meanIntensities)

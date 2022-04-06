import numpy as np
import matplotlib .pyplot as plt
from pathlib import Path
from vesuvio_analysis.core_functions.analysis_functions import calculateMeansAndStds

currentPath = Path(__file__).parent.absolute() 

bootPath = currentPath / "experiments" / "bootstrap_IC" / "back_bootstrap.npz"
bootData = np.load(bootPath)

bestPars = bootData["boot_samples"][:, :, 1:-2]
print(bestPars.shape)


def histSampleMeans(meanWidths, meanIntensities):
    print(meanWidths.shape)

    fig, axs = plt.subplots(1, 2)
    for mode, ax, means in zip(["Widhts", "Intensities"], axs, [meanWidths, meanIntensities]):

        ax.set_title(f"Histogram of Mean {mode}")
        print(f"\nBootstrap distribution of {mode}: \n")
        for i, bootHist in enumerate(means):

            leg = f"{mode} {i}: {np.mean(bootHist):>6.3f} \u00B1 {np.std(bootHist):<6.3f}"
            print(leg)
            ax.hist(bootHist, 20, histtype="step", label=leg)

        ax.legend()
    plt.show()


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


meanW0, meanI0, stdW0, stdI0 = calcBootMeans(bestPars)
# meanW1, meanI1, stdW1, stdI1 = calculateMeanWidhtsAndIntensities(bestPars)

# np.testing.assert_array_almost_equal(meanW0, meanW1)
# np.testing.assert_array_almost_equal(meanI0, meanI1)
# np.testing.assert_array_almost_equal(stdW0, stdW1)
# np.testing.assert_array_almost_equal(stdI0, stdI1)
# print("Tests passed! Array operations same as original.")

histSampleMeans(meanW0, meanI0)
# histSampleMeans(stdW, stdI)

import unittest
import numpy as np
from numpy.core.fromnumeric import size
import numpy.testing as nptest
from pathlib import Path

import matplotlib.pyplot as plt
from jupyterthemes import jtplot
jtplot.style()

currentPath = Path(__file__).absolute().parent  # Path to the repository
filePath = currentPath / "fixatures" / "adapted_original_1iter.npz"

def calcMeanWidthsAndIntensities(widths, intensities):
        widths = widths.T
        intensities = intensities.T
        noOfMasses = len(widths)

        meanWidths = np.nanmean(widths, axis=1).reshape(noOfMasses, 1)  
        stdWidths = np.nanstd(widths, axis=1).reshape(noOfMasses, 1)

        # Subtraction row by row
        widthDeviation = np.abs(widths - meanWidths)
        # Where True, replace by nan
        betterWidths = np.where(widthDeviation > stdWidths, np.nan, widths)
        betterIntensities = np.where(widthDeviation > stdWidths, np.nan, intensities)

        meanWidths = np.nanmean(betterWidths, axis=1)  
        stdWidths = np.nanstd(betterWidths, axis=1)

        # Not nansum(), to propagate nan
        normalization = np.sum(betterIntensities, axis=0)
        intensityRatios = betterIntensities / normalization

        meanIntensityRatios = np.nanmean(intensityRatios, axis=1)
        stdIntensityRatios = np.nanstd(intensityRatios, axis=1)
        return meanWidths, meanIntensityRatios


class TestPars(unittest.TestCase):
    def setUp(self):
        results = np.load(filePath)
        parameters = results["all_spec_best_par_chi_nit"][0]
        self.spec = parameters[:, 0]
        self.chi2 = parameters[:, -2]
        self.nit = parameters[:, -1]
        self.mainPars = parameters[:, 1:-2]
        self.intensities = parameters[:, 1:-2:3]
        self.widths = parameters[:, 2:-2:3]

    def test_nanPars(self):
        nanMask = np.isnan(self.mainPars)
        plt.figure()
        plt.imshow(nanMask, aspect="auto", cmap=plt.cm.binary, interpolation="nearest", norm=None)
        plt.title("np.isnan()")
        plt.xlabel("TOF")
        plt.ylabel("Spectra")
        plt.show()

        rowMask = np.any(nanMask, axis=1)
        print("\nSpectra with nan values:\n",
            self.spec[rowMask])

    def test_zeroPars(self):
        zerosMask = self.mainPars == 0
        plt.figure()
        plt.imshow(zerosMask, aspect="auto", cmap=plt.cm.binary, interpolation="nearest", norm=None)
        plt.title("Zeros mask")
        plt.xlabel("TOF")
        plt.ylabel("Spectra")
        plt.show()

        rowMask = np.any(zerosMask, axis=1)
        print("\nSpectra with zeros values:\n",
            self.spec[rowMask])

    def test_meanWidthsAndIntensities(self):
        results = np.load(filePath)
        oriMeanWidths = results["all_mean_widths"][0]
        oriMeanintensities = results["all_mean_intensities"][0]

        calcMeanWidths, calcMeanIntensities = calcMeanWidthsAndIntensities(self.widths, self.intensities)

        nptest.assert_almost_equal(oriMeanintensities, calcMeanIntensities)
        nptest.assert_almost_equal(oriMeanWidths, calcMeanWidths)


if __name__ == "__main__":
    unittest.main()
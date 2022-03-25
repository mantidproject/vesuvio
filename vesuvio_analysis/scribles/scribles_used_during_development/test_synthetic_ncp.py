import unittest
import numpy as np
from numpy.core.fromnumeric import size
import numpy.testing as nptest
from pathlib import Path

import matplotlib.pyplot as plt
from jupyterthemes import jtplot
jtplot.style()

currentPath = Path(__file__).absolute().parent  # Path to the repository
pathToSynthetic = currentPath / "runs_for_testing" / \
    "fit_synthetic_ncp_with_error_bars.npz"

class TestNCP(unittest.TestCase):
    def setUp(self):
        results = np.load(pathToSynthetic)
        self.ws = results["all_fit_workspaces"][0, :, :-1]
        self.ncp = results["all_tot_ncp"][0]
        self.chi2 = results["all_spec_best_par_chi_nit"][0][:, -2]
        

        if np.all(self.ws[np.isnan(self.ncp)] == 0):
            self.ws = np.where(self.ws==0, np.nan, self.ws)

        self.rtol = 0.01
        self.equal_nan = True

        self.mask = np.isclose(
            self.ncp, self.ws, rtol=self.rtol, equal_nan=self.equal_nan
        )
        self.idxToPlot = 55

    def test_diff_values(self):
        totalDiffMask = ~ self.mask
        noDiff = np.sum(totalDiffMask)
        maskSize = totalDiffMask.size
        print("\nNo of different ncp values:\n",
            noDiff, " out of ", maskSize,
            f"ie {100*noDiff/maskSize:.1f} %")
        
        print(self.ws[5, :5], self.ncp[5, :5])
        print("Mean chi2 of all spectrums:\n",
                np.nanmean(self.chi2))
        print("Maximum chi2: ", np.nanmax(self.chi2))
        chi2NoNans = self.chi2[~np.isnan(self.chi2)]
        print("Mean, taking out outliers:\n",
                np.nanmean(np.sort(chi2NoNans)[:-5]))

    def test_visual(self):
        plt.figure()
        plt.imshow(self.mask, aspect="auto", cmap=plt.cm.RdYlGn, interpolation="nearest", norm=None)
        plt.title("Comparison between sythetic ncp and fitted ncp")
        plt.xlabel("TOF")
        plt.ylabel("Spectra")
        plt.show()

    def test_plot(self):
        plt.figure()
        x = np.linspace(0, 1, len(self.ncp[0]))
        specIdx = self.idxToPlot
        plt.plot(x, self.ws[specIdx], label="synthetic ncp", linewidth = 2)
        plt.plot(x, self.ncp[specIdx], "--", label="fitted ncp", linewidth = 2)
        plt.ylabel("DataY")
        plt.title(f"IDX: {specIdx}")
        plt.legend()
        plt.show()


class TestMeanWidths(unittest.TestCase):
    def setUp(self):
        results = np.load(pathToSynthetic)
        self.meanwidths = results["all_mean_widths"][0]
    
    def test_ncp(self):
        print("\nMean widths:\n", self.meanwidths)


class TestMeanIntensities(unittest.TestCase):
    def setUp(self):
        results = np.load(pathToSynthetic)
        self.meanintensities = results["all_mean_intensities"][0]
    
    def test_ncp(self):
        print("\nMean intensities:\n", self.meanintensities), 


if __name__ == "__main__":
    unittest.main()
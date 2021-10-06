import unittest
import numpy as np
from numpy.core.fromnumeric import size
import numpy.testing as nptest
from pathlib import Path

import matplotlib.pyplot as plt
from jupyterthemes import jtplot
jtplot.style()

currentPath = Path(__file__).absolute().parent  # Path to the repository
pathToSynthetic = currentPath / "script_runs" / "testing_syn_ncp.npz"

class TestNCP(unittest.TestCase):
    def setUp(self):
        results = np.load(pathToSynthetic)
        self.ws = results["all_fit_workspaces"][0, :, :-1]
        self.ncp = results["all_tot_ncp"][0]

        if np.all(self.ws[np.isnan(self.ncp)] == 0):
            self.ws = np.where(self.ws==0, np.nan, self.ws)

        self.rtol = 0.01
        self.equal_nan = True

        self.mask = np.isclose(
            self.ncp, self.ws, rtol=self.rtol, equal_nan=self.equal_nan
        )
        self.idxToPlot = 0

    def test_diff_values(self):
        totalDiffMask = ~ self.mask
        noDiff = np.sum(totalDiffMask)
        maskSize = totalDiffMask.size
        print("\nNo of different ncp values:\n",
            noDiff, " out of ", maskSize,
            f"ie {100*noDiff/maskSize:.1f} %")
        
        print(self.ws[15, :5], self.ncp[15, :5])

    def test_visual(self):
        plt.figure()
        plt.imshow(self.mask, aspect="auto", cmap=plt.cm.RdYlGn, interpolation="nearest", norm=None)
        plt.title("Comparison between ws and ncp")
        plt.xlabel("TOF")
        plt.ylabel("spectrums")
        plt.show()

    def test_plot(self):
        plt.figure()
        x = np.linspace(0, 1, len(self.ncp[0]))
        specIdx = self.idxToPlot
        plt.plot(x, self.ws[specIdx], label="synthetic ncp", linewidth = 2)
        plt.plot(x, self.ncp[specIdx], "--", label="fitted ncp", linewidth = 2)
        plt.legend()
        plt.show()


if __name__ == "__main__":
    unittest.main()
import unittest
import numpy as np
from numpy.core.fromnumeric import size
import numpy.testing as nptest
from pathlib import Path

import matplotlib.pyplot as plt
from jupyterthemes import jtplot
jtplot.style()

currentPath = Path(__file__).absolute().parent  # Path to the repository
pathToOriginal = currentPath / "script_runs" / "ori_spec3-134_iter4_ncp.npz"
pathToOptimized = currentPath / "script_runs" / "testing_runs.npz"

class TestFitParameters(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        oriPars = originalResults["all_spec_best_par_chi_nit"][0]
        self.orispec = oriPars[:, 0]
        self.orichi2 = oriPars[:, -2]
        self.orinit = oriPars[:, -1]
        self.orimainPars = oriPars[:, 1:-2]
        self.oriintensities = self.orimainPars[:, 0::3]
        self.oriwidths = self.orimainPars[:, 1::3]
        self.oricenters = self.orimainPars[:, 2::3]

        optimizedResults = np.load(pathToOptimized)
        optPars = optimizedResults["all_spec_best_par_chi_nit"][0]
        self.optspec = optPars[:, 0]
        self.optchi2 = optPars[:, -2]
        self.optnit = optPars[:, -1]
        self.optmainPars = optPars[:, 1:-2]
        self.optintensities = self.optmainPars[:, 0::3]
        self.optwidths = self.optmainPars[:, 1::3]
        self.optcenters = self.optmainPars[:, 2::3]

        self.rtol = 0.0001
        self.equal_nan = True

    def test_mainPars(self):
        totalMask = np.isclose(
            self.orimainPars, self.optmainPars, rtol=self.rtol, equal_nan=self.equal_nan
            )
        totalDiffMask = ~ totalMask
        noDiff = np.sum(totalDiffMask)
        maskSize = totalDiffMask.size
        print("No of different pars:\n",
            noDiff, " out of ", maskSize,
            f"ie {100*noDiff/maskSize:.1f} %")
        
        plt.figure()
        plt.imshow(totalMask, aspect="auto", cmap=plt.cm.RdYlGn, 
                    interpolation="nearest", norm=None)
        plt.title("Comparison between ori and opt pars")
        plt.xlabel("Parameters")
        plt.ylabel("Spectra")
        plt.show()

    def test_chi2(self):
        totalMask = np.isclose(
            self.orichi2, self.optchi2, rtol=self.rtol, equal_nan=self.equal_nan
            )
        totalDiffMask = ~ totalMask
        noDiff = np.sum(totalDiffMask)
        maskSize = totalDiffMask.size
        print("No of different chi2:\n",
            noDiff, " out of ", maskSize,
            f"ie {100*noDiff/maskSize:.1f} %")

    def test_nit(self):
        totalMask = np.isclose(
            self.orinit, self.optnit, rtol=self.rtol, equal_nan=self.equal_nan
            )
        totalDiffMask = ~ totalMask
        noDiff = np.sum(totalDiffMask)
        maskSize = totalDiffMask.size
        print("No of different nit:\n",
            noDiff, " out of ", maskSize,
            f"ie {100*noDiff/maskSize:.1f} %")

class TestNcp(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        self.orincp = originalResults["all_tot_ncp"][0, :, :-1]
        
        optimizedResults = np.load(pathToOptimized)
        self.optncp = optimizedResults["all_tot_ncp"][0]

        self.rtol = 0.0001
        self.equal_nan = True

    def test_ncp(self):
        correctNansOri = np.where(
            (self.orincp==0) & np.isnan(self.optncp), np.nan, self.orincp
        )

        totalMask = np.isclose(
            correctNansOri, self.optncp, rtol=self.rtol, equal_nan=self.equal_nan
            )
        totalDiffMask = ~ totalMask
        noDiff = np.sum(totalDiffMask)
        maskSize = totalDiffMask.size
        print("No of different values ncp:\n",
            noDiff, " out of ", maskSize,
            f"ie {100*noDiff/maskSize:.1f} %")
        
        plt.figure()
        plt.imshow(totalMask, aspect="auto", cmap=plt.cm.RdYlGn, 
                    interpolation="nearest", norm=None)
        plt.title("Comparison between ori and opt ncp workspace")
        plt.xlabel("Parameters")
        plt.ylabel("Spectra")
        plt.show()

class TestMeanWidths(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        self.orimeanwidths = originalResults["all_mean_widths"][0]

        optimizedResults = np.load(pathToOptimized)
        self.optmeanwidths = optimizedResults["all_mean_widths"][0]
    
    def test_ncp(self):
        print("Mean widths:\n",
            "ori: ", self.orimeanwidths, 
            "\nopt: ", self.optmeanwidths)

class TestMeanIntensities(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        self.orimeanintensities = originalResults["all_mean_intensities"][0]

        optimizedResults = np.load(pathToOptimized)
        self.optmeanintensities = optimizedResults["all_mean_intensities"][0]
    
    def test_ncp(self):
        print("Mean widths:\n",
            "ori: ", self.orimeanintensities, 
            "\nopt: ", self.optmeanintensities)

# class TestFitWorkspaces(unittest.TestCase):
#     def setUp(self):
#         originalResults = np.load(pathToOriginal)
#         self.oriws = originalResults["all_fit_workspaces"][0]
        
#         optimizedResults = np.load(pathToOptimized)
#         self.optws = optimizedResults["all_fit_workspaces"][0]

#     def test_ws(self):
#         nptest.assert_array_equal(
#             self.optws, self.oriws
#         )

if __name__ == "__main__":
    unittest.main()
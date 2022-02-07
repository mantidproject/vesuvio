import unittest
import numpy as np
from numpy.core.fromnumeric import size
import numpy.testing as nptest
from pathlib import Path

import matplotlib.pyplot as plt
# from jupyterthemes import jtplot
# jtplot.style()
np.set_printoptions(suppress=True, precision=8, linewidth=150)
# plt.style.use('dark_background')

currentPath = Path(__file__).absolute().parent  # Path to the repository
print(currentPath)


pathToOriginal = currentPath / "starter_forward.npz" 
pathToOptimized = currentPath / "current_forward.npz" 


class TestFitParameters(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        oriPars = originalResults["all_spec_best_par_chi_nit"]
        self.orispec = oriPars[:, :, 0]
        self.orichi2 = oriPars[:, :, -2]
        self.orinit = oriPars[:, :, -1]
        self.orimainPars = oriPars[:, :, 1:-2]
        self.oriintensities = self.orimainPars[:, :, 0::3]
        self.oriwidths = self.orimainPars[:, :, 1::3]
        self.oricenters = self.orimainPars[:, :, 2::3]

        optimizedResults = np.load(pathToOptimized)
        optPars = optimizedResults["all_spec_best_par_chi_nit"]
        self.optspec = optPars[:, :, 0]
        self.optchi2 = optPars[:, :, -2]
        self.optnit = optPars[:, :, -1]
        self.optmainPars = optPars[:, :, 1:-2]
        self.optintensities = self.optmainPars[:, :, 0::3]
        self.optwidths = self.optmainPars[:, :, 1::3]
        self.optcenters = self.optmainPars[:, :, 2::3]

        self.rtol = 0.0001
        self.equal_nan = True

    def test_mainPars(self):
        nptest.assert_array_equal(self.orimainPars, self.optmainPars)

    def test_chi2(self):
        nptest.assert_array_equal(self.orichi2, self.optchi2)

    def test_nit(self):
        nptest.assert_array_equal(self.orinit, self.optnit)

    def test_intensities(self):
        nptest.assert_array_equal(self.oriintensities, self.optintensities)


def displayMask(mask, rtol, string):
    noDiff = np.sum(mask)
    maskSize = mask.size
    print("\nNo of different "+string+f", rtol={rtol}:\n",
        noDiff, " out of ", maskSize,
        f"ie {100*noDiff/maskSize:.1f} %")    


class TestNcp(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        self.orincp = originalResults["all_tot_ncp"]
        
        optimizedResults = np.load(pathToOptimized)
        self.optncp = optimizedResults["all_tot_ncp"]

        self.rtol = 0.001
        self.equal_nan = True

    def test_ncp(self):
        nptest.assert_array_equal(self.orincp, self.optncp)


class TestMeanWidths(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        self.orimeanwidths = originalResults["all_mean_widths"]

        optimizedResults = np.load(pathToOptimized)
        self.optmeanwidths = optimizedResults["all_mean_widths"]
    
    def test_widths(self):
        # nptest.assert_allclose(self.orimeanwidths, self.optmeanwidths)
        nptest.assert_array_equal(self.orimeanwidths, self.optmeanwidths)


class TestMeanIntensities(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        self.orimeanintensities = originalResults["all_mean_intensities"]

        optimizedResults = np.load(pathToOptimized)
        self.optmeanintensities = optimizedResults["all_mean_intensities"]

    def test_intensities(self):
        # nptest.assert_allclose(self.orimeanintensities, self.optmeanintensities)
        nptest.assert_array_equal(self.orimeanintensities, self.optmeanintensities)


class TestFitWorkspaces(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        self.oriws = originalResults["all_fit_workspaces"]
        
        optimizedResults = np.load(pathToOptimized)
        self.optws = optimizedResults["all_fit_workspaces"]

        self.decimal = 8

    def test_ws(self):
        nptest.assert_array_equal(self.optws, self.oriws)



if __name__ == "__main__":
    unittest.main()
import unittest
import numpy as np
import numpy.testing as nptest
from pathlib import Path

currentPath = Path(__file__).absolute().parent  # Path to the repository
pathToOriginal = currentPath / "fixatures" / "opt_spec3-134_iter4_ncp_nightlybuild.npz"
pathToOptimized = currentPath / "runs_for_testing" / "compare_with_original.npz"

class TestFitParameters(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        oriPars = originalResults["all_spec_best_par_chi_nit"][0]
        self.orispec = oriPars[:, 0]
        self.orichi2 = oriPars[:, -2]
        self.orinit = oriPars[:, -1]
        mainPars = oriPars[:, 1:-2]
        self.oriintensities = mainPars[:, 0::3]
        self.oriwidths = mainPars[:, 1::3]
        self.oricenters = mainPars[:, 2::3]

        optimizedResults = np.load(pathToOptimized)
        optPars = optimizedResults["all_spec_best_par_chi_nit"][0]
        self.optspec = optPars[:, 0]
        self.optchi2 = optPars[:, -2]
        self.optnit = optPars[:, -1]
        mainPars = optPars[:, 1:-2]
        self.optintensities = mainPars[:, 0::3]
        self.optwidths = mainPars[:, 1::3]
        self.optcenters = mainPars[:, 2::3]
    
    def test_chi2(self):
        nptest.assert_allclose(
            self.optchi2, self.orichi2, rtol=1e-5, equal_nan=True
            )

    def test_intensities(self):
        nptest.assert_allclose(
            self.optintensities, self.oriintensities, rtol=1e-5, equal_nan=True
            )
    def test_widths(self):
        nptest.assert_allclose(
            self.optwidths, self.oriwidths, rtol=1e-5, equal_nan=True
            )
    def test_centers(self):
        nptest.assert_allclose(
            self.optcenters, self.oricenters, rtol=1e-4, equal_nan=True
            )


class TestFitWorkspaces(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        self.oriws = originalResults["all_fit_workspaces"][0]
        
        optimizedResults = np.load(pathToOptimized)
        self.optws = optimizedResults["all_fit_workspaces"][0]

    def test_ws(self):
        nptest.assert_array_equal(
            self.optws, self.oriws
        )


class TestNcp(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        self.orincp = originalResults["all_tot_ncp"][0]
        
        optimizedResults = np.load(pathToOptimized)
        self.optncp = optimizedResults["all_tot_ncp"][0]

    def test_ncp(self):
        nptest.assert_allclose(
            self.optncp, self.orincp, rtol=1e-4, equal_nan=True
            )

class TestMeanWidths(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        self.orimeanwidths = originalResults["all_mean_widths"][0]

        optimizedResults = np.load(pathToOptimized)
        self.optmeanwidths = optimizedResults["all_mean_widths"][0]
    
    def test_ncp(self):
        nptest.assert_allclose(
            self.optmeanwidths, self.orimeanwidths, rtol=1e-5, equal_nan=True
            )

class TestMeanIntensities(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        self.orimeanintensities = originalResults["all_mean_intensities"][0]

        optimizedResults = np.load(pathToOptimized)
        self.optmeanintensities = optimizedResults["all_mean_intensities"][0]
    
    def test_ncp(self):
        nptest.assert_allclose(
            self.optmeanintensities, self.orimeanintensities, rtol=1e-5, equal_nan=True
            )


if __name__ == "__main__":
    unittest.main()
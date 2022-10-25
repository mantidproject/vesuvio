from vesuvio.vesuvio_analysis.core_functions.run_script import runScript
import unittest
import numpy as np
import numpy.testing as nptest
from pathlib import Path
from .tests_IC import scriptName, wsBackIC, wsFrontIC, bckwdIC, fwdIC, yFitIC


class BootstrapInitialConditions:  # Not used, but still need to pass as arg
    runBootstrap = False


class UserScriptControls:
    runRoutine = True
    procedure = "FORWARD"
    fitInYSpace = None


class AnalysisRunner:
    _benchmarkResults = None
    _currentResults = None

    @classmethod
    def get_benchmark_result(cls):
        if not AnalysisRunner._benchmarkResults:
            cls._load_benchmark_results()
        return AnalysisRunner._benchmarkResults

    @classmethod
    def get_current_result(cls):
        if not AnalysisRunner._currentResults:
            cls._run()
        return AnalysisRunner._currentResults

    @classmethod
    def _run(cls):
        bootIC = BootstrapInitialConditions
        userCtr = UserScriptControls

        scattRes, yfitRes = runScript(userCtr, scriptName, wsBackIC, wsFrontIC, bckwdIC, fwdIC, yFitIC, bootIC)

        wsFinal, forwardScatteringResults = scattRes

        # Test the results
        np.set_printoptions(suppress=True, precision=8, linewidth=150)

        currentResults = forwardScatteringResults
        AnalysisRunner._currentResults = currentResults

    @classmethod
    def _load_benchmark_results(cls):
        testPath = Path(__file__).absolute().parent
        benchmarkResults = np.load(str(testPath / "stored_analysis.npz"))
        AnalysisRunner._benchmarkResults = benchmarkResults


def displayMask(mask, rtol, string):
    noDiff = np.sum(mask)
    maskSize = mask.size
    print("\nNo of different "+string+f", rtol={rtol}:\n",
          noDiff, " out of ", maskSize,
          f"ie {100*noDiff/maskSize:.1f} %")


class TestFitParameters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmarkResults = AnalysisRunner.get_benchmark_result()
        cls.currentResults = AnalysisRunner.get_current_result()

    def setUp(self):
        oriPars = self.benchmarkResults["all_spec_best_par_chi_nit"]
        self.orispec = oriPars[:, :, 0]
        self.orichi2 = oriPars[:, :, -2]
        self.orinit = oriPars[:, :, -1]
        self.orimainPars = oriPars[:, :, 1:-2]
        self.oriintensities = self.orimainPars[:, :, 0::3]
        self.oriwidths = self.orimainPars[:, :, 1::3]
        self.oricenters = self.orimainPars[:, :, 2::3]

        optPars = self.currentResults.all_spec_best_par_chi_nit
        self.optspec = optPars[:, :, 0]
        self.optchi2 = optPars[:, :, -2]
        self.optnit = optPars[:, :, -1]
        self.optmainPars = optPars[:, :, 1:-2]
        self.optintensities = self.optmainPars[:, :, 0::3]
        self.optwidths = self.optmainPars[:, :, 1::3]
        self.optcenters = self.optmainPars[:, :, 2::3]

        self.rtol = 1e-7
        self.equal_nan = True

    def test_mainPars(self):
        for orip, optp in zip(self.orimainPars, self.optmainPars):
            mask = ~np.isclose(orip, optp, rtol=self.rtol, equal_nan=True)
            displayMask(mask, self.rtol, "Main Pars")
        nptest.assert_array_equal(self.orimainPars, self.optmainPars)

    def test_chi2(self):
        nptest.assert_array_equal(self.orichi2, self.optchi2)

    def test_nit(self):
        nptest.assert_array_equal(self.orinit, self.optnit)

    def test_intensities(self):
        nptest.assert_array_equal(self.oriintensities, self.optintensities)


class TestNcp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmarkResults = AnalysisRunner.get_benchmark_result()
        cls.currentResults = AnalysisRunner.get_current_result()

    def setUp(self):
        self.orincp = self.benchmarkResults["all_tot_ncp"]

        self.optncp = self.currentResults.all_tot_ncp

        self.rtol = 1e-7
        self.equal_nan = True

    def test_ncp(self):
        for orincp, optncp in zip(self.orincp, self.optncp):
            mask = ~np.isclose(orincp, optncp, rtol=self.rtol, equal_nan=True)
            displayMask(mask, self.rtol, "NCP")
        nptest.assert_array_equal(self.orincp, self.optncp)


class TestMeanWidths(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmarkResults = AnalysisRunner.get_benchmark_result()
        cls.currentResults = AnalysisRunner.get_current_result()

    def setUp(self):
        self.orimeanwidths = self.benchmarkResults["all_mean_widths"]

        self.optmeanwidths = self.currentResults.all_mean_widths

    def test_widths(self):
        # nptest.assert_allclose(self.orimeanwidths, self.optmeanwidths)
        nptest.assert_array_equal(self.orimeanwidths, self.optmeanwidths)


class TestMeanIntensities(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmarkResults = AnalysisRunner.get_benchmark_result()
        cls.currentResults = AnalysisRunner.get_current_result()

    def setUp(self):
        self.orimeanintensities = self.benchmarkResults["all_mean_intensities"]

        self.optmeanintensities = self.currentResults.all_mean_intensities

    def test_intensities(self):
        # nptest.assert_allclose(self.orimeanintensities, self.optmeanintensities)
        nptest.assert_array_equal(self.orimeanintensities, self.optmeanintensities)


class TestFitWorkspaces(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmarkResults = AnalysisRunner.get_benchmark_result()
        cls.currentResults = AnalysisRunner.get_current_result()

    def setUp(self):
        self.oriws = self.benchmarkResults["all_fit_workspaces"]

        self.optws = self.currentResults.all_fit_workspaces

        self.decimal = 8
        self.rtol = 1e-7
        self.equal_nan = True

    def test_FinalWS(self):
        for oriws, optws in zip(self.oriws, self.optws):
            mask = ~np.isclose(oriws, optws, rtol=self.rtol, equal_nan=True)
            displayMask(mask, self.rtol, "wsFinal")
        nptest.assert_array_equal(self.optws, self.oriws)


if __name__ == '__main__':
    unittest.main()

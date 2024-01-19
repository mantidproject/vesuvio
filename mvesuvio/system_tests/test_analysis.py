from mvesuvio.vesuvio_analysis.run_script import runScript
import unittest
import numpy as np
import numpy.testing as nptest
from pathlib import Path
from mvesuvio.scripts import handle_config
from mvesuvio.system_tests.analysis_inputs import (
    LoadVesuvioBackParameters,
    LoadVesuvioFrontParameters,
    BackwardInitialConditions,
    ForwardInitialConditions,
    YSpaceFitInitialConditions,
    BootstrapInitialConditions,
    UserScriptControls,
)


ipFilesPath = Path(handle_config.read_config_var("caching.ipfolder"))


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
        scattRes, yfitRes = runScript(
            UserScriptControls(),
            "test_expr",
            LoadVesuvioBackParameters(ipFilesPath),
            LoadVesuvioFrontParameters(ipFilesPath),
            BackwardInitialConditions(ipFilesPath),
            ForwardInitialConditions(ipFilesPath),
            YSpaceFitInitialConditions(),
            BootstrapInitialConditions(),
            True,
        )

        wsFinal, forwardScatteringResults = scattRes

        # Test the results
        np.set_printoptions(suppress=True, precision=8, linewidth=150)

        currentResults = forwardScatteringResults
        AnalysisRunner._currentResults = currentResults

    @classmethod
    def _load_benchmark_results(cls):
        testPath = Path(__file__).absolute().parent
        benchmarkResults = np.load(
            str(testPath / "stored_spec_144-182_iter_3_GC_MS.npz")
        )
        AnalysisRunner._benchmarkResults = benchmarkResults


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

    def test_chi2(self):
        nptest.assert_almost_equal(self.orichi2, self.optchi2, decimal=6)

    def test_nit(self):
        nptest.assert_almost_equal(self.orinit, self.optnit, decimal=-2)

    def test_intensities(self):
        nptest.assert_almost_equal(self.oriintensities, self.optintensities, decimal=2)

    def test_widths(self):
        nptest.assert_almost_equal(self.oriwidths, self.optwidths, decimal=2)

    def test_centers(self):
        nptest.assert_almost_equal(self.oricenters, self.optcenters, decimal=1)


class TestNcp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmarkResults = AnalysisRunner.get_benchmark_result()
        cls.currentResults = AnalysisRunner.get_current_result()

    def setUp(self):
        self.orincp = self.benchmarkResults["all_tot_ncp"][:, :, :-1]

        self.optncp = self.currentResults.all_tot_ncp

    def test_ncp(self):
        correctNansOri = np.where(
            (self.orincp == 0) & np.isnan(self.optncp), np.nan, self.orincp
        )
        nptest.assert_almost_equal(correctNansOri, self.optncp, decimal=4)


class TestMeanWidths(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmarkResults = AnalysisRunner.get_benchmark_result()
        cls.currentResults = AnalysisRunner.get_current_result()

    def setUp(self):
        self.orimeanwidths = self.benchmarkResults["all_mean_widths"]
        self.optmeanwidths = self.currentResults.all_mean_widths

    def test_widths(self):
        nptest.assert_almost_equal(self.orimeanwidths, self.optmeanwidths, decimal=5)


class TestMeanIntensities(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmarkResults = AnalysisRunner.get_benchmark_result()
        cls.currentResults = AnalysisRunner.get_current_result()

    def setUp(self):
        self.orimeanintensities = self.benchmarkResults["all_mean_intensities"]
        self.optmeanintensities = self.currentResults.all_mean_intensities

    def test_intensities(self):
        nptest.assert_almost_equal(self.orimeanintensities, self.optmeanintensities, decimal=6)


class TestFitWorkspaces(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmarkResults = AnalysisRunner.get_benchmark_result()
        cls.currentResults = AnalysisRunner.get_current_result()

    def setUp(self):
        self.oriws = self.benchmarkResults["all_fit_workspaces"]
        self.optws = self.currentResults.all_fit_workspaces

    def test_ws(self):
        nptest.assert_almost_equal(self.oriws, self.optws, decimal=6)

if __name__ == "__main__":
    unittest.main()

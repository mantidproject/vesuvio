import unittest
import numpy as np
import numpy.testing as nptest
from pathlib import Path
from mvesuvio.main.run_routine import Runner 
from mvesuvio.util import handle_config
import mvesuvio


class AnalysisRunner:
    _benchmarkResults = None
    _currentResults = None

    @classmethod
    def get_benchmark_result(cls):
        if not cls._benchmarkResults:
            cls._load_benchmark_results()
        return cls._benchmarkResults

    @classmethod
    def get_current_result(cls):
        if not cls._currentResults:
            mvesuvio.set_config(
                ip_folder=str(Path(handle_config.VESUVIO_PACKAGE_PATH).joinpath("config", "ip_files")),
                inputs_file=str(Path(__file__).absolute().parent.parent.parent / "data" / "analysis" / "inputs" / "analysis_test.py")
            )
            cls._run()
        return cls._currentResults

    @classmethod
    def _run(cls):
        scattRes, yfitRes = Runner(True).run()
        cls._currentResults = scattRes 
        return

    @classmethod
    def _load_benchmark_results(cls):
        benchmarkPath = Path(__file__).absolute().parent.parent.parent / "data" / "analysis" / "benchmark"
        benchmarkResults = np.load(
            str(benchmarkPath / "stored_spec_144-182_iter_1_GC_MS.npz")
        )
        cls._benchmarkResults = benchmarkResults


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

        np.set_printoptions(suppress=True, precision=8, linewidth=150)

    def test_chi2(self):
        nptest.assert_almost_equal(self.orichi2, self.optchi2, decimal=5)

    def test_nit(self):
        nptest.assert_almost_equal(self.orinit, self.optnit, decimal=-2)

    def test_intensities(self):
        nptest.assert_almost_equal(self.oriintensities, self.optintensities, decimal=2)

    def test_widths(self):
        nptest.assert_almost_equal(self.oriwidths, self.optwidths, decimal=3)

    def test_centers(self):
        nptest.assert_almost_equal(self.oricenters, self.optcenters, decimal=1)


class TestNcp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmarkResults = AnalysisRunner.get_benchmark_result()
        cls.currentResults = AnalysisRunner.get_current_result()

    def setUp(self):
        self.orincp = self.benchmarkResults["all_tot_ncp"]
        self.optncp = self.currentResults.all_tot_ncp

    def test_ncp(self):
        nptest.assert_almost_equal(self.orincp, self.optncp, decimal=4)


class TestMeanWidths(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmarkResults = AnalysisRunner.get_benchmark_result()
        cls.currentResults = AnalysisRunner.get_current_result()

    def setUp(self):
        self.orimeanwidths = self.benchmarkResults["all_mean_widths"]
        self.optmeanwidths = self.currentResults.all_mean_widths

    def test_widths(self):
        nptest.assert_almost_equal(self.orimeanwidths, self.optmeanwidths, decimal=4)


class TestMeanIntensities(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmarkResults = AnalysisRunner.get_benchmark_result()
        cls.currentResults = AnalysisRunner.get_current_result()

    def setUp(self):
        self.orimeanintensities = self.benchmarkResults["all_mean_intensities"]
        self.optmeanintensities = self.currentResults.all_mean_intensities

    def test_intensities(self):
        nptest.assert_almost_equal(self.orimeanintensities, self.optmeanintensities, decimal=4)


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

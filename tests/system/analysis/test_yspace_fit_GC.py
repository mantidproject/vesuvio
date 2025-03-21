from mvesuvio.main.run_routine import Runner 
from mantid.simpleapi import Load
from mantid.api import AnalysisDataService
from pathlib import Path
import numpy as np
import unittest
import numpy.testing as nptest
from mvesuvio.util import handle_config
import mvesuvio

np.set_printoptions(suppress=True, precision=8, linewidth=150)


class AnalysisRunner:
    _benchmarkResults = None
    _currentResults = None
    _input_data_path = Path(__file__).absolute().parent.parent.parent / "data" / "analysis" / "inputs"
    _benchmark_path = Path(__file__).absolute().parent.parent.parent / "data" / "analysis" / "benchmark"
    _workspaces_loaded = False

    @classmethod
    def load_workspaces(cls):
        if not cls._workspaces_loaded:
            cls._load_workspaces()
            cls._workspaces_loaded = True

    @classmethod
    def get_benchmark_result(cls):
        if not AnalysisRunner._benchmarkResults:
            cls._load_benchmark_results()
        return AnalysisRunner._benchmarkResults

    @classmethod
    def get_current_result(cls):
        if not AnalysisRunner._currentResults:
            mvesuvio.set_config(
                ip_folder=str(Path(handle_config.VESUVIO_PACKAGE_PATH).joinpath("config", "ip_files")),
                inputs_file=str(Path(__file__).absolute().parent.parent.parent / "data" / "analysis" / "inputs" / "yspace_gc_test.py")
            )
            cls._run()
        return AnalysisRunner._currentResults

    @classmethod
    def _load_workspaces(cls):
        AnalysisDataService.clear()
        Load(
            str(cls._input_data_path / "yspace_tests_fwd_1.nxs"),
            OutputWorkspace="yspace_gc_test_fwd_1"
        )
        Load(
            str(cls._input_data_path / "yspace_tests_fwd_1_total_ncp.nxs"),
            OutputWorkspace="yspace_gc_test_fwd_1_total_ncp"
        )
        Load(
            str(cls._input_data_path / "yspace_tests_fwd_1_1.0079_ncp.nxs"),
            OutputWorkspace="yspace_gc_test_fwd_1_1.0079_ncp"
        )
        Load(
            str(cls._input_data_path / "yspace_gauss_test_fwd_initial_parameters.nxs"),
            OutputWorkspace="yspace_gc_test_fwd_initial_parameters"
        )
        return

    @classmethod
    def _run(cls):
        cls.load_workspaces()
        scattRes, yfitRes = Runner(True).run()
        cls._currentResults = yfitRes

    @classmethod
    def _load_benchmark_results(cls):
        cls._benchmarkResults = np.load(
            str(cls._benchmark_path / "yspace_gc_test.npz")
        )


class TestSymSumYSpace(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.oridataY = AnalysisRunner.get_benchmark_result()["YSpaceSymSumDataY"]
        cls.oridataE = AnalysisRunner.get_benchmark_result()["YSpaceSymSumDataE"]

        cls.optdataY = AnalysisRunner.get_current_result().YSpaceSymSumDataY
        cls.optdataE = AnalysisRunner.get_current_result().YSpaceSymSumDataE

    def test_YSpaceDataY(self):
        nptest.assert_almost_equal(self.oridataY, self.optdataY)

    def test_YSpaceDataE(self):
        nptest.assert_almost_equal(self.oridataE, self.optdataE)


class TestResolution(unittest.TestCase):
    def setUp(self):
        self.orires = AnalysisRunner.get_benchmark_result()["resolution"]
        self.optres = AnalysisRunner.get_current_result().resolution

    def test_resolution(self):
        nptest.assert_almost_equal(self.orires, self.optres)


class TestHdataY(unittest.TestCase):
    def setUp(self):
        self.oriHdataY = AnalysisRunner.get_benchmark_result()["HdataY"]
        self.optHdataY = AnalysisRunner.get_current_result().HdataY

    def test_HdataY(self):
        nptest.assert_almost_equal(self.oriHdataY, self.optHdataY)


class TestFinalRawDataY(unittest.TestCase):
    def setUp(self):
        self.oriFinalDataY = AnalysisRunner.get_benchmark_result()["finalRawDataY"]
        self.optFinalDataY = AnalysisRunner.get_current_result().finalRawDataY

    def test_FinalDataY(self):
        nptest.assert_almost_equal(self.oriFinalDataY, self.optFinalDataY)


class TestFinalRawDataE(unittest.TestCase):
    def setUp(self):
        self.oriFinalDataE = AnalysisRunner.get_benchmark_result()["finalRawDataE"]
        self.optFinalDataE = AnalysisRunner.get_current_result().finalRawDataE

    def test_HdataE(self):
        nptest.assert_almost_equal(self.oriFinalDataE, self.optFinalDataE)


class Testpopt(unittest.TestCase):
    def setUp(self):
        self.oripopt = AnalysisRunner.get_benchmark_result()["popt"][:, :-1]    # Exclude Chi2 from test
        self.optpopt = AnalysisRunner.get_current_result().popt[:, :-1]

    def test_opt(self):
        nptest.assert_almost_equal(self.oripopt, self.optpopt)


class Testperr(unittest.TestCase):
    def setUp(self):
        self.oriperr = AnalysisRunner.get_benchmark_result()["perr"]
        self.optperr = AnalysisRunner.get_current_result().perr

    def test_perr(self):
        nptest.assert_almost_equal(self.oriperr, self.optperr, decimal=5)


if __name__ == "__main__":
    unittest.main()

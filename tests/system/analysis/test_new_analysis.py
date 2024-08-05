import unittest
import numpy as np
import numpy.testing as nptest
from pathlib import Path
from mvesuvio.util import handle_config

ipFilesPath = Path(handle_config.VESUVIO_PACKAGE_PATH).joinpath("config", "ip_files")

from mvesuvio.oop.AnalysisRoutine import AnalysisRoutine
from mvesuvio.oop.NeutronComptonProfile import NeutronComptonProfile
from mvesuvio.oop.analysis_helpers import loadRawAndEmptyWsFromUserPath, cropAndMaskWorkspace 

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

        ws = loadRawAndEmptyWsFromUserPath(
            userWsRawPath=str(Path(__file__).absolute().parent.parent.parent/"data"/"analysis"/"inputs"/"sample_test"/"input_ws"/"sample_test_raw_forward.nxs" ),
            # userWsRawPath='/home/ljg28444/Documents/user_0/some_new_experiment/some_new_exp/input_ws/some_new_exp_raw_forward.nxs',
            userWsEmptyPath=str(Path(__file__).absolute().parent.parent.parent/"data"/"analysis"/"inputs"/"sample_test"/"input_ws"/"sample_test_empty_forward.nxs" ),
            # userWsEmptyPath='/home/ljg28444/Documents/user_0/some_new_experiment/some_new_exp/input_ws/some_new_exp_empty_forward.nxs',
            tofBinning = "110.,1.,430",
            name='exp',
            scaleRaw=1,
            scaleEmpty=1,
            subEmptyFromRaw=False
        )
        cropedWs = cropAndMaskWorkspace(ws, firstSpec=144, lastSpec=182,
                                        maskedDetectors=[173, 174, 179],
                                        maskTOFRange=None)

        AR = AnalysisRoutine(cropedWs,
                             ip_file='/home/ljg28444/.mvesuvio/ip_files/ip2018_3.par',
                             number_of_iterations=3,
                             mask_spectra=[173, 174, 179],
                             multiple_scattering_correction=True,
                             vertical_width=0.1, horizontal_width=0.1, thickness=0.001,
                             transmission_guess=0.8537, 
                             multiple_scattering_order=2,
                             number_of_events=1.0e5,
                             gamma_correction=True,
                             mode_running='FORWARD')
            
        H = NeutronComptonProfile('H', mass=1.0079, intensity=1, width=4.7, center=0,
                                  intensity_bounds=[0, np.nan], width_bounds=[3, 6], center_bounds=[-3, 1]) 
        C = NeutronComptonProfile('C', mass=12, intensity=1, width=12.71, center=0,
                                  intensity_bounds=[0, np.nan], width_bounds=[12.71, 12.71], center_bounds=[-3, 1]) 
        S = NeutronComptonProfile('S', mass=16, intensity=1, width=8.76, center=0,
                                  intensity_bounds=[0, np.nan], width_bounds=[8.76, 8.76], center_bounds=[-3, 1]) 
        Co = NeutronComptonProfile('Co', mass=27, intensity=1, width=13.897, center=0,
                                  intensity_bounds=[0, np.nan], width_bounds=[13.897, 13.897], center_bounds=[-3, 1]) 

        AR.add_profiles(H, C, S, Co)
        AnalysisRunner._currentResults = AR.run()

        np.set_printoptions(suppress=True, precision=8, linewidth=150)


    @classmethod
    def _load_benchmark_results(cls):
        benchmarkPath = Path(__file__).absolute().parent.parent.parent / "data" / "analysis" / "benchmark"
        benchmarkResults = np.load(
            str(benchmarkPath / "stored_spec_144-182_iter_3_GC_MS.npz")
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

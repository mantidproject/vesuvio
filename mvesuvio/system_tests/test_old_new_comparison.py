from mvesuvio.vesuvio_analysis.run_script import runScript
import unittest
import numpy as np
import numpy.testing as nptest
from pathlib import Path
from mvesuvio.scripts import handle_config
from mvesuvio.system_tests.old_new_comparison_inputs import (
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
            "test_expr_comp",
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


def displayMaskAllIter(mask, rtol, string):
    print("\nNo of different " + string + f", rtol={rtol}:")
    for i, mask_i in enumerate(mask):
        noDiff = np.sum(mask_i)
        maskSize = mask_i.size
        print(
            f"iter {i}: ",
            noDiff,
            " out of ",
            maskSize,
            f"ie {100*noDiff/maskSize:.1f} %",
        )


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

        self.rtol = 0.0001
        self.equal_nan = True

    def xtest_mainPars(self):
        totalMask = np.isclose(
            self.orimainPars, self.optmainPars, rtol=self.rtol, equal_nan=self.equal_nan
        )
        totalDiffMask = ~totalMask
        displayMaskAllIter(totalDiffMask, self.rtol, "parameters")
        nptest.assert_allclose(self.orimainPars, self.optmainPars, self.rtol)

    def test_chi2(self):
        totalMask = np.isclose(
            self.orichi2, self.optchi2, rtol=self.rtol, equal_nan=self.equal_nan
        )
        totalDiffMask = ~totalMask
        displayMaskAllIter(totalDiffMask, self.rtol, "chi2")
        nptest.assert_allclose(self.orichi2, self.optchi2, self.rtol)

    def xtest_nit(self):
        totalMask = np.isclose(
            self.orinit, self.optnit, rtol=self.rtol, equal_nan=self.equal_nan
        )
        totalDiffMask = ~totalMask
        displayMaskAllIter(totalDiffMask, self.rtol, "nit")
        nptest.assert_allclose(self.orinit, self.optnit, self.rtol)

    def xtest_intensities(self):
        totalMask = np.isclose(
            self.oriintensities,
            self.optintensities,
            rtol=self.rtol,
            equal_nan=self.equal_nan,
        )
        totalDiffMask = ~totalMask
        displayMaskAllIter(totalDiffMask, self.rtol, "intensities")
        nptest.assert_allclose(self.oriintensities, self.optintensities, self.rtol)


class TestNcp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmarkResults = AnalysisRunner.get_benchmark_result()
        cls.currentResults = AnalysisRunner.get_current_result()

    def setUp(self):
        self.orincp = self.benchmarkResults["all_tot_ncp"][:, :, :-1]

        self.optncp = self.currentResults.all_tot_ncp

        self.rtol = 0.0001
        self.equal_nan = True

    def xtest_ncp(self):
        correctNansOri = np.where(
            (self.orincp == 0) & np.isnan(self.optncp), np.nan, self.orincp
        )

        totalMask = np.isclose(
            correctNansOri, self.optncp, rtol=self.rtol, equal_nan=self.equal_nan
        )
        totalDiffMask = ~totalMask
        displayMaskAllIter(totalDiffMask, self.rtol, "ncp")
        nptest.assert_allclose(self.orincp, self.optncp, self.rtol)


class TestMeanWidths(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmarkResults = AnalysisRunner.get_benchmark_result()
        cls.currentResults = AnalysisRunner.get_current_result()

    def setUp(self):
        self.orimeanwidths = self.benchmarkResults["all_mean_widths"]
        self.optmeanwidths = self.currentResults.all_mean_widths
        self.rtol = 0.0001

    def test_widths(self):
        nptest.assert_allclose(self.orimeanwidths, self.optmeanwidths, self.rtol)


class TestMeanIntensities(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmarkResults = AnalysisRunner.get_benchmark_result()
        cls.currentResults = AnalysisRunner.get_current_result()

    def setUp(self):
        self.orimeanintensities = self.benchmarkResults["all_mean_intensities"]
        self.optmeanintensities = self.currentResults.all_mean_intensities
        self.rtol = 0.001

    def test_intensities(self):
        nptest.assert_allclose(
            self.orimeanintensities, self.optmeanintensities, self.rtol
        )


class TestFitWorkspaces(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.benchmarkResults = AnalysisRunner.get_benchmark_result()
        cls.currentResults = AnalysisRunner.get_current_result()

    def setUp(self):
        self.oriws = self.benchmarkResults["all_fit_workspaces"]
        self.optws = self.currentResults.all_fit_workspaces

        self.decimal = 8
        self.rtol = 0.0001
        self.equal_nan = True

    def xtest_ws(self):
        totalMask = np.isclose(
            self.oriws, self.optws, rtol=self.rtol, equal_nan=self.equal_nan
        )
        totalDiffMask = ~totalMask
        displayMaskAllIter(totalDiffMask, self.rtol, "workspaces to be fitted")
        nptest.assert_allclose(self.oriws, self.optws, self.rtol)


if __name__ == "__main__":
    unittest.main()

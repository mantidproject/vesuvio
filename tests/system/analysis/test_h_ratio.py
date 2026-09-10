import runpy
import unittest
from unittest.mock import patch
from pathlib import Path
from mvesuvio.util import handle_config
from mvesuvio import ConfigArgInputs
from shutil import copytree
import mvesuvio
from mantid.simpleapi import LoadAscii, CompareWorkspaces


class TestHRatioRoutine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        handle_config.refresh_config_dir_and_contents()
        mvesuvio.main(ConfigArgInputs(analysis_inputs="", ip_folder=""))
        copytree(
            handle_config.PACKAGE_CONFIG_PATH / "experiment_template" / "reduction_inputs",
            handle_config.USER_CONFIG_PATH / "experiment_template" / "reduction_inputs",
            dirs_exist_ok=True
            )
        pass

    def setUp(self):
        pass

    def test_h_ratio_routine(self):
        reduction_script = handle_config.USER_CONFIG_PATH / "experiment_template" / "run_reduction.py"
        namespace = runpy.run_path(str(reduction_script), run_name="test_h_ratio_run_reduction")
        namespace["BackwardAnalysisInputs"].run_this_scattering_type = True
        namespace["ForwardAnalysisInputs"].run_this_scattering_type = True
        namespace["BackwardAnalysisInputs"].intensity_ratio_of_hydrogen_to_chosen_mass = 0
        namespace["BackwardAnalysisInputs"].number_of_iterations_for_corrections = 0
        namespace["ForwardAnalysisInputs"].number_of_iterations_for_corrections = 0
        with patch("builtins.input", return_value=""):
            namespace["main"]()

        benchmark_path = Path(__file__).absolute().parent.parent.parent / "data" / "analysis" / "benchmark" / "h_ratio" / "hydrogen_intensity_ratios_estimates"
        result_path = handle_config.USER_CONFIG_PATH / "experiment_template" / "hydrogen_intensity_ratios_estimates"

        bench_name = "bench_h_ratios"
        result_name = "result_h_ratios"

        LoadAscii(str(benchmark_path), Separator="CSV", OutputWorkspace=bench_name)
        LoadAscii(str(result_path), Separator="CSV", OutputWorkspace=result_name)
        (result, _messages) = CompareWorkspaces(bench_name, result_name, Tolerance=1e-3)
        self.assertTrue(result)
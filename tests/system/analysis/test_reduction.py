import runpy
import unittest
from pathlib import Path
from mvesuvio.util import handle_config
from mvesuvio import ConfigArgInputs
from shutil import copytree
import mvesuvio
from mantid.simpleapi import mtd, LoadAscii, AnalysisDataService, CompareWorkspaces, Load


class TestReduction(unittest.TestCase):
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

    def test_reduction_routine(self):
        reduction_script = handle_config.USER_CONFIG_PATH / "experiment_template" / "run_reduction.py"
        namespace = runpy.run_path(str(reduction_script), run_name="test_reduction_run_reduction")
        namespace["BackwardAnalysisInputs"].run_this_scattering_type = True
        namespace["ForwardAnalysisInputs"].run_this_scattering_type = True
        namespace["BackwardAnalysisInputs"].name = "back"
        namespace["ForwardAnalysisInputs"].name = "front"
        namespace["BackwardAnalysisInputs"].number_of_iterations_for_corrections = 0
        namespace["ForwardAnalysisInputs"].number_of_iterations_for_corrections = 1
        namespace["ForwardAnalysisInputs"].mask_of_time_of_flight_range = "110-140"
        namespace["main"]()

        AnalysisDataService.clear()

        benchmark_path = Path(__file__).absolute().parent.parent.parent / "data" / "analysis" / "benchmark" / "reduction"
        results_path = handle_config.USER_CONFIG_PATH / "experiment_template" / "reduction_outputs"

        for prefix, path in zip(("bench", "result"), (benchmark_path, results_path)):
            for p in path.iterdir():
                if p.is_dir():
                    continue
                if p.name.endswith("nxs"):
                    Load(str(p), OutputWorkspace=prefix+"_"+p.stem)
                    continue
                LoadAscii(str(p), Separator="CSV", OutputWorkspace=prefix+"_"+p.stem)

        for ws_name in mtd.getObjectNames():
            if ws_name.startswith('bench'):
                if ws_name.endswith('fit_results'):
                    # Fit results spectra by spectra very too much for comparison
                    # TODO: Find out why and fix it
                    continue
                else:
                    tol = 1e-3
                (result, messages) = CompareWorkspaces(ws_name, ws_name.replace("bench", "result"), Tolerance=tol)
                self.assertTrue(result, f"Comparison failed for workspace: {ws_name}")


if (__name__ == "__main__") or (__name__ == "mantidqt.widgets.codeeditor.execution"):
    unittest.main()

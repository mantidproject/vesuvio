
import runpy
import unittest
from pathlib import Path
from mvesuvio.util import handle_config
from mvesuvio import ConfigArgInputs
from shutil import copytree
import mvesuvio
from mantid.simpleapi import mtd, LoadAscii, AnalysisDataService, CompareWorkspaces, Load


class TestFitting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        handle_config.refresh_config_dir_and_contents()
        mvesuvio.main(ConfigArgInputs(experiment_dir="", ip_dir=""))
        copytree(
            handle_config.PACKAGE_CONFIG_PATH / "experiment_template" / "fitting_inputs",
            handle_config.USER_CONFIG_PATH / "experiment_template" / "fitting_inputs",
            dirs_exist_ok=True
            )
        pass

    def setUp(self):
        pass

    def test_fitting_routine(self):
        fitting_script = handle_config.USER_CONFIG_PATH / "experiment_template" / "run_fitting.py"
        namespace = runpy.run_path(str(fitting_script), run_name="test_fitting_run_fitting")
        namespace["BackwardFittingInputs"].run_this_fitting_type = False
        namespace["ForwardFittingInputs"].run_this_fitting_type = True
        namespace["ForwardFittingInputs"].fitting_model = "gauss"
        namespace["run_fitting"]()

        AnalysisDataService.clear()

        benchmark_path = Path(__file__).absolute().parent.parent.parent / "data" / "analysis" / "benchmark" / "fitting" / "gauss_fit"
        results_path = handle_config.USER_CONFIG_PATH / "experiment_template" / "fitting_outputs" / "gauss_fit"

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
                tol = 1e-3
                (result, messages) = CompareWorkspaces(ws_name, ws_name.replace("bench", "result"), Tolerance=tol)
                self.assertTrue(result, f"Comparison failed for workspace: {ws_name}")
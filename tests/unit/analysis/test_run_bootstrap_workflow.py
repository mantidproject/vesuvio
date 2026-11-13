import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

from mvesuvio.default_config.experiment_template import run_bootstrap
from mvesuvio.util.files_manager import FilesManager


class TestBootstrapWorkflow(unittest.TestCase):
    def tearDown(self):
        FilesManager._experiment_dir = None

    def test_runs_correction_between_reduction_and_fitting(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            back_dir = root / "backward"
            front_dir = root / "forward"
            back_dir.mkdir()
            front_dir.mkdir()
            back_path = back_dir / "back_1.nxs"
            front_path = front_dir / "front_1.nxs"
            back_path.touch()
            front_path.touch()
            FilesManager.set_experiment_dir(root)
            stages = Mock()
            with (
                patch.object(run_bootstrap.bootstrap_helpers, "get_bootstrap_input_directories", return_value=(root, back_dir, front_dir)),
                patch.object(run_bootstrap, "_run_reduction_with_injected_workspaces", stages.reduction),
                patch.object(run_bootstrap, "_run_fse_correction_with_injected_workspaces", stages.correction),
                patch.object(run_bootstrap, "_run_fitting_with_injected_workspaces", stages.fitting),
                patch.object(run_bootstrap.AnalysisDataService, "clear"),
                patch.object(run_bootstrap.plt, "close"),
            ):
                run_bootstrap.run_bootstrap()

            self.assertEqual(
                stages.mock_calls,
                [
                    call.reduction(str(back_path), str(front_path)),
                    call.correction(str(back_path), str(front_path)),
                    call.fitting(str(back_path), str(front_path)),
                ],
            )

    def test_injects_bootstrap_workspace_paths_into_correction(self):
        back_inputs = SimpleNamespace(overwrite_analysis_input_workspace="")
        front_inputs = SimpleNamespace(overwrite_analysis_input_workspace="")
        run_correction = Mock()
        namespace = {
            "BackwardFSEFittingInputs": back_inputs,
            "ForwardFSEFittingInputs": front_inputs,
            "run_fse_correction": run_correction,
        }
        with patch.object(run_bootstrap.runpy, "run_path", return_value=namespace):
            run_bootstrap._run_fse_correction_with_injected_workspaces("back.nxs", "front.nxs")

        self.assertEqual(back_inputs.overwrite_analysis_input_workspace, "back.nxs")
        self.assertEqual(front_inputs.overwrite_analysis_input_workspace, "front.nxs")
        run_correction.assert_called_once_with()

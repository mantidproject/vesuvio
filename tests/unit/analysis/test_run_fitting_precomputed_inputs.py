import builtins
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from mantid.simpleapi import AnalysisDataService, CreateWorkspace, SaveNexus

from mvesuvio.default_config.experiment_template.run_fitting import ForwardFittingInputs, run_fitting, run_y_space_reduction_and_fit
from mvesuvio.util.files_manager import FilesManager


class TestLocalFittingModels(unittest.TestCase):
    def test_package_import_errors_are_not_retried_as_script_imports(self):
        real_import = builtins.__import__

        def import_with_missing_dependency(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "run_reduction" and level == 1:
                raise ImportError("missing reduction dependency")
            return real_import(name, globals, locals, fromlist, level)

        for script in ("run_fitting", "run_fse_correction"):
            with (
                self.subTest(script=script),
                patch("builtins.__import__", side_effect=import_with_missing_dependency),
                patch("importlib.util.spec_from_file_location", side_effect=AssertionError("unexpected script fallback")),
                self.assertRaisesRegex(ImportError, "missing reduction dependency"),
            ):
                runpy.run_module(f"mvesuvio.default_config.experiment_template.{script}", run_name="package_test")


class TestRunFittingPrecomputedInputs(unittest.TestCase):
    def tearDown(self):
        AnalysisDataService.clear()
        FilesManager._experiment_dir = None

    @patch("mvesuvio.default_config.experiment_template.run_fitting.SaveAscii")
    @patch("mvesuvio.default_config.experiment_template.run_fitting.FitInYSpace")
    def test_uses_ads_fitting_workspaces_without_loading_from_disk(self, mock_fit_class, mock_save_ascii):
        with tempfile.TemporaryDirectory() as output_dir:
            FilesManager.set_experiment_dir(output_dir)
            AnalysisDataService.clear()

            ws_prefix = f"{ForwardFittingInputs.name_of_subtracted_workspace}_0_corr"
            CreateWorkspace(
                DataX=[0, 1, 2],
                DataY=[1, 2, 3],
                DataE=[1, 1, 1],
                NSpec=1,
                UnitX="TOF",
                OutputWorkspace=f"{ws_prefix}_ws_resolution",
            )
            CreateWorkspace(
                DataX=[0, 1, 2],
                DataY=[2, 3, 4],
                DataE=[1, 1, 1],
                NSpec=1,
                UnitX="TOF",
                OutputWorkspace=f"{ws_prefix}_joy",
            )
            CreateWorkspace(
                DataX=[0, 1, 2],
                DataY=[3, 4, 5],
                DataE=[1, 1, 1],
                NSpec=1,
                UnitX="TOF",
                OutputWorkspace=f"{ws_prefix}_joy_avg",
            )

            ForwardFittingInputs.number_of_iterations_for_corrections = 0

            with patch("mvesuvio.util.general_helpers.load_saved_fitting_input_workspaces") as mock_load:
                result = run_y_space_reduction_and_fit(ForwardFittingInputs)

            self.assertTrue(result)
            mock_load.assert_not_called()
            mock_fit_class.assert_called_once()
            fit_args = mock_fit_class.call_args.args
            self.assertEqual(fit_args[1].name(), f"{ws_prefix}_joy")
            self.assertEqual(fit_args[2].name(), f"{ws_prefix}_joy_avg")
            self.assertEqual(fit_args[3].name(), f"{ws_prefix}_ws_resolution")
            mock_fit_class.return_value.run.assert_called_once_with()
            mock_save_ascii.assert_called_once_with(
                f"{ws_prefix}_ws_resolution",
                str(Path(output_dir) / "fitting_outputs" / f"{ws_prefix}_ws_resolution"),
            )

    @patch("mvesuvio.default_config.experiment_template.run_fitting.SaveAscii")
    @patch("mvesuvio.default_config.experiment_template.run_fitting.FitInYSpace")
    def test_loads_fitting_workspaces_from_disk_when_missing_from_ads(self, mock_fit_class, mock_save_ascii):
        with tempfile.TemporaryDirectory() as output_dir:
            FilesManager.set_experiment_dir(output_dir)
            AnalysisDataService.clear()

            ws_prefix = "front_0_corr"
            output_directory = FilesManager.get_fse_correction_outputs_dir() / "front_0"
            output_directory.mkdir()
            for suffix in ("ws_resolution", "joy", "joy_avg"):
                workspace = CreateWorkspace(
                    DataX=[0, 1, 2],
                    DataY=[1, 2, 3],
                    DataE=[1, 1, 1],
                    NSpec=1,
                    UnitX="TOF",
                    OutputWorkspace=f"{ws_prefix}_{suffix}",
                )
                SaveNexus(workspace, str(output_directory / f"{ws_prefix}_{suffix}.nxs"))

            AnalysisDataService.clear()

            with patch(
                "mvesuvio.util.general_helpers.get_workspace_name_candidates",
                return_value=["front_0"],
            ):
                result = run_y_space_reduction_and_fit(ForwardFittingInputs)

            self.assertTrue(result)
            mock_fit_class.assert_called_once()
            mock_save_ascii.assert_called_once()
            self.assertTrue(AnalysisDataService.doesExist(f"{ws_prefix}_ws_resolution"))
            self.assertTrue(AnalysisDataService.doesExist(f"{ws_prefix}_joy"))
            self.assertTrue(AnalysisDataService.doesExist(f"{ws_prefix}_joy_avg"))

    @patch("mvesuvio.default_config.experiment_template.run_fitting.SaveAscii")
    @patch("mvesuvio.default_config.experiment_template.run_fitting.FitInYSpace")
    def test_skips_incomplete_candidate_folder_and_loads_next_complete_candidate(self, mock_fit_class, _):
        with tempfile.TemporaryDirectory() as output_dir:
            FilesManager.set_experiment_dir(output_dir)
            correction_outputs = FilesManager.get_fse_correction_outputs_dir()
            for candidate, suffixes in (
                ("first", ("ws_resolution", "joy")),
                ("second", ("ws_resolution", "joy", "joy_avg")),
            ):
                candidate_dir = correction_outputs / candidate
                candidate_dir.mkdir()
                for suffix in suffixes:
                    name = f"{candidate}_corr_{suffix}"
                    workspace = CreateWorkspace(DataX=[0, 1, 2], DataY=[1, 2, 3], NSpec=1, OutputWorkspace=name)
                    SaveNexus(workspace, str(candidate_dir / f"{name}.nxs"))
            AnalysisDataService.clear()

            with patch("mvesuvio.util.general_helpers.get_workspace_name_candidates", return_value=["first", "second"]):
                result = run_y_space_reduction_and_fit(ForwardFittingInputs)

            self.assertTrue(result)
            self.assertEqual(mock_fit_class.call_args.args[1].name(), "second_corr_joy")
            self.assertFalse(AnalysisDataService.doesExist("first_corr_joy"))

    @patch("mvesuvio.default_config.experiment_template.run_fitting.FitInYSpace")
    def test_does_not_combine_incomplete_candidate_folders_or_ads_with_disk(self, mock_fit_class):
        with tempfile.TemporaryDirectory() as output_dir:
            FilesManager.set_experiment_dir(output_dir)
            correction_outputs = FilesManager.get_fse_correction_outputs_dir()
            first_dir = correction_outputs / "first"
            second_dir = correction_outputs / "second"
            first_dir.mkdir()
            second_dir.mkdir()
            for folder, name in (
                (first_dir, "first_corr_ws_resolution"),
                (first_dir, "first_corr_joy"),
                (second_dir, "first_corr_joy_avg"),
                (second_dir, "second_corr_joy_avg"),
            ):
                workspace = CreateWorkspace(DataX=[0, 1, 2], DataY=[1, 2, 3], NSpec=1, OutputWorkspace=name)
                SaveNexus(workspace, str(folder / f"{name}.nxs"))
            AnalysisDataService.clear()
            CreateWorkspace(DataX=[0, 1, 2], DataY=[1, 2, 3], NSpec=1, OutputWorkspace="first_corr_joy_avg")

            with (
                patch("mvesuvio.util.general_helpers.get_workspace_name_candidates", return_value=["first", "second"]),
                patch("mvesuvio.default_config.experiment_template.run_fitting.logger") as mock_logger,
                patch("mvesuvio.util.general_helpers.Load") as mock_load,
            ):
                result = run_y_space_reduction_and_fit(ForwardFittingInputs)

            self.assertFalse(result)
            mock_load.assert_not_called()
            mock_fit_class.assert_not_called()
            mock_logger.warning.assert_called_once()

    @patch("mvesuvio.default_config.experiment_template.run_fitting.FitInYSpace")
    def test_skips_fit_when_precomputed_workspaces_missing(self, mock_fit_class):
        with tempfile.TemporaryDirectory() as output_dir:
            FilesManager.set_experiment_dir(output_dir)
            AnalysisDataService.clear()

            ForwardFittingInputs.name = "front"
            ForwardFittingInputs.number_of_iterations_for_corrections = 0

            with patch("mvesuvio.default_config.experiment_template.run_fitting.logger") as mock_logger:
                result = run_y_space_reduction_and_fit(ForwardFittingInputs)

        self.assertFalse(result)
        mock_fit_class.assert_not_called()
        mock_logger.warning.assert_called_once()

    @patch("mvesuvio.default_config.experiment_template.run_fitting.general_helpers.make_summarised_log_file")
    @patch("mvesuvio.default_config.experiment_template.run_fitting.FitInYSpace")
    def test_run_fitting_rejects_reduction_workspaces_in_ads_without_corrected_files(self, mock_fit_class, _):
        with tempfile.TemporaryDirectory() as output_dir:
            FilesManager.set_experiment_dir(output_dir)
            for suffix in ("ws_resolution", "joy", "joy_avg"):
                CreateWorkspace(
                    DataX=[0, 1, 2],
                    DataY=[1, 2, 3],
                    DataE=[1, 1, 1],
                    NSpec=1,
                    OutputWorkspace=f"front_0_{suffix}",
                )
            with (
                patch.object(ForwardFittingInputs, "name_of_subtracted_workspace", "front"),
                patch.object(ForwardFittingInputs, "overwrite_analysis_input_workspace", ""),
                patch.object(ForwardFittingInputs, "number_of_iterations_for_corrections", 0),
                patch.object(ForwardFittingInputs, "run_this_fitting_type", True),
                patch("mvesuvio.default_config.experiment_template.run_fitting.logger") as logger,
            ):
                run_fitting()

            mock_fit_class.assert_not_called()
            logger.warning.assert_called_once()


if (__name__ == "__main__") or (__name__ == "mantidqt.widgets.codeeditor.execution"):
    unittest.main()

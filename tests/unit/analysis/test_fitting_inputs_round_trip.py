import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np
from mantid.simpleapi import AnalysisDataService, CreateWorkspace, GroupWorkspaces, RenameWorkspace, mtd

from mvesuvio.default_config.experiment_template.run_fitting import load_saved_fitting_input_workspaces
from mvesuvio.default_config.experiment_template.run_fse_correction import (
    BackwardFSEFittingInputs,
    ForwardFSEFittingInputs,
    run_fse_correction,
    run_fse_correction_for_inputs,
    run_fse_fitting,
)
from mvesuvio.default_config.experiment_template.run_reduction import ForwardAnalysisInputs
from mvesuvio.globals import FSEFitWith, WORKSPACE_FSE_SUFFIX
from mvesuvio.util import fitting_helpers, reduction_helpers
from mvesuvio.util.files_manager import FilesManager
from mvesuvio.util.global_fit_builder import FSE_INTENSITY_PARAMETER, GlobalFitWithFSEBuilder
from mvesuvio.util.model_manager import model_manager


class TestFittingInputsRoundTrip(unittest.TestCase):
    def setUp(self):
        self.output_dir = tempfile.TemporaryDirectory()
        FilesManager.set_experiment_dir(self.output_dir.name)
        AnalysisDataService.clear()

    def tearDown(self):
        AnalysisDataService.clear()
        FilesManager._experiment_dir = None
        self.output_dir.cleanup()

    def test_fse_fitting_inputs_override_routine_and_fse_settings(self):
        expected_settings = {
            "run_this_fitting_type",
            "fitting_model",
            "fit_fse_with",
            "initial_fse_width",
            "tie_fse_width_to_model_param",
        }
        self.assertEqual(
            {key for key in vars(BackwardFSEFittingInputs) if not key.startswith("__")},
            expected_settings,
        )
        self.assertEqual(
            {key for key in vars(ForwardFSEFittingInputs) if not key.startswith("__")},
            expected_settings,
        )
        self.assertFalse(BackwardFSEFittingInputs.run_this_fitting_type)
        self.assertTrue(ForwardFSEFittingInputs.run_this_fitting_type)
        self.assertEqual(BackwardFSEFittingInputs.fitting_model, model_manager.name("GAUSS"))
        self.assertEqual(ForwardFSEFittingInputs.fitting_model, model_manager.name("GAUSS"))
        self.assertEqual(BackwardFSEFittingInputs.fit_fse_with, FSEFitWith.FREE_SHARED)
        self.assertEqual(ForwardFSEFittingInputs.fit_fse_with, FSEFitWith.TIED_TO_MODEL)
        for fitting_inputs in (BackwardFSEFittingInputs, ForwardFSEFittingInputs):
            self.assertGreater(fitting_inputs.initial_fse_width, 0)
            self.assertEqual(fitting_inputs.tie_fse_width_to_model_param, "sigma")

    def test_fse_fit_modes_configure_width_as_selected(self):
        x = np.tile(np.linspace(-10, 10, 21), (2, 1))
        y = np.ones_like(x)
        errors = np.ones_like(x)
        ws_joy = Mock()
        ws_joy.name.return_value = "test_joy"
        ws_resolution = Mock()
        ws_q = Mock()
        ws_q.extractY.return_value = y
        ws_fse = Mock()

        for fit_with in FSEFitWith:
            expected_suffix = {
                FSEFitWith.FREE_SHARED: "free_shared",
                FSEFitWith.FREE_UNSHARED: "free_unshared",
                FSEFitWith.TIED_TO_MODEL: "tied",
            }[fit_with]
            self.assertEqual(WORKSPACE_FSE_SUFFIX[fit_with], expected_suffix)
            class FittingInputs(ForwardFSEFittingInputs):
                fitting_model = "GAUSS"
                fit_fse_with = fit_with
                initial_fse_width = 1.0
                tie_fse_width_to_model_param = "sigma"
            with (
                patch(
                    "mvesuvio.default_config.experiment_template.run_fse_correction.extractWS",
                    side_effect=[(x, y, errors), (x, y, errors)],
                ),
                patch(
                    "mvesuvio.default_config.experiment_template.run_fse_correction.minimize",
                    side_effect=lambda _, parameters, **kwargs: SimpleNamespace(x=parameters),
                ),
                patch(
                    "mvesuvio.default_config.experiment_template.run_fse_correction.save_fse_fit_results",
                    return_value=(None, None, ws_fse),
                ) as save_results,
                patch(
                    "mvesuvio.default_config.experiment_template.run_fse_correction.subtract_fse_contribution"
                ) as subtract_fse,
                patch(
                    "mvesuvio.default_config.experiment_template.run_fse_correction."
                    "fitting_helpers.create_table_for_fse_fit_parameters"
                ) as create_parameter_table,
            ):
                builder, _ = run_fse_fitting(ws_resolution, ws_joy, ws_q, FittingInputs, "test_corr_joy")

            self.assertIsInstance(builder, GlobalFitWithFSEBuilder)
            save_results.assert_called_once()
            self.assertEqual(save_results.call_args.args[3], fit_with)
            subtract_fse.assert_called_once_with(ws_joy, ws_fse, "test_corr_joy")
            create_parameter_table.assert_called_once()
            self.assertEqual(create_parameter_table.call_args.args[0], f"test_joy_{expected_suffix}")
            if fit_with is FSEFitWith.FREE_SHARED:
                self.assertIn(FSE_INTENSITY_PARAMETER, builder.parameter_names)
                self.assertNotIn(f"{FSE_INTENSITY_PARAMETER}0", builder.parameter_names)
            elif fit_with is FSEFitWith.FREE_UNSHARED:
                self.assertIn(f"{FSE_INTENSITY_PARAMETER}0", builder.parameter_names)
                self.assertNotIn(FSE_INTENSITY_PARAMETER, builder.parameter_names)
            else:
                self.assertEqual(builder.model_width_parameter, "sigma")
                self.assertNotIn(FSE_INTENSITY_PARAMETER, builder.parameter_names)
            table_args = create_parameter_table.call_args.args
            width_names = [name for name in table_args[2] if name.startswith("fse_width")]
            expected_width_names = {
                FSEFitWith.FREE_SHARED: ["fse_width"],
                FSEFitWith.FREE_UNSHARED: ["fse_width0", "fse_width1"],
                FSEFitWith.TIED_TO_MODEL: [],
            }
            self.assertEqual(width_names, expected_width_names[fit_with])

    @staticmethod
    def _create_resolution_workspace(output_workspace):
        data_x = [(-25 + 0.5 * i) for i in range(101)]
        data_y = [1.0] * 100
        data_e = [1.0] * 100
        return CreateWorkspace(
            DataX=data_x,
            DataY=data_y,
            DataE=data_e,
            NSpec=1,
            UnitX="TOF",
            OutputWorkspace=output_workspace,
        )

    @staticmethod
    def _mock_calculate_resolution(mass, ws, rebin_range):
        fitting_helpers.VesuvioResolution(
            Workspace=ws,
            WorkspaceIndex=0,
            Mass=mass,
            OutputWorkspaceYSpace="tmp",
        )
        return TestFittingInputsRoundTrip._create_resolution_workspace(f"{ws.name()}_resolution")

    def test_save_and_load_fitting_inputs_round_trip(self):
        ws = CreateWorkspace(DataX=[0, 1, 2], DataY=[1, 2, 3], DataE=[1, 1, 1], NSpec=1, UnitX="TOF")
        ws_name = "front_0"
        RenameWorkspace(ws, ws_name)

        ncp_a = CreateWorkspace(DataX=[0, 1, 2], DataY=[1, 2, 3], DataE=[1, 1, 1], NSpec=1, UnitX="TOF")
        RenameWorkspace(ncp_a, f"{ws_name}_1.0079_ncp")

        ncp_b = CreateWorkspace(DataX=[0, 1, 2], DataY=[4, 5, 6], DataE=[1, 1, 1], NSpec=1, UnitX="TOF")
        RenameWorkspace(ncp_b, f"{ws_name}_12_ncp")

        ws_total = ncp_a + ncp_b
        RenameWorkspace(ws_total, f"{ws_name}_total_ncp")

        GroupWorkspaces(
            [f"{ws_name}_1.0079_ncp", f"{ws_name}_12_ncp", f"{ws_name}_total_ncp"],
            OutputWorkspace=f"{ws_name}_ncp_group",
        )

        ForwardAnalysisInputs.run_this_scattering_type = True
        ForwardAnalysisInputs.name_of_subtracted_workspace = "front"
        ForwardAnalysisInputs.overwrite_analysis_input_workspace = ""
        ForwardAnalysisInputs.number_of_iterations_for_corrections = 0
        ForwardAnalysisInputs.subtract_calculated_fse_from_data = False
        ForwardAnalysisInputs.do_symmetrisation = False

        def fake_y_space_reduction(ws_data, ws_ncp, lightest_mass, inputs):
            self.assertEqual(lightest_mass, 1.0079)
            joy = CreateWorkspace(DataX=[0, 1, 2], DataY=[1, 2, 3], DataE=[1, 1, 1], NSpec=1, UnitX="TOF", OutputWorkspace=f"{ws_data.name()}_joy")
            joy_avg = CreateWorkspace(DataX=[0, 1, 2], DataY=[1, 2, 3], DataE=[1, 1, 1], NSpec=1, UnitX="TOF", OutputWorkspace=f"{ws_data.name()}_joy_avg")
            q = CreateWorkspace(DataX=[0, 1, 2], DataY=[1, 2, 3], DataE=[1, 1, 1], NSpec=1, UnitX="TOF", OutputWorkspace=f"{ws_data.name()}_q_rebin")
            return joy, joy_avg, q

        with patch("mvesuvio.util.fitting_helpers.calculate_resolution", side_effect=self._mock_calculate_resolution):
            with patch("mvesuvio.util.fitting_helpers.VesuvioResolution") as resolution_mock:
                with patch.object(reduction_helpers, "y_space_reduction", side_effect=fake_y_space_reduction):
                    reduction_helpers.save_fitting_input_workspaces(ForwardAnalysisInputs)

        resolution_mock.assert_called_once()
        self.assertEqual(resolution_mock.call_args.kwargs["Workspace"].name(), ws_name)

        AnalysisDataService.clear()
        self.assertFalse((FilesManager.get_experiment_dir() / "fitting_inputs").exists())
        self.assertFalse((FilesManager.get_experiment_dir() / "fse_correction_inputs").exists())
        reduction_outputs = FilesManager.get_reduction_outputs_dir()
        self.assertTrue((reduction_outputs / f"{ws_name}_ws_resolution.nxs").exists())
        self.assertTrue((reduction_outputs / f"{ws_name}_joy.nxs").exists())
        self.assertTrue((reduction_outputs / f"{ws_name}_joy_avg.nxs").exists())
        self.assertTrue((reduction_outputs / f"{ws_name}_q.nxs").exists())
        (reduction_outputs / f"{ws_name}.nxs").write_text("not a derived workspace")

        load_saved_fitting_input_workspaces()

        self.assertFalse(AnalysisDataService.doesExist(f"{ws_name}_ws_resolution"))
        self.assertFalse(AnalysisDataService.doesExist(f"{ws_name}_joy"))
        self.assertFalse(AnalysisDataService.doesExist(f"{ws_name}_joy_avg"))
        self.assertFalse(AnalysisDataService.doesExist(f"{ws_name}_q"))

        AnalysisDataService.clear()
        fse_workspace_names = []
        fse_joy_values = []
        with patch(
            "mvesuvio.default_config.experiment_template.run_fse_correction.run_fse_fitting"
        ) as fse_fitting:
            def create_corrected_joy(ws_resolution, ws_joy, ws_q, _, output_workspace):
                fse_workspace_names.append((ws_resolution.name(), ws_joy.name(), ws_q.name()))
                fse_joy_values.append(ws_joy.extractY().copy())
                self.assertTrue(AnalysisDataService.doesExist(f"{ws_name}_joy_avg"))
                np.testing.assert_array_equal(mtd[f"{ws_name}_joy_avg"].extractY(), [[1, 2, 3]])
                self.assertEqual(output_workspace, f"{ws_name}_corr_joy")
                CreateWorkspace(
                    DataX=[0, 1, 2],
                    DataY=[9, 8, 7],
                    DataE=[1, 1, 1],
                    NSpec=1,
                    UnitX="TOF",
                    OutputWorkspace=output_workspace,
                )

            fse_fitting.side_effect = create_corrected_joy
            run_fse_correction()
            run_fse_correction()

        output_directory = FilesManager.get_fse_correction_outputs_dir()
        self.assertEqual(
            {path.name for path in output_directory.glob("*.nxs")},
            {f"{ws_name}_corr_ws_resolution.nxs", f"{ws_name}_corr_joy.nxs", f"{ws_name}_corr_joy_avg.nxs"},
        )
        self.assertEqual(fse_fitting.call_count, 2)
        self.assertEqual(fse_workspace_names, [(f"{ws_name}_ws_resolution", f"{ws_name}_joy", f"{ws_name}_q")] * 2)
        for values in fse_joy_values:
            np.testing.assert_array_equal(values, [[1, 2, 3]])

        np.testing.assert_array_equal(mtd[f"{ws_name}_joy"].extractY(), [[1, 2, 3]])
        np.testing.assert_array_equal(mtd[f"{ws_name}_joy_avg"].extractY(), [[1, 2, 3]])
        self.assertFalse(AnalysisDataService.doesExist(f"{ws_name}_joy_tied_fse_corrected"))
        self.assertTrue(AnalysisDataService.doesExist(f"{ws_name}_corr_ws_resolution"))
        np.testing.assert_array_equal(mtd[f"{ws_name}_corr_joy"].extractY(), [[9, 8, 7]])
        np.testing.assert_array_equal(
            mtd[f"{ws_name}_corr_ws_resolution"].extractY(), mtd[f"{ws_name}_ws_resolution"].extractY()
        )
        AnalysisDataService.clear()
        self.assertEqual(
            load_saved_fitting_input_workspaces(),
            {f"{ws_name}_corr_ws_resolution", f"{ws_name}_corr_joy", f"{ws_name}_corr_joy_avg"},
        )
        self.assertFalse(AnalysisDataService.doesExist(f"{ws_name}_ws_resolution"))
        self.assertFalse(AnalysisDataService.doesExist(f"{ws_name}_joy"))
        self.assertFalse(AnalysisDataService.doesExist(f"{ws_name}_joy_avg"))
        self.assertTrue(AnalysisDataService.doesExist(f"{ws_name}_corr_ws_resolution"))
        self.assertTrue(AnalysisDataService.doesExist(f"{ws_name}_corr_joy"))
        self.assertTrue(AnalysisDataService.doesExist(f"{ws_name}_corr_joy_avg"))
        np.testing.assert_array_equal(mtd[f"{ws_name}_corr_joy"].extractY(), [[9, 8, 7]])
        np.testing.assert_array_equal(mtd[f"{ws_name}_corr_joy_avg"].extractY(), [[9, 8, 7]])

    def test_fse_correction_runs_for_each_available_candidate(self):
        workspaces = {
            "first_ws_resolution": object(),
            "first_joy": object(),
            "first_q": object(),
            "first_corr_joy": object(),
            "second_ws_resolution": object(),
            "second_joy": object(),
            "second_q": object(),
            "second_corr_joy": object(),
        }
        with (
            patch(
                "mvesuvio.util.general_helpers.get_workspace_name_candidates",
                return_value=["first", "second"],
            ),
            patch(
                "mvesuvio.default_config.experiment_template.run_fse_correction.AnalysisDataService",
                Mock(doesExist=workspaces.__contains__),
            ),
            patch("mvesuvio.default_config.experiment_template.run_fse_correction.mtd", workspaces),
            patch(
                "mvesuvio.default_config.experiment_template.run_fse_correction.run_fse_fitting"
            ) as fse_fitting,
            patch(
                "mvesuvio.default_config.experiment_template.run_fse_correction.save_corrected_fitting_inputs"
            ) as save_corrected,
        ):
            run_fse_correction_for_inputs(ForwardFSEFittingInputs)

        self.assertEqual(
            fse_fitting.call_args_list,
            [
                call(workspaces["first_ws_resolution"], workspaces["first_joy"], workspaces["first_q"], ForwardFSEFittingInputs, "first_corr_joy"),
                call(workspaces["second_ws_resolution"], workspaces["second_joy"], workspaces["second_q"], ForwardFSEFittingInputs, "second_corr_joy"),
            ],
        )
        self.assertEqual(
            save_corrected.call_args_list,
            [
                call("first", workspaces["first_ws_resolution"], workspaces["first_corr_joy"]),
                call("second", workspaces["second_ws_resolution"], workspaces["second_corr_joy"]),
            ],
        )
        with (
            patch("mvesuvio.util.general_helpers.get_workspace_name_candidates", return_value=["first"]),
            patch("mvesuvio.default_config.experiment_template.run_fse_correction.AnalysisDataService", Mock(doesExist=workspaces.__contains__)),
            patch("mvesuvio.default_config.experiment_template.run_fse_correction.mtd", workspaces),
            patch("mvesuvio.default_config.experiment_template.run_fse_correction.run_fse_fitting") as fse_fitting,
            patch("mvesuvio.default_config.experiment_template.run_fse_correction.save_corrected_fitting_inputs"),
        ):
            run_fse_correction_for_inputs(
                ForwardFSEFittingInputs, {"first_ws_resolution", "first_joy", "first_q"}
            )
        fse_fitting.assert_called_once()

    def test_fse_correction_does_not_use_workspaces_missing_from_reduction_outputs(self):
        for suffix in ("ws_resolution", "joy", "q"):
            CreateWorkspace(
                DataX=[0, 1, 2], DataY=[1, 2, 3], DataE=[1, 1, 1], NSpec=1, OutputWorkspace=f"front_0_{suffix}"
            )
        with (
            patch.object(ForwardFSEFittingInputs, "name_of_subtracted_workspace", "front"),
            patch.object(ForwardFSEFittingInputs, "overwrite_analysis_input_workspace", ""),
            patch.object(ForwardFSEFittingInputs, "number_of_iterations_for_corrections", 0),
            patch.object(ForwardFSEFittingInputs, "run_this_fitting_type", True),
            patch("mvesuvio.default_config.experiment_template.run_fse_correction.run_fse_fitting") as fse_fitting,
            patch("mvesuvio.default_config.experiment_template.run_fse_correction.logger") as logger,
            patch("mvesuvio.default_config.experiment_template.run_fse_correction.general_helpers.make_summarised_log_file"),
        ):
            run_fse_correction()

        fse_fitting.assert_not_called()
        logger.warning.assert_called_once()

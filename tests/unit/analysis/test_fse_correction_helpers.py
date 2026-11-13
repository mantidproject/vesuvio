import unittest
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np

from mvesuvio.util.fse_correction_helpers import FSEFitWith, save_fse_fit_results, subtract_fse_contribution


class TestFSECorrectionHelpers(unittest.TestCase):
    def test_fse_result_workspace_names_include_fit_mode(self):
        ws_joy = Mock()
        ws_joy.name.return_value = "test_joy"
        builder = SimpleNamespace(
            convolved_model=Mock(return_value=np.zeros((1, 21))),
            fit_curve_errors=Mock(return_value=np.zeros((1, 21))),
            valid=np.ones((1, 21), dtype=bool),
            data_y=np.zeros((1, 21)),
            data_x=np.zeros((1, 21)),
            fse_contribution=Mock(return_value=np.zeros((1, 21))),
            fse_contribution_errors=Mock(return_value=np.zeros((1, 21))),
        )
        clones = []

        def clone_workspace(_, OutputWorkspace):
            clone = Mock()
            clone.name.return_value = OutputWorkspace
            clones.append(OutputWorkspace)
            return clone

        corrected = Mock()
        corrected.name.return_value = "test_joy_free_shared_fse_corrected"
        with (
            patch("mvesuvio.util.fse_correction_helpers.CloneWorkspace", side_effect=clone_workspace),
            patch("mvesuvio.util.fse_correction_helpers.SumSpectra") as sum_spectra,
            patch("mvesuvio.util.fse_correction_helpers.general_helpers.pass_data_into_ws"),
            patch("mvesuvio.util.fse_correction_helpers.Minus", return_value=corrected) as minus,
        ):
            ws_fit, ws_residuals, ws_fse = save_fse_fit_results(
                ws_joy, builder, SimpleNamespace(x=np.array([1.0])), FSEFitWith.FREE_SHARED, np.array([[0.1]])
            )
            ws_corrected = subtract_fse_contribution(ws_joy, ws_fse, FSEFitWith.FREE_SHARED)

        self.assertEqual(
            clones,
            [
                "test_joy_free_shared_fse_fit",
                "test_joy_free_shared_fse_residuals",
                "test_joy_free_shared_fse_contribution",
            ],
        )
        self.assertEqual(
            [summed.kwargs["OutputWorkspace"] for summed in sum_spectra.call_args_list],
            [
                "test_joy_sum",
                "test_joy_free_shared_fse_fit_sum",
                "test_joy_free_shared_fse_residuals_sum",
                "test_joy_free_shared_fse_contribution_sum",
                "test_joy_free_shared_fse_corrected_sum",
            ],
        )
        self.assertIs(ws_fit, sum_spectra.call_args_list[1].kwargs["InputWorkspace"])
        self.assertIs(ws_residuals, sum_spectra.call_args_list[2].kwargs["InputWorkspace"])
        self.assertIs(ws_fse, sum_spectra.call_args_list[3].kwargs["InputWorkspace"])
        self.assertIs(ws_corrected, corrected)
        minus.assert_called_once_with(
            LHSWorkspace=ws_joy,
            RHSWorkspace=ws_fse,
            OutputWorkspace="test_joy_free_shared_fse_corrected",
        )
        self.assertEqual(sum_spectra.call_args_list[-1], call(InputWorkspace=corrected, OutputWorkspace="test_joy_free_shared_fse_corrected_sum"))

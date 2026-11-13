import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call, patch

import numpy as np
from mantid.simpleapi import AnalysisDataService, CreateWorkspace, mtd

from mvesuvio.globals import FSEFitWith
from mvesuvio.util.fse_correction_helpers import (
    save_fse_corrected_outputs,
    save_fse_fit_results,
    subtract_fse_contribution,
)


class TestFSECorrectionHelpers(unittest.TestCase):
    def tearDown(self):
        AnalysisDataService.clear()

    def test_saves_corrected_fitting_outputs(self):
        ws_resolution = Mock()
        ws_corrected_joy = Mock()
        ws_corrected_resolution = Mock()
        ws_corrected_resolution.name.return_value = "candidate_corr_ws_resolution"
        ws_corrected_joy.name.return_value = "candidate_corr_joy"
        ws_joy_avg = Mock()
        ws_joy_avg.name.return_value = "candidate_corr_joy_avg"
        output_directory = Path("/tmp/fse-correction")
        with (
            patch("mvesuvio.util.fse_correction_helpers.FilesManager.get_fse_correction_outputs_dir", return_value=output_directory),
            patch("mvesuvio.util.fse_correction_helpers.CloneWorkspace", return_value=ws_corrected_resolution) as clone,
            patch("mvesuvio.util.fse_correction_helpers.SumSpectra", return_value=ws_joy_avg) as sum_spectra,
            patch("mvesuvio.util.fse_correction_helpers.SaveNexus") as save_nexus,
        ):
            save_fse_corrected_outputs("candidate", ws_resolution, ws_corrected_joy)

        clone.assert_called_once_with(
            InputWorkspace=ws_resolution,
            OutputWorkspace="candidate_corr_ws_resolution",
        )
        sum_spectra.assert_called_once_with(
            InputWorkspace=ws_corrected_joy,
            WeightedSum=True,
            MultiplyBySpectra=False,
            OutputWorkspace="candidate_corr_joy_avg",
        )
        self.assertEqual(
            save_nexus.call_args_list,
            [
                call(InputWorkspace=ws_corrected_resolution, Filename=f"{output_directory}/candidate_corr_ws_resolution.nxs"),
                call(InputWorkspace=ws_corrected_joy, Filename=f"{output_directory}/candidate_corr_joy.nxs"),
                call(InputWorkspace=ws_joy_avg, Filename=f"{output_directory}/candidate_corr_joy_avg.nxs"),
            ],
        )

    def test_fse_result_workspace_names_include_fit_mode(self):
        ws_joy = Mock()
        ws_joy.name.return_value = "test_joy"
        builder = SimpleNamespace(
            convolved_model=Mock(return_value=np.zeros((1, 21))),
            fit_curve_errors=Mock(return_value=np.zeros((1, 21))),
            valid=np.ones((1, 21), dtype=bool),
            data_y=np.zeros((1, 21)),
            data_x=np.zeros((1, 21)),
            fse_contribution=Mock(return_value=np.ones((1, 21))),
        )
        clones = []

        def clone_workspace(_, OutputWorkspace):
            clone = Mock()
            clone.name.return_value = OutputWorkspace
            clones.append(OutputWorkspace)
            return clone

        corrected = Mock()
        corrected.name.return_value = "test_corr_joy"
        ws_joy.extractY.return_value = np.ones((1, 21))
        corrected_data = np.ones((1, 21))
        corrected.extractY.return_value = corrected_data
        corrected.getNumberHistograms.return_value = 1
        corrected.dataY.side_effect = lambda index: corrected_data[index]
        with (
            patch("mvesuvio.util.fse_correction_helpers.CloneWorkspace", side_effect=clone_workspace),
            patch("mvesuvio.util.fse_correction_helpers.SumSpectra") as sum_spectra,
            patch("mvesuvio.util.fse_correction_helpers.general_helpers.pass_data_into_ws") as pass_data,
            patch("mvesuvio.util.fse_correction_helpers.Minus", return_value=corrected) as minus,
        ):
            ws_fit, ws_residuals, ws_fse = save_fse_fit_results(
                ws_joy, builder, SimpleNamespace(x=np.array([1.0])), FSEFitWith.FREE_SHARED, np.array([[0.1]])
            )
            ws_corrected = subtract_fse_contribution(ws_joy, ws_fse, "test_corr_joy")

        np.testing.assert_array_equal(pass_data.call_args_list[2].args[2], np.zeros((1, 21)))
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
                "test_corr_joy_sum",
            ],
        )
        self.assertIs(ws_fit, sum_spectra.call_args_list[1].kwargs["InputWorkspace"])
        self.assertIs(ws_residuals, sum_spectra.call_args_list[2].kwargs["InputWorkspace"])
        self.assertIs(ws_fse, sum_spectra.call_args_list[3].kwargs["InputWorkspace"])
        self.assertIs(ws_corrected, corrected)
        minus.assert_called_once_with(
            LHSWorkspace=ws_joy,
            RHSWorkspace=ws_fse,
            OutputWorkspace="test_corr_joy",
        )
        self.assertEqual(sum_spectra.call_args_list[-1], call(InputWorkspace=corrected, OutputWorkspace="test_corr_joy_sum"))

    def test_subtraction_creates_final_workspace_without_replacing_reduction_data(self):
        ws_joy = CreateWorkspace(DataX=[0, 1, 2], DataY=[4, 0, 6], DataE=[1, 1, 1], NSpec=1, OutputWorkspace="front_0_joy")
        ws_fse = CreateWorkspace(DataX=[0, 1, 2], DataY=[1, 2, 3], DataE=[0, 0, 0], NSpec=1, OutputWorkspace="front_0_fse")
        corrected = subtract_fse_contribution(ws_joy, ws_fse, "front_0_corr_joy")

        self.assertEqual(corrected.name(), "front_0_corr_joy")
        self.assertTrue(AnalysisDataService.doesExist("front_0_corr_joy_sum"))
        self.assertFalse(AnalysisDataService.doesExist("front_0_joy_tied_fse_corrected"))
        np.testing.assert_array_equal(corrected.extractY(), [[3, 0, 3]])
        np.testing.assert_array_equal(corrected.extractE(), [[1, 1, 1]])
        np.testing.assert_array_equal(mtd["front_0_joy"].extractY(), [[4, 0, 6]])

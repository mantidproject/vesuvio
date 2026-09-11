import tempfile
import unittest
from unittest.mock import patch

from mantid.simpleapi import AnalysisDataService, CreateWorkspace, GroupWorkspaces, RenameWorkspace

from mvesuvio.default_config.experiment_template.run_fitting import load_saved_fitting_input_workspaces
from mvesuvio.default_config.experiment_template.run_reduction import ForwardAnalysisInputs, save_fitting_input_workspaces
from mvesuvio.util import fitting_helpers
from mvesuvio.util.files_manager import FilesManager


class TestFittingInputsRoundTrip(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

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
        output_dir = tempfile.TemporaryDirectory()
        FilesManager.set_experiment_dir(output_dir.name)
        AnalysisDataService.clear()

        try:
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
            ForwardAnalysisInputs.name = "front"
            ForwardAnalysisInputs.number_of_iterations_for_corrections = 0
            ForwardAnalysisInputs.subtract_calculated_fse_from_data = False

            with patch("mvesuvio.util.fitting_helpers.calculate_resolution", side_effect=self._mock_calculate_resolution):
                with patch("mvesuvio.util.fitting_helpers.VesuvioResolution") as resolution_mock:
                    save_fitting_input_workspaces()

            resolution_mock.assert_called_once()
            self.assertEqual(resolution_mock.call_args.kwargs["Workspace"].name(), ws_name)

            AnalysisDataService.clear()
            self.assertTrue((FilesManager.get_fitting_inputs_dir() / f"{ws_name}_ws_resolution.nxs").exists())
            self.assertTrue((FilesManager.get_fitting_inputs_dir() / f"{ws_name}_ws_lighest_data.nxs").exists())
            self.assertTrue((FilesManager.get_fitting_inputs_dir() / f"{ws_name}_ws_lighest_ncp.nxs").exists())

            load_saved_fitting_input_workspaces()

            self.assertTrue(AnalysisDataService.doesExist(f"{ws_name}_ws_resolution"))
            self.assertTrue(AnalysisDataService.doesExist(f"{ws_name}_ws_lighest_data"))
            self.assertTrue(AnalysisDataService.doesExist(f"{ws_name}_ws_lighest_ncp"))
        finally:
            AnalysisDataService.clear()
            FilesManager._experiment_dir = None
            output_dir.cleanup()

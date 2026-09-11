
import os
import tempfile
import unittest
from pathlib import Path
from textwrap import dedent
from unittest.mock import patch, Mock, MagicMock, call

import numpy as np

from mvesuvio import globals
from mvesuvio.util import reduction_helpers
from mantid.simpleapi import CreateWorkspace, DeleteWorkspace


class TestReductionHelpers(unittest.TestCase):

    def test_extract_ws(self):
        data = [1, 2, 3]
        ws = CreateWorkspace(DataX=data, DataY=data, DataE=data, NSpec=1, UnitX="some_unit")

        dataX, dataY, dataE = reduction_helpers.extractWS(ws)
        np.testing.assert_array_equal([data], dataX)
        np.testing.assert_array_equal([data], dataY)
        np.testing.assert_array_equal([data], dataE)

        DeleteWorkspace(ws)

    def test_convert_dict_to_table(self):
        d = {'H': {'label': 'H', 'mass': 1, 'intensity': 1}}
        table = reduction_helpers._convert_dict_to_table(d)
        self.assertEqual(['label', 'mass', 'intensity'], table.getColumnNames())
        self.assertEqual({'label': 'H', 'mass': 1, 'intensity': 1}, table.row(0))

    def test_mask_time_of_flight_bins_with_zeros(self):
        data_x = np.arange(10).reshape(1, -1) * np.ones((3, 1))
        data_y = np.ones((3, 10))
        data_e = np.ones((3, 10))
        workspace_mock = MagicMock()
        workspace_mock.extractX.return_value = data_x
        workspace_mock.extractY.return_value = data_y
        workspace_mock.extractE.return_value = data_e

        actual_data_x = np.zeros((3, 10))
        actual_data_y = np.zeros((3, 10))
        actual_data_e = np.zeros((3, 10))

        workspace_mock.dataY.side_effect = lambda i: actual_data_y[i]
        workspace_mock.dataX.side_effect = lambda i: actual_data_x[i]
        workspace_mock.dataE.side_effect = lambda i: actual_data_e[i]

        workspace_mock.getNumberHistograms.return_value = 3
        reduction_helpers.mask_time_of_flight_bins_with_zeros(workspace_mock, '4.5-7.3')

        np.testing.assert_allclose(actual_data_x, data_x)
        np.testing.assert_allclose(actual_data_e, data_e)
        expected_data_y = np.ones((3, 10))
        expected_data_y[(data_x >= 4.5) & (data_x <= 7.3)] = 0
        np.testing.assert_allclose(actual_data_y, expected_data_y)


    def test_mask_time_of_flight_bins_with_zeros_several_ranges(self):
        data_x = np.arange(0, 10, 0.5).reshape(1, -1) * np.ones((3, 1))
        data_y = np.ones_like(data_x)
        data_e = np.ones_like(data_x)
        workspace_mock = MagicMock()
        workspace_mock.extractX.return_value = data_x
        workspace_mock.extractY.return_value = data_y
        workspace_mock.extractE.return_value = data_e

        actual_data_x = np.zeros_like(data_x)
        actual_data_y = np.zeros_like(data_x)
        actual_data_e = np.zeros_like(data_x)

        workspace_mock.dataY.side_effect = lambda i: actual_data_y[i]
        workspace_mock.dataX.side_effect = lambda i: actual_data_x[i]
        workspace_mock.dataE.side_effect = lambda i: actual_data_e[i]

        workspace_mock.getNumberHistograms.return_value = 3

        reduction_helpers.mask_time_of_flight_bins_with_zeros(workspace_mock, '1.1-3,3.5,  4.5-7.3, 9')

        np.testing.assert_allclose(actual_data_x, data_x)
        np.testing.assert_allclose(actual_data_e, data_e)
        expected_data_y = np.ones_like(data_x)
        expected_data_y[(data_x >= 1.1) & (data_x <= 3)] = 0
        expected_data_y[(data_x == 3.5)] = 0
        expected_data_y[(data_x >= 4.5) & (data_x <= 7.3)] = 0
        expected_data_y[(data_x == 9)] = 0
        np.testing.assert_allclose(actual_data_y, expected_data_y)

    def test_fix_profile_parameters_with_H(self):
        means_table_mock = MagicMock()
        means_table_mock.rowCount.return_value = 3
        means_table_mock.row.side_effect = [
            {"label": "16.0", "mass": 16.0, "mean_width": 8.974, "std_width": 1.401, "mean_intensity": 0.176, "std_intensity": 0.08722},
            {"label": "27.0", "mass": 27.0, "mean_width": 15.397, "std_width": 1.131, "mean_intensity": 0.305, "std_intensity": 0.04895},
            {"label": "12.0", "mass": 12.0, "mean_width": 13.932, "std_width": 0.314, "mean_intensity": 0.517, "std_intensity": 0.09531},
        ]
        profiles_table_mock = MagicMock()
        profiles_table_mock.rowCount.return_value = 4
        profiles_table_mock.row.side_effect = [
            {"label": "1.0079", "mass": 1.0078, "intensity": 1.0, "intensity_lb": 0, "intensity_ub": np.inf, "width": 4.699, "width_lb": 3, "width_ub": 6, "center": 0.0, "center_lb": -3, "center_ub": 1},
            {"label": "16.0", "mass": 16.0, "intensity": 1.0, "intensity_lb": 0, "intensity_ub": np.inf, "width": 12, "width_lb": 0, "width_ub": np.inf, "center": 0.0, "center_lb": -3, "center_ub": 1},
            {"label": "12.0", "mass": 12.0, "intensity": 1.0, "intensity_lb": 0, "intensity_ub": np.inf, "width": 8, "width_lb": 0, "width_ub": np.inf, "center": 0.0, "center_lb": -3, "center_ub": 1},
            {"label": "27.0", "mass": 27.0, "intensity": 1.0, "intensity_lb": 0, "intensity_ub": np.inf, "width": 13, "width_lb": 0, "width_ub": np.inf, "center": 0.0, "center_lb": -3, "center_ub": 1},
        ]

        result_table = reduction_helpers.fix_profile_parameters(means_table_mock, profiles_table_mock, h_ratio=14.7)
        self.assertEqual(
            result_table.row(0),
            {"label": "1.0079", "mass": 1.0077999830245972, "intensity": 0.8839251399040222, "intensity_lb": 0.0, "intensity_ub": np.inf, "width": 4.698999881744385, "width_lb": 3.0, "width_ub": 6.0, "center": 0.0, "center_lb": -3.0, "center_ub": 1.0},
        )

    def test_fix_profile_parameters_without_H(self):
        means_table_mock = MagicMock()
        means_table_mock.rowCount.return_value = 3
        means_table_mock.row.side_effect = [
            {"label": "16.0", "mass": 16.0, "mean_width": 8.974, "std_width": 1.401, "mean_intensity": 0.176, "std_intensity": 0.08722},
            {"label": "27.0", "mass": 27.0, "mean_width": 15.397, "std_width": 1.131, "mean_intensity": 0.305, "std_intensity": 0.04895},
            {"label": "12.0", "mass": 12.0, "mean_width": 13.932, "std_width": 0.314, "mean_intensity": 0.517, "std_intensity": 0.09531},
        ]
        profiles_table_mock = MagicMock()
        profiles_table_mock.rowCount.return_value = 3
        profiles_table_mock.row.side_effect = [
            {"label": "16.0", "mass": 16.0, "intensity": 1.0, "intensity_lb": 0, "intensity_ub": np.inf, "width": 12, "width_lb": 0, "width_ub": np.inf, "center": 0.0, "center_lb": -3, "center_ub": 1},
            {"label": "12.0", "mass": 12.0, "intensity": 1.0, "intensity_lb": 0, "intensity_ub": np.inf, "width": 8, "width_lb": 0, "width_ub": np.inf, "center": 0.0, "center_lb": -3, "center_ub": 1},
            {"label": "27.0", "mass": 27.0, "intensity": 1.0, "intensity_lb": 0, "intensity_ub": np.inf, "width": 13, "width_lb": 0, "width_ub": np.inf, "center": 0.0, "center_lb": -3, "center_ub": 1},
        ]

        result_table = reduction_helpers.fix_profile_parameters(means_table_mock, profiles_table_mock, h_ratio=14.7)
        self.assertEqual(result_table.row(0)["label"], "16.0")
        self.assertEqual(result_table.row(0)["intensity"], 0.17635270953178406)

    def test_calculate_h_ratio(self):
        means_table_mock = MagicMock()
        means_table_mock.column.side_effect = lambda x: [16, 1, 12] if x == "mass" else [0.1, 0.85, 0.05]
        h_ratio = reduction_helpers.calculate_h_ratio(means_table_mock, 12)
        self.assertEqual(h_ratio, 0.85 / 0.05)

    def test_is_hydrogen_present_bad_inputs(self):
        with self.assertRaises(AssertionError):
            reduction_helpers.is_hydrogen_present(np.array([1.01]))
        with self.assertRaises(AssertionError):
            reduction_helpers.is_hydrogen_present(np.array([2.0, 1.0, 12.0]))
        with self.assertRaises(AssertionError):
            reduction_helpers.is_hydrogen_present(np.array([1.0, 1.0078, 12.0]))

    def test_create_profiles_table(self):

        mock_ai = Mock()
        mock_ai.masses = [1, 12, 16]
        mock_ai.initial_fitting_parameters = [1, 5, 0, 1, 10, 0, 1, 13, 0]
        mock_ai.fitting_bounds = [[0, None], [2, 6], [-1, 3], [0, None], [8, 12], [-1, 3], [0, np.inf], [11, 15], [-1, 3]]

        with patch('mvesuvio.util.reduction_helpers.CreateEmptyTableWorkspace') as mock_create_table_ws:
            table_mock = MagicMock()
            mock_create_table_ws.return_value = table_mock

            reduction_helpers.create_profiles_table(table_mock, mock_ai)

            table_mock.addColumn.assert_has_calls([
                call(type='str', name='label'),
                call(type='float', name='mass'),
                call(type='float', name='intensity'),
                call(type='float', name='intensity_lb'),
                call(type='float', name='intensity_ub'),
                call(type='float', name='width'),
                call(type='float', name='width_lb'),
                call(type='float', name='width_ub'),
                call(type='float', name='center'),
                call(type='float', name='center_lb'),
                call(type='float', name='center_ub')
            ])
            table_mock.addRow.assert_has_calls([
                call(['1.0', 1.0, 1.0, 0.0, np.inf, 5.0, 2.0, 6.0, 0.0, -1.0, 3.0]),
                call(['12.0', 12.0, 1.0, 0.0, np.inf, 10.0, 8.0, 12.0, 0.0, -1.0, 3.0]),
                call(['16.0', 16.0, 1.0, 0.0, np.inf, 13.0, 11.0, 15.0, 0.0, -1.0, 3.0])
            ])

    def test_is_hydrogen_present_with_hydrogen(self):
        masses = np.array([1.0078, 12.0, 16.0])
        is_present = reduction_helpers.is_hydrogen_present(masses)
        self.assertTrue(is_present)


    def test_is_hydrogen_present_without_hydrogen(self):
        masses = np.array([2.0, 12.0, 16.0])
        is_present = reduction_helpers.is_hydrogen_present(masses)
        self.assertFalse(is_present)

    def test_is_hydrogen_present_one_mass_no_hydrogen(self):
        masses = np.array([2.0])
        is_present = reduction_helpers.is_hydrogen_present(masses)
        self.assertFalse(is_present)

    def test_ws_history_matches_inputs_invalid_path(self):
        path = Path("notthere.nxs")
        with patch('mvesuvio.util.reduction_helpers.logger') as mock_logger:
            match = reduction_helpers.ws_history_matches_inputs(0, 0, 0, path)
            mock_logger.notice.assert_has_calls([call('Cached workspace not found at notthere.nxs')])
            self.assertFalse(match)


    @patch('mvesuvio.util.reduction_helpers.Load')
    def test_ws_history_matches_inputs_bad_runs(self, mock_load):
        path = Mock()
        path.is_file.return_value = True
        props = {
            "Filename": "1234-1235",
            "Mode": "SingleDifference",
            "InstrumentParFile": "ip_par.txt"
        }
        mock_metadata = Mock()
        mock_metadata.getPropertyValue.side_effect = lambda key: props[key]
        mock_history = Mock()
        mock_history.getAlgorithmHistory.return_value = mock_metadata
        mock_ws = Mock()
        mock_ws.getHistory.return_value = mock_history
        mock_load.return_value = mock_ws

        with patch('mvesuvio.util.reduction_helpers.logger') as mock_logger:
            match = reduction_helpers.ws_history_matches_inputs("0000", "SingleDifference", "ip_par.txt", path)
            mock_logger.notice.assert_has_calls([call('Filename in saved workspace did not match: 1234-1235 and 0000')])
            self.assertFalse(match)


    @patch('mvesuvio.util.reduction_helpers.Load')
    def test_ws_history_matches_inputs_bad_mode(self, mock_load):
        path = Mock()
        path.is_file.return_value = True
        props = {
            "Filename": "1234-1235",
            "Mode": "SingleDifference",
            "InstrumentParFile": "ip_par.txt"
        }
        mock_metadata = Mock()
        mock_metadata.getPropertyValue.side_effect = lambda key: props[key]
        mock_history = Mock()
        mock_history.getAlgorithmHistory.return_value = mock_metadata
        mock_ws = Mock()
        mock_ws.getHistory.return_value = mock_history
        mock_load.return_value = mock_ws

        with patch('mvesuvio.util.reduction_helpers.logger') as mock_logger:
            match = reduction_helpers.ws_history_matches_inputs("1234-1235", "DoubleDifference", "ip_par.txt", path)
            mock_logger.notice.assert_has_calls([call('Mode in saved workspace did not match: SingleDifference and DoubleDifference')])
            self.assertFalse(match)

    @patch('mvesuvio.util.reduction_helpers.Load')
    def test_ws_history_matches_inputs_bad_ipfile(self, mock_load):
        path = Mock()
        path.is_file.return_value = True
        props = {
            "Filename": "1234-1235",
            "Mode": "SingleDifference",
            "InstrumentParFile": "ip_par.txt"
        }
        mock_metadata = Mock()
        mock_metadata.getPropertyValue.side_effect = lambda key: props[key]
        mock_history = Mock()
        mock_history.getAlgorithmHistory.return_value = mock_metadata
        mock_ws = Mock()
        mock_ws.getHistory.return_value = mock_history
        mock_load.return_value = mock_ws

        with patch('mvesuvio.util.reduction_helpers.logger') as mock_logger:
            match = reduction_helpers.ws_history_matches_inputs("1234-1235", "SingleDifference", "new_par.txt", path)
            mock_logger.notice.assert_has_calls([call('IP files in saved workspace did not match: ip_par.txt and new_par.txt')])
            self.assertFalse(match)

    @patch('mvesuvio.util.reduction_helpers.DeleteWorkspace')
    @patch('mvesuvio.util.reduction_helpers.Load')
    def test_ws_history_matches_good_inputs(self, mock_load, mock_delete):
        path = Mock()
        path.is_file.return_value = True
        props = {
            "Filename": "1234-1235",
            "Mode": "SingleDifference",
            "InstrumentParFile": "ip_par.txt"
        }
        mock_metadata = Mock()
        mock_metadata.getPropertyValue.side_effect = lambda key: props[key]
        mock_history = Mock()
        mock_history.getAlgorithmHistory.return_value = mock_metadata
        mock_ws = Mock()
        mock_ws.getHistory.return_value = mock_history
        mock_load.return_value = mock_ws

        with patch('mvesuvio.util.reduction_helpers.logger') as mock_logger:
            match = reduction_helpers.ws_history_matches_inputs("1234-1235", "SingleDifference", "ip_par.txt", path)
            mock_logger.notice.assert_has_calls([call('\nLocally saved workspace metadata matched with analysis inputs.\n')])
            self.assertTrue(match)


    @patch('mvesuvio.util.reduction_helpers.SaveNexus')
    @patch('mvesuvio.util.reduction_helpers.LoadVesuvio')
    def test_save_ws_from_load_vesuvio_backward(self, mock_load_vesuvio, mock_save_nexus):
        path = Path(f'notthere/raw_{globals.BACKWARD_TAG}.nxs')
        reduction_helpers.save_ws_from_load_vesuvio("1234", "SingleDifference", "ipfile.txt", path)
        mock_load_vesuvio.assert_has_calls([
            call(Filename='1234', SpectrumList='3-134', Mode='SingleDifference', InstrumentParFile='ipfile.txt', OutputWorkspace=f'raw_{globals.BACKWARD_TAG}.nxs', LoadLogFiles=False)
        ])
        args, kwargs = mock_save_nexus.call_args
        self.assertEqual(kwargs["Filename"], str(path.absolute()))


    @patch('mvesuvio.util.reduction_helpers.SaveNexus')
    @patch('mvesuvio.util.reduction_helpers.LoadVesuvio')
    def test_save_ws_from_load_vesuvio_forward(self, mock_load_vesuvio, mock_save_nexus):
        path = Path(f'notthere/raw_{globals.FORWARD_TAG}.nxs')
        reduction_helpers.save_ws_from_load_vesuvio("1234", "SingleDifference", "ipfile.txt", path)
        mock_load_vesuvio.assert_has_calls([
            call(Filename='1234', SpectrumList="135-198", Mode='SingleDifference', InstrumentParFile='ipfile.txt', OutputWorkspace=f'raw_{globals.FORWARD_TAG}.nxs', LoadLogFiles=False)
        ])
        args, kwargs = mock_save_nexus.call_args
        self.assertEqual(kwargs["Filename"], str(path.absolute()))


    def test_load_and_save_input_ws_if_not_on_path_backward(self):
        class BackwardInputs:
            name = "backward"
            mode = "SingleDifference"
            runs = "1234"
            empty_runs = "5678"
            instrument_parameters_file = "ipfile.txt"

        reduction_inputs_dir = Path("/tmp/reduction_inputs")
        raw_filename = "experiment_raw_backward.nxs"
        empty_filename = "experiment_empty_backward.nxs"
        raw_path = reduction_inputs_dir / raw_filename
        empty_path = reduction_inputs_dir / empty_filename

        with patch.object(reduction_helpers.FilesManager, "get_reduction_inputs_dir", return_value=reduction_inputs_dir), \
            patch.object(reduction_helpers.FilesManager, "get_backward_raw_filename", return_value=raw_filename), \
            patch.object(reduction_helpers.FilesManager, "get_backward_empty_filename", return_value=empty_filename), \
            patch.object(reduction_helpers.FilesManager, "get_instrument_parameters_dir", return_value=Path("/tmp/ip")), \
            patch.object(reduction_helpers, "ws_history_matches_inputs", side_effect=[False, True]) as mock_matches, \
            patch.object(reduction_helpers, "save_ws_from_load_vesuvio") as mock_save_ws:
            result_raw_path, result_empty_path = reduction_helpers.load_and_save_input_ws_if_not_on_path(BackwardInputs)

        self.assertEqual(result_raw_path, raw_path)
        self.assertEqual(result_empty_path, empty_path)
        mock_matches.assert_has_calls([
            call("1234", "SingleDifference", "ipfile.txt", raw_path),
            call("5678", "SingleDifference", "ipfile.txt", empty_path),
        ])
        mock_save_ws.assert_has_calls([
            call("1234", "SingleDifference", "/tmp/ip/ipfile.txt", raw_path),
        ])


    def test_load_and_save_input_ws_if_not_on_path_forward_uses_cached_files(self):
        class ForwardInputs:
            name = "forward"
            mode = "SingleDifference"
            runs = "2345"
            empty_runs = "6789"
            instrument_parameters_file = "ipfile.txt"

        reduction_inputs_dir = Path("/tmp/reduction_inputs")
        raw_filename = "experiment_raw_forward.nxs"
        empty_filename = "experiment_empty_forward.nxs"
        raw_path = reduction_inputs_dir / raw_filename
        empty_path = reduction_inputs_dir / empty_filename

        with patch.object(reduction_helpers.FilesManager, "get_reduction_inputs_dir", return_value=reduction_inputs_dir), \
            patch.object(reduction_helpers.FilesManager, "get_forward_raw_filename", return_value=raw_filename), \
            patch.object(reduction_helpers.FilesManager, "get_forward_empty_filename", return_value=empty_filename), \
            patch.object(reduction_helpers.FilesManager, "get_instrument_parameters_dir", return_value=Path("/tmp/ip")), \
            patch.object(reduction_helpers, "ws_history_matches_inputs", return_value=True) as mock_matches, \
            patch.object(reduction_helpers, "save_ws_from_load_vesuvio") as mock_save_ws:
            result_raw_path, result_empty_path = reduction_helpers.load_and_save_input_ws_if_not_on_path(ForwardInputs)

        self.assertEqual(result_raw_path, raw_path)
        self.assertEqual(result_empty_path, empty_path)
        mock_matches.assert_has_calls([
            call("2345", "SingleDifference", "ipfile.txt", raw_path),
            call("6789", "SingleDifference", "ipfile.txt", empty_path),
        ])
        mock_save_ws.assert_not_called()

    def test_convert_to_list_of_spectrum_numbers_string(self):
        res = reduction_helpers.convert_to_list_of_spectrum_numbers("1, 3-6, 8-8, 10")
        self.assertEqual(res, [1, 3, 4, 5, 6, 8, 10])


    def test_convert_to_list_of_spectrum_numbers_list_mixed(self):
        res = reduction_helpers.convert_to_list_of_spectrum_numbers(["1", 3, 5, 6, "7"])
        self.assertEqual(res, [1, 3, 5, 6, 7])


    def test_convert_to_list_of_spectrum_numbers_list_integers(self):
        res = reduction_helpers.convert_to_list_of_spectrum_numbers([1, 3, 5, 6, 7])
        self.assertEqual(res, [1, 3, 5, 6, 7])

    def test_make_summarised_log_file(self):

        with tempfile.NamedTemporaryFile(delete=False) as mock_mantid_log_file, tempfile.NamedTemporaryFile(delete=False) as mock_summary_file:
            mock_mantid_log_file.write(
                dedent(
                    """
                    2025-01-08 10:48:44,832 [0] Notice CreateWorkspace - CreateWorkspace started (child)
                    2025-01-08 10:48:44,844 [0] Notice CreateWorkspace - CreateWorkspace successful, Duration 0.01 seconds
                    2025-01-08 10:48:44,860 [0] Notice VesuvioAnalysisRoutine -
                    Fitting neutron compton profiles ...
                    2025-01-08 10:48:45,319 [0] Notice VesuvioAnalysisRoutine - Fit spectrum 148: ✓
                    2025-01-08 10:48:45,517 [0] Warning Python - Values in x were outside bounds during a minimize step, clipping to bounds
                    2025-01-08 10:48:45,623 [0] Notice VesuvioAnalysisRoutine - Fit spectrum 151: ✓
                    2025-01-08 10:48:48,568 [0] Notice CreateEmptyTableWorkspace - CreateEmptyTableWorkspace started (child)
                    2025-01-08 10:48:48,570 [0] Notice CreateEmptyTableWorkspace - CreateEmptyTableWorkspace successful, Duration 0.00 seconds
                    2025-01-08 10:48:48,573 [0] Notice Python - Table analysis_inputs_fwd_0_means:
                    2025-01-08 10:48:48,574 [0] Notice Python -  ----------------------------------------------------------------
                    2025-01-08 10:48:48,576 [0] Notice Python - |label |mass   |mean_width|std_width|mean_intensity|std_intensity|
                    2025-01-08 10:48:48,578 [0] Notice Python - |1.0079|1.00790|5.29627   |0.19464  |0.91410       |0.00862      |
                    2025-01-08 10:48:48,584 [0] Notice Python -  ----------------------------------------------------------------
                    2025-01-08 10:48:48,588 [0] Notice VesuvioAnalysisRoutine - VesuvioAnalysisRoutine successful, Duration 3.89 seconds
                    2025-01-08 10:48:49,390 [0] Notice Python -
                    Shared Parameters: ['sigma']
                    2025-01-08 10:48:49,391 [0] Notice Python -
                    Unshared Parameters: ['A', 'x0']
                    """
                ).encode()
            )

            mock_mantid_log_file_path = mock_mantid_log_file.name
            mock_summary_file_path = mock_summary_file.name

        with patch("mvesuvio.util.reduction_helpers.FilesManager.get_mantid_log_file", return_value=mock_mantid_log_file_path):
            with patch("mvesuvio.util.reduction_helpers.FilesManager.get_summarised_log_file", return_value=mock_summary_file_path):
                reduction_helpers.make_summarised_log_file()

        with open(mock_summary_file_path, "rb") as summary_file:
            current_log_file_content = summary_file.read()

        self.assertEqual(
            dedent(
                """
                2025-01-08 10:48:44,860 [0] Notice VesuvioAnalysisRoutine -
                Fitting neutron compton profiles ...
                2025-01-08 10:48:45,319 [0] Notice VesuvioAnalysisRoutine - Fit spectrum 148: ✓
                2025-01-08 10:48:45,623 [0] Notice VesuvioAnalysisRoutine - Fit spectrum 151: ✓
                2025-01-08 10:48:48,573 [0] Notice Python - Table analysis_inputs_fwd_0_means:
                2025-01-08 10:48:48,574 [0] Notice Python -  ----------------------------------------------------------------
                2025-01-08 10:48:48,576 [0] Notice Python - |label |mass   |mean_width|std_width|mean_intensity|std_intensity|
                2025-01-08 10:48:48,578 [0] Notice Python - |1.0079|1.00790|5.29627   |0.19464  |0.91410       |0.00862      |
                2025-01-08 10:48:48,584 [0] Notice Python -  ----------------------------------------------------------------
                2025-01-08 10:48:48,588 [0] Notice VesuvioAnalysisRoutine - VesuvioAnalysisRoutine successful, Duration 3.89 seconds
                2025-01-08 10:48:49,390 [0] Notice Python -
                Shared Parameters: ['sigma']
                2025-01-08 10:48:49,391 [0] Notice Python -
                Unshared Parameters: ['A', 'x0']
                             """
            ).encode(),
            current_log_file_content,
        )

        os.remove(mock_mantid_log_file_path)
        os.remove(mock_summary_file_path)



import sys

from unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5 import EVSCalibrationFit, DETECTOR_RANGE, \
     ENERGY_ESTIMATE, BRAGG_PEAK_CROP_RANGE, BRAGG_FIT_WINDOW_RANGE, RECOIL_PEAK_CROP_RANGE, RECOIL_FIT_WINDOW_RANGE, \
     RESONANCE_PEAK_CROP_RANGE, RESONANCE_FIT_WINDOW_RANGE
from mock import MagicMock, patch, call
from functools import partial

import unittest
import numpy as np


class TestVesuvioCalibrationFit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        self.set_cell_list = []

    @staticmethod
    def setup_mtd_mock(mock_obj, mock_dict):
        d = {}
        for key, return_obj in mock_dict.items():
            d[key] = return_obj
            mock_obj.__getitem__.side_effect = d.__getitem__

    def side_effect_set_cell(self, arg1, arg2, value):
        self.set_cell_list.append((arg1, arg2, value))

    @staticmethod
    def side_effect_cell(row_index, col_index, peaks):
        return peaks[row_index]

    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.mtd')
    def test_filter_peaks_perfect_match(self, mock_mtd):
        alg = EVSCalibrationFit()

        peak_table = 'peak_table'
        peak_estimates_list = [9440, 13351, 15417]
        return_mock_obj_peak_table = MagicMock()
        found_peaks = [9440, 13351, 15417]
        return_mock_obj_peak_table.column.return_value = found_peaks
        return_mock_obj_peak_table.cell.side_effect = partial(self.side_effect_cell, peaks=found_peaks)

        table_to_overwrite = 'overwrite_table'
        return_mock_obj_overwrite_table = MagicMock()
        return_mock_obj_overwrite_table.setRowCount = MagicMock()
        return_mock_obj_overwrite_table.columnCount.return_value = 1
        return_mock_obj_overwrite_table.rowCount.return_value = len(peak_estimates_list)
        return_mock_obj_overwrite_table.setCell.side_effect = self.side_effect_set_cell

        mtd_mock_dict = {'peak_table': return_mock_obj_peak_table, 'overwrite_table': return_mock_obj_overwrite_table}

        self.setup_mtd_mock(mock_mtd, mtd_mock_dict)

        linear_bg_coeffs = (0, 0)
        alg._filter_found_peaks(peak_table, peak_estimates_list, table_to_overwrite, linear_bg_coeffs)
        self.assertEqual([(0, 0, found_peaks[0]), (1, 0, found_peaks[1]), (2, 0, found_peaks[2])], self.set_cell_list)

    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.mtd')
    def test_filter_peaks_no_match(self, mock_mtd):
        alg = EVSCalibrationFit()

        peak_table = 'peak_table'
        peak_estimates_list = [9440, 13351, 15417]
        return_mock_obj_peak_table = MagicMock()
        found_peaks = []
        return_mock_obj_peak_table.column.return_value = found_peaks
        return_mock_obj_peak_table.cell.side_effect = partial(self.side_effect_cell, peaks=found_peaks)
        mtd_mock_dict = {'peak_table': return_mock_obj_peak_table}

        table_to_overwrite = 'overwrite_table'

        self.setup_mtd_mock(mock_mtd, mtd_mock_dict)

        linear_bg_coeffs = (0, 0)
        alg._filter_found_peaks(peak_table, peak_estimates_list, table_to_overwrite, linear_bg_coeffs)
        self.assertEqual([], self.set_cell_list)

    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.mtd')
    def test_filter_peaks_one_match(self, mock_mtd):
        alg = EVSCalibrationFit()

        peak_table = 'peak_table'
        peak_estimates_list = [9440, 13351, 15417]
        return_mock_obj_peak_table = MagicMock()
        found_peaks = [13000]
        return_mock_obj_peak_table.column.return_value = found_peaks
        return_mock_obj_peak_table.cell.side_effect = partial(self.side_effect_cell, peaks=found_peaks)

        table_to_overwrite = 'overwrite_table'
        return_mock_obj_overwrite_table = MagicMock()
        return_mock_obj_overwrite_table.setRowCount = MagicMock()
        return_mock_obj_overwrite_table.columnCount.return_value = 1
        return_mock_obj_overwrite_table.rowCount.return_value = len(peak_estimates_list)
        return_mock_obj_overwrite_table.setCell.side_effect = self.side_effect_set_cell

        mtd_mock_dict = {'peak_table': return_mock_obj_peak_table, 'overwrite_table': return_mock_obj_overwrite_table}

        self.setup_mtd_mock(mock_mtd, mtd_mock_dict)

        linear_bg_coeffs = (0, 0)
        alg._func_param_names = {"Position": 'LorentzPos'}
        alg._filter_found_peaks(peak_table, peak_estimates_list, table_to_overwrite, linear_bg_coeffs)
        self.assertEqual([('LorentzPos', 0, peak_estimates_list[0]), (1, 0, found_peaks[0]), ('LorentzPos', 2, peak_estimates_list[2])],
                         self.set_cell_list)

    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.mtd')
    def test_filter_peaks_two_match(self, mock_mtd):
        alg = EVSCalibrationFit()

        peak_table = 'peak_table'
        peak_estimates_list = [9440, 13351, 15417]
        return_mock_obj_peak_table = MagicMock()
        found_peaks = [9000, 16000]
        return_mock_obj_peak_table.column.return_value = found_peaks
        return_mock_obj_peak_table.cell.side_effect = partial(self.side_effect_cell, peaks=found_peaks)

        table_to_overwrite = 'overwrite_table'
        return_mock_obj_overwrite_table = MagicMock()
        return_mock_obj_overwrite_table.setRowCount = MagicMock()
        return_mock_obj_overwrite_table.columnCount.return_value = 1
        return_mock_obj_overwrite_table.rowCount.return_value = len(peak_estimates_list)
        return_mock_obj_overwrite_table.setCell.side_effect = self.side_effect_set_cell

        mtd_mock_dict = {'peak_table': return_mock_obj_peak_table, 'overwrite_table': return_mock_obj_overwrite_table}

        self.setup_mtd_mock(mock_mtd, mtd_mock_dict)

        linear_bg_coeffs = (0, 0)
        alg._func_param_names = {"Position": 'LorentzPos'}
        alg._filter_found_peaks(peak_table, peak_estimates_list, table_to_overwrite, linear_bg_coeffs)
        self.assertEqual([(0, 0, found_peaks[0]), ('LorentzPos', 1, peak_estimates_list[1]), (2, 0, found_peaks[1])],
                         self.set_cell_list)

    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.mtd')
    def test_filter_peaks_does_not_include_higher_found_peak(self, mock_mtd):
        alg = EVSCalibrationFit()

        peak_table = 'peak_table'
        peak_estimates_list = [9440, 13351, 15417]
        return_mock_obj_peak_table = MagicMock()
        found_peaks = [9440, 15417, 16000]
        return_mock_obj_peak_table.column.return_value = found_peaks
        return_mock_obj_peak_table.cell.side_effect = partial(self.side_effect_cell, peaks=found_peaks)

        table_to_overwrite = 'overwrite_table'
        return_mock_obj_overwrite_table = MagicMock()
        return_mock_obj_overwrite_table.setRowCount = MagicMock()
        return_mock_obj_overwrite_table.columnCount.return_value = 1
        return_mock_obj_overwrite_table.rowCount.return_value = len(peak_estimates_list)
        return_mock_obj_overwrite_table.setCell.side_effect = self.side_effect_set_cell

        mtd_mock_dict = {'peak_table': return_mock_obj_peak_table, 'overwrite_table': return_mock_obj_overwrite_table}

        self.setup_mtd_mock(mock_mtd, mtd_mock_dict)

        linear_bg_coeffs = (0, 0)
        alg._func_param_names = {"Position": 'LorentzPos'}
        alg._filter_found_peaks(peak_table, peak_estimates_list, table_to_overwrite, linear_bg_coeffs)
        self.assertEqual([(0, 0, found_peaks[0]), ('LorentzPos', 1, peak_estimates_list[1]), (2, 0, found_peaks[1])],
                         self.set_cell_list)

    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.mtd')
    def test_filter_peaks_does_not_include_lower_found_peak(self, mock_mtd):
        alg = EVSCalibrationFit()

        peak_table = 'peak_table'
        peak_estimates_list = [9440, 13351, 15417]
        return_mock_obj_peak_table = MagicMock()
        found_peaks = [8000, 9440, 15417]
        return_mock_obj_peak_table.column.return_value = found_peaks
        return_mock_obj_peak_table.cell.side_effect = partial(self.side_effect_cell, peaks=found_peaks)

        table_to_overwrite = 'overwrite_table'
        return_mock_obj_overwrite_table = MagicMock()
        return_mock_obj_overwrite_table.setRowCount = MagicMock()
        return_mock_obj_overwrite_table.columnCount.return_value = 1
        return_mock_obj_overwrite_table.rowCount.return_value = len(peak_estimates_list)
        return_mock_obj_overwrite_table.setCell.side_effect = self.side_effect_set_cell

        mtd_mock_dict = {'peak_table': return_mock_obj_peak_table, 'overwrite_table': return_mock_obj_overwrite_table}

        self.setup_mtd_mock(mock_mtd, mtd_mock_dict)

        linear_bg_coeffs = (0, 0)
        alg._func_param_names = {"Position": 'LorentzPos'}
        alg._filter_found_peaks(peak_table, peak_estimates_list, table_to_overwrite, linear_bg_coeffs)
        self.assertEqual([(0, 0, found_peaks[1]), ('LorentzPos', 1, peak_estimates_list[1]), (2, 0, found_peaks[2])],
                         self.set_cell_list)

    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.mtd')
    def test_filter_peaks_handles_multiple_peaks(self, mock_mtd):
        alg = EVSCalibrationFit()

        peak_table = 'peak_table'
        peak_estimates_list = [9440, 13351, 15417]
        return_mock_obj_peak_table = MagicMock()
        found_peaks = [8000, 9445, 13000, 13355, 15415, 16000]
        return_mock_obj_peak_table.column.return_value = found_peaks
        return_mock_obj_peak_table.cell.side_effect = partial(self.side_effect_cell, peaks=found_peaks)

        table_to_overwrite = 'overwrite_table'
        return_mock_obj_overwrite_table = MagicMock()
        return_mock_obj_overwrite_table.setRowCount = MagicMock()
        return_mock_obj_overwrite_table.columnCount.return_value = 1
        return_mock_obj_overwrite_table.rowCount.return_value = len(peak_estimates_list)
        return_mock_obj_overwrite_table.setCell.side_effect = self.side_effect_set_cell

        mtd_mock_dict = {'peak_table': return_mock_obj_peak_table, 'overwrite_table': return_mock_obj_overwrite_table}

        self.setup_mtd_mock(mock_mtd, mtd_mock_dict)

        linear_bg_coeffs = (0, 0)
        alg._filter_found_peaks(peak_table, peak_estimates_list, table_to_overwrite, linear_bg_coeffs)
        self.assertEqual([(0, 0, found_peaks[1]), (1, 0, found_peaks[3]), (2, 0, found_peaks[4])], self.set_cell_list)

    #Found peaks sometimes returns 'zero' peaks, usually at the end of the table workspace.
    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.mtd')
    def test_filter_peaks_handles_zero_position_in_found_peaks(self, mock_mtd):
        alg = EVSCalibrationFit()

        peak_table = 'peak_table'
        peak_estimates_list = [9440, 13351, 15417]
        return_mock_obj_peak_table = MagicMock()
        found_peaks = [9440, 13351, 0]
        return_mock_obj_peak_table.column.return_value = found_peaks
        return_mock_obj_peak_table.cell.side_effect = partial(self.side_effect_cell, peaks=found_peaks)

        table_to_overwrite = 'overwrite_table'
        return_mock_obj_overwrite_table = MagicMock()
        return_mock_obj_overwrite_table.setRowCount = MagicMock()
        return_mock_obj_overwrite_table.columnCount.return_value = 1
        return_mock_obj_overwrite_table.rowCount.return_value = len(peak_estimates_list)
        return_mock_obj_overwrite_table.setCell.side_effect = self.side_effect_set_cell

        mtd_mock_dict = {'peak_table': return_mock_obj_peak_table, 'overwrite_table': return_mock_obj_overwrite_table}

        self.setup_mtd_mock(mock_mtd, mtd_mock_dict)

        linear_bg_coeffs = (0, 0)
        alg._func_param_names = {"Position": 'LorentzPos'}
        alg._filter_found_peaks(peak_table, peak_estimates_list, table_to_overwrite, linear_bg_coeffs)
        self.assertEqual([(0, 0, found_peaks[0]), (1, 0, found_peaks[1]), ('LorentzPos', 2, peak_estimates_list[2])], self.set_cell_list)

    def test_estimate_bragg_peak_positions(self):
        def side_effect(arg1, arg2):
            if arg1 == 'L0':
                return 11.05
            elif arg1 == 'L1':
                return 0.5505
            elif arg1 == 't0':
                return -0.2
            elif arg1 == 'theta':
                return 139.5371

        alg = EVSCalibrationFit()
        alg._spec_list = [22]
        alg._read_param_column = MagicMock()
        alg._read_param_column.side_effect = side_effect
        alg._d_spacings = np.array([1.75, 2.475, 2.858])

        estimated_positions = alg._estimate_bragg_peak_positions()
        np.testing.assert_almost_equal([9629.84, 13619.43, 15727.03], estimated_positions.flatten().tolist(), 0.01)
        print(estimated_positions)

    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.mtd')
    def test_check_nans_false(self, mock_mtd):
        alg = EVSCalibrationFit()
        table_ws = 'table_ws'
        data = [9440, np.nan, 15417]
        return_mock_obj_table_ws = MagicMock()
        return_mock_obj_table_ws.column.return_value = data
        return_mock_obj_table_ws.columnCount.return_value = len(data)

        mtd_mock_dict = {'table_ws': return_mock_obj_table_ws}

        self.setup_mtd_mock(mock_mtd, mtd_mock_dict)

        self.assertFalse(alg._check_nans(table_ws))


    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.mtd')
    def test_check_nans_true(self, mock_mtd):
        alg = EVSCalibrationFit()
        table_ws = 'table_ws'
        data = [9440, 13351, 15417]
        return_mock_obj_table_ws = MagicMock()
        return_mock_obj_table_ws.column.return_value = data
        return_mock_obj_table_ws.columnCount.return_value = len(data)

        mtd_mock_dict = {'table_ws': return_mock_obj_table_ws}

        self.setup_mtd_mock(mock_mtd, mtd_mock_dict)

        self.assertTrue(alg._check_nans(table_ws))


    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.ReplaceSpecialValues')
    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.EVSCalibrationFit._load_to_ads_and_crop')
    def test_preprocess_no_bg(self, mock_load_to_ads_and_crop, mock_replace_special_values):
        test_run_numbers = [1, 2, 3, 4]
        test_crop_range = [3, 10]
        test_sample_ws_name = "test_ws"

        alg = EVSCalibrationFit()
        alg._ws_crop_range = test_crop_range
        alg._sample_run_numbers = test_run_numbers
        alg._sample = test_sample_ws_name
        alg._bkg_run_numbers = []
        alg._preprocess()
        mock_load_to_ads_and_crop.assert_called_once_with(test_run_numbers, test_sample_ws_name, test_crop_range[0],
                                                          test_crop_range[-1])
        mock_replace_special_values.assert_called_once_with(test_sample_ws_name, NaNValue=0, NaNError=0, InfinityValue=0,
                                                            InfinityError=0, OutputWorkspace=test_sample_ws_name)

    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.EVSCalibrationFit._normalise_sample_by_background')
    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.ReplaceSpecialValues')
    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.EVSCalibrationFit._load_to_ads_and_crop')
    def test_preprocess_with_bg(self, mock_load_to_ads_and_crop, mock_replace_special_values, mock_normalise_sample):
        test_run_numbers = [1, 2, 3, 4]
        test_bg_run_numbers = [5, 6]
        test_crop_range = [3, 10]
        test_sample_ws_name = "test_ws"

        alg = EVSCalibrationFit()
        alg._ws_crop_range = test_crop_range
        alg._sample_run_numbers = test_run_numbers
        alg._sample = test_sample_ws_name
        alg._bkg_run_numbers = test_bg_run_numbers
        alg._background = test_bg_run_numbers[0]
        alg._preprocess()
        mock_load_to_ads_and_crop.assert_has_calls([call(test_run_numbers, test_sample_ws_name, test_crop_range[0],
                                                         test_crop_range[-1]),
                                                    call(test_bg_run_numbers, test_bg_run_numbers[0], test_crop_range[0],
                                                         test_crop_range[-1])])
        mock_normalise_sample.assert_called_once()
        mock_replace_special_values.assert_called_once_with(test_sample_ws_name, NaNValue=0, NaNError=0, InfinityValue=0,
                                                            InfinityError=0, OutputWorkspace=test_sample_ws_name)

    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.CropWorkspace')
    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.EVSCalibrationFit._load_files')
    def test_load_to_ads_and_crop(self, mock_load_files, mock_crop_workspace):
        alg = EVSCalibrationFit()
        run_numbers = [1, 2, 3, 4]
        output = "test_ws"
        xmin = 3
        xmax = 10

        alg._load_to_ads_and_crop(run_numbers, output, xmin, xmax)
        mock_load_files.assert_called_once_with(run_numbers, output)
        mock_crop_workspace.assert_called_once_with(output, XMin=xmin, XMax=xmax, OutputWorkspace=output)

    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.DeleteWorkspace')
    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.Divide')
    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.RebinToWorkspace')
    def test_normalise_sample_by_background(self, mock_rebin, mock_divide, mock_delete):
        alg = EVSCalibrationFit()
        sample_ws = 'test_ws'
        bg_ws = 'bg_ws'

        alg._sample = sample_ws
        alg._background = bg_ws

        alg._normalise_sample_by_background()
        mock_rebin.assert_called_once_with(WorkspaceToRebin=bg_ws, WorkspaceToMatch=sample_ws, OutputWorkspace=bg_ws)
        mock_divide.assert_called_once_with(sample_ws, bg_ws, OutputWorkspace=sample_ws)
        mock_delete.assert_called_once_with(bg_ws)

    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.DeleteWorkspace')
    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.Plus')
    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.EVSCalibrationFit._load_file')
    def test_load_files(self, mock_load_file, mock_plus, mock_delete):
        alg = EVSCalibrationFit()
        ws_numbers = ['1-4']  # Note this is parsed as '1-3', is this intentional?
        output_name = 'test_ws'
        alg._load_files(ws_numbers, output_name)
        mock_load_file.assert_has_calls([call('1', output_name), call('2', '__EVS_calib_temp_ws'),
                                         call('3', '__EVS_calib_temp_ws')])
        mock_plus.assert_has_calls([call(output_name, '__EVS_calib_temp_ws', OutputWorkspace=output_name),
                                    call(output_name, '__EVS_calib_temp_ws', OutputWorkspace=output_name)])
        mock_delete.assert_has_calls([call('__EVS_calib_temp_ws'), call('__EVS_calib_temp_ws')])

    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.LoadRaw')
    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.LoadVesuvio')
    def test_load_file_vesuvio(self, mock_load_vesuvio, mock_load_raw):
        alg = EVSCalibrationFit()
        ws_name = 'test_file'
        output_name = 'test_ws'
        mode = 'FoilOut'
        spec_list = [3, 4, 5, 6]
        alg._mode = mode
        alg._spec_list = spec_list

        alg._load_file(ws_name, output_name)
        mock_load_vesuvio.assert_called_once_with(Filename=ws_name, Mode=mode, OutputWorkspace=output_name,
                                                  SpectrumList="%d-%d" % (spec_list[0], spec_list[-1]),
                                                  EnableLogging=False)
        mock_load_raw.assert_not_called()

    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.ConvertToDistribution')
    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.LoadRaw')
    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.LoadVesuvio')
    def test_load_file_raw(self, mock_load_vesuvio, mock_load_raw, mock_convert_to_dist):
        alg = EVSCalibrationFit()
        ws_name = 'test_file'
        output_name = 'test_ws'
        mode = 'FoilOut'
        spec_list = [3, 4, 5, 6]
        alg._mode = mode
        alg._spec_list = spec_list
        mock_load_vesuvio.side_effect = RuntimeError()

        alg._load_file(ws_name, output_name)
        mock_load_raw.assert_called_once_with('EVS' + ws_name + '.raw', OutputWorkspace=output_name,
                                              SpectrumMin=spec_list[0], SpectrumMax=spec_list[-1],
                                              EnableLogging=False)
        mock_convert_to_dist.assert_called_once_with(output_name, EnableLogging=False)

    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.GroupWorkspaces')
    def test_fit_peaks(self, group_workspaces_mock):
        alg = EVSCalibrationFit()
        alg._estimate_peak_positions = MagicMock(return_value=np.asarray([[5, 10, 15], [2.5, 7.5, 10.5]]))
        alg._create_parameter_table_and_output_headers = MagicMock(return_value=['a', 'b', 'c'])
        alg._fit_peak = MagicMock(side_effect=lambda a, b, c, d, e: f'fit_ws_{a}_{b}')
        alg._output_workspace_name = 'output_ws_name'
        alg._fit_peaks()
        alg._create_parameter_table_and_output_headers.assert_has_calls([call('output_ws_name_Peak_0_Parameters'),
                                                                         call('output_ws_name_Peak_1_Parameters')])
        alg._fit_peak.assert_has_calls([call(0, 0, 5, 'output_ws_name_Peak_0_Parameters', ['a', 'b', 'c']),
                                        call(0, 1, 10, 'output_ws_name_Peak_0_Parameters', ['a', 'b', 'c']),
                                        call(0, 2, 15, 'output_ws_name_Peak_0_Parameters', ['a', 'b', 'c']),
                                        call(1, 0, 2.5, 'output_ws_name_Peak_1_Parameters', ['a', 'b', 'c']),
                                        call(1, 1, 7.5, 'output_ws_name_Peak_1_Parameters', ['a', 'b', 'c']),
                                        call(1, 2, 10.5, 'output_ws_name_Peak_1_Parameters', ['a', 'b', 'c'])])
        self.assertEqual([['fit_ws_0_0', 'fit_ws_0_1', 'fit_ws_0_2'], ['fit_ws_1_0', 'fit_ws_1_1', 'fit_ws_1_2']],
                         alg._peak_fit_workspaces)
        group_workspaces_mock.assert_called_once_with(['output_ws_name_Peak_0_Parameters', 'output_ws_name_Peak_1_Parameters'],
                                                      OutputWorkspace='output_ws_name_Peak_Parameters')

    @staticmethod
    def _setup_alg_mocks_fit_peak():
        alg = EVSCalibrationFit()
        alg._find_peaks_and_output_params = MagicMock(return_value='peak_params')
        alg._build_function_string = MagicMock(return_value='function_string')
        alg._find_fit_x_window = MagicMock(return_value=(0, 10))
        alg._output_fit_params_to_table_ws = MagicMock()
        alg._sample = 'sample'
        alg._output_workspace_name = 'output'
        alg._spec_list = [30]
        alg._prog_reporter = MagicMock()
        alg._del_fit_workspaces = MagicMock()

        return alg

    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.Fit')
    def test_fit_peak(self, mock_fit):
        alg = self._setup_alg_mocks_fit_peak()

        fit_workspace = MagicMock()
        fit_workspace.name.return_value = 'fws'
        fit_results = ('success', 0.5, 'ncm', 'fit_params', fit_workspace, 'func', 'cost_func')
        mock_fit.return_value = fit_results

        fit_workspace_name = alg._fit_peak(0, 0, 5, 'output_Peak_0_Parameters', ['a', 'b', 'c'])

        alg._find_peaks_and_output_params.assert_called_once_with(0, 30, 0, 5)
        alg._build_function_string.assert_called_once_with('peak_params')
        alg._find_fit_x_window.assert_called_once_with('peak_params')
        mock_fit.assert_called_once_with(Function='function_string', InputWorkspace='sample',
                                         IgnoreInvalidData=True, StartX=0, EndX=10, WorkspaceIndex=0,
                                         CalcErrors=True, Output='__output_Peak_0_Spec_0',
                                         Minimizer='Levenberg-Marquardt,RelError=1e-8')
        alg._output_fit_params_to_table_ws.assert_called_once_with(30, 'fit_params', 'output_Peak_0_Parameters',
                                                                   ['a', 'b', 'c'])
        alg._del_fit_workspaces.assert_called_once_with('ncm', 'fit_params', fit_workspace)

        self.assertEqual(fit_workspace_name, 'fws')

    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.mtd')
    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.DeleteWorkspace')
    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.FindPeaks')
    def test_find_peaks_and_output_params(self, find_peaks_mock, delete_workspace_mock, mtd_mock):
        alg = EVSCalibrationFit()
        alg._sample = 'sample'
        find_peak_params_ret_val = {'a': 1, 'b': 2, 'c': 3}
        alg._get_find_peak_parameters = MagicMock(return_value=find_peak_params_ret_val)
        peak_table_name_ws = MagicMock()
        peak_table_name_ws.rowCount.return_value = 5
        mtd_mock.__getitem__.return_value = peak_table_name_ws

        alg._find_peaks_and_output_params(0, 30, 3, 5)

        alg._get_find_peak_parameters.assert_called_once_with(30, [5])
        find_peaks_mock.assert_called_once_with(InputWorkspace='sample', WorkspaceIndex=3, PeaksList='__sample_peaks_table_0_3',
                                                **find_peak_params_ret_val)
        peak_table_name_ws.rowCount.assert_called_once()
        delete_workspace_mock.assert_called_once_with('__sample_peaks_table_0_3')

    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.sys')
    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.logger.error')
    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.mtd')
    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.FindPeaks')
    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.EVSCalibrationFit._get_find_peak_parameters')
    def test_find_peaks_and_output_params_no_peaks_found(self, find_peak_params_mock, find_peaks_mock, mtd_mock, logger_mock,
                                                         sys_mock):
        alg = EVSCalibrationFit()
        alg._sample = 'sample'
        find_peak_params_ret_val = {'a': 4, 'b': 5, 'c': 6}
        find_peak_params_mock.return_value = find_peak_params_ret_val
        peak_table_name_ws = MagicMock()
        peak_table_name_ws.rowCount.return_value = 0
        mtd_mock.__getitem__.return_value = peak_table_name_ws
        sys_mock.exit.side_effect = Exception("Emulate Sys Exit")

        with self.assertRaises(Exception):
            alg._find_peaks_and_output_params(2, 32, 2, 7.5)

        find_peak_params_mock.assert_called_once_with(32, [7.5])
        find_peaks_mock.assert_called_once_with(InputWorkspace='sample', WorkspaceIndex=2, PeaksList='__sample_peaks_table_2_2',
                                                **find_peak_params_ret_val)
        peak_table_name_ws.rowCount.assert_called_once()
        logger_mock.assert_called_once()
        sys_mock.exit.assert_called_once()

    def test_find_fit_x_window(self):
        alg = EVSCalibrationFit()
        alg._func_param_names = {'Position': 'Position_key'}
        alg._fit_window_range = 2

        xmin, xmax = alg._find_fit_x_window({'Position_key': 5})
        self.assertEqual(3, xmin)
        self.assertEqual(7, xmax)

    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.logger.warning')
    def test_find_fit_x_window_position_less_than_1(self, logger_mock):
        alg = EVSCalibrationFit()
        alg._func_param_names = {'Position': 'Position_key'}
        alg._fit_window_range = 2

        xmin, xmax = alg._find_fit_x_window({'Position_key': 0})
        self.assertEqual(None, xmin)
        self.assertEqual(None, xmax)
        logger_mock.assert_called_once()

    def _params_column_side_effect(self, col_index):
        if col_index == 0:
            return ['a', 'b', 'c']
        elif col_index == 1:
            return [1, 2, 3]
        elif col_index == 2:
            return [0.1, 0.2, 0.3]
        else:
            raise ValueError("incorrect column index supplied")

    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.mtd')
    def test_output_fit_params_to_table_ws(self, mtd_mock):
        alg = EVSCalibrationFit()
        spec_num = 45
        params_mock = MagicMock()
        params_mock.column.side_effect = self._params_column_side_effect
        output_table_name = 'output_table'
        output_table_headers = ['a', 'b', 'c']
        output_table_ws_mock = MagicMock()
        mtd_mock.__getitem__.return_value = output_table_ws_mock

        alg._output_fit_params_to_table_ws(spec_num , params_mock, output_table_name, output_table_headers)
        mtd_mock.__getitem__.assert_called_once_with(output_table_name)
        output_table_ws_mock.addRow.assert_called_once_with([spec_num, 1, 0.1, 2, 0.2, 3, 0.3])

    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.DeleteWorkspace')
    def test_del_fit_workspace(self, del_ws_mock):
        alg = EVSCalibrationFit()
        alg._create_output = True
        alg._del_fit_workspaces('ncm', 'params', 'fws')
        del_ws_mock.assert_has_calls([call('ncm'), call('params')])

    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.DeleteWorkspace')
    def test_del_fit_workspace_create_output_true(self, del_ws_mock):
        alg = EVSCalibrationFit()
        alg._create_output = False
        alg._del_fit_workspaces('ncm', 'params', 'fws')
        del_ws_mock.assert_has_calls([call('ncm'), call('params'), call('fws')])


if __name__ == '__main__':
    unittest.main()

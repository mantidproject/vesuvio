from unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5 import EVSCalibrationFit, DETECTOR_RANGE, \
     ENERGY_ESTIMATE, BRAGG_PEAK_CROP_RANGE, BRAGG_FIT_WINDOW_RANGE, RECOIL_PEAK_CROP_RANGE, RECOIL_FIT_WINDOW_RANGE, \
     RESONANCE_PEAK_CROP_RANGE, RESONANCE_FIT_WINDOW_RANGE
from mock import MagicMock, patch
from functools import partial
from mantid.kernel import IntArrayProperty, StringArrayProperty, FloatArrayProperty

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

    def test_PyInit_property_defaults(self):
        alg = EVSCalibrationFit()
        alg.PyInit()
        properties = {'Samples': [], 'Background': [], 'Mode': 'FoilOut', 'Function': 'Gaussian', 'SpectrumRange': DETECTOR_RANGE,
                      'Mass': 207.19, 'DSpacings': [], 'Energy': [ENERGY_ESTIMATE], 'InstrumentParameterFile': '',
                      'PeakType': '', 'InstrumentParameterWorkspace': None, 'CreateOutput': False, 'OutputWorkspace': ''}
        for prop in properties:
            expected_value = alg.getProperty(prop).value
            if type(expected_value) == np.ndarray:
                expected_value = list(expected_value)
            self.assertEqual(expected_value, properties[prop], f'Property {prop}. Expected: {expected_value},'
                                                               f'Actual: {properties[prop]}')

    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.EVSCalibrationFit._setup_spectra_list')
    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.EVSCalibrationFit._setup_run_numbers_and_output_workspace')
    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.EVSCalibrationFit._setup_function_type')
    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.EVSCalibrationFit._setup_parameter_workspace')
    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.EVSCalibrationFit._setup_peaks_and_set_crop_and_fit_ranges')
    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.EVSCalibrationFit._setup_class_variables_from_properties')
    def test_setup_calls_all_functions(self, mock_setup_vars, mock_setup_peaks, mock_setup_param_ws, mock_setup_fn_type,
                                        mock_setup_run_nos, mock_setup_spec):
        alg = EVSCalibrationFit()
        alg._setup()
        mock_setup_vars.assert_called_once()
        mock_setup_peaks.assert_called_once()
        mock_setup_param_ws.assert_called_once()
        mock_setup_fn_type.assert_called_once()
        mock_setup_run_nos.assert_called_once()
        mock_setup_spec.assert_called_once()

    def test_setup_class_variables_from_properties(self):
        alg = EVSCalibrationFit()
        alg.PyInit()

        mode = alg.getProperty("Mode").value
        energy = alg.getProperty("Energy").value
        mass = alg.getProperty("Mass").value
        create_output = alg.getProperty("CreateOutput").value

        alg._setup_class_variables_from_properties()
        self.assertEqual(mode, alg._mode)
        self.assertEqual(energy, alg._energy_estimates)
        self.assertEqual(mass, alg._sample_mass)
        self.assertEqual(create_output, alg._create_output)

    def test_setup_spectra_list(self):
        test_spec_list = [3, 4, 5, 6]
        alg = EVSCalibrationFit()
        alg.declareProperty(IntArrayProperty('SpectrumRange', test_spec_list))

        alg._setup_spectra_list()
        self.assertEqual([test_spec_list[0], test_spec_list[-1]], alg._spec_list)

    def test_setup_spectra_list_one_spec(self):
        test_spec_list = [3]
        alg = EVSCalibrationFit()
        alg.declareProperty(IntArrayProperty('SpectrumRange', test_spec_list))

        alg._setup_spectra_list()
        self.assertEqual([test_spec_list[0]], alg._spec_list)

    def test_setup_spectra_list_no_spec(self):
        test_spec_list = []
        alg = EVSCalibrationFit()
        alg.declareProperty(IntArrayProperty('SpectrumRange', test_spec_list))
        self.assertRaises(ValueError, alg._setup_spectra_list)

    def test_setup_run_numbers_and_output_workspace(self):
        test_sample_list = [3, 4, 5, 6]
        test_bg_list = [7, 8, 9, 10]
        test_ws_name = 'test_ws'
        alg = EVSCalibrationFit()
        alg.declareProperty(StringArrayProperty('Samples', test_sample_list))
        alg.declareProperty(StringArrayProperty('Background', test_bg_list))
        alg.declareProperty('OutputWorkspace', test_ws_name)

        alg._setup_run_numbers_and_output_workspace()
        self.assertEqual(str(test_bg_list[0]), alg._background)
        self.assertEqual(test_ws_name + '_Sample_' + '_'.join([str(x) for x in test_sample_list]), alg._sample)

    def test_setup_run_numbers_and_output_workspace_no_bg(self):
        test_sample_list = [3, 4, 5, 6]
        test_bg_list = []
        test_ws_name = 'test_ws'
        alg = EVSCalibrationFit()
        alg.declareProperty(StringArrayProperty('Samples', test_sample_list))
        alg.declareProperty(StringArrayProperty('Background', test_bg_list))
        alg.declareProperty('OutputWorkspace', test_ws_name)

        alg._setup_run_numbers_and_output_workspace()
        self.assertFalse(hasattr(alg, '_background'))
        self.assertEqual(test_ws_name + '_Sample_' + '_'.join([str(x) for x in test_sample_list]), alg._sample)

    def test_setup_run_numbers_and_output_workspace_no_sample(self):
        test_sample_list = []
        test_bg_list = [7, 8, 9, 10]
        test_ws_name = 'test_ws'
        alg = EVSCalibrationFit()
        alg.declareProperty(StringArrayProperty('Samples', test_sample_list))
        alg.declareProperty(StringArrayProperty('Background', test_bg_list))
        alg.declareProperty('OutputWorkspace', test_ws_name)

        self.assertRaises(ValueError, alg._setup_run_numbers_and_output_workspace)

    def test_setup_function_type_gaussian(self):
        alg = EVSCalibrationFit()
        alg.declareProperty('Function', 'Gaussian')
        alg._setup_function_type()
        expected_param_names = {'Height': 'Height', 'Width': 'Sigma', 'Position': 'PeakCentre'}
        self.assertEqual(expected_param_names, alg._func_param_names)
        self.assertEqual({k: v+"_Err" for k, v in expected_param_names.items()}, alg._func_param_names_error)

    def test_setup_function_type_voigt(self):
        alg = EVSCalibrationFit()
        alg.declareProperty('Function', 'Voigt')
        alg._setup_function_type()
        expected_param_names = {'Height': 'LorentzAmp', 'Position': 'LorentzPos', 'Width': 'LorentzFWHM', 'Width_2': 'GaussianFWHM'}
        self.assertEqual(expected_param_names, alg._func_param_names)
        self.assertEqual({k: v+"Err" for k, v in expected_param_names.items()}, alg._func_param_names_error)

    def test_setup_function_type_error(self):
        alg = EVSCalibrationFit()
        alg.declareProperty('Function', 'Error')
        self.assertRaises(ValueError, alg._setup_function_type)

    def test_setup_parameter_workspace(self):
        alg = EVSCalibrationFit()
        alg.declareProperty('InstrumentParameterWorkspace', 'test_ws')
        alg.declareProperty('InstrumentParameterFile', '')
        alg._setup_parameter_workspace()
        self.assertEqual('test_ws', alg._param_workspace)
        self.assertEqual('test_ws', alg._param_table)

    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.load_instrument_parameters')
    def test_setup_parameter_workspace_no_ws(self, mock_load_instrument_parameters):
        alg = EVSCalibrationFit()
        alg.declareProperty('InstrumentParameterWorkspace', '')
        alg.declareProperty('InstrumentParameterFile', '/c/test_param_dir/file')
        alg._setup_parameter_workspace()
        mock_load_instrument_parameters.assert_called_once()
        self.assertEqual('file', alg._param_table)

    def test_setup_peaks_and_set_crop_and_fit_ranges_bragg(self):
        test_d_spacings = [3.5, 1.5, 2.5]
        alg = EVSCalibrationFit()
        alg.declareProperty(FloatArrayProperty('DSpacings', test_d_spacings))
        alg.declareProperty('PeakType', 'Bragg')
        alg._setup_peaks_and_set_crop_and_fit_ranges()
        self.assertEqual(sorted(test_d_spacings), list(alg._d_spacings))
        self.assertEqual(BRAGG_PEAK_CROP_RANGE, alg._ws_crop_range)
        self.assertEqual(BRAGG_FIT_WINDOW_RANGE, alg._fit_window_range)

    def test_setup_peaks_and_set_crop_and_fit_ranges_recoil(self):
        test_d_spacings = []
        alg = EVSCalibrationFit()
        alg.declareProperty(FloatArrayProperty('DSpacings', test_d_spacings))
        alg.declareProperty('PeakType', 'Recoil')
        alg._setup_peaks_and_set_crop_and_fit_ranges()
        self.assertEqual(RECOIL_PEAK_CROP_RANGE, alg._ws_crop_range)
        self.assertEqual(RECOIL_FIT_WINDOW_RANGE, alg._fit_window_range)

    def test_setup_peaks_and_set_crop_and_fit_ranges_resonance(self):
        test_d_spacings = []
        alg = EVSCalibrationFit()
        alg.declareProperty(FloatArrayProperty('DSpacings', test_d_spacings))
        alg.declareProperty('PeakType', 'Resonance')
        alg._setup_peaks_and_set_crop_and_fit_ranges()
        self.assertEqual(RESONANCE_PEAK_CROP_RANGE, alg._ws_crop_range)
        self.assertEqual(RESONANCE_FIT_WINDOW_RANGE, alg._fit_window_range)

if __name__ == '__main__':
    unittest.main()

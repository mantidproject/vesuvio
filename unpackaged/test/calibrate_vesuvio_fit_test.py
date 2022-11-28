from unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5 import EVSCalibrationFit
from mock import MagicMock, patch
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

        alg._filter_found_peaks_by_estimated(peak_table, peak_estimates_list, table_to_overwrite)
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

        alg._filter_found_peaks_by_estimated(peak_table, peak_estimates_list, table_to_overwrite)
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

        alg._filter_found_peaks_by_estimated(peak_table, peak_estimates_list, table_to_overwrite)
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

        alg._filter_found_peaks_by_estimated(peak_table, peak_estimates_list, table_to_overwrite)
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

        alg._filter_found_peaks_by_estimated(peak_table, peak_estimates_list, table_to_overwrite)
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

        alg._filter_found_peaks_by_estimated(peak_table, peak_estimates_list, table_to_overwrite)
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

        alg._filter_found_peaks_by_estimated(peak_table, peak_estimates_list, table_to_overwrite)
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

        alg._filter_found_peaks_by_estimated(peak_table, peak_estimates_list, table_to_overwrite)
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
    def test_check_nans_true(self, mock_mtd):
        alg = EVSCalibrationFit()
        table_ws = 'table_ws'
        data = [9440, np.nan, 15417]
        return_mock_obj_table_ws = MagicMock()
        return_mock_obj_table_ws.column.return_value = data
        return_mock_obj_table_ws.columnCount.return_value = len(data)

        mtd_mock_dict = {'table_ws': return_mock_obj_table_ws}

        self.setup_mtd_mock(mock_mtd, mtd_mock_dict)

        self.assertTrue(alg._check_nans(table_ws))


    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.mtd')
    def test_check_nans_false(self, mock_mtd):
        alg = EVSCalibrationFit()
        table_ws = 'table_ws'
        data = [9440, 13351, 15417]
        return_mock_obj_table_ws = MagicMock()
        return_mock_obj_table_ws.column.return_value = data
        return_mock_obj_table_ws.columnCount.return_value = len(data)

        mtd_mock_dict = {'table_ws': return_mock_obj_table_ws}

        self.setup_mtd_mock(mock_mtd, mtd_mock_dict)

        self.assertFalse(alg._check_nans(table_ws))

    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.mtd')
    def test_check_nans_str(self, mock_mtd):
        alg = EVSCalibrationFit()
        table_ws = 'table_ws'
        data = ['str1', 'str2', 'str3']
        return_mock_obj_table_ws = MagicMock()
        return_mock_obj_table_ws.column.return_value = data
        return_mock_obj_table_ws.columnCount.return_value = len(data)

        mtd_mock_dict = {'table_ws': return_mock_obj_table_ws}

        self.setup_mtd_mock(mock_mtd, mtd_mock_dict)

        self.assertFalse(alg._check_nans(table_ws))

if __name__ == '__main__':
    unittest.main()

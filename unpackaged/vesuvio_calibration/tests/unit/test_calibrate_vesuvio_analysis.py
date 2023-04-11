from unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5 import EVSCalibrationAnalysis
from mock import MagicMock, patch

import unittest
import numpy as np


class TestVesuvioCalibrationAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        pass

    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.mtd')
    def test_identify_invalid_spectra_no_invalid(self, mock_mtd):
        alg = EVSCalibrationAnalysis()
        ws_mock = MagicMock()
        mock_column_output_dict = {'f1.GaussianFWHM': [1, 2, 3], 'f1.GaussianFWHM_Err': [0.1, 0.2, 0.3],
                                   'f1.LorentzFWHM': [1.1, 2.1, 3.1], 'f1.LorentzFWHM_Err': [0.11, 0.21, 0.31],
                                   'f1.LorentzAmp': [1.2, 2.2, 3.2], 'f1.LorentzAmp_Err': [0.12, 0.22, 0.32]}
        ws_mock.column.side_effect = lambda key: mock_column_output_dict[key]
        mock_mtd.__getitem__.return_value = ws_mock

        np.testing.assert_equal(np.argwhere([]), alg._identify_invalid_spectra('peak_table', [5, 10, 20], [0.1, 0.15, 0.2], [0, 2]))

    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.mtd')
    def test_identify_invalid_spectra_nan_in_errors(self, mock_mtd):
        alg = EVSCalibrationAnalysis()
        ws_mock = MagicMock()
        mock_column_output_dict = {'f1.GaussianFWHM': [1, 2, 3], 'f1.GaussianFWHM_Err': [np.nan, 0.2, 0.3],
                                   'f1.LorentzFWHM': [1.1, 2.1, 3.1], 'f1.LorentzFWHM_Err': [0.11, 0.21, 0.31],
                                   'f1.LorentzAmp': [1.2, 2.2, 3.2], 'f1.LorentzAmp_Err': [0.12, 0.22, np.nan]}
        ws_mock.column.side_effect = lambda key: mock_column_output_dict[key]
        mock_mtd.__getitem__.return_value = ws_mock

        np.testing.assert_equal(np.argwhere(np.array([True, False, True])),
                                alg._identify_invalid_spectra('peak_table', [5, 10, 20], [0.1, 0.15, 0.2], [0, 2]))

    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.mtd')
    def test_identify_invalid_spectra_inf_in_errors(self, mock_mtd):
        alg = EVSCalibrationAnalysis()
        ws_mock = MagicMock()
        mock_column_output_dict = {'f1.GaussianFWHM': [1, 2, 3], 'f1.GaussianFWHM_Err': [0.1, 0.2, 0.3],
                                   'f1.LorentzFWHM': [1.1, 2.1, 3.1], 'f1.LorentzFWHM_Err': [np.inf, 0.21, 0.31],
                                   'f1.LorentzAmp': [1.2, 2.2, 3.2], 'f1.LorentzAmp_Err': [0.12, np.inf, 0.32]}
        ws_mock.column.side_effect = lambda key: mock_column_output_dict[key]
        mock_mtd.__getitem__.return_value = ws_mock

        np.testing.assert_equal(np.argwhere(np.array([True, True, False])),
                                alg._identify_invalid_spectra('peak_table', [5, 10, 20], [0.1, 0.15, 0.2], [0, 2]))

    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.mtd')
    def test_identify_invalid_spectra_error_greater_than_peak(self, mock_mtd):
        alg = EVSCalibrationAnalysis()
        ws_mock = MagicMock()
        mock_column_output_dict = {'f1.GaussianFWHM': [1, 2, 3], 'f1.GaussianFWHM_Err': [0.1, 0.2, 0.3],
                                   'f1.LorentzFWHM': [1.1, 2.1, 3.1], 'f1.LorentzFWHM_Err': [0.11, 10, 0.31],
                                   'f1.LorentzAmp': [1.2, 2.2, 3.2], 'f1.LorentzAmp_Err': [0.12, 0.22, 10]}
        ws_mock.column.side_effect = lambda key: mock_column_output_dict[key]
        mock_mtd.__getitem__.return_value = ws_mock

        np.testing.assert_equal(np.argwhere(np.array([False, True, True])),
                                alg._identify_invalid_spectra('peak_table', [5, 10, 20], [0.1, 0.15, 0.2], [0, 2]))


if __name__ == '__main__':
    unittest.main()

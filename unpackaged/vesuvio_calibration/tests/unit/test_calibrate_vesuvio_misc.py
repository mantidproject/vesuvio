from unpackaged.vesuvio_calibration.calibration_scripts.calibrate_vesuvio_helper_functions import EVSMiscFunctions
from mock import MagicMock, patch

import unittest
import numpy as np


class TestVesuvioCalibrationMisc(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        pass

    def test_generate_header_function_gaussian(self):
        header = EVSMiscFunctions.generate_fit_function_header("Gaussian")
        self.assertEqual({'Height': 'Height', 'Width': 'Sigma', 'Position': 'PeakCentre'}, header)

    def test_generate_header_function_gaussian_with_error(self):
        header = EVSMiscFunctions.generate_fit_function_header("Gaussian", error=True)
        self.assertEqual({'Height': 'Height_Err', 'Width': 'Sigma_Err', 'Position': 'PeakCentre_Err'}, header)

    def test_generate_header_function_voigt(self):
        header = EVSMiscFunctions.generate_fit_function_header("Voigt")
        self.assertEqual({'Height': 'LorentzAmp', 'Position': 'LorentzPos', 'Width': 'LorentzFWHM', 'Width_2': 'GaussianFWHM'}, header)

    def test_generate_header_function_voigt_with_error(self):
        header = EVSMiscFunctions.generate_fit_function_header("Voigt", error=True)
        self.assertEqual({'Height': 'LorentzAmpErr', 'Position': 'LorentzPosErr', 'Width': 'LorentzFWHMErr', 'Width_2': 'GaussianFWHMErr'},
                         header)

    def test_generate_header_function_invalid(self):
        with self.assertRaises(ValueError):
            EVSMiscFunctions.generate_fit_function_header("Invalid")

    @patch('unpackaged.vesuvio_calibration.calibration_scripts.calibrate_vesuvio_helper_functions.mtd')
    def test_identify_invalid_spectra_no_invalid(self, mock_mtd):
        ws_mock = MagicMock()
        mock_column_output_dict = {'f1.GaussianFWHM': [1, 2, 3], 'f1.GaussianFWHM_Err': [0.1, 0.2, 0.3],
                                   'f1.LorentzFWHM': [1.1, 2.1, 3.1], 'f1.LorentzFWHM_Err': [0.11, 0.21, 0.31],
                                   'f1.LorentzAmp': [1.2, 2.2, 3.2], 'f1.LorentzAmp_Err': [0.12, 0.22, 0.32]}
        ws_mock.column.side_effect = lambda key: mock_column_output_dict[key]
        mock_mtd.__getitem__.return_value = ws_mock

        np.testing.assert_equal(np.argwhere([]),
                                EVSMiscFunctions.identify_invalid_spectra('peak_table', [5, 10, 20], [0.1, 0.15, 0.2], [0, 2]))

    @patch('unpackaged.vesuvio_calibration.calibration_scripts.calibrate_vesuvio_helper_functions.mtd')
    def test_identify_invalid_spectra_nan_in_errors(self, mock_mtd):
        ws_mock = MagicMock()
        mock_column_output_dict = {'f1.GaussianFWHM': [1, 2, 3], 'f1.GaussianFWHM_Err': [np.nan, 0.2, 0.3],
                                   'f1.LorentzFWHM': [1.1, 2.1, 3.1], 'f1.LorentzFWHM_Err': [0.11, 0.21, 0.31],
                                   'f1.LorentzAmp': [1.2, 2.2, 3.2], 'f1.LorentzAmp_Err': [0.12, 0.22, np.nan]}
        ws_mock.column.side_effect = lambda key: mock_column_output_dict[key]
        mock_mtd.__getitem__.return_value = ws_mock

        np.testing.assert_equal(np.argwhere(np.array([True, False, True])),
                                EVSMiscFunctions.identify_invalid_spectra('peak_table', [5, 10, 20], [0.1, 0.15, 0.2], [0, 2]))

    @patch('unpackaged.vesuvio_calibration.calibration_scripts.calibrate_vesuvio_helper_functions.mtd')
    def test_identify_invalid_spectra_inf_in_errors(self, mock_mtd):
        ws_mock = MagicMock()
        mock_column_output_dict = {'f1.GaussianFWHM': [1, 2, 3], 'f1.GaussianFWHM_Err': [0.1, 0.2, 0.3],
                                   'f1.LorentzFWHM': [1.1, 2.1, 3.1], 'f1.LorentzFWHM_Err': [np.inf, 0.21, 0.31],
                                   'f1.LorentzAmp': [1.2, 2.2, 3.2], 'f1.LorentzAmp_Err': [0.12, np.inf, 0.32]}
        ws_mock.column.side_effect = lambda key: mock_column_output_dict[key]
        mock_mtd.__getitem__.return_value = ws_mock

        np.testing.assert_equal(np.argwhere(np.array([True, True, False])),
                                EVSMiscFunctions.identify_invalid_spectra('peak_table', [5, 10, 20], [0.1, 0.15, 0.2], [0, 2]))

    @patch('unpackaged.vesuvio_calibration.calibration_scripts.calibrate_vesuvio_helper_functions.mtd')
    def test_identify_invalid_spectra_error_greater_than_peak(self, mock_mtd):
        ws_mock = MagicMock()
        mock_column_output_dict = {'f1.GaussianFWHM': [1, 2, 3], 'f1.GaussianFWHM_Err': [0.1, 0.2, 0.3],
                                   'f1.LorentzFWHM': [1.1, 2.1, 3.1], 'f1.LorentzFWHM_Err': [0.11, 10, 0.31],
                                   'f1.LorentzAmp': [1.2, 2.2, 3.2], 'f1.LorentzAmp_Err': [0.12, 0.22, 10]}
        ws_mock.column.side_effect = lambda key: mock_column_output_dict[key]
        mock_mtd.__getitem__.return_value = ws_mock

        np.testing.assert_equal(np.argwhere(np.array([False, True, True])),
                                EVSMiscFunctions.identify_invalid_spectra('peak_table', [5, 10, 20], [0.1, 0.15, 0.2], [0, 2]))


if __name__ == '__main__':
    unittest.main()

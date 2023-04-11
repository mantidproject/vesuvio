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

    @staticmethod
    def _mtd_col_side_effect(header_str):
        if header_str == 'f1.GaussianFWHM':
            return [1, 2, 3]
        if header_str == 'f1.GaussianFWHM_Err':
            return [0.1, 0.2, 0.3]
        if header_str == 'f1.LorentzFWHM':
            return [1.1, 2.1, 3.1]
        if header_str == 'f1.LorentzFWHM_Err':
            return [0.11, 0.21, 0.31]
        if header_str == 'f1.LorentzAmp':
            return [1.2, 2.2, 3.2]
        if header_str == 'f1.LorentzAmp_Err':
            return [0.12, 0.22, 0.32]

    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.mtd')
    def test_identify_invalid_spectra_no_invalid(self, mock_mtd):
        alg = EVSCalibrationAnalysis()
        ws_mock = MagicMock()
        ws_mock.column.side_effect = self._mtd_col_side_effect
        mock_mtd.__getitem__.return_value = ws_mock

        np.testing.assert_equal(np.argwhere([]), alg._identify_invalid_spectra('peak_table', [5, 10, 20], [0.1, 0.15, 0.2], [0, 2]))





if __name__ == '__main__':
    unittest.main()

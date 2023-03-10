from unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5 import generate_fit_function_header
from mock import MagicMock, patch

import unittest


class TestVesuvioCalibrationMisc(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        pass

    def test_generate_header_function_gaussian(self):
        header = generate_fit_function_header("Gaussian")
        self.assertEqual({'Height': 'Height', 'Width': 'Sigma', 'Position': 'PeakCentre'}, header)

    def test_generate_header_function_gaussian_with_error(self):
        header = generate_fit_function_header("Gaussian", error=True)
        self.assertEqual({'Height': 'Height_Err', 'Width': 'Sigma_Err', 'Position': 'PeakCentre_Err'}, header)

    def test_generate_header_function_voigt(self):
        header = generate_fit_function_header("Voigt")
        self.assertEqual({'Height': 'LorentzAmp', 'Position': 'LorentzPos', 'Width': 'LorentzFWHM', 'Width_2': 'GaussianFWHM'}, header)

    def test_generate_header_function_voigt_with_error(self):
        header = generate_fit_function_header("Voigt", error=True)
        self.assertEqual({'Height': 'LorentzAmpErr', 'Position': 'LorentzPosErr', 'Width': 'LorentzFWHMErr', 'Width_2': 'GaussianFWHMErr'},
                         header)

    def test_generate_header_function_invalid(self):
        with self.assertRaises(ValueError):
            generate_fit_function_header("Invalid")


if __name__ == '__main__':
    unittest.main()

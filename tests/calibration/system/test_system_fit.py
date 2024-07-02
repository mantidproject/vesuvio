import unittest
import numpy as np
import scipy.constants
import scipy.stats

from mantid.api import WorkspaceGroup, AlgorithmFactory
from mantid.simpleapi import mtd
from mock import patch
from tests.calibration.testhelpers.algorithms import create_algorithm
from tests.calibration.testhelpers.system_test_base import EVSCalibrationTest, TestConstants
from tests.calibration.testhelpers.system_test_misc_functions import assert_allclose_excluding_bad_detectors
from tools.calibration_scripts.calibrate_vesuvio_helper_functions import EVSMiscFunctions, EVSGlobals
from tools.calibration_scripts.calibrate_vesuvio_fit import EVSCalibrationFit
from os import path


class TestEVSCalibrationFit(EVSCalibrationTest):

    @classmethod
    def setUpClass(cls):
        # Register Algorithm
        AlgorithmFactory.subscribe(EVSCalibrationFit)

    def setUp(self):
        self._function = 'Voigt'
        test_directory = path.dirname(path.dirname(__file__))
        self._parameter_file = path.join(test_directory, 'data', 'IP0005.par')
        self._calibrated_params = self.load_ip_file()
        self._energy_estimates = np.array([EVSGlobals.ENERGY_ESTIMATE])
        self._alg = None

        # Switches set to none, activated if necessary in the load file side_effect
        self._L0_fit_active = None
        self._E1_fit_active = None
        self._current_run = None

        # Lists in order of call of EVSCalibrationFit in the EVSCalibrationAnalysis function
        self._mode = ['FoilOut']
        self._spec_range = [EVSGlobals.FRONTSCATTERING_RANGE]
        self._E1_fit = [False]
        self._L0_fit = [False]

    @patch('tools.calibration_scripts.calibrate_vesuvio_fit.EVSCalibrationFit._load_file')
    def test_fit_bragg_peaks_copper(self, load_file_mock):
        self._setup_copper_test()
        self._spec_range = [EVSGlobals.DETECTOR_RANGE]
        self._output_workspace = "copper_bragg_peak_fit"

        load_file_mock.side_effect = self._load_file_side_effect

        expected_values = self._calculate_theta_peak_positions()
        params_table = self._run_evs_calibration_fit("Bragg")
        self._assert_fitted_positions_match_expected(expected_values, params_table, {15: TestConstants.IGNORE_DETECTOR})

    @patch('tools.calibration_scripts.calibrate_vesuvio_fit.EVSCalibrationFit._load_file')
    def test_fit_bragg_peaks_lead(self, load_file_mock):
        self._setup_lead_test()
        self._spec_range = [EVSGlobals.DETECTOR_RANGE]
        self._output_workspace = "lead_bragg_peak_fit"

        load_file_mock.side_effect = self._load_file_side_effect

        expected_values = self._calculate_theta_peak_positions()
        params_table = self._run_evs_calibration_fit("Bragg")
        self._assert_fitted_positions_match_expected(expected_values, params_table, {145: 0.27, 158: 0.15, 190:
                                                                                    TestConstants.IGNORE_DETECTOR})

    @patch('tools.calibration_scripts.calibrate_vesuvio_fit.EVSCalibrationFit._load_file')
    def test_fit_peaks_copper_E1(self, load_file_mock):
        self._setup_copper_test()
        self._E1_fit_active = True
        self._E1_fit = [True]
        self._output_workspace = "copper_peak_fit"
        self._mode = ['SingleDifference']

        load_file_mock.side_effect = self._load_file_side_effect

        expected_values = self._calculate_energy_peak_positions()
        params_table = self._run_evs_calibration_fit("Recoil")
        self._assert_fitted_positions_match_expected(expected_values, params_table, {38:  0.12})

    @patch('tools.calibration_scripts.calibrate_vesuvio_fit.EVSCalibrationFit._load_file')
    def test_fit_peaks_lead_E1(self, load_file_mock):
        self._setup_lead_test()
        self._E1_fit_active = True
        self._E1_fit = [True]
        self._output_workspace = "lead_peak_fit"
        self._mode = ['SingleDifference']

        load_file_mock.side_effect = self._load_file_side_effect

        expected_values = self._calculate_energy_peak_positions()
        params_table = self._run_evs_calibration_fit("Recoil")
        self._assert_fitted_positions_match_expected(expected_values, params_table, {38:  0.12})

    @patch('tools.calibration_scripts.calibrate_vesuvio_fit.EVSCalibrationFit._load_file')
    def test_fit_frontscattering_uranium(self, load_file_mock):
        self._setup_uranium_test()
        self._run_range = EVSGlobals.U_FRONTSCATTERING_SAMPLE
        self._background = EVSGlobals.U_FRONTSCATTERING_BACKGROUND
        self._spec_range = [EVSGlobals.FRONTSCATTERING_RANGE]
        self._output_workspace = 'uranium_peak_fit_front'

        load_file_mock.side_effect = self._load_file_side_effect

        expected_values = self._calculate_energy_peak_positions()
        params_table = self._run_evs_calibration_fit("Recoil")
        self._assert_fitted_positions_match_expected(expected_values, params_table)

    @patch('tools.calibration_scripts.calibrate_vesuvio_fit.EVSCalibrationFit._load_file')
    def test_fit_backscattering_uranium(self, load_file_mock):
        self._setup_uranium_test()
        self._run_range = EVSGlobals.U_BACKSCATTERING_SAMPLE
        self._background = EVSGlobals.U_BACKSCATTERING_BACKGROUND
        self._spec_range = [EVSGlobals.BACKSCATTERING_RANGE]
        self._output_workspace = 'uranium_peak_fit_back'

        load_file_mock.side_effect = self._load_file_side_effect

        expected_values = self._calculate_energy_peak_positions()
        params_table = self._run_evs_calibration_fit("Recoil")
        self._assert_fitted_positions_match_expected(expected_values, params_table, default_rel_tol=0.01)

    def _assert_fitted_positions_match_expected(self, expected_positions, params_table, rel_tolerance=None,
                                                default_rel_tol=TestConstants.DEFAULT_RELATIVE_TOLERANCE):
        """
        Check that each of the fitted peak positions match the expected 
        positions in time of flight calculated from the parameter file
        within a small tolerance.
        """
        rel_tolerance = {} if not rel_tolerance else rel_tolerance

        if isinstance(params_table, WorkspaceGroup):
            params_table = params_table.getNames()[0]
            params_table = mtd[params_table]

        column_names = self._find_all_peak_positions(params_table)

        error_dict = {}
        for name, expected_position in zip(column_names, expected_positions):        
            position = np.array(params_table.column(name))
            
            self.assertFalse(np.isnan(position).any())
            self.assertFalse(np.isinf(position).any())
            error_list = assert_allclose_excluding_bad_detectors(expected_position, position, rel_tolerance, default_rel_tol)
            if error_list:
                error_dict[name] = error_list
        if error_dict:
            raise AssertionError(error_dict)

    def _create_evs_calibration_fit(self, peak_type):
        args = {
                "OutputWorkspace": self._output_workspace, "Samples": self._run_range,
                "Background": self._background_run_range(), "InstrumentParameterFile": self._parameter_file,
                "DSpacings": self._get_d_spacings(), "Energy": self._energy_estimates, "CreateOutput": False,
                "Function": self._function, "Mode": self._mode[0], "SpectrumRange": self._spec_range[0],
                "PeakType": peak_type
                }

        self._alg = create_algorithm("EVSCalibrationFit", **args)

    def _execute_evs_calibration_fit(self):
        self._alg.execute()
        return mtd[self._output_workspace + '_Peak_Parameters']

    def _run_evs_calibration_fit(self, peak_type):
        self._create_evs_calibration_fit(peak_type)
        return self._execute_evs_calibration_fit()

    @staticmethod
    def _find_all_peak_positions(params_table):
        filter_errors_func = lambda name: ('LorentzPos' in name or 'PeakCentre' in name) and 'Err' not in name
        column_names = params_table.getColumnNames()
        column_names = filter(filter_errors_func, column_names)
        return column_names

    def _calculate_energy_peak_positions(self):
        """ 
        Using the calibrated values to calculate the expected
        position of the L0/L1/E1 peak in time of flight.
        """
        lower, upper = self._spec_range[0][0] - EVSGlobals.DETECTOR_RANGE[0], self._spec_range[0][1] - EVSGlobals.DETECTOR_RANGE[0] +1

        L0 = self._calibrated_params['L0'][lower:upper]
        L1 = self._calibrated_params['L1'][lower:upper]
        t0 = self._calibrated_params['t0'][lower:upper]
        thetas = self._calibrated_params['theta'][lower:upper]
        r_theta = EVSMiscFunctions.calculate_r_theta(self._mass, thetas)

        t0 /= 1e+6

        energy_estimates = np.copy(self._energy_estimates)
        energy_estimates = energy_estimates.reshape(1, energy_estimates.size).T
        energy_estimates = energy_estimates * EVSGlobals.MEV_CONVERSION

        v1 = np.sqrt(2 * energy_estimates / scipy.constants.m_n)
        tof = ((L0 * r_theta + L1) / v1) + t0
        tof *= 1e+6

        return tof

    def _calculate_theta_peak_positions(self):
        """ 
        Using the calibrated values of theta calculate the expected
        peak position in time of flight.
        """
        L0 = self._calibrated_params['L0']
        L1 = self._calibrated_params['L1']
        t0 = self._calibrated_params['t0']
        thetas = self._calibrated_params['theta']

        t0 /= 1e+6

        d_spacings = np.copy(self._d_spacings)
        d_spacings *= 1e-10
        d_spacings = d_spacings.reshape(1, d_spacings.size).T

        lambdas = 2 * d_spacings * np.sin(np.radians(thetas) / 2)
        tof = (lambdas * scipy.constants.m_n * (L0 + L1)) / scipy.constants.h + t0
        tof *= 1e+6

        return tof


if __name__ == '__main__':
    unittest.main()

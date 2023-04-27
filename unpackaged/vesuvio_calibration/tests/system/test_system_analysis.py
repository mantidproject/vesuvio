import unittest
import numpy as np

from mantid.api import AlgorithmFactory
from mantid.simpleapi import mtd
from mock import patch
from tests.testhelpers.algorithms import create_algorithm
from tests.testhelpers.system_test_base import EVSCalibrationTest, TestConstants
from calibration_scripts.calibrate_vesuvio_helper_functions import EVSGlobals
from tests.testhelpers.system_test_misc_functions import assert_allclose_excluding_bad_detectors
from calibration_scripts.calibrate_vesuvio_analysis import EVSCalibrationAnalysis
from calibration_scripts.calibrate_vesuvio_fit import EVSCalibrationFit
from copy import copy, deepcopy
from os import path


class TestEVSCalibrationAnalysis(EVSCalibrationTest):

    @classmethod
    def setUpClass(cls):
        # Register Algorithms
        AlgorithmFactory.subscribe(EVSCalibrationFit)
        AlgorithmFactory.subscribe(EVSCalibrationAnalysis)

    def setUp(self):
        test_directory = path.dirname(path.dirname(__file__))
        self._parameter_file = path.join(test_directory, 'data', 'IP0005.par')
        self._calibrated_params = self.load_ip_file()
        self._iterations = 1
        self._alg = None

        # Switches set to none, activated if necessary in the load file side_effect
        self._L0_fit_active = None
        self._E1_fit_active = None
        self._current_run = None

        # Lists in order of call of EVSCalibrationFit in the EVSCalibrationAnalysis function
        self._mode = ['FoilOut', 'SingleDifference', 'SingleDifference']
        self._spec_range = [EVSGlobals.DETECTOR_RANGE, EVSGlobals.BACKSCATTERING_RANGE, EVSGlobals.FRONTSCATTERING_RANGE]
        self._E1_fit = [False, True, True]
        self._L0_fit = [False]

    @patch('calibration_scripts.calibrate_vesuvio_fit.EVSCalibrationFit._load_file')
    def test_copper(self, load_file_mock):
        self._setup_copper_test()
        self._output_workspace = "copper_analysis_test"

        load_file_mock.side_effect = self._load_file_side_effect

        params_table = self._run_evs_calibration_analysis()

        #  Specify detectors tolerances set by user, then update with those to mask as invalid.
        detector_specific_r_tols = {"L1": {116: 0.65, 170: 0.75}}
        detector_specific_r_tols["L1"].update({k: TestConstants.INVALID_DETECTOR for k in [138, 141, 146, 147, 156, 158, 160, 163, 164,
                                                                                           165, 167, 168, 169, 170, 182, 191, 192]})
        self._assert_parameters_match_expected(params_table, detector_specific_r_tols)

    @patch('calibration_scripts.calibrate_vesuvio_fit.EVSCalibrationFit._load_file')
    def test_lead(self, load_file_mock):
        self._setup_lead_test()
        self._output_workspace = "lead_analysis_test"

        load_file_mock.side_effect = self._load_file_side_effect

        params_table = self._run_evs_calibration_analysis()

        #  Specify detectors tolerances set by user, then update with those to mask as invalid.
        detector_specific_r_tols = {"L1": {116: 0.25, 170: 0.70}, "Theta": {156: 0.19}}
        detector_specific_r_tols["L1"].update({k: TestConstants.INVALID_DETECTOR for k in [137, 141, 143, 144, 145, 146, 161, 170, 171,
                                                                                           178, 180, 182, 183]})
        self._assert_parameters_match_expected(params_table, detector_specific_r_tols)

    @patch('calibration_scripts.calibrate_vesuvio_fit.EVSCalibrationFit._load_file')
    def test_niobium(self, load_file_mock):
        self._setup_niobium_test()
        self._output_workspace = "niobium_analysis_test"

        load_file_mock.side_effect = self._load_file_side_effect

        params_table = self._run_evs_calibration_analysis()

        #  Specify detectors tolerances set by user, then update with those to mask as invalid.
        detector_specific_r_tols = {"L1": {116: 0.25, 170: TestConstants.IGNORE_DETECTOR, 171: TestConstants.IGNORE_DETECTOR}}
        detector_specific_r_tols["L1"].update({k: TestConstants.INVALID_DETECTOR for k in [3, 41, 44, 48, 49, 65, 73, 89, 99, 100, 102,
                                                                                           110, 114, 118, 123, 126, 131, 138, 141, 143,
                                                                                           146, 147, 151, 154, 156, 157, 159, 160, 162,
                                                                                           163, 166, 170, 171, 172, 178, 179, 180, 181,
                                                                                           182, 186, 187, 189, 191]})
        self._assert_parameters_match_expected(params_table, detector_specific_r_tols)

    @patch('calibration_scripts.calibrate_vesuvio_fit.EVSCalibrationFit._load_file')
    def test_copper_with_uranium(self, load_file_mock):
        self._setup_copper_test()
        self._output_workspace = "copper_analysis_test"

        load_file_mock.side_effect = self._load_file_side_effect

        params_table = self._run_evs_calibration_analysis()

        #  Specify detectors tolerances set by user, then update with those to mask as invalid.
        detector_specific_r_tols = {"L1": {116: 0.25, 170: 0.75}}
        detector_specific_r_tols["L1"].update({k: TestConstants.INVALID_DETECTOR for k in [138, 141, 146, 147, 156, 158, 160, 163, 164,
                                                                                           165, 167, 168, 169, 170, 182, 191, 192]})
        self._assert_parameters_match_expected(params_table, detector_specific_r_tols)

    @patch('calibration_scripts.calibrate_vesuvio_fit.EVSCalibrationFit._load_file')
    def test_lead_with_uranium(self, load_file_mock):
        self._setup_lead_test()
        self._output_workspace = "lead_analysis_test"

        load_file_mock.side_effect = self._load_file_side_effect

        params_table = self._run_evs_calibration_analysis()

        #  Specify detectors tolerances set by user, then update with those to mask as invalid.
        detector_specific_r_tols = {"L1": {116: 0.25, 170: 0.70}, "Theta": {156: 0.19}}
        detector_specific_r_tols["L1"].update({k: TestConstants.INVALID_DETECTOR for k in [137, 141, 143, 144, 145, 146, 161, 170, 171,
                                                                                           178, 180, 182, 183]})
        self._assert_parameters_match_expected(params_table, detector_specific_r_tols)

    @patch('calibration_scripts.calibrate_vesuvio_fit.EVSCalibrationFit._load_file')
    def test_copper_with_l0_calc(self, load_file_mock):
        self._setup_copper_test()
        self._L0_fit = [True, True, True, False, False, False]
        self._output_workspace = "copper_analysis_test"
        self._mode = ['FoilOut', 'FoilOut', 'FoilOut', 'FoilOut', 'SingleDifference', 'SingleDifference']
        self._spec_range = [EVSGlobals.FRONTSCATTERING_RANGE, EVSGlobals.FRONTSCATTERING_RANGE, EVSGlobals.BACKSCATTERING_RANGE,
                            EVSGlobals.DETECTOR_RANGE, EVSGlobals.BACKSCATTERING_RANGE, EVSGlobals.FRONTSCATTERING_RANGE]
        self._E1_fit = [False, False, False, False, True, True]

        load_file_mock.side_effect = self._load_file_side_effect

        params_table = self._run_evs_calibration_analysis()

        #  Specify detectors tolerances set by user, then update with those to mask as invalid.
        detector_specific_r_tols = {"L1": {116: 0.45, 170: 0.75, 171: 0.15, 178: 0.15}}
        detector_specific_r_tols["L1"].update({k: TestConstants.INVALID_DETECTOR for k in [138, 141, 146, 147, 156, 158, 160, 163, 164,
                                                                                           165, 167, 168, 169, 170, 182, 191, 192]})
        self._assert_parameters_match_expected(params_table, detector_specific_r_tols)

    @patch('calibration_scripts.calibrate_vesuvio_fit.EVSCalibrationFit._load_file')
    def test_copper_with_multiple_iterations(self, load_file_mock):
        self._setup_copper_test()
        self._iterations = 2
        self._output_workspace = "copper_analysis_test"
        self._mode = self._mode + self._mode
        self._spec_range = self._spec_range + self._spec_range
        self._E1_fit = self._E1_fit + self._E1_fit

        load_file_mock.side_effect = self._load_file_side_effect

        params_table = self._run_evs_calibration_analysis()

        #  Specify detectors tolerances set by user, then update with those to mask as invalid.
        detector_specific_r_tols = {"L1": {116: 0.25, 170: TestConstants.IGNORE_DETECTOR}, "Theta": {156: 0.14, 158: 0.14, 167: 0.2,
                                                                                                     170: 0.5, 182: 0.3}}
        detector_specific_r_tols["L1"].update({k: TestConstants.INVALID_DETECTOR for k in [138, 141, 146, 147, 156, 158, 160, 163, 164,
                                                                                           165, 167, 168, 169, 170, 182, 191, 192]})
        self._assert_parameters_match_expected(params_table, detector_specific_r_tols)

    @patch('calibration_scripts.calibrate_vesuvio_fit.EVSCalibrationFit._load_file')
    def test_copper_create_output(self, load_file_mock):
        self._setup_copper_test()
        self._output_workspace = "copper_analysis_test"

        load_file_mock.side_effect = self._load_file_side_effect

        self._create_evs_calibration_alg()
        self._alg.setProperty("CreateOutput", True)
        params_table = self._execute_evs_calibration_analysis()

        #  Specify detectors tolerances set by user, then update with those to mask as invalid.
        detector_specific_r_tols = {"L1": {116: 0.65, 170: 0.75}}
        detector_specific_r_tols["L1"].update({k: TestConstants.INVALID_DETECTOR for k in [138, 141, 146, 147, 156, 158, 160, 163, 164,
                                                                                           165, 167, 168, 169, 170, 182, 191, 192]})
        self._assert_parameters_match_expected(params_table, detector_specific_r_tols)

    @patch('calibration_scripts.calibrate_vesuvio_fit.EVSCalibrationFit._load_file')
    def test_copper_create_invalid_detectors_specified(self, load_file_mock):
        self._setup_copper_test()
        self._output_workspace = "copper_analysis_test"

        load_file_mock.side_effect = self._load_file_side_effect

        self._create_evs_calibration_alg()
        self._alg.setProperty("InvalidDetectors", [3, 5, 7, 141, 144, 149, 150, 159, 161, 163, 166, 167, 168, 170, 171, 172, 173, 185, 194,
                                                   195])
        params_table = self._execute_evs_calibration_analysis()

        #  Specify detectors tolerances set by user, then update with those to mask as invalid.
        detector_specific_r_tols = {"L1": {116: 0.65, 170: 0.75}}
        detector_specific_r_tols["L1"].update({k: TestConstants.INVALID_DETECTOR for k in [138, 141, 146, 147, 156, 158, 160, 163,
                                                                                           164, 165, 167, 168, 169, 170, 182, 191, 192]})
        self.assertRaises(AssertionError, self._assert_parameters_match_expected, *[params_table, detector_specific_r_tols])

        detector_specific_r_tols["L1"].update({k: TestConstants.INVALID_DETECTOR for k in [0, 2, 4]})
        self._assert_parameters_match_expected(params_table, detector_specific_r_tols)

    @patch('calibration_scripts.calibrate_vesuvio_fit.EVSCalibrationFit._load_file')
    def test_copper_with_individual_and_global_fit(self, load_file_mock):
        self._setup_copper_test()
        self._output_workspace = "copper_analysis_test"

        load_file_mock.side_effect = self._load_file_side_effect

        self._create_evs_calibration_alg()
        self._alg.setProperty("SharedParameterFitType", "Both")
        params_table = self._execute_evs_calibration_analysis()

        #  Specify detectors tolerances set by user, then update with those to mask as invalid.
        detector_specific_r_tols = {"L1": {116: 0.65, 170: 0.75}}
        detector_specific_r_tols["L1"].update({k: TestConstants.INVALID_DETECTOR for k in [138, 141, 146, 147, 156, 158, 160, 163, 164,
                                                                                           165, 167, 168, 169, 170, 182, 191, 192]})
        self._assert_parameters_match_expected(params_table, detector_specific_r_tols, "Both")
        self._assert_E1_parameters_match_expected(params_table, 3e-3, "Both")

    @patch('unpackaged.vesuvio_calibration.calibration_scripts.calibrate_vesuvio_fit.EVSCalibrationFit._load_file')
    def test_copper_with_global_fit(self, load_file_mock):
        self._setup_copper_test()
        self._output_workspace = "copper_analysis_test"

        load_file_mock.side_effect = self._load_file_side_effect

        self._create_evs_calibration_alg()
        self._alg.setProperty("SharedParameterFitType", "Shared")
        params_table = self._execute_evs_calibration_analysis()

        #  Specify detectors tolerances set by user, then update with those to mask as invalid.
        detector_specific_r_tols = {"L1": {116: 0.65, 170: 0.75}}
        detector_specific_r_tols["L1"].update({k: TestConstants.INVALID_DETECTOR for k in [138, 141, 146, 147, 156, 158, 160, 163, 164,
                                                                                           165, 167, 168, 169, 170, 182, 191, 192]})
        self._assert_parameters_match_expected(params_table, detector_specific_r_tols, "Shared")
        self._assert_E1_parameters_match_expected(params_table, 3e-3, "Shared")

    def _assert_theta_parameters_match_expected(self, params_table, rel_tolerance):
        thetas = params_table.column('theta')
        actual_thetas = self._calibrated_params['theta']

        self.assertFalse(np.isnan(thetas).any())
        self.assertFalse(np.isinf(thetas).any())
        return assert_allclose_excluding_bad_detectors(actual_thetas, thetas, rel_tolerance)

    def _assert_L1_parameters_match_expected(self, params_table, rel_tolerance):
        L1 = params_table.column('L1')
        actual_L1 = self._calibrated_params['L1']

        #  Filter invalid detectors then mask
        invalid_detectors = [(k, rel_tolerance.pop(k))[0] for k, v in copy(rel_tolerance).items() if v == TestConstants.INVALID_DETECTOR]
        invalid_detector_mask = np.zeros(len(L1))
        for index in invalid_detectors:
            invalid_detector_mask[index] = 1

        self.assertFalse(np.isnan(np.ma.masked_array(L1, mask=invalid_detector_mask)).any())
        self.assertFalse(np.isinf(L1).any())
        return assert_allclose_excluding_bad_detectors(np.ma.masked_array(actual_L1, mask=invalid_detector_mask),
                                                       np.ma.masked_array(L1, mask=invalid_detector_mask), rel_tolerance)

    def _assert_E1_parameters_match_expected(self, params_table, rel_tolerance, fit_type):
        if fit_type != "Shared":
            E1 = params_table.column('E1')[0]
            self.assertAlmostEqual(E1, EVSGlobals.ENERGY_ESTIMATE, delta=EVSGlobals.ENERGY_ESTIMATE*rel_tolerance)

        if fit_type != "Individual":
            global_E1 = params_table.column('global_E1')[0]
            self.assertAlmostEqual(global_E1, EVSGlobals.ENERGY_ESTIMATE, delta=EVSGlobals.ENERGY_ESTIMATE*rel_tolerance)

    def _assert_parameters_match_expected(self, params_table, tolerances=None, fit_type="Individual"):
        rel_tol_theta, rel_tol_L1 = self._extract_tolerances(deepcopy(tolerances))
        theta_errors = self._assert_theta_parameters_match_expected(params_table, rel_tol_theta)
        L1_errors = None
        if fit_type != 'Shared':
            L1_errors = self._assert_L1_parameters_match_expected(params_table, rel_tol_L1)

        if theta_errors or L1_errors:
            raise AssertionError(f"Theta: {theta_errors})\n L1: {L1_errors}")

    @staticmethod
    def _extract_tolerances(tolerances: dict) -> (dict, dict):
        theta_tol = {}
        L1_tol = {}
        if tolerances:
            if "Theta" in tolerances:
                theta_tol = tolerances["Theta"]
                if TestConstants.INVALID_DETECTOR in theta_tol.values():
                    raise ValueError('INVALID DETECTORS ONLY RELATE TO L1 TOLERANCES')
            if "L1" in tolerances:
                L1_tol = tolerances["L1"]
        return theta_tol, L1_tol

    def _create_evs_calibration_alg(self):
        args = {
                "OutputWorkspace": self._output_workspace, "Samples": self._run_range, "Background": self._background,
                "InstrumentParameterFile": self._parameter_file, "Mass": self._mass, "DSpacings": self._d_spacings,
                "Iterations": self._iterations, "CalculateL0": True in self._L0_fit
                }

        self._alg = create_algorithm("EVSCalibrationAnalysis", **args)

    def _execute_evs_calibration_analysis(self):
        self._alg.execute()
        last_fit_iteration = self._output_workspace + '_Iteration_%d' % (self._iterations-1)
        return mtd[last_fit_iteration]

    def _run_evs_calibration_analysis(self):
        self._create_evs_calibration_alg()
        return self._execute_evs_calibration_analysis()

    def tearDown(self):
        mtd.clear()


if __name__ == '__main__':
    unittest.main()

import unittest
import numpy as np
import scipy.constants
import scipy.stats

from mantid.api import WorkspaceGroup
from mantid.simpleapi import *
from mock import patch
from unpackaged.vesuvio_calibration.tests.testhelpers.algorithms import create_algorithm
from unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5 import (calculate_r_theta, FRONTSCATTERING_RANGE, DETECTOR_RANGE,
                                                                                  BACKSCATTERING_RANGE, ENERGY_ESTIMATE, MEV_CONVERSION,
                                                                                  U_FRONTSCATTERING_SAMPLE, U_FRONTSCATTERING_BACKGROUND,
                                                                                  U_BACKSCATTERING_SAMPLE, U_BACKSCATTERING_BACKGROUND,
                                                                                  U_MASS, U_PEAK_ENERGIES)


D_SPACINGS_COPPER = [2.0865, 1.807, 1.278, 1.0897]
D_SPACINGS_LEAD = [1.489, 1.750, 2.475, 2.858]
D_SPACINGS_NIOBIUM = [2.3356, 1.6515, 1.3484, 1.1678]

MASS_COPPER = 63.546
MASS_LEAD = 207.19
MASS_NIOBIUM = 92.906

#  Detectors which appear to give consistently bad data, to which fitting is not possible.
#  170 - Copper/Lead
#  171 - Niobium
IGNORE_DETECTORS = [170, 171]


class EVSCalibrationTest(unittest.TestCase):
    def load_ip_file(self):
        param_names = ['spectrum', 'theta', 't0', 'L0', 'L1']
        file_data = np.loadtxt(self._parameter_file, skiprows=3, usecols=[0,2,3,4,5], unpack=True)

        params = {}
        for name, column in zip(param_names, file_data):
            params[name] = column

        return params

    def _setup_copper_test(self):
        self._run_range = [17087, 17088]
        self._background = [17086]
        #  Mass of copper in amu
        self._mass = MASS_COPPER
        #  d-spacings of a copper sample
        self._d_spacings = np.array(D_SPACINGS_COPPER)
        self._d_spacings.sort()
        self._energy_estimates = np.array(ENERGY_ESTIMATE)

    def _setup_lead_test(self):
        self._run_range = [17083, 17084]
        self._background = [17086]
        #  Mass of a lead in amu
        self._mass = MASS_LEAD
        #  d-spacings of a lead sample
        self._d_spacings = np.array(D_SPACINGS_LEAD)
        self._d_spacings.sort()
        self._energy_estimates = np.array(ENERGY_ESTIMATE)

    def _setup_niobium_test(self):
        self._run_range = [17089, 17090, 17091]
        self._background = [17086]
        #  Mass of a lead in amu
        self._mass = MASS_NIOBIUM
        #  d-spacings of a lead sample
        self._d_spacings = np.array(D_SPACINGS_NIOBIUM)
        self._d_spacings.sort()
        self._energy_estimates = np.array(ENERGY_ESTIMATE)

    def _setup_uranium_test(self):
        self._function = 'Gaussian'
        self._mass = U_MASS
        self._d_spacings = []
        self._energy_estimates = np.array(U_PEAK_ENERGIES)
        self._energy_estimates.sort()

    def _select_mode(self):
        self._selected_mode = self._mode.pop(0) if len(self._mode) > 1 else self._mode[0]

    def _select_spec_range(self):
        self._selected_spec_range = self._spec_range.pop(0) if len(self._spec_range) > 1 else self._spec_range[0]

    def _select_active_fits(self):
        self._E1_fit_active = self._E1_fit.pop(0) if len(self._E1_fit) > 1 else self._E1_fit[0]
        self._L0_fit_active = self._L0_fit.pop(0) if len(self._L0_fit) > 1 else self._L0_fit[0]

    def _load_file_side_effect(self, *args):
        """
        Replaces the _load_file function in the calibration script, allowing locally stored test files to be loaded
        rather than requiring access to the archive.

        The arguments to the load algorithms are specific for each run of the EVSCalibrationFit algorithm. As the
        EVSCalibration algorithm consists of multiple calls to EVSCalibrationFit, and each call to EVSCalibrationFi
         calls _load_file a differing number of times, the input to this side effect must vary for each call.

         This is handled through lists, whose first element is popped following each run of EVSCalibration fit.
        """

        sample_no = args[0]
        output_name = args[1]

        if self._current_run is None or self._current_run >= self._total_runs():
            #  set variables that vary in consecutive runs of EVSCalibrationFit
            self._select_mode()
            self._select_spec_range()
            self._select_active_fits()
            self._current_run = 1
        else:
            self._current_run += 1

        try:
            self._load_file_vesuvio(sample_no, output_name)
        except RuntimeError:
            self._load_file_raw(sample_no, output_name)
        print('Load Successful')

    def _total_runs(self):
        """
        Calculates the total runs in the current call to EVSCalibrationFit. This varies depending upon the samples
        specified, and whether the call is a fit to calculate either L0 or E1.
        """

        if self._L0_fit_active:
            #  First 2 runs use frontscattering, rest use backscattering.
            if len(self._L0_fit) > 3:
                run_range = U_FRONTSCATTERING_SAMPLE
                background = U_FRONTSCATTERING_BACKGROUND
            else:
                run_range = U_BACKSCATTERING_SAMPLE
                background = U_BACKSCATTERING_BACKGROUND
        else:
            run_range = self._run_range
            background = self._background

        run_no = len(run_range) if self._E1_fit_active else len(run_range) + len(background)
        return run_no

    def _background_run_range(self):
        background = '' if self._E1_fit_active else self._background
        return background

    def _get_d_spacings(self):
        d_spacings = [] if (self._E1_fit_active or self._L0_fit_active) else self._d_spacings
        return d_spacings

    def _load_file_vesuvio(self, sample_no, output_name):
        print("Attempting LoadVesuvio")
        try:
            prefix = 'EVS'
            filename = f'{os.path.dirname(__file__)}\data\{prefix}{sample_no}.raw'
            LoadVesuvio(Filename=filename, Mode=self._selected_mode, OutputWorkspace=output_name,
                        SpectrumList="%d-%d" % (self._selected_spec_range[0], self._selected_spec_range[1]),
                        EnableLogging=False)
        except RuntimeError:
            prefix = 'VESUVIO000'
            filename = f'{os.path.dirname(__file__)}\data\{prefix}{sample_no}.raw'
            LoadVesuvio(Filename=filename, Mode=self._selected_mode, OutputWorkspace=output_name,
                        SpectrumList="%d-%d" % (self._selected_spec_range[0], self._selected_spec_range[1]),
                        EnableLogging=False)

    def _load_file_raw(self, sample_no, output_name):
        print("Attempting LoadRaw")
        try:
            prefix = 'EVS'
            filename = f'{os.path.dirname(__file__)}\data\{prefix}{sample_no}.raw'
            LoadRaw(filename, OutputWorkspace=output_name, SpectrumMin=self._selected_spec_range[0],
                    SpectrumMax=self._selected_spec_range[-1], EnableLogging=False)
        except RuntimeError as err:
            print(err)
            prefix = 'VESUVIO000'
            filename = f'{os.path.dirname(__file__)}\data\{prefix}{sample_no}.raw'
            LoadRaw(filename, OutputWorkspace=output_name, SpectrumMin=self._selected_spec_range[0],
                    SpectrumMax=self._selected_spec_range[-1], EnableLogging=False)
        ConvertToDistribution(output_name, EnableLogging=False)

    def tearDown(self):
        mtd.clear()


class EVSCalibrationAnalysisTests(EVSCalibrationTest):

    def setUp(self):
        self._parameter_file = f'{os.path.dirname(__file__)}\data\IP0005.par'
        self._calibrated_params = self.load_ip_file()
        self._iterations = 1
        self._alg = None

        # Switches set to none, activated if necessary in the load file side_effect
        self._L0_fit_active = None
        self._E1_fit_active = None
        self._current_run = None

        # Lists in order of call of EVSCalibrationFit in the EVSCalibrationAnalysis function
        self._mode = ['FoilOut', 'SingleDifference', 'SingleDifference']
        self._spec_range = [DETECTOR_RANGE, BACKSCATTERING_RANGE, FRONTSCATTERING_RANGE]
        self._E1_fit = [False, True, True]
        self._L0_fit = [False]

    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.EVSCalibrationFit._load_file')
    def test_copper(self, load_file_mock):
        self._setup_copper_test()
        self._output_workspace = "copper_analysis_test"

        load_file_mock.side_effect = self._load_file_side_effect

        params_table = self.run_evs_calibration_analysis()
        self.assert_theta_parameters_match_expected(params_table)
        self.assert_L1_parameters_match_expected(params_table)

    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.EVSCalibrationFit._load_file')
    def test_lead(self, load_file_mock):
        self._setup_lead_test()
        self._output_workspace = "lead_analysis_test"

        load_file_mock.side_effect = self._load_file_side_effect

        params_table = self.run_evs_calibration_analysis()
        self.assert_theta_parameters_match_expected(params_table)
        self.assert_L1_parameters_match_expected(params_table)

    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.EVSCalibrationFit._load_file')
    def test_niobium(self, load_file_mock):
        self._setup_niobium_test()
        self._output_workspace = "niobium_analysis_test"

        load_file_mock.side_effect = self._load_file_side_effect

        params_table = self.run_evs_calibration_analysis()
        self.assert_theta_parameters_match_expected(params_table)
        self.assert_L1_parameters_match_expected(params_table)

    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.EVSCalibrationFit._load_file')
    def test_copper_with_uranium(self, load_file_mock):
        self._setup_copper_test()
        self._output_workspace = "copper_analysis_test"

        load_file_mock.side_effect = self._load_file_side_effect

        params_table = self.run_evs_calibration_analysis()
        self.assert_theta_parameters_match_expected(params_table)
        self.assert_L1_parameters_match_expected(params_table)

    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.EVSCalibrationFit._load_file')
    def test_lead_with_uranium(self, load_file_mock):
        self._setup_lead_test()
        self._output_workspace = "lead_analysis_test"

        load_file_mock.side_effect = self._load_file_side_effect

        params_table = self.run_evs_calibration_analysis()
        self.assert_theta_parameters_match_expected(params_table)
        self.assert_L1_parameters_match_expected(params_table)

    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.EVSCalibrationFit._load_file')
    def test_copper_with_l0_calc(self, load_file_mock):
        self._setup_copper_test()
        self._L0_fit = [True, True, True, False, False, False]
        self._output_workspace = "copper_analysis_test"
        self._mode = ['FoilOut', 'FoilOut', 'FoilOut', 'FoilOut', 'SingleDifference', 'SingleDifference']
        self._spec_range = [FRONTSCATTERING_RANGE, FRONTSCATTERING_RANGE, BACKSCATTERING_RANGE, DETECTOR_RANGE,
                            BACKSCATTERING_RANGE, FRONTSCATTERING_RANGE]
        self._E1_fit = [False, False, False, False, True, True]

        load_file_mock.side_effect = self._load_file_side_effect

        params_table = self.run_evs_calibration_analysis()
        self.assert_theta_parameters_match_expected(params_table)
        self.assert_L1_parameters_match_expected(params_table, rel_tolerance=0.4)  # Element 116

    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.EVSCalibrationFit._load_file')
    def test_copper_with_multiple_iterations(self, load_file_mock):
        self._setup_copper_test()
        self._iterations = 2
        self._output_workspace = "copper_analysis_test"
        self._mode = self._mode + self._mode
        self._spec_range = self._spec_range + self._spec_range
        self._E1_fit = self._E1_fit + self._E1_fit

        load_file_mock.side_effect = self._load_file_side_effect

        params_table = self.run_evs_calibration_analysis()
        self.assert_theta_parameters_match_expected(params_table)
        self.assert_L1_parameters_match_expected(params_table)

    def tearDown(self):
        mtd.clear()

    #------------------------------------------------------------------
    # Misc helper functions
    #------------------------------------------------------------------

    def assert_theta_parameters_match_expected(self, params_table, rel_tolerance=0.2):
        thetas = params_table.column('theta')
        actual_thetas = self._calibrated_params['theta']

        self.assertFalse(np.isnan(thetas).any())
        self.assertFalse(np.isinf(thetas).any())
        np.testing.assert_allclose(actual_thetas, thetas, rtol=rel_tolerance)

    def assert_L1_parameters_match_expected(self, params_table, rel_tolerance=0.2):
        L1 = params_table.column('L1')
        actual_L1 = self._calibrated_params['L1']

        self.assertFalse(np.isnan(L1).any())
        self.assertFalse(np.isinf(L1).any())
        _assert_allclose_excluding_bad_detectors(actual_L1, L1, rtol=rel_tolerance, atol=0)

    def create_evs_calibration_alg(self):
        args = {
                "OutputWorkspace": self._output_workspace, "Samples": self._run_range, "Background": self._background,
                "InstrumentParameterFile": self._parameter_file, "Mass": self._mass, "DSpacings": self._d_spacings,
                "Iterations": self._iterations, "CalculateL0": True in self._L0_fit
                }

        self._alg = create_algorithm("EVSCalibrationAnalysis", **args)

    def execute_evs_calibration_analysis(self):
        self._alg.execute()
        last_fit_iteration = self._output_workspace + '_Iteration_%d' % (self._iterations-1)
        return mtd[last_fit_iteration]

    def run_evs_calibration_analysis(self):
        self.create_evs_calibration_alg()
        return self.execute_evs_calibration_analysis()


class EVSCalibrationFitTests(EVSCalibrationTest):
    
    def setUp(self):
        self._function = 'Voigt'
        self._parameter_file = f'{os.path.dirname(__file__)}\data\IP0005.par'
        self._calibrated_params = self.load_ip_file()
        self._energy_estimates = np.array([ENERGY_ESTIMATE])
        self._alg = None

        # Switches set to none, activated if necessary in the load file side_effect
        self._L0_fit_active = None
        self._E1_fit_active = None
        self._current_run = None

        # Lists in order of call of EVSCalibrationFit in the EVSCalibrationAnalysis function
        self._mode = ['FoilOut']
        self._spec_range = [FRONTSCATTERING_RANGE]
        self._E1_fit = [False]
        self._L0_fit = [False]


    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.EVSCalibrationFit._load_file')
    def test_fit_bragg_peaks_copper(self, load_file_mock):
        self._setup_copper_test()
        self._spec_range = [DETECTOR_RANGE]
        self._output_workspace = "copper_bragg_peak_fit"

        load_file_mock.side_effect = self._load_file_side_effect

        expected_values = self.calculate_theta_peak_positions()
        params_table = self.run_evs_calibration_fit("Bragg")
        self.assert_fitted_positions_match_expected(expected_values, params_table, rel_tolerance=0.61)

    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.EVSCalibrationFit._load_file')
    def test_fit_bragg_peaks_lead(self, load_file_mock):
        self._setup_lead_test()
        self._spec_range = [DETECTOR_RANGE]
        self._output_workspace = "lead_bragg_peak_fit"

        load_file_mock.side_effect = self._load_file_side_effect

        expected_values = self.calculate_theta_peak_positions()
        params_table = self.run_evs_calibration_fit("Bragg")
        self.assert_fitted_positions_match_expected(expected_values, params_table, rel_tolerance=0.52)

    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.EVSCalibrationFit._load_file')
    def test_fit_peaks_copper_E1(self, load_file_mock):
        self._setup_copper_test()
        self._E1_fit_active = True
        self._E1_fit = [True]
        self._output_workspace = "copper_peak_fit"
        self._mode = ['SingleDifference']

        load_file_mock.side_effect = self._load_file_side_effect

        expected_values = self.calculate_energy_peak_positions()
        params_table = self.run_evs_calibration_fit("Recoil")
        self.assert_fitted_positions_match_expected(expected_values, params_table, rel_tolerance=0.12)

    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.EVSCalibrationFit._load_file')
    def test_fit_peaks_lead_E1(self, load_file_mock):
        self._setup_lead_test()
        self._E1_fit_active = True
        self._E1_fit = [True]
        self._output_workspace = "lead_peak_fit"
        self._mode = ['SingleDifference']

        load_file_mock.side_effect = self._load_file_side_effect

        expected_values = self.calculate_energy_peak_positions()
        params_table = self.run_evs_calibration_fit("Recoil")
        self.assert_fitted_positions_match_expected(expected_values, params_table, rel_tolerance=0.12)

    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.EVSCalibrationFit._load_file')
    def test_fit_frontscattering_uranium(self, load_file_mock):
        self._setup_uranium_test()
        self._run_range = U_FRONTSCATTERING_SAMPLE
        self._background = U_FRONTSCATTERING_BACKGROUND
        self._spec_range = [FRONTSCATTERING_RANGE]
        self._output_workspace = 'uranium_peak_fit_front'

        load_file_mock.side_effect = self._load_file_side_effect

        expected_values = self.calculate_energy_peak_positions()
        params_table = self.run_evs_calibration_fit("Recoil")
        self.assert_fitted_positions_match_expected(expected_values, params_table, rel_tolerance=0.07)

    @patch('unpackaged.vesuvio_calibration.calibrate_vesuvio_uranium_martyn_MK5.EVSCalibrationFit._load_file')
    def test_fit_backscattering_uranium(self, load_file_mock):
        self._setup_uranium_test()
        self._run_range = U_BACKSCATTERING_SAMPLE
        self._background = U_BACKSCATTERING_BACKGROUND
        self._spec_range = [BACKSCATTERING_RANGE]
        self._output_workspace = 'uranium_peak_fit_back'

        load_file_mock.side_effect = self._load_file_side_effect

        expected_values = self.calculate_energy_peak_positions()
        params_table = self.run_evs_calibration_fit("Recoil")
        self.assert_fitted_positions_match_expected(expected_values, params_table, rel_tolerance=0.01, ignore_zero=True)
        
    #------------------------------------------------------------------
    # Misc Helper functions
    #------------------------------------------------------------------

    def assert_fitted_positions_match_expected(self, expected_positions, params_table, rel_tolerance=1e-7, abs_tolerance=0, ignore_zero=False):
        """
        Check that each of the fitted peak positions match the expected 
        positions in time of flight calculated from the parameter file
        within a small tolerance.
        """
        if isinstance(params_table, WorkspaceGroup):
            params_table = params_table.getNames()[0]
            params_table = mtd[params_table]

        column_names = self.find_all_peak_positions(params_table)

        for name, expected_position in zip(column_names, expected_positions):        
            position = np.array(params_table.column(name))
            
            if ignore_zero:
                expected_position, position = self.mask_bad_detector_readings(expected_position, position)
            
            self.assertFalse(np.isnan(position).any())
            self.assertFalse(np.isinf(position).any())
            np.testing.assert_allclose(expected_position, position, rtol=rel_tolerance, atol=abs_tolerance)

    def mask_bad_detector_readings(self, expected_positions, actual_positions):
        """
        Mask values that are very close to zero.

        This handles the case where some of the uranium runs have a missing entry
        for one of the detectors.
        """
        zero_mask = np.where(actual_positions > 1e-10)
        expected_positions = expected_positions[zero_mask]
        actual_positions = actual_positions[zero_mask]
        return expected_positions, actual_positions

    def create_evs_calibration_fit(self, peak_type):
        args = {
                "OutputWorkspace": self._output_workspace, "Samples": self._run_range,
                "Background": self._background_run_range(), "InstrumentParameterFile": self._parameter_file,
                "DSpacings": self._get_d_spacings(), "Energy": self._energy_estimates, "CreateOutput": False,
                "Function": self._function, "Mode": self._mode[0], "SpectrumRange": self._spec_range[0],
                "PeakType": peak_type
                }

        self._alg = create_algorithm("EVSCalibrationFit", **args)

    def execute_evs_calibration_fit(self):
        self._alg.execute()
        return mtd[self._output_workspace + '_Peak_Parameters']

    def run_evs_calibration_fit(self, peak_type):
        self.create_evs_calibration_fit(peak_type)
        return self.execute_evs_calibration_fit()

    @staticmethod
    def find_all_peak_positions(params_table):
        filter_errors_func = lambda name: ('LorentzPos' in name or 'PeakCentre' in name) and 'Err' not in name
        column_names = params_table.getColumnNames()
        column_names = filter(filter_errors_func, column_names)
        return column_names

    def calculate_energy_peak_positions(self):
        """ 
        Using the calibrated values to calculate the expected
        position of the L0/L1/E1 peak in time of flight.
        """
        lower, upper = self._spec_range[0][0] - DETECTOR_RANGE[0], self._spec_range[0][1] - DETECTOR_RANGE[0] +1

        L0 = self._calibrated_params['L0'][lower:upper]
        L1 = self._calibrated_params['L1'][lower:upper]
        t0 = self._calibrated_params['t0'][lower:upper]
        thetas = self._calibrated_params['theta'][lower:upper]
        r_theta = calculate_r_theta(self._mass, thetas)

        t0 /= 1e+6

        energy_estimates = np.copy(self._energy_estimates)
        energy_estimates = energy_estimates.reshape(1, energy_estimates.size).T
        energy_estimates = energy_estimates * MEV_CONVERSION

        v1 = np.sqrt(2 * energy_estimates / scipy.constants.m_n)
        tof = ((L0 * r_theta + L1) / v1) + t0
        tof *= 1e+6

        return tof

    def calculate_theta_peak_positions(self):
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


def _assert_allclose_excluding_bad_detectors(expected_position, position, rtol, atol):
    np.set_printoptions(threshold=sys.maxsize)
    test_failures = []
    for i, (elem_m, elem_n) in enumerate(zip(expected_position, position)):
        if i not in IGNORE_DETECTORS:
            try:
                np.testing.assert_allclose(elem_m, elem_n, rtol, atol)
            except AssertionError:
                test_failures.append(f"Element {i}: Expected {elem_m}, found {elem_n}. atol "
                                     f"{abs(elem_n-elem_m)}, rtol {abs(elem_n-elem_m)/elem_n}")
    if test_failures:
        raise AssertionError(test_failures)


if __name__ == '__main__':
    unittest.main()

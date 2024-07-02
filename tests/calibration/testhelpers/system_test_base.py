import unittest
import numpy as np
from mantid.simpleapi import LoadVesuvio, LoadRaw, mtd, ConvertToDistribution
from tools.calibration_scripts.calibrate_vesuvio_helper_functions import EVSGlobals
from os import path


class TestConstants:
    D_SPACINGS_COPPER = [2.0865, 1.807, 1.278, 1.0897]
    D_SPACINGS_LEAD = [1.489, 1.750, 2.475, 2.858]
    D_SPACINGS_NIOBIUM = [2.3356, 1.6515, 1.3484, 1.1678]

    MASS_COPPER = 63.546
    MASS_LEAD = 207.19
    MASS_NIOBIUM = 92.906

    DEFAULT_RELATIVE_TOLERANCE = 0.1
    IGNORE_DETECTOR = 100  # Detectors to be ignored by the system test, as specified by the user
    INVALID_DETECTOR = 101  # Detectors identified as invalid by the script


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
        self._mass = TestConstants.MASS_COPPER
        #  d-spacings of a copper sample
        self._d_spacings = np.array(TestConstants.D_SPACINGS_COPPER)
        self._d_spacings.sort()
        self._energy_estimates = np.array(EVSGlobals.ENERGY_ESTIMATE)

    def _setup_lead_test(self):
        self._run_range = [17083, 17084]
        self._background = [17086]
        #  Mass of a lead in amu
        self._mass = TestConstants.MASS_LEAD
        #  d-spacings of a lead sample
        self._d_spacings = np.array(TestConstants.D_SPACINGS_LEAD)
        self._d_spacings.sort()
        self._energy_estimates = np.array(EVSGlobals.ENERGY_ESTIMATE)

    def _setup_niobium_test(self):
        self._run_range = [17089, 17090, 17091]
        self._background = [17086]
        #  Mass of a lead in amu
        self._mass = TestConstants.MASS_NIOBIUM
        #  d-spacings of a lead sample
        self._d_spacings = np.array(TestConstants.D_SPACINGS_NIOBIUM)
        self._d_spacings.sort()
        self._energy_estimates = np.array(EVSGlobals.ENERGY_ESTIMATE)

    def _setup_uranium_test(self):
        self._function = 'Gaussian'
        self._mass = EVSGlobals.U_MASS
        self._d_spacings = []
        self._energy_estimates = np.array(EVSGlobals.U_PEAK_ENERGIES)
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
                run_range = EVSGlobals.U_FRONTSCATTERING_SAMPLE
                background = EVSGlobals.U_FRONTSCATTERING_BACKGROUND
            else:
                run_range = EVSGlobals.U_BACKSCATTERING_SAMPLE
                background = EVSGlobals.U_BACKSCATTERING_BACKGROUND
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
        test_directory = path.dirname(path.dirname(__file__))
        try:
            prefix = 'EVS'
            filename = str(path.join(test_directory, 'data', f'{prefix}{sample_no}.raw'))
            LoadVesuvio(Filename=filename, Mode=self._selected_mode, OutputWorkspace=output_name,
                        SpectrumList="%d-%d" % (self._selected_spec_range[0], self._selected_spec_range[1]),
                        EnableLogging=False)
        except RuntimeError:
            prefix = 'VESUVIO000'
            filename = str(path.join(test_directory, 'data', f'{prefix}{sample_no}.raw'))
            LoadVesuvio(Filename=filename, Mode=self._selected_mode, OutputWorkspace=output_name,
                        SpectrumList="%d-%d" % (self._selected_spec_range[0], self._selected_spec_range[1]),
                        EnableLogging=False)

    def _load_file_raw(self, sample_no, output_name):
        print("Attempting LoadRaw")
        test_directory = path.dirname(path.dirname(__file__))
        try:
            prefix = 'EVS'
            filename = str(path.join(test_directory, 'data', f'{prefix}{sample_no}.raw'))
            LoadRaw(filename, OutputWorkspace=output_name, SpectrumMin=self._selected_spec_range[0],
                    SpectrumMax=self._selected_spec_range[-1], EnableLogging=False)
        except RuntimeError as err:
            print(err)
            prefix = 'VESUVIO000'
            filename = str(path.join(test_directory, 'data', f'{prefix}{sample_no}.raw'))
            LoadRaw(filename, OutputWorkspace=output_name, SpectrumMin=self._selected_spec_range[0],
                    SpectrumMax=self._selected_spec_range[-1], EnableLogging=False)
        ConvertToDistribution(output_name, EnableLogging=False)

    def tearDown(self):
        mtd.clear()

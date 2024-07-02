from mantid.kernel import StringArrayProperty, Direction, StringListValidator, IntArrayBoundedValidator, IntArrayProperty,\
     FloatArrayBoundedValidator, StringMandatoryValidator, IntBoundedValidator,FloatArrayProperty
from mantid.api import FileProperty, FileAction, PythonAlgorithm,AlgorithmManager
from mantid.simpleapi import CreateEmptyTableWorkspace, DeleteWorkspace, ReplaceSpecialValues, GroupWorkspaces, mtd,\
     ConvertTableToMatrixWorkspace, ConjoinWorkspaces, Transpose, PlotPeakByLogValue,RenameWorkspace
from calibration_scripts.calibrate_vesuvio_helper_functions import EVSGlobals, EVSMiscFunctions,\
     InvalidDetectors

import os
import sys
import scipy.constants
import scipy.stats
import numpy as np


class EVSCalibrationAnalysis(PythonAlgorithm):

    def summary(self):
        return "Calculates the calibration parameters for the EVS intrument."

    def category(self):
        return "VesuvioCalibration"

    def PyInit(self):

        self.declareProperty(StringArrayProperty("Samples", Direction.Input),
                             doc="Sample run numbers to fit peaks to.")

        self.declareProperty(StringArrayProperty("Background", Direction.Input),
                             doc="Run numbers to use as a background.")

        self.declareProperty(FileProperty('InstrumentParameterFile', '', action=FileAction.Load, extensions=["par"]),
                             doc="Filename of the instrument parameter file.")

        self.declareProperty('Mass', sys.float_info.max,
                             doc="Mass of the sample in amu to be used when calculating energy. Default is Pb: 207.19")

        greater_than_zero_float = FloatArrayBoundedValidator()
        greater_than_zero_float.setLower(0)
        self.declareProperty(FloatArrayProperty('DSpacings', [], greater_than_zero_float, Direction.Input),
                             doc="List of d-spacings used to estimate the positions of peaks in TOF.")

        self.declareProperty(FloatArrayProperty('E1FixedValueAndError', [], greater_than_zero_float, Direction.Input),
                             doc="Value at which to fix E1 and E1 error (form: E1 value, E1 Error). If no input is provided,"
                                 "values will be calculated.")

        detector_validator = IntArrayBoundedValidator()
        detector_validator.setLower(EVSGlobals.DETECTOR_RANGE[0])
        detector_validator.setUpper(EVSGlobals.DETECTOR_RANGE[-1])
        self.declareProperty(IntArrayProperty('InvalidDetectors', [], detector_validator, Direction.Input),
                             doc="List of detectors to be marked as invalid (3-198), to be excluded from analysis calculations.")

        self.declareProperty('Iterations', 2, validator=IntBoundedValidator(lower=1),
                             doc="Number of iterations to perform. Default is 2.")

        shared_fit_type_validator = StringListValidator(["Individual", "Shared", "Both"])
        self.declareProperty('SharedParameterFitType', "Individual", doc='Calculate shared parameters using an individual and/or'
                                                                         'global fit.', validator=shared_fit_type_validator)

        self.declareProperty('CreateOutput', False,
                             doc="Whether to create output from fitting.")

        self.declareProperty('CalculateL0', False,
                             doc="Whether to calculate L0 or just use the values from the parameter file.")

        self.declareProperty('CreateIPFile', False,
                             doc="Whether to save the output as an IP file. \
          This file will use the same name as the OutputWorkspace and will be saved to the default save directory.")

        self.declareProperty('OutputWorkspace', '', StringMandatoryValidator(),
                             doc="Name to call the output workspace.")

    def PyExec(self):
        self._setup()
        self._current_workspace = self._output_workspace_name + '_Iteration_0'
        self._create_calib_parameter_table(self._current_workspace)

        # PEAK FUNCTIONS
        self._theta_peak_function = 'Gaussian'
        self._theta_func_param_names = EVSMiscFunctions.generate_fit_function_header(self._theta_peak_function)
        self._theta_func_param_names_error = EVSMiscFunctions.generate_fit_function_header(self._theta_peak_function, error=True)

        if self._calc_L0:
            # calibrate L0 from the fronstscattering detectors, use the value of the L0 calibrated from frontscattering detectors  for all detectors
            L0_fit = self._current_workspace + '_L0'
            self._run_calibration_fit(Samples=EVSGlobals.U_FRONTSCATTERING_SAMPLE, Background=EVSGlobals.U_FRONTSCATTERING_BACKGROUND,
                                      SpectrumRange=EVSGlobals.FRONTSCATTERING_RANGE, InstrumentParameterWorkspace=self._param_table,
                                      Mass=EVSGlobals.U_MASS, Energy=EVSGlobals.U_PEAK_ENERGIES, OutputWorkspace=L0_fit,
                                      CreateOutput=self._create_output, PeakType='Resonance')
            self._L0_peak_fits = L0_fit + '_Peak_Parameters'
            self._calculate_incident_flight_path(self._L0_peak_fits, EVSGlobals.FRONTSCATTERING_RANGE)

            # calibrate t0 from the front scattering detectors
            t0_fit_front = self._current_workspace + '_t0_front'
            self._run_calibration_fit(Samples=EVSGlobals.U_FRONTSCATTERING_SAMPLE, Background=EVSGlobals.U_FRONTSCATTERING_BACKGROUND,
                                      SpectrumRange=EVSGlobals.FRONTSCATTERING_RANGE, InstrumentParameterWorkspace=self._param_table,
                                      Mass=EVSGlobals.U_MASS, Energy=EVSGlobals.U_PEAK_ENERGIES, OutputWorkspace=t0_fit_front,
                                      CreateOutput=self._create_output, PeakType='Resonance')
            t0_peak_fits_front = t0_fit_front + '_Peak_Parameters'
            self._calculate_time_delay(t0_peak_fits_front, EVSGlobals.FRONTSCATTERING_RANGE)

            # calibrate t0 from the backscattering detectors
            t0_fit_back = self._current_workspace + '_t0_back'
            self._run_calibration_fit(Samples=EVSGlobals.U_BACKSCATTERING_SAMPLE, Background=EVSGlobals.U_BACKSCATTERING_BACKGROUND,
                                      SpectrumRange=EVSGlobals.BACKSCATTERING_RANGE, InstrumentParameterWorkspace=self._param_table,
                                      Mass=EVSGlobals.U_MASS, Energy=EVSGlobals.U_PEAK_ENERGIES, OutputWorkspace=t0_fit_back,
                                      CreateOutput=self._create_output, PeakType='Resonance')
            t0_peak_fits_back = t0_fit_back + '_Peak_Parameters'
            self._calculate_time_delay(t0_peak_fits_back, EVSGlobals.BACKSCATTERING_RANGE)
        else:
            # Just copy values over from parameter file
            t0 = EVSMiscFunctions.read_table_column(self._param_table, 't0', EVSGlobals.DETECTOR_RANGE)
            L0 = EVSMiscFunctions.read_table_column(self._param_table, 'L0', EVSGlobals.DETECTOR_RANGE)
            self._set_table_column(self._current_workspace, 't0', t0)
            self._set_table_column(self._current_workspace, 'L0', L0)

        # repeatedly fit L1, E1 and Theta parameters
        table_group = []
        for i in range(self._iterations):
            # calibrate theta for all detectors
            theta_fit = self._current_workspace + '_theta'
            self._run_calibration_fit(Samples=self._samples, Background=self._background, Function=self._theta_peak_function,
                                      Mode='FoilOut', SpectrumRange=EVSGlobals.DETECTOR_RANGE,
                                      InstrumentParameterWorkspace=self._param_table, DSpacings=self._d_spacings, OutputWorkspace=theta_fit,
                                      CreateOutput=self._create_output, PeakType='Bragg')
            self._theta_peak_fits = theta_fit + '_Peak_Parameters'
            self._calculate_scattering_angle(self._theta_peak_fits, EVSGlobals.DETECTOR_RANGE)

            # calibrate  E1 for backscattering detectors and use the backscattering averaged value for all detectors
            E1_fit_back = self._current_workspace + '_E1_back'
            invalid_detectors = \
            self._run_calibration_fit(Samples=self._samples, Function='Voigt', Mode='SingleDifference',
                                      SpectrumRange=EVSGlobals.BACKSCATTERING_RANGE, InstrumentParameterWorkspace=self._param_table,
                                      Mass=self._sample_mass, OutputWorkspace=E1_fit_back, CreateOutput=self._create_output,
                                      PeakType='Recoil', SharedParameterFitType=self._shared_parameter_fit_type,
                                      InvalidDetectors=self._invalid_detectors.get_all_invalid_detectors())
            self._invalid_detectors.add_invalid_detectors(invalid_detectors)

            E1_peak_fits_back = mtd[self._current_workspace + '_E1_back_Peak_Parameters'].getNames()
            self._calculate_final_energy(E1_peak_fits_back, EVSGlobals.BACKSCATTERING_RANGE, self._shared_parameter_fit_type != "Individual")

            # calibrate L1 for backscattering detectors based on the averaged E1 value  and calibrated theta values
            self._calculate_final_flight_path(E1_peak_fits_back[0], EVSGlobals.BACKSCATTERING_RANGE)

            # calibrate L1 for frontscattering detectors based on the averaged E1 value  and calibrated theta values
            E1_fit_front = self._current_workspace + '_E1_front'
            invalid_detectors += \
            self._run_calibration_fit(Samples=self._samples, Function='Voigt', Mode='SingleDifference',
                                      SpectrumRange=EVSGlobals.FRONTSCATTERING_RANGE, InstrumentParameterWorkspace=self._param_table,
                                      Mass=self._sample_mass, OutputWorkspace=E1_fit_front, CreateOutput=self._create_output,
                                      PeakType='Recoil', SharedParameterFitType=self._shared_parameter_fit_type,
                                      InvalidDetectors=self._invalid_detectors.get_all_invalid_detectors())
            self._invalid_detectors.add_invalid_detectors(invalid_detectors)

            E1_peak_fits_front = mtd[self._current_workspace + '_E1_front_Peak_Parameters'].getNames()
            self._calculate_final_flight_path(E1_peak_fits_front[0], EVSGlobals.FRONTSCATTERING_RANGE)

            # make the fitted parameters for this iteration the input to the next iteration.
            table_group.append(self._current_workspace)
            self._param_table = self._current_workspace

            if i < self._iterations -1:
                self._current_workspace = self._output_workspace_name + '_Iteration_%d' % ( i +1)
                self._create_calib_parameter_table(self._current_workspace)

                # copy over L0 and t0 parameters to new table
                t0 = EVSMiscFunctions.read_table_column(self._param_table, 't0', EVSGlobals.DETECTOR_RANGE)
                t0_error = EVSMiscFunctions.read_table_column(self._param_table, 't0_Err', EVSGlobals.DETECTOR_RANGE)
                L0 = EVSMiscFunctions.read_table_column(self._param_table, 'L0', EVSGlobals.DETECTOR_RANGE)
                L0_error = EVSMiscFunctions.read_table_column(self._param_table, 'L0_Err', EVSGlobals.DETECTOR_RANGE)

                self._set_table_column(self._current_workspace, 't0', t0)
                self._set_table_column(self._current_workspace, 'L0', L0)
                self._set_table_column(self._current_workspace, 't0_Err', t0_error)
                self._set_table_column(self._current_workspace, 'L0_Err', L0_error)

        GroupWorkspaces(','.join(table_group), OutputWorkspace=self._output_workspace_name)

        if self._make_IP_file:
            ws_name = mtd[self._output_workspace_name].getNames()[-1]
            self._save_instrument_parameter_file(ws_name, EVSGlobals.DETECTOR_RANGE)

    def _run_calibration_fit(self, *args, **kwargs):
        """
          Runs EVSCalibrationFit using the AlgorithmManager.

          This allows the calibration script to be run directly from the
          script window after Mantid has started.

          @param args - positional arguments to the algorithm
          @param kwargs - key word arguments to the algorithm
          @returns - returns list of invalid detectors so set in overarching alg
        """
        from mantid.simpleapi import set_properties
        alg = AlgorithmManager.create('EVSCalibrationFit')
        alg.initialize()
        alg.setRethrows(True)

        set_properties(alg, *args, **kwargs)
        alg.execute()

        return alg.getProperty("InvalidDetectorsOut").value.tolist()

    def _calculate_time_delay(self, table_name, spec_list):
        """
          Calculate time delay from frontscattering detectors.

          @param table_name - name of table containing fitted parameters for the peak centres
          @param spec_list - spectrum range to calculate t0 for.
        """
        t0_param_table = self._current_workspace + '_t0_Parameters'
        self._fit_linear(table_name, t0_param_table)

        t0 = np.asarray(mtd[t0_param_table].column('A0'))
        t0_error = np.asarray(mtd[t0_param_table].column('A0_Err'))

        self._set_table_column(self._current_workspace, 't0', t0, spec_list)
        self._set_table_column(self._current_workspace, 't0_Err', t0_error, spec_list)

        DeleteWorkspace(t0_param_table)

    def _calculate_incident_flight_path(self, table_name, spec_list):
        """
          Calculate incident flight path from frontscattering detectors.
          This takes the average value of a fit over all detectors for the value of L0
          and t0.

          @param table_name - name of table containing fitted parameters for the peak centres
          @param spec_list - spectrum range to calculate t0 for.
        """
        L0_param_table = self._current_workspace + '_L0_Parameters'
        self._fit_linear(table_name, L0_param_table)

        L0 = np.asarray(mtd[L0_param_table].column('A1'))

        spec_range = EVSGlobals.DETECTOR_RANGE[1 ] +1 - EVSGlobals.DETECTOR_RANGE[0]
        mean_L0 = np.empty(spec_range)
        L0_error = np.empty(spec_range)

        mean_L0.fill(np.mean(L0))
        L0_error.fill(scipy.stats.sem(L0))

        self._set_table_column(self._current_workspace, 'L0', mean_L0)
        self._set_table_column(self._current_workspace, 'L0_Err', L0_error)

        DeleteWorkspace(L0_param_table)

    def _calculate_final_flight_path(self, peak_table, spec_list):
        """
          Calculate the final flight path using the values for energy.
          This also uses the old value for L1 loaded from the parameter file.

          @param spec_list - spectrum range to calculate t0 for.
        """

        E1 = EVSMiscFunctions.read_table_column(self._current_workspace, 'E1', spec_list)
        t0 = EVSMiscFunctions.read_table_column(self._current_workspace, 't0', spec_list)
        t0_error = EVSMiscFunctions.read_table_column(self._current_workspace, 't0_Err', spec_list)
        L0 = EVSMiscFunctions.read_table_column(self._current_workspace, 'L0', spec_list)
        theta = EVSMiscFunctions.read_table_column(self._current_workspace, 'theta', spec_list)
        r_theta = EVSMiscFunctions.calculate_r_theta(self._sample_mass, theta)

        peak_centres = self._invalid_detectors.filter_peak_centres_for_invalid_detectors(spec_list, peak_table)

        delta_t = (peak_centres - t0) / 1e+6
        delta_t_error = t0_error / 1e+6

        E1 *= EVSGlobals.MEV_CONVERSION
        v1 = np.sqrt( 2 * E1 /scipy.constants.m_n)
        L1 = v1 * delta_t - L0 * r_theta
        L1_error = v1 * delta_t_error

        self._set_table_column(self._current_workspace, 'L1', L1, spec_list)
        self._set_table_column(self._current_workspace, 'L1_Err', L1_error, spec_list)

    def _calculate_scattering_angle(self, table_name, spec_list):
        """
          Calculate the total scattering angle using the previous calculated parameters.

          @param table_name - name of table containing fitted parameters for the peak centres
          @param spec_list - spectrum range to calculate t0 for.
        """

        t0 = EVSMiscFunctions.read_table_column(self._current_workspace, 't0', spec_list)
        L0 = EVSMiscFunctions.read_table_column(self._current_workspace, 'L0', spec_list)
        L1 = EVSMiscFunctions.read_table_column(self._param_table, 'L1', spec_list)
        L1_nan_to_num = np.nan_to_num(L1)

        t0 /= 1e+6

        d_spacings = np.asarray(self._d_spacings)
        d_spacings *= 1e-10
        d_spacings = d_spacings.reshape(1, d_spacings.size).T

        peak_centres = []
        pos_str = self._theta_func_param_names["Position"]
        err_str = self._theta_func_param_names_error["Position"]
        col_names = [name for name in mtd[table_name].getColumnNames() if pos_str in name and err_str not in name]
        for name in col_names:
            t = np.asarray(mtd[table_name].column(name))
            peak_centres.append(t)

        peak_centres = np.asarray(peak_centres)
        masked_peak_centres = np.ma.masked_array(peak_centres, np.logical_or(peak_centres <= 2000, peak_centres >= 20000))
        masked_peak_centres /= 1e+6

        sin_theta = ((masked_peak_centres - t0) * scipy.constants.h) / \
                    (scipy.constants.m_n * d_spacings * 2 * (L0 + L1_nan_to_num))
        theta = np.arcsin(sin_theta) * 2
        theta = np.degrees(theta)

        masked_theta = np.nanmean(theta, axis=0)
        theta_error = np.nanstd(theta, axis=0)

        self._set_table_column(self._current_workspace, 'theta', masked_theta, spec_list)
        self._set_table_column(self._current_workspace, 'theta_Err', theta_error, spec_list)

    def _calculate_final_energy(self, peak_table, spec_list, calculate_global):
        """
          Calculate the final energy using the fitted peak centres of a run.

          @param table_name - name of table containing fitted parameters for the peak centres
          @param spec_list - spectrum range to calculate t0 for.
        """

        spec_range = EVSGlobals.DETECTOR_RANGE[1] + 1 - EVSGlobals.DETECTOR_RANGE[0]
        mean_E1 = np.empty(spec_range)
        E1_error = np.empty(spec_range)
        global_E1 = np.empty(spec_range)
        global_E1_error = np.empty(spec_range)

        if not self._E1_value_and_error:
            t0 = EVSMiscFunctions.read_table_column(self._current_workspace, 't0', spec_list)
            L0 = EVSMiscFunctions.read_table_column(self._current_workspace, 'L0', spec_list)

            L1 = EVSMiscFunctions.read_table_column(self._param_table, 'L1', spec_list)
            theta = EVSMiscFunctions.read_table_column(self._current_workspace, 'theta', spec_list)
            r_theta = EVSMiscFunctions.calculate_r_theta(self._sample_mass, theta)

            peak_centres = self._invalid_detectors.filter_peak_centres_for_invalid_detectors(spec_list, peak_table[0])

            delta_t = (peak_centres - t0) / 1e+6
            v1 = (L0 * r_theta + L1) / delta_t

            E1 = 0.5 * scipy.constants.m_n * v1 ** 2
            E1 /= EVSGlobals.MEV_CONVERSION

            mean_E1_val = np.nanmean(E1)
            E1_error_val = np.nanstd(E1)

        else:
            mean_E1_val = self._E1_value_and_error[0]
            E1_error_val = self._E1_value_and_error[1]

        mean_E1.fill(mean_E1_val)
        E1_error.fill(E1_error_val)

        self._set_table_column(self._current_workspace, 'E1', mean_E1)
        self._set_table_column(self._current_workspace, 'E1_Err', E1_error)

        if calculate_global:  # This fn will need updating for the only global option
            peak_centre = EVSMiscFunctions.read_fitting_result_table_column(peak_table[1], 'f1.LorentzPos', [0])
            peak_centre = [peak_centre] * len(peak_centres)

            delta_t = (peak_centre - t0) / 1e+6
            v1 = (L0 * r_theta + L1) / delta_t
            E1 = 0.5 * scipy.constants.m_n * v1 ** 2
            E1 /= EVSGlobals.MEV_CONVERSION

            global_E1_val = np.nanmean(E1)
            global_E1_error_val = np.nanstd(E1)

            global_E1.fill(global_E1_val)
            global_E1_error.fill(global_E1_error_val)

            self._set_table_column(self._current_workspace, 'global_E1', global_E1)
            self._set_table_column(self._current_workspace, 'global_E1_Err', global_E1_error)

    def _setup(self):
        """
          Setup algorithm.
        """
        self._samples = self.getProperty("Samples").value
        self._background = self.getProperty("Background").value
        self._param_file = self.getProperty("InstrumentParameterFile").value
        self._sample_mass = self.getProperty("Mass").value
        self._d_spacings = self.getProperty("DSpacings").value.tolist()
        self._E1_value_and_error = self.getProperty("E1FixedValueAndError").value.tolist()
        self._invalid_detectors = InvalidDetectors(self.getProperty("InvalidDetectors").value.tolist())
        self._shared_parameter_fit_type = self.getProperty("SharedParameterFitType").value
        self._calc_L0 = self.getProperty("CalculateL0").value
        self._make_IP_file = self.getProperty("CreateIPFile").value
        self._output_workspace_name = self.getPropertyValue("OutputWorkspace")
        self._iterations = self.getProperty("Iterations").value
        self._create_output = self.getProperty("CreateOutput").value

        if len(self._samples) == 0:
            raise ValueError("You must supply at least one sample run number.")

        # if len(self._background) == 0:
        #  raise ValueError("You must supply at least one background run number.")

        self._d_spacings.sort()

        self._param_table = '__EVS_calib_analysis_parameters'
        EVSMiscFunctions.load_instrument_parameters(self._param_file, self._param_table)

    def _create_calib_parameter_table(self, ws_name):
        # create table for calculated parameters
        CreateEmptyTableWorkspace(OutputWorkspace=ws_name)
        table_ws = mtd[ws_name]
        table_ws.addColumn('int', 'Spectrum')

        for value in range(EVSGlobals.DETECTOR_RANGE[0], EVSGlobals.DETECTOR_RANGE[1] + 1):
            table_ws.addRow([value])

        column_names = ['t0', 't0_Err', 'L0', 'L0_Err', 'L1', 'L1_Err', 'E1', 'E1_Err', 'theta', 'theta_Err']
        for name in column_names:
            table_ws.addColumn('double', name)

    def _fit_linear(self, table_workspace_group, output_table):
        """
          Create a workspace wth the fitted peak_centres on the y and corresponding neutron velocity
          on the x. The intercept is the value of t0 and the gradient is the value of L0/L-Total.

          @param table_workspace_group - workspace group containing the fitted parameters of the peaks.
          @param output_table - name to call the fit workspace.
        """
        # extract fit data to workspace
        peak_workspaces = []
        for i, param_ws in enumerate(mtd[table_workspace_group].getNames()):
            temp_peak_data = '__temp_peak_ws_%d' % i
            ConvertTableToMatrixWorkspace(InputWorkspace=param_ws, OutputWorkspace=temp_peak_data,
                                          ColumnX='Spectrum', ColumnY='f1.PeakCentre')
            peak_workspaces.append(temp_peak_data)

        # create workspace of peaks
        peak_workspace = table_workspace_group + '_Workspace'
        RenameWorkspace(peak_workspaces[0], OutputWorkspace=peak_workspace)
        for temp_ws in peak_workspaces[1:]:
            ConjoinWorkspaces(peak_workspace, temp_ws, CheckOverlapping=False)
        Transpose(peak_workspace, OutputWorkspace=peak_workspace)

        num_spectra = mtd[peak_workspace].getNumberHistograms()
        plot_peak_indicies = ';'.join([peak_workspace + ',i' + str(i) for i in range(num_spectra)])

        for i in range(num_spectra):
            mtd[peak_workspace].setX(i, np.asarray(EVSGlobals.U_NEUTRON_VELOCITY))

        ReplaceSpecialValues(peak_workspace, NaNValue=0, NaNError=0, InfinityValue=0, InfinityError=0,
                             OutputWorkspace=peak_workspace)

        # perform linear fit on peak centres
        func_string = 'name=LinearBackground, A0=0, A1=0;'
        PlotPeakByLogValue(Input=plot_peak_indicies, Function=func_string,
                           FitType='Individual', CreateOutput=False, OutputWorkspace=output_table)
        DeleteWorkspace(peak_workspace)

    def _set_table_column(self, table_name, column_name, data, spec_list=None):
        """
          Add a set of data to a table workspace

          @param table_name - name of the table workspace to modify
          @param column_name - name of the column to add data to
          @param data - data array to add to the table
          @param spec_list - range of rows to add the data too
        """
        table_ws = mtd[table_name]

        if column_name not in table_ws.getColumnNames():
            table_ws.addColumn('double', column_name)

        if spec_list == None:
            offset = 0
        else:
            if len(data) < (spec_list[1] + 1 - spec_list[0]):
                raise ValueError("Not enough data from spectrum range.")

            offset = spec_list[0] - EVSGlobals.DETECTOR_RANGE[0]

        if isinstance(data, np.ndarray):
            data = data.tolist()

        for i, value in enumerate(data):
            table_ws.setCell(column_name, offset + i, value)

    def _save_instrument_parameter_file(self, ws_name, spec_list):
        """
          Save the calibrated parameters to a tab delimited instrument parameter file.

          @param ws_name - name of the workspace to save the IP file from.
          @param spec_list - spectrum range to save to file.
        """
        file_header = '\t'.join(['plik', 'det', 'theta', 't0', 'L0', 'L1'])
        fmt = "%d  %d  %.4f  %.4f  %.3f  %.4f"

        det = EVSMiscFunctions.read_table_column(ws_name, 'Spectrum', spec_list)
        t0 = EVSMiscFunctions.read_table_column(ws_name, 't0', spec_list)
        L0 = EVSMiscFunctions.read_table_column(ws_name, 'L0', spec_list)
        L1 = EVSMiscFunctions.read_table_column(ws_name, 'L1', spec_list)
        theta = EVSMiscFunctions.read_table_column(ws_name, 'theta', spec_list)

        # pad the start of the file with dummy data for the monitors
        file_data = np.asarray([[1, 1, 0, 0, 0, 0], [2, 2, 0, 0, 0, 0]])
        file_data = np.append(file_data, np.column_stack((det, det, theta, t0, L0, L1)), axis=0)

        workdir = config['defaultsave.directory']
        file_path = os.path.join(workdir, self._output_workspace_name + '.par')

        with open(file_path, 'wb') as f_handle:
            np.savetxt(f_handle, file_data, header=file_header, fmt=fmt)

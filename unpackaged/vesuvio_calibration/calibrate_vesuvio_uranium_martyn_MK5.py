"""
  Calibration algorithms for the VESUVIO spectrometer. This file provides two Mantid algorithms: EVSCalibrationFit and EVSCalibrationAnalysis. EVSCalibrationFit
  is used to fit m peaks to n spectra. The positions of the peaks are esitmated using the supplied instrument parameter file and the d-spacings of the sample (if provided)
  and provides support for both Voigt and Gaussian functions. EVSCalibrationAnalysis uses EVSCalibrationFit to calculate instrument parameters using the output of
  the fitting and the and an existing instrument parameter file.

  The procedures used here are based upon those descibed in: Calibration of an electron volt neutron spectrometer, Nuclear Instruments and Methods in Physics Research A
  (15 October 2010), doi:10.1016/j.nima.2010.09.079 by J. Mayers, M. A. Adams
"""

from mantid.kernel import StringArrayProperty, Direction, StringListValidator, IntArrayBoundedValidator, IntArrayProperty,\
     FloatArrayBoundedValidator, FloatArrayMandatoryValidator, StringMandatoryValidator, IntBoundedValidator,\
     FloatArrayProperty, logger
from mantid.api import FileProperty, FileAction, ITableWorkspaceProperty, PropertyMode, Progress, TextAxis, PythonAlgorithm, \
     AlgorithmFactory, WorkspaceFactory, AlgorithmManager
from mantid.simpleapi import CreateEmptyTableWorkspace, DeleteWorkspace, CropWorkspace, RebinToWorkspace, Divide,\
     ReplaceSpecialValues, FindPeaks, GroupWorkspaces, mtd, Plus, LoadVesuvio, LoadRaw, ConvertToDistribution,\
     FindPeakBackground, ExtractSingleSpectrum, SumSpectra, AppendSpectra, ConvertTableToMatrixWorkspace,\
     ConjoinWorkspaces, Transpose, PlotPeakByLogValue, CloneWorkspace, Fit, RenameWorkspace

from functools import partial

import os
import sys
import scipy.constants
import scipy.stats
import numpy as np

# Configuration for Uranium runs / Indium runs
#----------------------------------------------------------------------------------------
#Uranium sample & background run numbers
U_FRONTSCATTERING_SAMPLE = [14025] #[14025]  for U foil in the beam,   [19129, 19130] for In foil in the beam

#  ['12570']  or [19132, 19134, 19136, 19138, 19140, 19142,19144, 19146, 19148, 19150, 19152] or [42209, 42210, 42211, 42212, 42213, 42214, 42215, 42216, 42217, 42218, 42219, 42220, 42221, 42222, 42223, 42224, 42225, 42226, 42227,42228] for Pb 2mm with U foil out 
U_FRONTSCATTERING_BACKGROUND = [42209, 42210, 42211, 42212, 42213, 42214, 42215, 42216, 42217, 42218, 42219, 42220, 42221, 42222, 42223, 42224, 42225, 42226, 42227,42228] 

#  ['12570']  or [19132, 19134, 19136, 19138, 19140, 19142,19144, 19146, 19148, 19150, 19152] or [42209, 42210, 42211, 42212, 42213, 42214, 42215, 42216, 42217, 42218, 42219, 42220, 42221, 42222, 42223, 42224, 42225, 42226, 42227,42228] for Pb 2mm with U foil out  
U_BACKSCATTERING_SAMPLE = [42209, 42210, 42211, 42212, 42213, 42214, 42215, 42216, 42217, 42218, 42219, 42220, 42221, 42222, 42223, 42224, 42225, 42226, 42227,42228]

# ['12571'] or [42229,42230,42231,42232,42233,42234,42235,42236,42237,42238,42239,42240,42241,42242,42243,42244,42245,42246,42247,42248,42249,42250,42251,42252,42253] or  [19131, 19133, 19135, 19137, 19139, 19141, 19143, 19145, 19147, 19149, 19151]  for Pb 2mm with U foil in 
U_BACKSCATTERING_BACKGROUND = [42229,42230,42231,42232,42233,42234,42235,42236,42237,42238,42239,42240,42241,42242,42243,42244,42245,42246,42247,42248,42249,42250,42251,42252,42253]

#peak enegy for U/In in mev
U_PEAK_ENERGIES = [36684, 20874, 6672] #  [36684, 20874, 6672] for uranium, [39681,22723,14599,12056,9088,3855] for indium
#U mass/ In mass  in amu
U_MASS=238.0289 # 113 for indium

#misc global variables
#----------------------------------------------------------------------------------------
#full range of both the front and backscattering banks
FRONTSCATTERING_RANGE = [135,198]
BACKSCATTERING_RANGE = [3,134]
DETECTOR_RANGE = [BACKSCATTERING_RANGE[0], FRONTSCATTERING_RANGE[1]]

#file loading modes
MODES=["SingleDifference", "DoubleDifference", "ThickDifference", "FoilOut", "FoilIn", "FoilInOut"]

# Different peak types that will be fit
PEAK_TYPES = ["Resonance", "Recoil", "Bragg"]

# self._fit_window_range applies bith to fitting the resonances and the lead recoil peaks
# it is defined as the range left and right from the peak centre (i. e. the whole fitting window is twice the fitting range)

BRAGG_PEAK_CROP_RANGE = (2000, 20000)
BRAGG_FIT_WINDOW_RANGE = 500
BRAGG_PEAK_POSITION_TOLERANCE = 1000

RECOIL_PEAK_CROP_RANGE = (300, 500)
RECOIL_FIT_WINDOW_RANGE = 300

RESONANCE_PEAK_CROP_RANGE =(100, 350)
RESONANCE_FIT_WINDOW_RANGE = 50

PEAK_HEIGHT_RELATIVE_THRESHOLD = 0.25

#energy used to estimate peak position
ENERGY_ESTIMATE = 4897.3

#physical constants
#----------------------------------------------------------------------------------------
#convert to 1 / v and scale for fitting
U_NEUTRON_VELOCITY = np.array([83769.7, 63190.5, 35725.4]) #np.array([83769.7, 63190.5, 35725.4]) # for U  # np.array([87124.5,65929.8,52845.8,48023.1,41694.9,27155.7])  # for indium
U_NEUTRON_VELOCITY = 1.0 / U_NEUTRON_VELOCITY
U_NEUTRON_VELOCITY *= 1e+6


#mass of a neutron in amu
NEUTRON_MASS_AMU = scipy.constants.value("neutron mass in u")
#1 meV in Joules
MEV_CONVERSION = 1.602176487e-22

#misc helper functions used by both algorithms
#----------------------------------------------------------------------------------------

def calculate_r_theta(sample_mass, thetas):
  """
    Returns the ratio of the final neutron velocity to the initial neutron velocity
    as a function of the scattering angle theta and atomic mass of lead and a neutron.

    @param sample_mass - mass of the sample in amu
    @param thetas - vector containing the values of theta
    @return the ratio of final and incident velocities
  """
  rad_theta = np.radians(thetas)
  r_theta = (np.cos(rad_theta) + np.sqrt( (sample_mass / NEUTRON_MASS_AMU)**2 - np.sin(rad_theta)**2 )) / ((sample_mass / NEUTRON_MASS_AMU) +1)

  return r_theta

#----------------------------------------------------------------------------------------
# The IP text file load function skips the first 3 rows of the text file.
# This assumes that there is one line for the file header and 2 lines for
# parameters of monitor 1 and monitor 2, respectively.
# One has to be careful here as some IP files have no header!
# In such a case one has to add header
# Otherwise, the algorithm will read the number of spectra to be fitted
# for the calibration of of L0, t0, L1, theta from the vectors:
# FRONTSCATTERING_RANGE, BACKSCATTERING_RANGE and DETECTOR_RANGE
# This number will be different from the number of fits performed
# (which will be given by the length of the variable table_name)
# The program will start running but will crash after fitting L0 and t0

def load_instrument_parameters(file_path, table_name):
  """
    Load instrument parameters from file into a table workspace

    @param file_path - the location of the file on disk
    @return the name of the table workspace.
  """

  file_data = np.loadtxt(file_path, skiprows=3, usecols=[0,2,3,4,5])

  CreateEmptyTableWorkspace(OutputWorkspace=table_name)
  table_ws = mtd[table_name]

  table_ws.addColumn('double', 'Spectrum')
  table_ws.addColumn('double', 'theta')
  table_ws.addColumn('double', 't0')
  table_ws.addColumn('double', 'L0')
  table_ws.addColumn('double', 'L1')

  for row in file_data:
    table_ws.addRow(row.tolist())

  return table_name

#----------------------------------------------------------------------------------------

def read_table_column(table_name, column_name, spec_list=DETECTOR_RANGE):
  """
    Read a column from a table workspace representing the instrument parameter file and return the data as an array.

    @param table_name - name of the table workspace
    @param column_name - name of the column to select
    @param spec_list - range of spectra to use
    @return numpy array of values in the spec_list range
  """

  offset = DETECTOR_RANGE[0]
  if len(spec_list) > 1:
    lower, upper = spec_list
  else:
    lower = spec_list[0]
    upper = spec_list[0]

  column_values = mtd[table_name].column(column_name)

  return np.array(column_values[lower-offset:upper+1-offset])
  
#----------------------------------------------------------------------------------------

def read_fitting_result_table_column(table_name, column_name, spec_list):
  """
    Read a column from a table workspace resulting from fitting and return the data as an array.

    @param table_name - name of the table workspace
    @param column_name - name of the column to select
    @param spec_list - range of spectra to use
    @return numpy array of values in the spec_list range
  """

  offset = spec_list[0]
  if len(spec_list) > 1:
    lower, upper = spec_list
  else:
    lower = spec_list[0]
    upper = spec_list[0]

  column_values = mtd[table_name].column(column_name)

  return np.array(column_values[lower-offset:upper+1-offset])
  
#----------------------------------------------------------------------------------------

def generate_fit_function_header(function_type, error=False):
    if function_type == 'Voigt':
        error_str = "Err" if error else ""
        func_header = {'Height': 'LorentzAmp', 'Position': 'LorentzPos', 'Width': 'LorentzFWHM', 'Width_2': 'GaussianFWHM'}
    elif function_type == 'Gaussian':
        error_str = "_Err" if error else ""
        func_header = {'Height': 'Height', 'Width': 'Sigma', 'Position': 'PeakCentre'}
    else:
        raise ValueError("Unsupported fit function type: %s" % function_type)

    return {k: v+error_str for k,v in func_header.items()}

#----------------------------------------------------------------------------------------


class EVSCalibrationFit(PythonAlgorithm):

  def summary(self):
    return "Fits peaks to a list of spectra using the mass or the d-spacings (for bragg peaks) of the sample."

  def category(self):
        return "VesuvioCalibration"

  def PyInit(self):

    self.declareProperty(StringArrayProperty("Samples", Direction.Input),
      doc="Sample run numbers to fit peaks to.")

    self.declareProperty(StringArrayProperty("Background", Direction.Input),
      doc="Run numbers to use as a background.")

    self.declareProperty('Mode', 'FoilOut', StringListValidator(MODES),
      doc="Mode to load files with. This is passed to the LoadVesuvio algorithm. Default is FoilOut.")

    self.declareProperty('Function', 'Gaussian', StringListValidator(['Gaussian', 'Voigt']),
      doc="Function to fit each of the spectra with. Default is Gaussian")

    spectrum_validator = IntArrayBoundedValidator()
    spectrum_validator.setLower(DETECTOR_RANGE[0])
    spectrum_validator.setUpper(DETECTOR_RANGE[1])
    self.declareProperty(IntArrayProperty('SpectrumRange', DETECTOR_RANGE, spectrum_validator, Direction.Input),
      doc='Spectrum range to use. Default is the total range (%d,%d)' % tuple(DETECTOR_RANGE))

    self.declareProperty('Mass', 207.19,
      doc="Mass of the sample in amu to be used when calculating energy. Default is Pb: 207.19")

    greaterThanZero = FloatArrayBoundedValidator()
    greaterThanZero.setLower(0)
    self.declareProperty(FloatArrayProperty('DSpacings', [], greaterThanZero, Direction.Input),
      doc="List of d-spacings used to estimate the positions of bragg peaks in TOF.")


    self.declareProperty(FloatArrayProperty('Energy', [ENERGY_ESTIMATE], FloatArrayMandatoryValidator(), Direction.Input),
      doc='List of estimated expected energies for peaks. Optional: the default is %f' % ENERGY_ESTIMATE)

    self.declareProperty(FileProperty('InstrumentParameterFile', '', action=FileAction.OptionalLoad, extensions=["par"]),
      doc='Filename of the instrument parameter file.')

    self.declareProperty('PeakType', '', StringListValidator(PEAK_TYPES),
                         doc='Choose the peak type that is being fitted.'
                         'Note that supplying a set of dspacings overrides the setting here')

    self.declareProperty(ITableWorkspaceProperty("InstrumentParameterWorkspace", "", Direction.Input, PropertyMode.Optional),
      doc='Workspace contain instrument parameters.')

    self.declareProperty('CreateOutput', False,
      doc='Create fitting workspaces for each of the parameters.')

    self.declareProperty('OutputWorkspace', '', StringMandatoryValidator(),
      doc="Name to call the output workspace.")

#----------------------------------------------------------------------------------------

  def PyExec(self):
    self._setup()
    self._preprocess()

    if self._fitting_bragg_peaks:
        self._fit_bragg_peaks()
    else:
        self._fit_peaks()

    #create output of fit if required.
    if self._create_output and not self._fitting_bragg_peaks:
      self._generate_fit_workspaces()

    #clean up workspaces
    if self._param_file != "":
      DeleteWorkspace(self._param_table)

#----------------------------------------------------------------------------------------

  def _setup(self):
    """
      Setup parameters for fitting.
    """
    self._sample_run_numbers = self.getProperty("Samples").value
    self._bkg_run_numbers = self.getProperty("Background").value
    self._peak_function = self.getProperty("Function").value
    self._mode = self.getProperty("Mode").value
    self._energy_estimates = self.getProperty("Energy").value
    self._sample_mass = self.getProperty("Mass").value
    self._d_spacings = self.getProperty("DSpacings").value
    self._param_file = self.getPropertyValue('InstrumentParameterFile')
    self._peak_type = self.getPropertyValue('PeakType')
    self._param_workspace = self.getPropertyValue('InstrumentParameterWorkspace')
    self._spec_list = self.getProperty("SpectrumRange").value.tolist()
    self._output_workspace_name = self.getPropertyValue("OutputWorkspace")
    self._create_output = self.getProperty("CreateOutput").value

    self._d_spacings.sort()

    #validate spectra list
    if len(self._spec_list) > 2:
      self._spec_list = [self._spec_list[0], self._spec_list[-1]]
    elif len(self._spec_list) == 1:
      self._spec_list = [self._spec_list[0]]
    elif len(self._spec_list) < 1:
      raise ValueError("You must specify a spectrum range.")

    #validate sample run numbers
    if len(self._sample_run_numbers) == 0:
      raise ValueError("You must supply at least one sample run number.")

    self._sample = self._output_workspace_name + '_Sample_' + '_'.join(self._sample_run_numbers)
    if len(self._bkg_run_numbers) > 0:
      self._background = '' + self._bkg_run_numbers[0]

    #fit function type
    self._func_param_names = generate_fit_function_header(self._peak_function)
    self._func_param_names_error = generate_fit_function_header(self._peak_function, error=True)

    #validate instrument parameter workspace/file
    if self._param_workspace != "":
      self._param_table = self._param_workspace
    elif self._param_file != "":
      base = os.path.basename(self._param_file)
      self._param_table = os.path.splitext(base)[0]
      load_instrument_parameters(self._param_file, self._param_table)

    # check peak types
    self._fitting_recoil_peaks = False
    self._fitting_resonance_peaks = False
    self._fitting_bragg_peaks = len(self._d_spacings) > 0
    if self._fitting_bragg_peaks:
        self._ws_crop_range = BRAGG_PEAK_CROP_RANGE
        self._fit_window_range = BRAGG_FIT_WINDOW_RANGE          
    else:
      if self._peak_type == "Recoil":
        self._fitting_recoil_peaks = True
        self._ws_crop_range = RECOIL_PEAK_CROP_RANGE
        self._fit_window_range = RECOIL_FIT_WINDOW_RANGE
      elif self._peak_type == "Resonance":
        self._fitting_resonance_peaks = True
        self._ws_crop_range = RESONANCE_PEAK_CROP_RANGE
        self._fit_window_range = RESONANCE_FIT_WINDOW_RANGE

#----------------------------------------------------------------------------------------

  def _preprocess(self):
    """
      Preprocess a workspace. This include optionally dividing by a background
    """
    self._load_files(self._sample_run_numbers, self._sample)

    xmin, xmax = self._ws_crop_range
    CropWorkspace(self._sample, XMin=xmin, XMax=xmax, OutputWorkspace=self._sample)

    if len(self._bkg_run_numbers) > 0:
      self._load_files(self._bkg_run_numbers, self._background)

      CropWorkspace(self._background, XMin=xmin, XMax=xmax, OutputWorkspace=self._background)
      RebinToWorkspace(WorkspaceToRebin=self._background, WorkspaceToMatch=self._sample, OutputWorkspace=self._background)
      Divide(self._sample, self._background, OutputWorkspace=self._sample)

      DeleteWorkspace(self._background)

    ReplaceSpecialValues(self._sample, NaNValue=0, NaNError=0, InfinityValue=0, InfinityError=0, OutputWorkspace=self._sample)

#----------------------------------------------------------------------------------------

  def _fit_bragg_peaks(self):
    #ESTIMATE PEAK POSITIONS USING D SPACING
    peak_positions = self._estimate_bragg_peak_positions()
    num_peaks, num_spectra = peak_positions.shape
    self._prog_reporter = Progress(self, 0.0, 1.0, num_spectra)

    #CREATE OUTPUT PARAMETERS TABLE
    peaks_table = self._output_workspace_name + '_Peak_Parameters'
    CreateEmptyTableWorkspace(OutputWorkspace=peaks_table)

    #PARAM NAMES FOR LINEAR BACKGROUND
    param_names = ['f0.A0', 'f0.A1']

    #ADD OTHER PARAM NAMES DEPENDING UPON CHOSEN PEAK FUNCTION
    for i in range(num_peaks):
      param_names += ['f' + str(i) + '.' + name for name in self._func_param_names.values()]

    #GENERATE ERROR NAMES FROM PARAM NAMES, ZIP TO GET COL NAMES
    err_names = [name + '_Err' for name in param_names]
    col_names = [element for tupl in zip(param_names, err_names) for element in tupl]

    #CREATE OUTPUT TABLE FROM COL NAMES, STARTING WITH SPECTRA COLUMN
    mtd[peaks_table].addColumn('int', 'Spectrum')
    for name in col_names:
        mtd[peaks_table].addColumn('double', name)

    output_workspaces = []
    #LOOP PEAK ESTIMATES FOR EACH SPECTRUM
    for i, peak_estimates_list in enumerate(peak_positions.transpose()):
        self._prog_reporter.report("Fitting to spectrum %d" % i)

        spec_number = self._spec_list[0]+i
        peak_table = self._sample + '_peaks_table_%d' % spec_number
        find_peak_params = self._get_find_peak_parameters(spec_number, peak_estimates_list)
        logger.notice(str(i) + '   ' + str(find_peak_params))
        #FIND PEAKS USING ESTIMATES, CONSTRAINED BY PEAK PARAMETERS
        FindPeaks(InputWorkspace=self._sample, WorkspaceIndex=i, PeaksList=peak_table, **find_peak_params)

        fit_output_name = self._output_workspace_name + '_Spec_%d' % spec_number
        status, chi2, ncm, params, fws, func, cost_func, xMin, xMax = self._fit_found_peaks(peak_table, peak_estimates_list, i, fit_output_name)
        status = "peaks invalid" if not self._check_fitted_peak_validity(fit_output_name + '_Parameters', peak_estimates_list, peak_height_rel_threshold=PEAK_HEIGHT_RELATIVE_THRESHOLD) else status

        if not status == "success":
            # FIND PEAKS RELATIVELY UNCONSTRAINED (u) BY PEAK PARAMETERS
            self._prog_reporter.report("Fitting to spectrum %d without constraining parameters" % i)
            peak_table_u = self._sample + '_peaks_table_unconstrained%d' % spec_number
            find_peak_params_u = self._get_find_peak_parameters(spec_number, None, unconstrained=True)
            try:  # Needed as sometimes "Matrix A is singular" occurs, have been unable to work out why.
                FindPeaks(InputWorkspace=self._sample, WorkspaceIndex=i, PeaksList=peak_table_u, **find_peak_params_u)
                peaks_found = True
            except ValueError:
                logger.error("This error relates to the unconstrained workflow. Result from standard workflow will be used.")
                peaks_found = False

            if peaks_found:
                peak_table_temp = peak_table + "_temp"
                CloneWorkspace(InputWorkspace=mtd[peak_table], OutputWorkspace=peak_table_temp)

                # FILTER FOUND PEAKS USING ESTIMATED POSITIONS AND MINIMUM HEIGHT
                linear_bg_coeffs = self._calc_linear_bg_coefficients()
                self._filter_found_peaks(peak_table_u, peak_estimates_list, peak_table_temp, linear_bg_coeffs,
                                         peak_height_rel_threshold=PEAK_HEIGHT_RELATIVE_THRESHOLD)
                fit_output_name_u = fit_output_name + "_unconstrained"
                status_u, chi2_u, ncm_u, params_u, fws_u, func_u, cost_func_u, xMin, xMax = self._fit_found_peaks(peak_table_temp, None, i,
                                                                                                                  fit_output_name_u,
                                                                                                                  (xMin, xMax))
                status_u = "peaks invalid" if not self._check_fitted_peak_validity(fit_output_name_u + '_Parameters', peak_estimates_list,
                                                                                   peak_height_rel_threshold=PEAK_HEIGHT_RELATIVE_THRESHOLD) else status_u
                if chi2_u < chi2 and status_u != "peaks invalid":
                    params = params_u
                    status = "unused"  # mark initial fit as unused
                    self._prog_reporter.report("Fit to spectrum %d without constraining parameters successful" % i)
                else:
                    DeleteWorkspace(fit_output_name_u + '_Workspace')
                    DeleteWorkspace(fit_output_name_u + '_NormalisedCovarianceMatrix')
                    DeleteWorkspace(fit_output_name_u + '_Parameters')

                DeleteWorkspace(peak_table + "_temp")
                DeleteWorkspace(peak_table_u)
        DeleteWorkspace(peak_table)

        fit_values = dict(zip(params.column(0), params.column(1)))
        fit_errors = dict(zip(params.column(0), params.column(2)))

        row_values = [fit_values['f0.A0'], fit_values['f0.A1']]
        row_errors = [fit_errors['f0.A0'], fit_errors['f0.A1']]
        row_values += [fit_values['f1.' + name] for name in param_names[2:]]
        row_errors += [fit_errors['f1.' + name] for name in param_names[2:]]
        row = [element for tupl in zip(row_values, row_errors) for element in tupl]

        mtd[peaks_table].addRow([spec_number] + row)

        DeleteWorkspace(fit_output_name + '_NormalisedCovarianceMatrix')
        DeleteWorkspace(fit_output_name + '_Parameters')
        if status == "unused":
            DeleteWorkspace(fit_output_name_u + '_NormalisedCovarianceMatrix')
            DeleteWorkspace(fit_output_name_u + '_Parameters')

        if status == "unused" or not self._create_output:
            DeleteWorkspace(fit_output_name + '_Workspace')
        if status == "unused" and not self._create_output:
            DeleteWorkspace(fit_output_name_u + '_Workspace')
        elif status == "unused":
            output_workspaces.append(fit_output_name_u + '_Workspace')
        else:
            output_workspaces.append(fit_output_name + '_Workspace')

    if self._create_output:
        GroupWorkspaces(','.join(output_workspaces), OutputWorkspace=self._output_workspace_name + "_Peak_Fits")

#----------------------------------------------------------------------------------------

  def _calc_linear_bg_coefficients(self):
    temp_bg_fit = Fit(Function="(name=LinearBackground,A0=0,A1=0)", InputWorkspace=self._sample, Output="temp_bg_fit", CreateOutput=True)
    linear_bg_coeffs = temp_bg_fit.OutputParameters.cell("Value", 0), temp_bg_fit.OutputParameters.cell("Value", 1)
    DeleteWorkspace('temp_bg_fit_Workspace')
    DeleteWorkspace('temp_bg_fit_NormalisedCovarianceMatrix')
    DeleteWorkspace('temp_bg_fit_Parameters')
    return linear_bg_coeffs

#----------------------------------------------------------------------------------------

  def _check_fitted_peak_validity(self, table_name, estimated_peaks, peak_height_abs_threshold=0, peak_height_rel_threshold=0):
    check_nans = self._check_nans(table_name)
    check_peak_positions = self._check_peak_positions(table_name, estimated_peaks)
    check_peak_heights = self._check_peak_heights(table_name, peak_height_abs_threshold, peak_height_rel_threshold)
    
    if check_nans and check_peak_positions and check_peak_heights:
        return True
    else:
        return False
    
#----------------------------------------------------------------------------------------

  def _check_nans(self, table_name):
    table_ws = mtd[table_name]
    for i in table_ws.column("Value"):
        if np.isnan(i):
            print(f"nan found in value common, indicates invalid peak")
            return False
    return True
    
#----------------------------------------------------------------------------------------
    
  def _check_peak_positions(self, table_name, estimated_positions):
    pos_str = self._func_param_names["Position"]

    i = 0
    invalid_positions = []
    for name, value in zip(mtd[table_name].column("Name"), mtd[table_name].column("Value")):
        if pos_str in name:
            if not (value >= estimated_positions[i] - BRAGG_PEAK_POSITION_TOLERANCE and value <= estimated_positions[i] + BRAGG_PEAK_POSITION_TOLERANCE):
                invalid_positions.append(value)
            i += 1

    if len(invalid_positions) > 0:
        print(f"Invalid peak positions found: {invalid_positions}")
        return False
    else:
        return True

#----------------------------------------------------------------------------------------

  def _evaluate_peak_height_against_bg(self, height, position, linear_bg_A0, linear_bg_A1, rel_threshold, abs_threshold):
    bg = position * linear_bg_A1 + linear_bg_A0
    required_height = bg*rel_threshold + abs_threshold
    if height < required_height:
        return False
    else:
        return True

#----------------------------------------------------------------------------------------

  def _check_peak_heights(self, table_name, abs_threshold_over_bg, rel_threshold_over_bg):
    height_str = self._func_param_names["Height"]
    pos_str = self._func_param_names["Position"]
    
    peak_heights = []
    peak_positions = []
    for name, value in zip(mtd[table_name].column("Name"), mtd[table_name].column("Value")):
        if height_str in name:
            peak_heights.append(value)
        elif pos_str in name:
            peak_positions.append(value)
        elif name == "f0.A0":
            linear_bg_A0 = value
        elif name == "f0.A1":
            linear_bg_A1 = value

    for height, pos in zip(peak_heights, peak_positions):
        if not self._evaluate_peak_height_against_bg(height, pos, linear_bg_A0, linear_bg_A1, abs_threshold_over_bg, rel_threshold_over_bg):
            print(f"Peak height threshold not met. Found height: {height}")
            return False
    return True

#----------------------------------------------------------------------------------------

  def _filter_found_peaks(self, peak_table, peak_estimates_list, table_to_overwrite, linear_bg_coeffs, peak_height_abs_threshold=0, peak_height_rel_threshold=0):
    peak_estimate_deltas = []
    linear_bg_A0, linear_bg_A1 = linear_bg_coeffs
    #with estimates for spectrum, loop through all peaks, get and store delta from peak estimates
    for peak_estimate_index, peak_estimate in enumerate(peak_estimates_list):
        for position_index, (position, height) in enumerate(zip(mtd[peak_table].column(2), mtd[peak_table].column(1))):
            if not position == 0 and self._evaluate_peak_height_against_bg(height, position, linear_bg_A0, linear_bg_A1, peak_height_abs_threshold, peak_height_rel_threshold):
                peak_estimate_deltas.append((peak_estimate_index, position_index, abs(position - peak_estimate)))

    #loop through accendings delta, assign peaks until there are none left to be assigned.
    peak_estimate_deltas.sort(key=partial(self._get_x_elem, elem=2))
    index_matches = []
    for peak_estimate_index, position_index, delta in peak_estimate_deltas:
        #if all estimated peaks matched, break
        if len(index_matches)==len(peak_estimates_list):
            break
        #assign match for smallest delta if that position or estimate has not been already matched
        if position_index not in [i[1] for i in index_matches] and \
           peak_estimate_index not in [i[0] for i in index_matches] and \
           all(x>position_index for x in [i[1] for i in index_matches if i[0]>peak_estimate_index]) and \
           all(x<position_index for x in [i[1] for i in index_matches if i[0]<peak_estimate_index]):
              index_matches.append((peak_estimate_index, position_index))

    if len(index_matches) > 0:
        index_matches.sort(key=partial(self._get_x_elem, elem=1))
        mtd[table_to_overwrite].setRowCount(len(peak_estimates_list))
        for col_index in range(mtd[table_to_overwrite].columnCount()):
            match_n = 0
            for row_index in range(mtd[table_to_overwrite].rowCount()):
                if row_index in [x[0] for x in index_matches]:
                    position_row_index = index_matches[match_n][1]
                    mtd[table_to_overwrite].setCell(row_index,col_index, mtd[peak_table].cell(position_row_index, col_index))
                    match_n+=1
                else: #otherwise just use estimate position
                    pos_str = self._func_param_names["Position"]
                    mtd[table_to_overwrite].setCell(pos_str, row_index, peak_estimates_list[row_index])

#----------------------------------------------------------------------------------------
  def _get_x_elem(self, input_list, elem):
    return input_list[elem]

#----------------------------------------------------------------------------------------

  def _fit_found_peaks(self, peak_table, peak_estimates_list, workspace_index, fit_output_name, xLimits=None):
    #get parameters from the peaks table
    peak_params = []
    prefix = ''#'f0.'
    position = prefix + self._func_param_names['Position']

    if peak_estimates_list is not None: #If no peak estimates list, we are doing an unconstrained fit
        #Don't yet understand what this is doing here
        #print(peak_estimates_list)
        self._set_table_column(peak_table, position, peak_estimates_list, spec_list=None)
        unconstrained=False
    else:
        unconstrained=True

    for peak_index in range(mtd[peak_table].rowCount()):
            peak_params.append(mtd[peak_table].row(peak_index))

    #build function string
    func_string = self._build_multiple_peak_function(peak_params, workspace_index, unconstrained)

    #select min and max x range for fitting
    positions = [params[position] for params in peak_params]

    if len(positions) < 1:
        raise RuntimeError("FindPeaks could not find a peaks with the given parameters: %s" % find_peak_params)

    if xLimits:
        xmin, xmax = xLimits
    else:
        xmin = min(peak_estimates_list) - BRAGG_PEAK_POSITION_TOLERANCE - self._fit_window_range
        xmax = max(peak_estimates_list) + BRAGG_PEAK_POSITION_TOLERANCE + self._fit_window_range

    #xmin, xmax = None, None

    #fit function to workspace
    return Fit(Function=func_string, InputWorkspace=self._sample, IgnoreInvalidData=True, StartX=xmin, EndX=xmax,
               WorkspaceIndex=workspace_index, CalcErrors=True, Output=fit_output_name, Minimizer='Levenberg-Marquardt,AbsError=0,RelError=1e-8') \
               + (xmin,) + (xmax,)

#----------------------------------------------------------------------------------------

  def _fit_peaks(self):
    """
      Fit peaks to time of flight data.

      This uses the Mantid algorithms FindPeaks and Fit. Estimates for the centre of the peaks
      are found using either the energy or d-spacings provided. It creates a group workspace
      containing one table workspace per peak with parameters for each detector.
    """

    #ESTIMATE PEAK POSITIONS FROM ENERGY ESTIMATES AND PREVIOUS PARAMETERS.
    peak_positions = self._estimate_peak_positions()
    num_peaks, num_spectra = peak_positions.shape
    self._prog_reporter = Progress(self,0.0,1.0,num_peaks*num_spectra)

    self._parameter_tables = []
    self._fit_workspaces = []
    #LOOP PEAK ESTIMATES FOR EACH SPECTRUM
    for i, peak_estimates_list in enumerate(peak_positions):
      #create parameter table
      peaks_table = self._output_workspace_name + '_Peak_%d_Parameters' % i
      param_names = self._create_parameter_table(peaks_table)

      #fit every spectrum
      self._peak_fit_workspaces = []
      #LOOP THROUGH PEAK ESTIMATES FOR THIS SPECTRUM
      for j, peak_centre in enumerate(peak_estimates_list):
        spec_number = self._spec_list[0]+j
        self._prog_reporter.report("Fitting peak %d to spectrum %d" % (i,spec_number))

        #find inital parameters given the estimate position
        peak_table = '__' + self._sample + '_peaks_table_%d_%d' % (i,j)
        find_peak_params = self._get_find_peak_parameters(spec_number, [peak_centre])
        #FIND DATA PEAK RELATED TO ESTIMATED PEAK
        FindPeaks(InputWorkspace=self._sample, WorkspaceIndex=j, PeaksList=peak_table, **find_peak_params)

        #extract data from table
        if mtd[peak_table].rowCount() > 0:
          peak_params = mtd[peak_table].row(0)
          DeleteWorkspace(peak_table)
        else:
          logger.error('FindPeaks could not find any peaks matching the parameters:\n' + str(find_peak_params))
          sys.exit()

        #build fit function string
        func_string = self._build_function_string(peak_params)
        fit_output_name = '__' + self._output_workspace_name + '_Peak_%d_Spec_%d' % (i,j)

        #find x window
        xmin, xmax = None, None
        position = '' + self._func_param_names['Position']
        if peak_params[position] > 0:
          xmin = peak_params[position]-self._fit_window_range
          xmax = peak_params[position]+self._fit_window_range
        else:
          logger.warning('Could not specify fit window. Using full spectrum x range.')

        status, chi2, ncm, params, fws, func, cost_func = Fit(Function=func_string, InputWorkspace=self._sample, IgnoreInvalidData=True,
                                                              StartX=xmin, EndX=xmax, WorkspaceIndex=j,
                                                              CalcErrors=True, Output=fit_output_name,
                                                              Minimizer='Levenberg-Marquardt,RelError=1e-8')

        #output fit parameters to table workspace
        fit_values = dict(zip(params.column(0), params.column(1)))
        fit_errors = dict(zip(params.column(0), params.column(2)))

        row_values = [fit_values[name] for name in param_names]
        row_errors = [fit_errors[name] for name in param_names]
        row = [element for tupl in zip(row_values, row_errors) for element in tupl]

        mtd[peaks_table].addRow([spec_number] + row)

        self._parameter_tables.append(peaks_table)
        self._peak_fit_workspaces.append(fws.name())

        DeleteWorkspace(ncm)
        DeleteWorkspace(params)
        if not self._create_output:
          DeleteWorkspace(fws)

      self._fit_workspaces.append(self._peak_fit_workspaces)

    GroupWorkspaces(self._parameter_tables, OutputWorkspace=self._output_workspace_name + '_Peak_Parameters')

    
#----------------------------------------------------------------------------------------
    
  def _get_find_peak_parameters(self, spec_number, peak_centre, unconstrained=False):
    """
      Get find peak parameters

      @param spec_num - the current spectrum number being fitted
      @return dictionary of parameters for find peaks
    """

    find_peak_params = {}
    find_peak_params['PeakFunction'] = self._peak_function
    find_peak_params['RawPeakParameters'] = True
    find_peak_params['BackgroundType'] = 'Linear'

    if not unconstrained:
        find_peak_params['PeakPositions'] = peak_centre

    if self._fitting_bragg_peaks and not unconstrained:
        find_peak_params['PeakPositionTolerance'] = BRAGG_PEAK_POSITION_TOLERANCE

        if spec_number >= FRONTSCATTERING_RANGE[0]:
            find_peak_params['FWHM'] = 70
        else:
            find_peak_params['FWHM'] = 5

    elif not self._fitting_bragg_peaks:
      if self._fitting_resonance_peaks:
        # 25 seems to be able to fit the final peak in the first backscattering spectrum
        if spec_number >= FRONTSCATTERING_RANGE[0]:
            half_peak_window = 20
        else:
            half_peak_window = 25
      else:
        # #ust be recoil
        half_peak_window = 60

      fit_windows = [[peak-half_peak_window, peak+half_peak_window] for peak in peak_centre]
      # flatten the list: http://stackoverflow.com/questions/952914/making-a-flat-list-out-of-list-of-lists-in-python
      find_peak_params['FitWindows'] = [peak for pair in fit_windows for peak in pair]

    return find_peak_params

#----------------------------------------------------------------------------------------

  def _create_parameter_table(self, peaks_table):
      """
        Create a table workspace with headers corresponding to the parameters
        output from fitting.

        @param peak_table - name to call the parameter table.
        @return param_names - name of the columns
      """
      CreateEmptyTableWorkspace(OutputWorkspace=peaks_table)

      param_names = ['f0.A0', 'f0.A1']
      param_names += ['f1.' + name for name in self._func_param_names.values()]

      err_names = [name + '_Err' for name in param_names]
      col_names = [element for tupl in zip(param_names, err_names) for element in tupl]

      mtd[peaks_table].addColumn('int', 'Spectrum')
      for name in col_names:
        mtd[peaks_table].addColumn('double', name)

      return param_names

#----------------------------------------------------------------------------------------

  def _estimate_peak_positions(self):
    """
      Estimate the position of a peak based on a given energy and previous calibrated parameters.

      @return list of estimates in tof for the peak centres
    """
    L0 = self._read_param_column('L0', self._spec_list)
    L1 = self._read_param_column('L1', self._spec_list)
    t0 = self._read_param_column('t0', self._spec_list)
    t0 /= 1e+6
    thetas = self._read_param_column('theta', self._spec_list)
    r_theta = calculate_r_theta(self._sample_mass, thetas)

    self._energy_estimates = self._energy_estimates.reshape(1, self._energy_estimates.size).T
    self._energy_estimates *= MEV_CONVERSION

    v1 = np.sqrt(2 * self._energy_estimates / scipy.constants.m_n)
    if self._fitting_recoil_peaks:
      # for recoil peaks:
      tof = ((L0 * r_theta + L1) / v1) + t0
    elif self._fitting_resonance_peaks:
      # for resonances:
      tof = (L0 / v1) + t0
    else:
      raise RuntimeError("Cannot estimate peak positions for unknown peak type")

    tof *= 1e+6
    return tof

#----------------------------------------------------------------------------------------

  def _estimate_bragg_peak_positions(self):
    """
      Estimate the position of bragg peaks in TOF using estimates of the parameters
      and d-spacings of the sample.

      The number of peaks will depend on the number of d-spacings provided.

      @return estimates of peak positions in TOF for all spectra
    """

    L0 = self._read_param_column('L0', self._spec_list)
    L1 = self._read_param_column('L1', self._spec_list)
    t0 = self._read_param_column('t0', self._spec_list)
    thetas = self._read_param_column('theta', self._spec_list)

    t0 /= 1e+6

    self._d_spacings *= 1e-10
    self._d_spacings = self._d_spacings.reshape(1, self._d_spacings.size).T

    lambdas = 2 * self._d_spacings * np.sin(np.radians(thetas) / 2)
    
    L1_nan_to_num = np.nan_to_num(L1)
    
    print(L1_nan_to_num)
    
    tof = (lambdas * scipy.constants.m_n * (L0 + L1_nan_to_num)) / scipy.constants.h + t0
    tof *= 1e+6

    return tof

#----------------------------------------------------------------------------------------

  def _load_files(self, ws_numbers, output_name):
    """
      Load a list of run numbers and sum each of the runs.

      @param ws_numbers - list of run numbers to load.
      @param output_name - name to call the final workspace.
    """

    run_numbers = [run for run in self._parse_run_numbers(ws_numbers)]

    self._load_file(run_numbers[0], output_name)
    temp_ws_name = '__EVS_calib_temp_ws'

    for run_number in run_numbers[1:]:
      self._load_file(run_number, temp_ws_name)
      Plus(output_name, temp_ws_name, OutputWorkspace=output_name)
      DeleteWorkspace(temp_ws_name)

#----------------------------------------------------------------------------------------

  def _parse_run_numbers(self, run_numbers):
      """
        Converts a list of mixed single runs and run ranges
        into a single flat list of run numbers.

        @param run_numbers - list of mixed single runs and run ranges
        @return iterator to each run number in the flat list of runs
      """
      for run in run_numbers:
        if '-' in run:
          sample_range = run.split('-')
          sample_range = map(int, sample_range)

          for i in range(*sample_range):
            yield str(i)

        else:
          yield run

#----------------------------------------------------------------------------------------

  def _load_file(self, ws_name, output_name):
    """
      Load a file into a workspace.

      This will attempt to use LoadVesuvio, but will fall back on LoadRaw if LoadVesuvio fails.
      @param ws_name - name of the run to load.
      @param output_name - name to call the loaded workspace
    """

    try:
      LoadVesuvio(Filename=ws_name, Mode=self._mode, OutputWorkspace=output_name,
                  SpectrumList="%d-%d" % (self._spec_list[0], self._spec_list[-1]),
                  EnableLogging=False)
    except RuntimeError:
      LoadRaw('EVS' + ws_name + '.raw', OutputWorkspace=output_name,
              SpectrumMin=self._spec_list[0], SpectrumMax=self._spec_list[-1],
              EnableLogging=False)
      ConvertToDistribution(output_name, EnableLogging=False)

#----------------------------------------------------------------------------------------

  def _build_linear_background_function(self, peak, tie_gradient=False):
    """
      Builds a string describing a peak function that can be passed to Fit.
      This will be either a Gaussian or Lorentzian shape.

      @param peaks - dictionary containing the parameters for the function
      @return the function string
    """

    bkg_intercept, bkg_graident = peak['A0'], peak['A1']
    fit_func = 'name=LinearBackground, A0=%f, A1=%f' % (bkg_intercept, bkg_graident)
    fit_func += ';'

    return fit_func

#----------------------------------------------------------------------------------------

  def _build_peak_function(self, peak):
    """
      Builds a string describing a linear background function that can be passed to Fit.

      @param peaks - dictionary containing the parameters for the function
      @return the function string
    """

    fit_func = 'name=%s, ' % self._peak_function

    prefix = ''#f0.'
    func_list = [name + '=' + str(peak[prefix + name]) for name in self._func_param_names.values()]
    fit_func += ', '.join(func_list) + ';'

    return fit_func

#----------------------------------------------------------------------------------------

  def _build_multiple_peak_function(self, peaks, workspace_index, unconstrained):
    """
      Builds a string describing a composite function that can be passed to Fit.
      This will be a linear background with multiple peaks of either a Gaussian or Voigt shape.

      @param peaks - list of dictionaries containing the parameters for the function
      @return the function string
    """

    if(len(peaks) == 0):
        return ''

    if not unconstrained:
        fit_func = self._build_linear_background_function(peaks[0], False)
    else:
        pos_str = self._func_param_names["Position"]
        if len(peaks) > 1:
            fit_window = [peaks[0][pos_str] - 1000, peaks[len(peaks)-1][pos_str] + 1000]
        else:
            fit_window = [peaks[0][pos_str] - 1000, peaks[0][pos_str] + 1000]
        FindPeakBackground(InputWorkspace=self._sample, WorkspaceIndex=workspace_index, FitWindow=fit_window, BackgroundType="Linear", OutputWorkspace="temp_fit_bg")
        fit_func = f'name=LinearBackground, A0={mtd["temp_fit_bg"].cell("bkg0",0)}, A1={mtd["temp_fit_bg"].cell("bkg1",0)};'
        DeleteWorkspace("temp_fit_bg")

    fit_func += '('
    for i, peak in enumerate(peaks):
        fit_func += self._build_peak_function(peak)

    fit_func = fit_func[:-1]
    fit_func += ');'

    return fit_func

#----------------------------------------------------------------------------------------

  def _build_function_string(self, peak):
    """
      Builds a string describing a composite function that can be passed to Fit.
      This will be a linear background with either a Gaussian or Voigt function.

      @param peak - dictionary containing the parameters for the function
      @return the function string
    """

    if(len(peak) == 0):
        return ''

    fit_func = self._build_linear_background_function(peak)
    fit_func += self._build_peak_function(peak)

    return fit_func

#----------------------------------------------------------------------------------------

  def _read_param_column(self, column_name, spec_list=DETECTOR_RANGE):
    """
      Read a column from a table workspace and return the data as an array.

      @param column_name - name of the column to select
      @param spec_list - range of spectra to use
      @return numpy array of values in the spec_list range
    """

    return read_table_column(self._param_table, column_name, spec_list)

#----------------------------------------------------------------------------------------

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
      if len(data) < (spec_list[1]+1 - spec_list[0]):
        raise ValueError("Not enough data from spectrum range.")

      offset = spec_list[0] - DETECTOR_RANGE[0]

    if isinstance(data, np.ndarray):
      data = data.tolist()

    for i, value in enumerate(data):
      table_ws.setCell(column_name, offset+i, value)

#----------------------------------------------------------------------------------------

  def _generate_fit_workspaces(self):
    """
      Output the fit workspace for each spectrum fitted.
    """
    fit_workspaces = map(list, zip(*self._fit_workspaces))
    group_ws_names = []
    for i, peak_fits in enumerate(fit_workspaces):
      ws_name = self._output_workspace_name + '_%d_Workspace' % i
      ExtractSingleSpectrum(InputWorkspace=self._sample, WorkspaceIndex=i, OutputWorkspace=ws_name)

      #transfer fits to individual spectrum
      peak_collection = WorkspaceFactory.create(mtd[ws_name], NVectors=len(peak_fits))
      data_x = mtd[ws_name].readX(0)

      ax = TextAxis.create(len(peak_fits)+2)
      ax.setLabel(0, "Data")

      for j, peak in enumerate(peak_fits):
        peak_x = mtd[peak].readX(0)
        peak_y = mtd[peak].readY(1)
        peak_e = mtd[peak].readE(1)

        index = int(np.where(data_x == peak_x[0])[0])
        data_y = peak_collection.dataY(j)
        data_y[index:index + peak_y.size] = peak_y

        data_e = peak_collection.dataE(j)
        data_e[index:index + peak_e.size] = peak_e

        peak_collection.setX(j, data_x)
        peak_collection.setY(j, data_y)
        peak_collection.setE(j, data_e)

        ax.setLabel(j+1, "Peak_%d" % (j+1))

        DeleteWorkspace(peak)

      peak_ws = "__tmp_peak_workspace"
      mtd.addOrReplace(peak_ws, peak_collection)
      AppendSpectra(ws_name, peak_ws, ValidateInputs=False, OutputWorkspace=ws_name)

      #create sum of peak fits
      temp_sum_workspace = '__temp_sum_ws'
      SumSpectra(InputWorkspace=peak_ws, OutputWorkspace=peak_ws)
      AppendSpectra(ws_name, peak_ws, ValidateInputs=False, OutputWorkspace=ws_name)
      ax.setLabel(mtd[ws_name].getNumberHistograms()-1, "Total")
      DeleteWorkspace(peak_ws)

      mtd[ws_name].replaceAxis(1, ax)
      group_ws_names.append(ws_name)

    GroupWorkspaces(group_ws_names, OutputWorkspace=self._output_workspace_name + "_Peak_Fits")


#----------------------------------------------------------------------------------------

AlgorithmFactory.subscribe(EVSCalibrationFit)

#########################################################################################

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

    greaterThanZero = FloatArrayBoundedValidator()
    greaterThanZero.setLower(0)
    self.declareProperty(FloatArrayProperty('DSpacings', [], greaterThanZero, Direction.Input),
      doc="List of d-spacings used to estimate the positions of peaks in TOF.")

    self.declareProperty('Iterations', 2, validator=IntBoundedValidator(lower=1),
      doc="Number of iterations to perform. Default is 2.")

    self.declareProperty('CreateOutput', False,
      doc="Whether to create output from fitting.")

    self.declareProperty('CalculateL0', False,
      doc="Whether to calculate L0 or just use the values from the parameter file.")

    self.declareProperty('CreateIPFile', False,
      doc="Whether to save the output as an IP file. \
          This file will use the same name as the OutputWorkspace and will be saved to the default save directory.")

    self.declareProperty('OutputWorkspace', '', StringMandatoryValidator(),
      doc="Name to call the output workspace.")

#----------------------------------------------------------------------------------------

  def PyExec(self):
    self._setup()
    self._current_workspace = self._output_workspace_name + '_Iteration_0'
    self._create_calib_parameter_table(self._current_workspace)

    #PEAK FUNCTIONS
    self._theta_peak_function = 'Gaussian'
    self._theta_func_param_names = generate_fit_function_header(self._theta_peak_function)
    self._theta_func_param_names_error = generate_fit_function_header(self._theta_peak_function, error=True)

    if self._calc_L0:
      #calibrate L0 from the fronstscattering detectors, use the value of the L0 calibrated from frontscattering detectors  for all detectors
      L0_fit = self._current_workspace + '_L0'
      self._run_calibration_fit(Samples=U_FRONTSCATTERING_SAMPLE, Background=U_FRONTSCATTERING_BACKGROUND, SpectrumRange=FRONTSCATTERING_RANGE,
                                InstrumentParameterWorkspace=self._param_table, Mass=U_MASS, Energy=U_PEAK_ENERGIES, OutputWorkspace=L0_fit,
                                CreateOutput=self._create_output, PeakType='Resonance')
      self._L0_peak_fits = L0_fit + '_Peak_Parameters'
      self._calculate_incident_flight_path(self._L0_peak_fits, FRONTSCATTERING_RANGE)
      
      #calibrate t0 from the front scattering detectors
      t0_fit_front = self._current_workspace + '_t0_front'
      self._run_calibration_fit(Samples=U_FRONTSCATTERING_SAMPLE, Background=U_FRONTSCATTERING_BACKGROUND, SpectrumRange=FRONTSCATTERING_RANGE,
                                InstrumentParameterWorkspace=self._param_table, Mass=U_MASS, Energy=U_PEAK_ENERGIES, OutputWorkspace=t0_fit_front,
                                CreateOutput=self._create_output, PeakType='Resonance')
      t0_peak_fits_front = t0_fit_front + '_Peak_Parameters'
      self._calculate_time_delay(t0_peak_fits_front, FRONTSCATTERING_RANGE)

      #calibrate t0 from the backscattering detectors
      t0_fit_back = self._current_workspace + '_t0_back'
      self._run_calibration_fit(Samples=U_BACKSCATTERING_SAMPLE, Background=U_BACKSCATTERING_BACKGROUND, SpectrumRange=BACKSCATTERING_RANGE,
                                InstrumentParameterWorkspace=self._param_table, Mass=U_MASS, Energy=U_PEAK_ENERGIES, OutputWorkspace=t0_fit_back,
                                CreateOutput=self._create_output, PeakType='Resonance')
      t0_peak_fits_back = t0_fit_back + '_Peak_Parameters'
      self._calculate_time_delay(t0_peak_fits_back, BACKSCATTERING_RANGE)
    else:
      #Just copy values over from parameter file
      t0 = read_table_column(self._param_table, 't0', DETECTOR_RANGE)
      L0 = read_table_column(self._param_table, 'L0', DETECTOR_RANGE)
      self._set_table_column(self._current_workspace, 't0', t0)
      self._set_table_column(self._current_workspace, 'L0', L0)

    #repeatedly fit L1, E1 and Theta parameters
    table_group = []
    for i in range(self._iterations):

      #calibrate theta for all detectors
      theta_fit = self._current_workspace + '_theta'
      self._run_calibration_fit(Samples=self._samples, Background=self._background, Function=self._theta_peak_function, Mode='FoilOut', SpectrumRange=DETECTOR_RANGE,
                                InstrumentParameterWorkspace=self._param_table, DSpacings=self._d_spacings, OutputWorkspace=theta_fit, CreateOutput=self._create_output,
                                PeakType='Bragg')
      self._theta_peak_fits = theta_fit + '_Peak_Parameters'
      self._calculate_scattering_angle(self._theta_peak_fits, DETECTOR_RANGE)
      
      #calibrate  E1 for backscattering detectors and use the backscattering averaged value for all detectors    
      E1_fit_back = self._current_workspace + '_E1_back'
      self._run_calibration_fit(Samples=self._samples, Function='Voigt', Mode='SingleDifference', SpectrumRange=BACKSCATTERING_RANGE,
                                InstrumentParameterWorkspace=self._param_table, Mass=self._sample_mass, OutputWorkspace=E1_fit_back, CreateOutput=self._create_output,
                                PeakType='Recoil')
      
      
      E1_peak_fits_back = mtd[self._current_workspace + '_E1_back_Peak_Parameters'].getNames()[0]
      self._calculate_final_energy(E1_peak_fits_back, BACKSCATTERING_RANGE)
      
      # calibrate L1 for backscattering detectors based on the averaged E1 value  and calibrated theta values 
      self._calculate_final_flight_path(E1_peak_fits_back, BACKSCATTERING_RANGE)
      
      # calibrate L1 for frontscattering detectors based on the averaged E1 value  and calibrated theta values 
      E1_fit_front = self._current_workspace + '_E1_front'
      self._run_calibration_fit(Samples=self._samples, Function='Voigt', Mode='SingleDifference', SpectrumRange=FRONTSCATTERING_RANGE,
                                InstrumentParameterWorkspace=self._param_table, Mass=self._sample_mass, OutputWorkspace=E1_fit_front, CreateOutput=self._create_output,
                                PeakType='Recoil')
      
      E1_peak_fits_front = mtd[self._current_workspace + '_E1_front_Peak_Parameters'].getNames()[0]
      self._calculate_final_flight_path(E1_peak_fits_front, FRONTSCATTERING_RANGE)
  

      #make the fitted parameters for this iteration the input to the next iteration.
      table_group.append(self._current_workspace)
      self._param_table = self._current_workspace

      if i < self._iterations-1:
        self._current_workspace = self._output_workspace_name + '_Iteration_%d' % (i+1)
        self._create_calib_parameter_table(self._current_workspace)

        #copy over L0 and t0 parameters to new table
        t0 = read_table_column(self._param_table, 't0', DETECTOR_RANGE)
        t0_error = read_table_column(self._param_table, 't0_Err', DETECTOR_RANGE)
        L0 = read_table_column(self._param_table, 'L0', DETECTOR_RANGE)
        L0_error = read_table_column(self._param_table, 'L0_Err', DETECTOR_RANGE)

        self._set_table_column(self._current_workspace, 't0', t0)
        self._set_table_column(self._current_workspace, 'L0', L0)
        self._set_table_column(self._current_workspace, 't0_Err', t0_error)
        self._set_table_column(self._current_workspace, 'L0_Err', L0_error)

    GroupWorkspaces(','.join(table_group), OutputWorkspace=self._output_workspace_name)

    if self._make_IP_file:
      ws_name = mtd[self._output_workspace_name].getNames()[-1]
      self._save_instrument_parameter_file(ws_name, DETECTOR_RANGE)

#----------------------------------------------------------------------------------------

  def _run_calibration_fit(self, *args, **kwargs):
    """
      Runs EVSCalibrationFit using the AlgorithmManager.

      This allows the calibration script to be run directly from the
      script window after Mantid has started.

      @param args - positional arguments to the algorithm
      @param kwargs - key word arguments to the algorithm
    """
    from mantid.simpleapi import set_properties
    alg = AlgorithmManager.create('EVSCalibrationFit')
    alg.initialize()
    alg.setRethrows(True)
    set_properties(alg, *args, **kwargs)
    alg.execute()
    
#----------------------------------------------------------------------------------------

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

#----------------------------------------------------------------------------------------

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

    spec_range = DETECTOR_RANGE[1]+1 - DETECTOR_RANGE[0]
    mean_L0 = np.empty(spec_range)
    L0_error = np.empty(spec_range)

    mean_L0.fill(np.mean(L0))
    L0_error.fill(scipy.stats.sem(L0))

    self._set_table_column(self._current_workspace, 'L0', mean_L0)
    self._set_table_column(self._current_workspace, 'L0_Err', L0_error)

    DeleteWorkspace(L0_param_table)

#----------------------------------------------------------------------------------------

  def _calculate_final_flight_path(self, peak_table, spec_list):
    """
      Calculate the final flight path using the values for energy.
      This also uses the old value for L1 loaded from the parameter file.

      @param spec_list - spectrum range to calculate t0 for.
    """

    E1 = read_table_column(self._current_workspace, 'E1', spec_list)
    t0 = read_table_column(self._current_workspace, 't0', spec_list)
    t0_error = read_table_column(self._current_workspace, 't0_Err', spec_list)
    L0 = read_table_column(self._current_workspace, 'L0', spec_list)
    theta = read_table_column(self._current_workspace, 'theta', spec_list)
    r_theta = calculate_r_theta(self._sample_mass, theta)
    
    peak_centres = read_fitting_result_table_column(peak_table, 'f1.LorentzPos', spec_list)
    peak_centres_errors = read_fitting_result_table_column(peak_table, 'f1.LorentzPos_Err', spec_list)
    peak_Gaussian_FWHM = read_fitting_result_table_column(peak_table, 'f1.GaussianFWHM', spec_list)
    peak_Gaussian_FWHM_errors = read_fitting_result_table_column(peak_table, 'f1.GaussianFWHM_Err', spec_list)
    peak_Lorentz_FWHM = read_fitting_result_table_column(peak_table, 'f1.LorentzFWHM', spec_list)
    peak_Lorentz_FWHM_errors = read_fitting_result_table_column(peak_table, 'f1.LorentzFWHM_Err', spec_list)
    peak_Lorentz_Amp = read_fitting_result_table_column(peak_table, 'f1.LorentzAmp', spec_list)
    peak_Lorentz_Amp_errors = read_fitting_result_table_column(peak_table, 'f1.LorentzAmp_Err', spec_list)
    
    badSpecs=np.argwhere((np.isinf(peak_Lorentz_Amp_errors)) | (np.isnan(peak_Lorentz_Amp_errors))  | \
    (np.isinf(peak_centres_errors)) | (np.isnan(peak_centres_errors))  | \
    (np.isnan(peak_Gaussian_FWHM_errors)) | (np.isinf(peak_Gaussian_FWHM_errors)) | \
    (np.isnan(peak_Lorentz_FWHM_errors)) | (np.isinf(peak_Lorentz_FWHM_errors)) | \
    (np.isnan(peak_Lorentz_Amp_errors)) | (np.isinf(peak_Lorentz_Amp_errors)) | \
    (np.absolute(peak_Gaussian_FWHM_errors) > np.absolute(peak_Gaussian_FWHM)) | \
    (np.absolute(peak_Lorentz_FWHM_errors) > np.absolute(peak_Lorentz_FWHM)) | \
    (np.absolute(peak_Lorentz_Amp_errors) > np.absolute(peak_Lorentz_Amp)) | \
    (np.absolute(peak_centres_errors) > np.absolute(peak_centres)))
    
    peak_centres[badSpecs] = np.nan


    delta_t = (peak_centres - t0) / 1e+6
    
    delta_t_error = t0_error / 1e+6


    E1 *= MEV_CONVERSION
    v1 = np.sqrt(2*E1/scipy.constants.m_n)
    L1 = v1 * delta_t - L0 * r_theta
    L1_error = v1 * delta_t_error 



    self._set_table_column(self._current_workspace, 'L1', L1, spec_list)
    self._set_table_column(self._current_workspace, 'L1_Err', L1_error, spec_list)

#----------------------------------------------------------------------------------------

  def _calculate_scattering_angle(self, table_name, spec_list):
    """
      Calculate the total scattering angle using the previous calculated parameters.

      @param table_name - name of table containing fitted parameters for the peak centres
      @param spec_list - spectrum range to calculate t0 for.
    """
    t0 = read_table_column(self._current_workspace, 't0', spec_list)
    L0 = read_table_column(self._current_workspace, 'L0', spec_list)
    L1 = read_table_column(self._param_table, 'L1', spec_list)
    L1_nan_to_num = np.nan_to_num(L1)
    spec = read_table_column(self._current_workspace, 'Spectrum', spec_list)

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
    masked_peak_centres = np.ma.masked_array(peak_centres, np.logical_or(peak_centres<=2000,peak_centres>=20000)) 
    masked_peak_centres /=  1e+6

    sin_theta = ((masked_peak_centres - t0) * scipy.constants.h) / (scipy.constants.m_n * d_spacings * 2 * (L0+L1_nan_to_num))
    theta = np.arcsin(sin_theta) * 2
    theta = np.degrees(theta)

    masked_theta = np.nanmean(theta, axis=0)
    theta_error = np.nanstd(theta, axis=0)

    self._set_table_column(self._current_workspace, 'theta', masked_theta, spec_list)
    self._set_table_column(self._current_workspace, 'theta_Err', theta_error, spec_list)

#----------------------------------------------------------------------------------------

 

  def _calculate_final_energy(self, peak_table, spec_list):
    """
      Calculate the final energy using the fitted peak centres of a run.

      @param table_name - name of table containing fitted parameters for the peak centres
      @param spec_list - spectrum range to calculate t0 for.
    """
    t0 = read_table_column(self._current_workspace, 't0', spec_list)
    L0 = read_table_column(self._current_workspace, 'L0', spec_list)

    L1 = read_table_column(self._param_table, 'L1', spec_list)
    theta = read_table_column(self._current_workspace, 'theta', spec_list)
    r_theta = calculate_r_theta(self._sample_mass, theta)

    peak_centres = read_fitting_result_table_column(peak_table, 'f1.LorentzPos', spec_list)
    peak_centres_errors = read_fitting_result_table_column(peak_table, 'f1.LorentzPos_Err', spec_list)
    peak_Gaussian_FWHM = read_fitting_result_table_column(peak_table, 'f1.GaussianFWHM', spec_list)
    peak_Gaussian_FWHM_errors = read_fitting_result_table_column(peak_table, 'f1.GaussianFWHM_Err', spec_list)
    peak_Lorentz_FWHM = read_fitting_result_table_column(peak_table, 'f1.LorentzFWHM', spec_list)
    peak_Lorentz_FWHM_errors = read_fitting_result_table_column(peak_table, 'f1.LorentzFWHM_Err', spec_list)
    peak_Lorentz_Amp = read_fitting_result_table_column(peak_table, 'f1.LorentzAmp', spec_list)
    peak_Lorentz_Amp_errors = read_fitting_result_table_column(peak_table, 'f1.LorentzAmp_Err', spec_list)
    
    badSpecs=np.argwhere((np.isinf(peak_Lorentz_Amp_errors)) | (np.isnan(peak_Lorentz_Amp_errors))  | \
    (np.isinf(peak_centres_errors)) | (np.isnan(peak_centres_errors))  | \
    (np.isnan(peak_Gaussian_FWHM_errors)) | (np.isinf(peak_Gaussian_FWHM_errors)) | \
    (np.isnan(peak_Lorentz_FWHM_errors)) | (np.isinf(peak_Lorentz_FWHM_errors)) | \
    (np.isnan(peak_Lorentz_Amp_errors)) | (np.isinf(peak_Lorentz_Amp_errors)) | \
    (np.absolute(peak_Gaussian_FWHM_errors) > np.absolute(peak_Gaussian_FWHM)) | \
    (np.absolute(peak_Lorentz_FWHM_errors) > np.absolute(peak_Lorentz_FWHM)) | \
    (np.absolute(peak_Lorentz_Amp_errors) > np.absolute(peak_Lorentz_Amp)) | \
    (np.absolute(peak_centres_errors) > np.absolute(peak_centres)))
    
    peak_centres[badSpecs] = np.nan

    delta_t = (peak_centres - t0) / 1e+6
    v1 = (L0 * r_theta + L1) / delta_t

    E1 = 0.5*scipy.constants.m_n*v1**2
    E1 /= MEV_CONVERSION
    
    spec_range = DETECTOR_RANGE[1]+1 - DETECTOR_RANGE[0]
    mean_E1 = np.empty(spec_range)
    E1_error = np.empty(spec_range)

    #mean_E1.fill(np.nanmean(E1))
    #E1_error.fill(np.nanstd(E1))
    
    mean_E1.fill(np.nanmean(4897.3))
    E1_error.fill(np.nanmean(0.4))

    self._set_table_column(self._current_workspace, 'E1', mean_E1)
    self._set_table_column(self._current_workspace, 'E1_Err', E1_error)



#----------------------------------------------------------------------------------------

  def _setup(self):
    """
      Setup algorithm.
    """
    self._samples = self.getProperty("Samples").value
    self._background = self.getProperty("Background").value
    self._param_file = self.getProperty("InstrumentParameterFile").value
    self._sample_mass = self.getProperty("Mass").value
    self._d_spacings = self.getProperty("DSpacings").value.tolist()
    self._calc_L0 = self.getProperty("CalculateL0").value
    self._make_IP_file = self.getProperty("CreateIPFile").value
    self._output_workspace_name = self.getPropertyValue("OutputWorkspace")
    self._iterations = self.getProperty("Iterations").value
    self._create_output = self.getProperty("CreateOutput").value

    if len(self._samples) == 0:
      raise ValueError("You must supply at least one sample run number.")

    #if len(self._background) == 0:
    #  raise ValueError("You must supply at least one background run number.")

    self._d_spacings.sort()

    self._param_table = '__EVS_calib_analysis_parameters'
    load_instrument_parameters(self._param_file, self._param_table)

#----------------------------------------------------------------------------------------

  def _create_calib_parameter_table(self, ws_name):
    #create table for calculated parameters
    CreateEmptyTableWorkspace(OutputWorkspace=ws_name)
    table_ws = mtd[ws_name]
    table_ws.addColumn('int', 'Spectrum')

    for value in range(DETECTOR_RANGE[0], DETECTOR_RANGE[1]+1):
      table_ws.addRow([value])

    column_names = ['t0','t0_Err','L0','L0_Err','L1','L1_Err','E1','E1_Err','theta','theta_Err']
    for name in column_names:
      table_ws.addColumn('double', name)

#----------------------------------------------------------------------------------------

  def _fit_linear(self, table_workspace_group, output_table):
    """
      Create a workspace wth the fitted peak_centres on the y and corresponding neutron velocity
      on the x. The intercept is the value of t0 and the gradient is the value of L0/L-Total.

      @param table_workspace_group - workspace group containing the fitted parameters of the peaks.
      @param output_table - name to call the fit workspace.
    """
    #extract fit data to workspace
    peak_workspaces = []
    for i, param_ws in enumerate(mtd[table_workspace_group].getNames()):
      temp_peak_data = '__temp_peak_ws_%d' % i
      ConvertTableToMatrixWorkspace(InputWorkspace=param_ws, OutputWorkspace=temp_peak_data,
                                    ColumnX='Spectrum', ColumnY='f1.PeakCentre')
      peak_workspaces.append(temp_peak_data)

    #create workspace of peaks
    peak_workspace = table_workspace_group + '_Workspace'
    RenameWorkspace(peak_workspaces[0], OutputWorkspace=peak_workspace)
    for temp_ws in peak_workspaces[1:]:
      ConjoinWorkspaces(peak_workspace, temp_ws, CheckOverlapping=False)
    Transpose(peak_workspace, OutputWorkspace=peak_workspace)

    num_spectra = mtd[peak_workspace].getNumberHistograms()
    plot_peak_indicies = ';'.join([peak_workspace + ',i' + str(i) for i in range(num_spectra)])

    for i in range(num_spectra):
      mtd[peak_workspace].setX(i, np.asarray(U_NEUTRON_VELOCITY))

    ReplaceSpecialValues(peak_workspace, NaNValue=0, NaNError=0, InfinityValue=0, InfinityError=0,
                         OutputWorkspace=peak_workspace)

    #perform linear fit on peak centres
    func_string = 'name=LinearBackground, A0=0, A1=0;'
    PlotPeakByLogValue(Input=plot_peak_indicies, Function=func_string,
                       FitType='Individual', CreateOutput=False, OutputWorkspace=output_table)
    DeleteWorkspace(peak_workspace)

#----------------------------------------------------------------------------------------
 

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
      if len(data) < (spec_list[1]+1 - spec_list[0]):
        raise ValueError("Not enough data from spectrum range.")

      offset = spec_list[0] - DETECTOR_RANGE[0]

    if isinstance(data, np.ndarray):
      data = data.tolist()

    for i, value in enumerate(data):
      table_ws.setCell(column_name, offset+i, value)

#----------------------------------------------------------------------------------------

  def _save_instrument_parameter_file(self, ws_name, spec_list):
    """
      Save the calibrated parameters to a tab delimited instrument parameter file.

      @param ws_name - name of the workspace to save the IP file from.
      @param spec_list - spectrum range to save to file.
    """
    file_header = '\t'.join(['plik', 'det', 'theta', 't0', 'L0', 'L1'])
    fmt = "%d  %d  %.4f  %.4f  %.3f  %.4f"

    det = read_table_column(ws_name, 'Spectrum', spec_list)
    t0 = read_table_column(ws_name, 't0', spec_list)
    L0 = read_table_column(ws_name, 'L0', spec_list)
    L1 = read_table_column(ws_name, 'L1', spec_list)
    theta = read_table_column(ws_name, 'theta', spec_list)

    #pad the start of the file with dummy data for the monitors
    file_data = np.asarray([[1,1,0,0,0,0], [2,2,0,0,0,0]])
    file_data = np.append(file_data, np.column_stack((det, det, theta, t0, L0, L1)), axis=0)

    workdir = config['defaultsave.directory']
    file_path = os.path.join(workdir, self._output_workspace_name+'.par')

    with open(file_path, 'wb') as f_handle:
      np.savetxt(f_handle, file_data, header = file_header, fmt=fmt)

#----------------------------------------------------------------------------------------

AlgorithmFactory.subscribe(EVSCalibrationAnalysis)

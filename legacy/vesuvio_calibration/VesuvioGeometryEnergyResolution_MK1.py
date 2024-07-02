#pylint: disable=no-init,too-many-instance-attributes

from mantid.kernel import *
from mantid.api import *
from mantid.simpleapi import *

import math
import numpy as np
import scipy.constants as sc
import scipy.stats as stats

#==============================================================================

LOG_FIT = False

MODES = ['SingleDifference', 'DoubleDifference', 'ThickDifference']
BACKSCATTERING = range(3, 135)
FRONTSCATTERING = range(135, 199)

U_FRONTSCATTERING_SAMPLE = ['14025']
U_FRONTSCATTERING_BACKGROUND = ['12570']

U_BACKSCATTERING_SAMPLE = ['12570']
U_BACKSCATTERING_BACKGROUND = ['12571']

NEUTRON_MASS_AMU = sc.value('neutron mass in u')
U_MASS_IN_AMU = 238.0289
PB_MASS_IN_AMU = 207.19

nominal_E1 = 4897 # meV
v1 = np.sqrt(nominal_E1 / 5.2276e-6)  # m/sec

# Uranium peak details taken from: 10.1016/j.nima.2010.09.079
U_PEAKS = [
#    Er      dEr    vr          t      dTr
#    meV     meV    m/usec      usec   usec
    [36684., 133.6, 83768.0e-6, 131.  ,0.24],
    [20874., 91.7,  63190.0e-6, 173.8 ,0.38],
    [6672.,  52.4,  35725.0e-6, 307.7 ,1.21]
]

#==============================================================================

class VesuvioGeometryEnergyResolution(PythonAlgorithm):

    _ipf_filename = None
    _evs_ws = None
    _evs = None
    _sample_pos = None
    _l0_dist = None
    _delta_L0 = None
    _delta_L1 = None
    _delta_t0 = None
    _delta_theta_ana = None
    _delta_theta_mc = None
    _output_table = None

#------------------------------------------------------------------------------

    def summary(self):
        return "Calculates the geometry and energy resolution for VESUVIO."

#------------------------------------------------------------------------------

    def PyInit(self):
        # Sample data
        self.declareProperty(StringArrayProperty("Samples",  Direction.Input),
                             doc="Sample run numbers to fit peaks to.")

        self.declareProperty(StringArrayProperty("Background", Direction.Input),
                             doc="Run numbers to use as a background.")

        self.declareProperty(FloatArrayProperty("DSpacings", "1.750,2.475,2.858",
                                                direction=Direction.Input),
                             doc="List of dSpacings")

        self.declareProperty("MonteCarloEvents", 1000000,
                             doc="Number of events to use in Monte Carlo calculations")

        # Optional parameter file
        self.declareProperty(FileProperty("InstrumentParFile", "", action=FileAction.Load,
                                          extensions=["dat", "par"]),
                             doc="A parameter file. If provided the values are used to correct "
                                 "the default instrument values and attach the t0 values to each "
                                 "detector")

        # Output parameter tables
        self.declareProperty(ITableWorkspaceProperty("Resolution", "", Direction.Output),
                             doc="Resolution values per bank")

        # Output value table
        self.declareProperty(ITableWorkspaceProperty("OutputWorkspace", "", Direction.Output),
                             doc="Mean resolution parameters")

#------------------------------------------------------------------------------

    def PyExec(self):
        self._ipf_filename = self.getPropertyValue("InstrumentParFile")
        self._evs_ws = LoadEmptyVesuvio(InstrumentParFile=self._ipf_filename)

        # Calculate source to sample distance
        self._evs = self._evs_ws.getInstrument()
        source_pos = self._evs.getSource().getPos()
        self._sample_pos = self._evs.getSample().getPos()
        self._l0_dist = source_pos.distance(self._sample_pos)

        # Create mean output table
        self._output_table = CreateEmptyTableWorkspace(OutputWorkspace=self.getPropertyValue("OutputWorkspace"))

        self._output_table.addColumn('str',   'Parameter')
        self._output_table.addColumn('float', 'Mean')
        self._output_table.addColumn('float', 'StdDev')
        self._output_table.addColumn('float', 'StdErr')
        self._output_table.addColumn('float', 'WeightedMean')
        self._output_table.addColumn('float', 'WeightedErr')

        # Calculate resolutions
        data = {}

        # Calculate dL1
        data.update(self._process_l1())

        # Calculate weighted mean of dTheta for Monte Carlo values
        self._delta_theta_mc = self._weighted_mean(data['dTheta MC (deg)'])

        # Calculate dL and dt0
        data.update(self._process_l_t0())

        # Average of L0 for front scattering detectors (incident flight path
        # resolution is identical for both banks)
        front_dL0_sample = data['SD dL/dL0 (cm)'][len(BACKSCATTERING):]
        self._delta_L0 = self._weighted_mean(front_dL0_sample) 
        logger.notice('dL0 weighted mean = {0}'.format(self._delta_L0))

        # Calculate weighted mean of dL1
        back_dL_sample = data['SD dL/dL0 (cm)'][:len(BACKSCATTERING)]
        back_dL1 = np.sqrt(back_dL_sample**2 - self._delta_L0**2)
        self._delta_L1 = self._weighted_mean(back_dL1) 
        logger.notice('dL1 weighted mean = {0}'.format(self._delta_L1))

        # Output dL1
        self._add_param_table_row('SD Back dL1 (cm)', back_dL1)
        back_dL1.resize(len(BACKSCATTERING) + len(FRONTSCATTERING))
        data['SD Back dL1 (cm)'] = back_dL1


        # Calculate weighted mean of dt0
        self._delta_t0 = self._weighted_mean(data['SD dt0 (usec)'])
        logger.notice('dt0 weighted mean = {0}'.format(self._delta_t0))

        # Calculate dTheta and effective width
        data.update(self._process_theta_dw())

        # Calculate weighted mean of dTheta for analytical values
        self._delta_theta_ana = self._weighted_mean(data['SD dTheta ANA (deg)'])
        logger.notice('dTheta analytical weighted mean = {0}'.format(self._delta_theta_ana))

        # Calculate dE1_Gauss and dE1_Lorentz
        data.update(self._process_de1())

        # Create the table with all parameters
        data_table = CreateEmptyTableWorkspace(OutputWorkspace=self.getPropertyValue('Resolution'))
        data_table.addColumn('int', 'Spectrum')
        param_names = data.keys()
        for param_name in param_names:
            data_table.addColumn('float', param_name)

        for spec in BACKSCATTERING:
            row = [spec]
            row.extend([data[n][data_table.rowCount()] for n in param_names])
            data_table.addRow(row)

        for spec in FRONTSCATTERING:
            row = [spec]
            row.extend([data[n][data_table.rowCount()] for n in param_names])
            data_table.addRow(row)

        # Output workspaces
        self.setProperty("OutputWorkspace", self._output_table)
        self.setProperty("Resolution", data_table)
        
        return data

#------------------------------------------------------------------------------

    def _process_l1(self):
        """
        Caluclates delta L1.

        @return Dictionary of resolution data.
        """
        resolution, l1_res, theta_res = VesuvioL1ThetaResolution(PARFile=self._ipf_filename,L1BinWidth=0.001, ThetaBinWidth=0.001,
                                                                 NumEvents=self.getProperty('MonteCarloEvents').value)

        DeleteWorkspace(l1_res)
        DeleteWorkspace(theta_res)

        data = {
            'dL1 MC (cm)': resolution.dataY(1),
            'dTheta MC (deg)': resolution.dataY(3)
        }
        self._add_param_table_row('Back dL1 MC (cm)', data['dL1 MC (cm)'][:len(BACKSCATTERING)])
        self._add_param_table_row('Front dL1 MC (cm)', data['dL1 MC (cm)'][len(BACKSCATTERING):])
        self._add_param_table_row('Back dTheta MC (deg)', data['dTheta MC (deg)'][:len(BACKSCATTERING)])
        self._add_param_table_row('Front dTheta MC (deg)', data['dTheta MC (deg)'][len(BACKSCATTERING):])

        DeleteWorkspace(resolution)

        return data

#------------------------------------------------------------------------------

    def _process_l_t0(self):
        """
        Caluclates delta L for backscattering or delta L0 for forward
        scattering and delta t0.

        @return Dictionary of resolution data.
        """

        alg = AlgorithmManager.create("EVSCalibrationFit")
        alg.initialize()
        alg.setRethrows(True)
        alg.setLogging(LOG_FIT)
        alg.setProperty('Samples', U_FRONTSCATTERING_SAMPLE)
        alg.setProperty('Background', U_FRONTSCATTERING_BACKGROUND)
        alg.setProperty('SpectrumRange', [FRONTSCATTERING[0], FRONTSCATTERING[-1]])
        alg.setProperty('Mass', U_MASS_IN_AMU)
        alg.setProperty('Energy', [p[0] for p in U_PEAKS])
        alg.setProperty('Function', 'Gaussian')
        alg.setProperty('PeakType', 'Resonance')
        alg.setProperty('InstrumentParameterFile', self._ipf_filename)
        alg.setProperty('OutputWorkspace', '__l0_fit')
        alg.setProperty('CreateOutput', False)
        alg.execute()

        alg = AlgorithmManager.create("EVSCalibrationFit")
        alg.initialize()
        alg.setRethrows(True)
        alg.setLogging(LOG_FIT)
        alg.setProperty('Samples', U_BACKSCATTERING_SAMPLE)
        alg.setProperty('Background', U_BACKSCATTERING_BACKGROUND)
        alg.setProperty('SpectrumRange', [BACKSCATTERING[0], BACKSCATTERING[-1]])
        alg.setProperty('Mass', U_MASS_IN_AMU)
        alg.setProperty('Energy', [p[0] for p in U_PEAKS])
        alg.setProperty('Function', 'Gaussian')
        alg.setProperty('PeakType', 'Resonance')
        alg.setProperty('InstrumentParameterFile', self._ipf_filename)
        alg.setProperty('OutputWorkspace', '__t0_fit')
        alg.setProperty('CreateOutput', False)
        alg.execute()

        t0_peak_params = mtd['__t0_fit_Peak_Parameters']
        l0_peak_params = mtd['__l0_fit_Peak_Parameters']


        # Calculate resolution
        back_params = self._calculate_l_t0(BACKSCATTERING, t0_peak_params, 2)
        forward_params = self._calculate_l_t0(FRONTSCATTERING, l0_peak_params, 3)

        # Get data from param tables
        max_float = np.finfo(float).max
        back_l_data = np.clip(back_params.column('A1'), 0, max_float)
        back_t0_data = np.clip(back_params.column('A0'), 0, max_float)
        forward_l0_data = np.clip(forward_params.column('A1'), 0, max_float)
        forward_t0_data = np.clip(forward_params.column('A0'), 0, max_float)

        self._add_param_table_row('SD Back dL (cm)', back_l_data*100)
        self._add_param_table_row('SD Forward dL0 (cm)', forward_l0_data*100)
        self._add_param_table_row('SD Back dt0 (usec)', back_t0_data)
        self._add_param_table_row('SD Forward dt0 (usec)', forward_t0_data)

        data = {
            'SD dt0 (usec)': np.hstack(np.array([back_t0_data, forward_t0_data])),
            'SD dL/dL0 (cm)': np.hstack(np.array([back_l_data*100, forward_l0_data*100]))
        }

        return data

#------------------------------------------------------------------------------

    def _calculate_l_t0(self, detectors, calibration_params, range_limit):
        """
        Calculates resolution for given detectors.

        @param detectors List of detector numbers
        @param calibration_params Parameters from calibration fits
        @prama range_limit number of resonances to take into account
        @return Workspace containing resolution parameters per detector
        """
        # Create a workspace of the three peaks against 1/v1^2
        wks = WorkspaceFactory.Instance().create('Workspace2D', len(detectors), range_limit, range_limit)        

        for detector in detectors:
            det_index = detector - detectors[0]

            x_data = []
            for peak in range(range_limit):
                u_peak = U_PEAKS[peak]
                x_data.append(1.0/(u_peak[2])**2)

                params = calibration_params.getItem(peak)
                sigma = params.column('f1.Sigma')[det_index]
                sigma_err = params.column('f1.Sigma_Err')[det_index]

                wks.dataY(det_index)[peak] = (sigma ** 2) - (u_peak[4]**2)
                wks.dataE(det_index)[peak] = 2*sigma*sigma_err
                #wks.dataE(det_index)[peak] = 1

            wks.setX(det_index, np.array(x_data))

        AnalysisDataService.Instance().addOrReplace('__bank_data', wks)

       
        # Perform a linear fit of each spectra
        fit = AlgorithmManager.Instance().create('PlotPeakByLogValue')
        fit.initialize()
        fit.setChild(True)
        fit.setProperty('Function', 'name=LinearBackground')
        fit.setProperty('Input', '__bank_data,v0:132')
        fit.setProperty('OutputWorkspace', 'backscattering_params')
        fit.execute()

        DeleteWorkspace('__bank_data')
        params = fit.getProperty('OutputWorkspace').value
        
        # Process fit parameters
        for index, detector in enumerate(detectors):
            params.setCell(index, 0, detector)

            t0_val = params.cell(index, 1)
            l_dist = params.cell(index, 3)

            # Set negative values to zero, otherwise take square root
            if t0_val > 0:
                t0_val = np.sqrt(t0_val)
            else:
                t0_val = 0

            if l_dist > 0:
                l_dist = np.sqrt(l_dist)
            else:
                l_dist = 0

            params.setCell(index, 1, t0_val)
            params.setCell(index, 3, l_dist)

        return params
#------------------------------------------------------------------------------

    def _process_theta_dw(self):
        """
        Caluclates delta theta in degrees and effecitve detector width as seen
        by the sample in cm.

        @return Dictionary of resolution data.
        """

        # Run fitting
        alg = AlgorithmManager.create("EVSCalibrationFit")
        alg.initialize()
        alg.setRethrows(True)
        alg.setLogging(LOG_FIT)
        alg.setProperty('Samples', self.getPropertyValue('Samples'))
        alg.setProperty('Background', self.getPropertyValue('Background'))
        alg.setProperty('Function', 'Gaussian')
        alg.setProperty('Mode', 'FoilOut')
        alg.setProperty('PeakType', 'Bragg')
        alg.setProperty('InstrumentParameterFile', self.getPropertyValue('InstrumentParFile'))
        alg.setProperty('DSpacings', self.getProperty('DSpacings').value)
        alg.setProperty('OutputWorkspace', 'delta_theta')
        alg.setProperty('CreateOutput', False)
        alg.execute()
        params = mtd['delta_theta_Peak_Parameters']

        back_theta_data = []
        forward_theta_data = []
        back_dw_data = []
        forward_dw_data = []

        for spec_no in BACKSCATTERING:
            delta_theta, effective_width = self._calc_delta_theta(params, spec_no)
            back_theta_data.append(delta_theta)
            back_dw_data.append(effective_width)

        back_theta_data = np.array(back_theta_data)
        back_dw_data = np.array(back_dw_data)

        for spec_no in FRONTSCATTERING:
            delta_theta, effective_width = self._calc_delta_theta(params, spec_no)
            forward_theta_data.append(delta_theta)
            forward_dw_data.append(effective_width)

        forward_theta_data = np.array(forward_theta_data)
        forward_dw_data = np.array(forward_dw_data)

        self._add_param_table_row('SD Back dTheta ANA (deg)', back_theta_data)
        self._add_param_table_row('SD Forward dTheta ANA (deg)', forward_theta_data)
        self._add_param_table_row('SD Back Effective Width (cm)', back_dw_data)
        self._add_param_table_row('SD Forward Effective Width (cm)', forward_dw_data)

        data = {
            'SD dTheta ANA (deg)': np.hstack(np.array([back_theta_data, forward_theta_data])),
            'SD Effective Width (cm)': np.hstack(np.array([back_dw_data, forward_dw_data]))
        }

        return data

#------------------------------------------------------------------------------

    def _calc_delta_theta(self, params, spec_no):
        """
        Calculates delta Theta for a given spectrum number using Gaussian fits
        to (4) Bragg peaks for a Lead sample.

        @param params Fit parameter table
        @param spec_no Spectrum no to calculate for
        @return Average delta Theta over all fitted peaks
        """
        spec_col_idx = params.column('Spectrum').index(spec_no)

        sigma_col_names = [c for c in params.getColumnNames() if 'Sigma' in c and 'Err' not in c]
        peak_centre_col_names = [c for c in params.getColumnNames() if 'PeakCentre' in c and 'Err' not in c]

        l1_dist = self._get_l1(spec_no)

        delta_thetas = []
        delta_w = []
        
        for sigma_name, peak_centre_name in zip(sigma_col_names, peak_centre_col_names):
            sigma = params.cell(sigma_name, spec_col_idx)
            peak_centre = params.cell(peak_centre_name, spec_col_idx)
            theta = self._get_theta(spec_no)
            rad_theta = np.radians(theta)
           
            # One can replace the part (self._delta_L0 **2 + self._delta_L1**2)/ (100*self._l0_dist + 100*l1_dist)**2 with numerical result from McStas dT/T simulation of water moderator.
            # This simulation gives, in the asymptotic region of thermal neutron (2meV -- 1 eV) the values of 0.00115 from pure Gaussian fit and 0.000786 from Ikeda Carpenter function fit. 
            # Alternative calculation:
            #it is assumed that dT/T is proportional to dT/L for a constant wave length (velocity) of a neutron. In consequence, 
            #knowing that McStas simulation assumes TOF monitor at 8.54m from the moderator and assuming    L = L0+L1 = 11.7 m,
            #dT/T at the sample position  = 8.54/11.7 dT/T = 0.7299 dT/T  from McStas simulation. 0.7299*0.000786 = 0.000573
            # or 0.7299*0.00115 = 0.000839.
            
            if  theta>90:
                delta_theta = 2*math.tan(rad_theta/2) * np.sqrt((sigma / peak_centre)**2 - 0.00115**2)
            elif theta<90:
                delta_theta = 2*math.tan(rad_theta/2) * (sigma / peak_centre)
            else:
                raise RuntimeError()
            
            delta_thetas.append(np.degrees(abs(delta_theta)))
            delta_w.append(abs(delta_theta) * l1_dist * 100)

        delta_thetas = np.array(delta_thetas)
        masked_delta_thetas = np.ma.masked_array(delta_thetas, np.isnan(delta_thetas)) 
        delta_thetas_avg = np.mean(masked_delta_thetas, axis=0)
        logger.debug('Spectrum {0}: dThetas={1}, mean={2}'.format(spec_no, delta_thetas, delta_thetas_avg))

        delta_w = np.array(delta_w)
        masked_delta_w = np.ma.masked_array(delta_w, np.isnan(delta_w)) 
        delta_w_avg = np.mean(masked_delta_w, axis=0)

        return (delta_thetas_avg, delta_w_avg)

#------------------------------------------------------------------------------

    def _process_de1(self):
        """
        Calculates delta E1 for Lorentzian and Gaussian components.

        @return Dictionary of resolution data.
        """
        data = {}
        

        for mode in MODES:
            
            short_mode = ''.join([c for c in mode if c.isupper()])

            if mode == 'SingleDifference':
                
                self._energy_res_fit_back(mode)

                de1_gauss_ana_back, de1_gauss_mc_back, de1_lorentz_back = self._calculate_de1(mtd['{0}_Peaks_back_Peak_0_Parameters'.format(mode)])

                self._add_param_table_row('%s Back dE1_Gauss ANA st. dev. (meV)' % short_mode,
                                          de1_gauss_ana_back)
                self._add_param_table_row('%s Back dE1_Gauss MC st. dev. (meV)' % short_mode,
                                          de1_gauss_mc_back)
                self._add_param_table_row('%s Back dE1_Lorentz HWHM (meV)' % short_mode,
                                          de1_lorentz_back)
                                          
                
                self._energy_res_fit_front(mode)

                de1_gauss_ana_front, de1_gauss_mc_front, de1_lorentz_front = self._calculate_de1(mtd['{0}_Peaks_front_Peak_0_Parameters'.format(mode)])
                
                self._add_param_table_row('%s Forward dE1_Gauss ANA st. dev. (meV)' % short_mode,
                                          de1_gauss_ana_front)
                self._add_param_table_row('%s Forward dE1_Gauss MC st. dev. (meV)' % short_mode,
                                          de1_gauss_mc_front)
                self._add_param_table_row('%s Forward dE1_Lorentz HWHM (meV)' % short_mode,
                                          de1_lorentz_front)            

                data['%s dE1_Gauss ANA st. dev. (meV)' % short_mode] = np.hstack(np.array([de1_gauss_ana_back, de1_gauss_ana_front]))
                data['%s dE1_Gauss MC st. dev. (meV)' % short_mode] = np.hstack(np.array([de1_gauss_mc_back, de1_gauss_mc_front]))
                data['%s dE1_Lorentz HWHM (meV)' % short_mode] = np.hstack(np.array([de1_lorentz_back, de1_lorentz_front]))
             
                
            else:
                
                self._energy_res_fit_back(mode)

                de1_gauss_ana_back, de1_gauss_mc_back, de1_lorentz_back = self._calculate_de1(mtd['{0}_Peaks_back_Peak_0_Parameters'.format(mode)])



                self._add_param_table_row('%s Back dE1_Gauss ANA st. dev. (meV)' % short_mode,
                                          de1_gauss_ana_back)
                self._add_param_table_row('%s Back dE1_Gauss MC st. dev. (meV)' % short_mode,
                                          de1_gauss_mc_back)
                self._add_param_table_row('%s Back dE1_Lorentz HWHM (meV)' % short_mode,
                                          de1_lorentz_back)
                
                de1_gauss_ana_front =  np.empty(len(FRONTSCATTERING))
                de1_gauss_ana_front.fill(0)
                
                de1_gauss_mc_front =  np.empty(len(FRONTSCATTERING))
                de1_gauss_mc_front.fill(0)
                
                de1_lorentz_front =  np.empty(len(FRONTSCATTERING))
                de1_lorentz_front.fill(0)
                
                
                data['%s dE1_Gauss ANA st. dev. (meV)' % short_mode] =  np.hstack(np.array([de1_gauss_ana_back, de1_gauss_ana_front]))
                data['%s dE1_Gauss MC st. dev. (meV)' % short_mode] = np.hstack(np.array([de1_gauss_mc_back, de1_gauss_mc_front]))
                data['%s dE1_Lorentz HWHM (meV)' % short_mode] = np.hstack(np.array([de1_lorentz_back, de1_lorentz_front]))

        return data

#------------------------------------------------------------------------------

    def _energy_res_fit_back(self, mode):
        """
        Performs calibration fitting.

        @param mode Mode in which to fit data
        """
        ws_name = '{0}_Peaks_back'.format(mode)

        alg = AlgorithmManager.create('EVSCalibrationFit')
        alg.initialize()
        alg.setRethrows(True)
        alg.setLogging(LOG_FIT)
        alg.setProperty('Samples', self.getPropertyValue('Samples'))
        alg.setProperty('SpectrumRange', [BACKSCATTERING[0], BACKSCATTERING[-1]])
        alg.setProperty('Mode', mode)
        alg.setProperty('Function', 'Voigt')
        alg.setProperty('PeakType', 'Recoil')
        alg.setProperty('InstrumentParameterFile', self.getPropertyValue('InstrumentParFile'))
        alg.setProperty('CreateOutput', False)
        alg.setProperty('OutputWorkspace', ws_name)
        alg.execute()
        
        
#------------------------------------------------------------------------------

    def _energy_res_fit_front(self, mode):
        """
        Performs calibration fitting.

        @param mode Mode in which to fit data
        """
        ws_name = '{0}_Peaks_front'.format(mode)

        alg = AlgorithmManager.create('EVSCalibrationFit')
        alg.initialize()
        alg.setRethrows(True)
        alg.setLogging(LOG_FIT)
        alg.setProperty('Samples', self.getPropertyValue('Samples'))
        alg.setProperty('SpectrumRange', [FRONTSCATTERING[0], FRONTSCATTERING[-1]])
        alg.setProperty('Mode', mode)
        alg.setProperty('Function', 'Voigt')
        alg.setProperty('PeakType', 'Recoil')
        alg.setProperty('InstrumentParameterFile', self.getPropertyValue('InstrumentParFile'))
        alg.setProperty('CreateOutput', False)
        alg.setProperty('OutputWorkspace', ws_name)
        alg.execute()

#------------------------------------------------------------------------------

    def _calculate_de1(self, parameters):
        """
        Calculates delta-E1 for Lorentzian and Gaussian components, using both
        the analytical and Monte Carlo obtained values for dTheta.

        @param parameters Fit parameters
        @return Tuple of (dE1_Gauss Analytical, dE1_Gauss Monte Carlo, dE1_Lorentz)
        """
        spectrum_nos = parameters.column('Spectrum')
        positions = parameters.column('f1.LorentzPos')
        widths_lorentz = parameters.column('f1.LorentzFWHM')
        widths_gauss = parameters.column('f1.GaussianFWHM')

        de1_gauss_ana = []
        de1_gauss_mc = []
        de1_lorentz = []

        for spec, pos, fwhm_lorentz, fwhm_gauss in zip(spectrum_nos, positions, widths_lorentz, widths_gauss):
            l1_dist = self._get_l1(spec-1)
            theta = self._get_theta(spec-1)

            # Lorentzian component
            if pos == 0.0 or fwhm_lorentz == 0.0:
                delta_e1_lorentz = np.nan
            else:
                # HWHM e1_lorentz
                delta_e1_lorentz = 0.5*self._convert_time_to_energy(l1_dist, pos, fwhm_lorentz)

            # Gaussian component
            if pos == 0.0 or fwhm_gauss == 0.0:
                delta_e1_gauss_ana = np.nan
                delta_e1_gauss_mc = np.nan
            else:
                
                #fwhm=2*sqrt(2*log(2))*stdev = 2.3548*stdev
                #hwhm = sqrt(2*log(2))*stdev = 1.1774*stdev
                
                # Total width of the recoil peak fitted FWHM e1_gauss in meV
                delta_e1_gauss = self._convert_time_to_energy(l1_dist, pos, fwhm_gauss)
                
                # stdev delta_t0 in meV
                delta_t0_gauss = self._convert_time_to_energy(l1_dist, pos, self._delta_t0)
                
                # stdev L0  in meV
                delta_L0_gauss = self._convert_L0_to_energy(l1_dist, pos, self._delta_L0*0.01)
                
                
                # stdev L1 in meV
                delta_L1_gauss = self._convert_L1_to_energy(l1_dist, pos, self._delta_L1*0.01)
                
                
                # st. dev.^2  sub_val
                sub_val = delta_L0_gauss**2 + delta_L1_gauss**2 + delta_t0_gauss**2
                
                # stdev theta_ana in meV
                delta_theta_ana_gauss = self._convert_angle_to_energy(l1_dist, pos, theta, self._delta_theta_ana)
                
                # stdev theta_mc in meV
                delta_theta_mc_gauss = self._convert_angle_to_energy(l1_dist, pos, theta, self._delta_theta_mc)
                
                # kinetic energy of lead based on Debye model for TD=105K and T =300K is equal to 38.899 meV and st. dev. of mom. distr. of 35.851 inv. Ang.
                # FWHM e1_gauss, st. dev^2 self._delta_theta_ana, st.dev^2 sub_val
                vel_ana = (delta_e1_gauss/2.3548)**2 - delta_theta_ana_gauss**2 - sub_val - 38.899
                # FWHM e1_gauss, st. dev^2 self._delta_theta_mc, st.dev^2 sub_val
                vel_mc = (delta_e1_gauss/2.3548)**2 - delta_theta_mc_gauss**2 - sub_val - 38.899

                delta_e1_gauss_ana = np.nan if vel_ana < 0 else math.sqrt(vel_ana) 
                delta_e1_gauss_mc = np.nan if vel_mc < 0 else math.sqrt(vel_mc) 

            de1_gauss_ana.append(delta_e1_gauss_ana)
            de1_gauss_mc.append(delta_e1_gauss_mc)
            de1_lorentz.append(delta_e1_lorentz)

        return (np.array(de1_gauss_ana), np.array(de1_gauss_mc), np.array(de1_lorentz))


#------------------------------------------------------------------------------

    def _convert_time_to_energy(self, l1_dist, position, t_width):
        """
        Converts a time distribution width to energy distribution width

        @param l1_dist is the distance from sample to detector (m)
        @param position is the peak position in time of flight (us)
        @param t_width is the width of time of flight distribution (us)
        """
        
        d_time = position * sc.micro # us to s
        t_width_time = t_width * sc.micro # us to s
        
        # Calculate width of peak in meV
        e_width = 2*5.2276e-6 *t_width_time* self._l0_dist**2 /(d_time - l1_dist/v1) **3

        return e_width
        
#------------------------------------------------------------------------------

    def _convert_L0_to_energy(self, l1_dist, position, l_width):
        """
        Converts a L0 distance distribution width to energy distribution width

        @param l1_dist Distance from sample to detector (m)
        @param position is the peak position in time of flight (us)
        @param l_width is the width of distance distribution (m)
        """
        
        d_time = position * sc.micro # us to s
        
        # Calculate width of peak in meV
        e_width= 2*5.2276e-6 *self._l0_dist* l_width/ (d_time - l1_dist/v1) **2

        return e_width
        
#------------------------------------------------------------------------------

    def _convert_L1_to_energy(self, l1_dist, position, l_width):
        """
        Converts a L1 distance distribution width to energy distribution width

        @param l1_dist Distance from sample to detector (m)
        @param position is the peak position in time of flight (us)
        @param l_width is the width of distance distribution (m)
        """
        
        d_time = position * sc.micro # us to s
        
        # Calculate width of peak in meV
        e_width= 2*5.2276e-6 *self._l0_dist**2* l_width/v1/ (d_time - l1_dist/v1) **3

        return e_width
        
#------------------------------------------------------------------------------

    def _convert_angle_to_energy(self, l1_dist, position, theta, theta_width):
        """
        Converts an angular distribution width to energy distribution width

        @param l1_dist Distance from sample to detector (m)
        @param position is the peak position in time of flight (us)
        @param theta is the scattering angle (degrees)
        @param theta_width is the width of angle distribution (degrees)
        """
        
        d_time = position * sc.micro # us to s
        rad_theta = np.radians(theta)
        rad_theta_width = np.radians(theta_width)
        r_theta = self._calculate_r_theta(PB_MASS_IN_AMU, theta)
        
        # Calculate width of peak in meV
        e_width= 2*5.2276e-6* rad_theta_width*v1**2*np.sin(rad_theta)/np.sqrt( (PB_MASS_IN_AMU / NEUTRON_MASS_AMU)**2 - np.sin(rad_theta)**2 )/ (r_theta) **2

        return e_width

#------------------------------------------------------------------------------

    def _calculate_r_theta(self, mass, thetas):
        """
        Returns the ratio of the final neutron velocity to the initial neutron velocity
        as a function of the scattering angle theta and atomic mass of a scatterer and a neutron.

        @param mass - mass of the scatterer in amu
        @param thetas - vector containing the values of theta
        @return the ratio of final and incident velocities
        """
        rad_theta = np.radians(thetas)
        r_theta = (np.cos(rad_theta) + np.sqrt( (mass / NEUTRON_MASS_AMU)**2 - np.sin(rad_theta)**2 )) / ((mass / NEUTRON_MASS_AMU) +1)

        return r_theta
  
#------------------------------------------------------------------------------

    def _get_l1(self, spec_no):
        """
        Gets the L1 distance for a given spectrum number.

        @param det_no The detector number
        @return The L1 distance
        """
        ws_idx = self._evs_ws.getIndexFromSpectrumNumber(spec_no)
        det_pos = self._evs_ws.getDetector(ws_idx).getPos()
        dist = self._sample_pos.distance(det_pos)
        return dist

#------------------------------------------------------------------------------

    def _get_theta(self, spec_no):
        """
        Gets Theta (in degrees) for a given spectrum number.

        @param spec_no Spectrum number containing detector
        @return Theta in degrees for detector
        """
        ws_idx = self._evs_ws.getIndexFromSpectrumNumber(spec_no)
        detector = self._evs_ws.getDetector(ws_idx)
        theta = self._evs_ws.detectorTwoTheta(detector)
        return math.degrees(theta)

#------------------------------------------------------------------------------

    def _add_param_table_row(self, name, data):
        """
        Adds a parameter row to the output table.

        @param name Name of the parameter
        @param data Data values
        """
        data = data[np.isfinite(data)]
        mean = np.mean(data)
        weights = 1.0 / (np.repeat(mean, data.size) - data)**2
        weighted_data = data * weights / np.sum(weights)

        # In the event that a fit went wrong and all weights sum to zero
        try:
            weighted_mean = np.average(data, weights=weights)
            weighted_error = stats.sem(weighted_data)
        except ZeroDivisionError:
            weighted_mean = np.nan
            weighted_error = np.nan

        self._output_table.addRow([name,
                                   mean,
                                   np.std(data),
                                   stats.sem(data),
                                   weighted_mean,
                                   weighted_error])

#------------------------------------------------------------------------------
    def _weighted_mean(self, data):
        """
        Calculates weighted mean.

        @param data Numpy array of data
        @return Weighted mean
        """
        data = data[np.isfinite(data)]
        mean = np.mean(data)
        weights = 1.0 / (np.repeat(mean, data.size) - data)**2
        return np.average(data, weights=weights)

#==============================================================================

AlgorithmFactory.subscribe(VesuvioGeometryEnergyResolution)


"""
  Calibration routines for the VESUVIO spectrometer.

  The procedures used here are based upon those descibed in: Calibration of an electron volt neutron spectrometer, Nuclear Instruments and Methods in Physics Research A
  (15 October 2010), doi:10.1016/j.nima.2010.09.079 by J. Mayers, M. A. Adams
"""
from __future__ import print_function

import sys

import numpy as np
from numpy import ma
import scipy
from scipy.optimize import leastsq
import scipy.stats as mstats_basic

import mantid
from mantid import AlgorithmManager, logger
from six import iteritems

# ----------------------------------------------------------------------------------------------------------------------
# Global constants
# ----------------------------------------------------------------------------------------------------------------------
# Uranium foil in cryostat. Uranium resonance peaks help determine L0 for all detectors
# & t0 for forward scattering detectors
U_SAMPLE = ['14025'] # ['14025'] U foil in cryostat, # At present, the algortihm cannot find Bragg peak when working with the run ['17154']  for indium foil in cryostat
# Lead sample with uranium foil in incident beam. Provides Uranium absorption lines for calibration of
# L1 & t0 on the backscattering detectors
PB_SAMPLE_U_FOIL =['12571']# At present, the load_run and nperiods function cannot find Vesuvvio raw data with VESUVIO000 prefixes such as ['42229','42230','42231','42232','42233','42234','42235','42236','42237','42238','42239','42240','42241','42242','42243','42244','42245','42246','42247','42248','42249','42250','42251','42252','42253']#['12571']
# Lead sample with no U-foil in beam. Used for normalising the above to remove spectral shape
PB_SAMPLE =['12570']# At present, the load_run and nperiods function cannot find Vesuvvio raw data with VESUVIO000 prefixes such as['42209','42210','42211','42212','42213','42214','42215','42216','42217','42218','42219','42220','42221','42222','42223','42224','42225','42226','42227','42228'] #['12570']


# Approximate Uranium resonance peaks. Order must be decreasing
U_RESON_MEV = np.array([36684, 20874, 6672]) #  [36684, 20874, 6672] for Uranium # [39681,22723,14599] for indium
# Nominal neutron velocities at 3 Uranium resonance peaks (metres/us)
VEL_U_RES = 1e-6 * np.sqrt(
    2 * 1e-3 * U_RESON_MEV * scipy.constants.value("electron volt") / (scipy.constants.value("neutron mass")))
INV_VEL_U_RES = 1. / VEL_U_RES
# Fit windows for Uranium resonances in forward spectra
U_RESON_TOF_FOR = [131, 173, 307]
U_RESON_WINDOWS_FOR = [125, 140, 165, 180, 300., 315.]
# Fit windows for Uranium resonances in backward spectra
U_RESON_TOF_BACK = [138, 183, 325]
U_RESON_WINDOWS_BACK = [130, 148, 175, 191, 317., 332.]

# Fit window for Lead recoil peak in backward spectra
PB_TOF = [378]
PB_WINDOWS = [350, 400]

# Neutron mass
N_MASS = scipy.constants.m_n
PLANCK = scipy.constants.h
N_MASS_AMU = scipy.constants.value("neutron mass in u")
MEV = 1e-3 * scipy.constants.value("electron volt")

# Bank spectra
FORWARD_SPECTRA = [135,198]
N_FORWARD = FORWARD_SPECTRA[1] - FORWARD_SPECTRA[0] + 1
BACKWARD_SPECTRA = [3, 134]
N_BACKWARD = BACKWARD_SPECTRA[1] - BACKWARD_SPECTRA[0] + 1

# Crop range for TOF data
CROP_RANGE = [50, 560]
U_PARAMS ='50,1,560'
# Crop range for TOF data
CROP_RANGE_PB = [3000, 20000]#CROP_RANGE_PB = [1600, 20000]
PARAMS_PB='3000,50,20000'#PARAMS_PB='1600,50,20000'

# Properties controlling fitting bragg peaks in Lead
BRAGG_POSITION_TOL_BACK = 500
BRAGG_POSITION_TOL_FWD = 500
# ~Number of points across peak
BRAGG_PEAK_FWHM_BACK = 7
BRAGG_PEAK_FWHM_FWD = 7


# Debug. If true intermediate workspaces are made visible
DEBUG = True


# ----------------------------------------------------------------------------------------------------------------------
# Real work
# ----------------------------------------------------------------------------------------------------------------------

def calibrate(pb_runs, bkgd_run, ip_file, pb_mass, pb_dspacings, iterations,
              output_ip_file):
    logger.notice("Starting calibration procedure")
    logger.notice("Loading IP file")
    # Load current IP file to use parameters as starting point for iterative procedure
    current_ip = np.loadtxt(ip_file, skiprows=3, usecols=[0, 2, 3, 4, 5])

    # Calibrate L0 using forward scattering banks and use this value for all detectors
    # Also calibrates t0 for the forward banks
    l0, t0_forward, bad_fwd = calibrate_l0_t0fwd(U_SAMPLE, PB_SAMPLE)

    # calibrate L1, E1, & t0 for back scattering detectors
    t0, l1 = None, None
    thetas = current_ip[:, 1]
    for _ in range(iterations):
        l1, e1, t0_backward, bad_back = calibrate_l1_e1_t0back(PB_SAMPLE_U_FOIL, PB_SAMPLE, pb_runs,
                                                     l0, t0_forward, thetas, pb_mass, bad_fwd)

        # calibrate theta based on all previous values
        t0 = np.concatenate([t0_backward, t0_forward])
        bad_spectra = bad_back + bad_fwd
        thetas_iter = calibrate_theta(pb_runs, bkgd_run, l0, l1, t0, e1, thetas,
                                      pb_dspacings, bad_spectra)
        thetas = thetas_iter
    #endfor

    # Write out file
    write_ip_file(output_ip_file, thetas, t0, l0, l1)


def calibrate_l0_t0fwd(sample, background):
    logger.notice("Calibrating L0 & t0 for forward banks")
    # Load u_sample_fwd and normalise by lead
    u_sample_fwd = load_and_normalise_sample(sample, background, FORWARD_SPECTRA, CROP_RANGE, U_PARAMS)
    debug_ws('u_sample_fwd', u_sample_fwd)

    # Find all the peaks in the Uranium spectrum and keep those closest to the known resonance energies
    nspectra = u_sample_fwd.getNumberHistograms()
    l0_forward, t0_forward = np.zeros(nspectra), np.zeros(nspectra)
    bad_fwd = []
    logger.notice("Fitting to U resonance peaks for {0} spectra".format(nspectra))
    for i in range(nspectra):
        spectra_no = u_sample_fwd.getSpectrum(i).getSpectrumNo()
        t_res = find_uranium_resonances(u_sample_fwd, i, U_RESON_TOF_FOR, U_RESON_WINDOWS_FOR)
        if np.all(np.isfinite(t_res)):
            gradient, t0i = linear_least_squares(INV_VEL_U_RES, t_res)
            l0_forward[i] = gradient
            t0_forward[i] = t0i
        else:
            logger.warning("    Spectra {0}: Unable to find uranium resonances. Excluding.".format(spectra_no))
            bad_fwd.append(spectra_no)
        logger.notice(
            "    Spectra {0}: t_r={1}, L0={2}, t0={3}".format(spectra_no, t_res, l0_forward[i], t0_forward[i]))
    ##endfor
    l0_avg = np.nanmean(l0_forward)
    delta_l0 = np.nanstd(l0_forward)
    logger.notice("Mean L0={0}, dL0={1}".format(l0_avg, delta_l0))
    t0_avg = np.nanmean(t0_forward)
    delta_t0 = np.nanstd(t0_forward)
    logger.notice("Mean t0={0}, dt0={1}".format(t0_avg, delta_t0))
    return l0_avg, t0_forward, bad_fwd


def calibrate_l1_e1_t0back(pb_u_foil, background, pb_runs, l0, t0_fwd, thetas, pb_mass, bad_fwd=None):
    logger.notice("Calibrating L1, E1 & t0 for backscattering banks")
    # Dividing sample by background produces troughs (absorption lines ) rather than Peaks and the mantid
    # fitting cannot handle this. Inverting the normalisation to the peaks as peaks. This does
    # not affect the peak position
    u_reson_back = load_and_normalise_sample(pb_u_foil, background, BACKWARD_SPECTRA, CROP_RANGE, U_PARAMS, invert_divide=True)
    debug_ws('u_reson_back', u_reson_back)
    pb_peak_back = load_runs(pb_runs, 'SingleDifference', BACKWARD_SPECTRA, CROP_RANGE, U_PARAMS)
    debug_ws('pb_peak_back', pb_peak_back)

    # Compute rtheta for all spectra
    mass_ratio = pb_mass / N_MASS_AMU
    theta_back = thetas[:N_BACKWARD]
    rtheta_back = (np.cos(theta_back) + np.sqrt(np.square(mass_ratio) - np.square(np.sin(theta_back)))) / (
        1 + mass_ratio)

    nspectra = u_reson_back.getNumberHistograms()
    l1_back, t0_back, e1_back = np.zeros(nspectra), np.zeros(nspectra), np.zeros(nspectra)
    bad_back = []
    logger.notice("Fitting to U absorption peaks and Pb recoil peak for {0} forward spectra".format(nspectra))
    for i in range(nspectra):
        spectra_no = u_reson_back.getSpectrum(i).getSpectrumNo()
        t_u_res = find_uranium_resonances(u_reson_back, i, U_RESON_TOF_BACK, U_RESON_WINDOWS_BACK)
        if np.all(np.isfinite(t_u_res)):
            gradient, t0i = linear_least_squares(INV_VEL_U_RES, t_u_res)
            l1_back[i] = (gradient - l0) * rtheta_back[i]
            t0_back[i] = t0i
            # lead recoil
            t_pb_rec = find_lead_peak(pb_peak_back, i, PB_TOF, PB_WINDOWS)
            v1 = 1e6 * (l0 * rtheta_back[i] + l1_back[i]) / (t_pb_rec - t0i)
            e1_back[i] = 0.5 * N_MASS * np.square(v1) / MEV
        else:
            logger.warning("    Spectra {0}: Unable to find uranium resonances. Excluding.".format(spectra_no))
            bad_back.append(spectra_no)
        logger.notice("    Spectra {0}: t_u={1}, L1={2}, t0={3}, t_pb_rec={4}, E1={5}".format(spectra_no, t_u_res,
                                                                                              l1_back[i], t0_back[i],
                                                                                              t_pb_rec, e1_back[i]))
    # endfor
    l1_back_avg = np.nanmean(l1_back)
    delta_l1_back = np.nanstd(l1_back)
    logger.notice("Mean L1={0}, dL1={1}".format(l1_back_avg, delta_l1_back))
    t0_back_avg = np.nanmean(t0_back)
    delta_t0_back = np.nanstd(t0_back)
    logger.notice("Mean t0={0}, dt0={1}".format(t0_back_avg, delta_t0_back))
    e1_avg = np.nanmean(e1_back)
    delta_e1_back = np.nanstd(e1_back)
    logger.notice("Mean e1={0}, de1={1}".format(e1_avg, delta_e1_back))

    logger.notice("Calibrating L1 for forward banks")
    pb_peak_fwd = load_runs(pb_runs, 'SingleDifference', FORWARD_SPECTRA, CROP_RANGE,U_PARAMS)
    debug_ws('pb_peak_fwd', pb_peak_fwd)

    theta_fwd = thetas[N_BACKWARD:]
    rtheta_fwd = (np.cos(theta_fwd) + np.sqrt(np.square(mass_ratio) - np.square(np.sin(theta_fwd)))) / (1 + mass_ratio)
    v1_avg = np.sqrt(2 * e1_avg * MEV / N_MASS)
    nspectra = pb_peak_fwd.getNumberHistograms()
    logger.notice("Fitting to Pb recoil peak for {0} forward spectra".format(nspectra))
    l1_fwd = np.zeros(nspectra)
    for i in range(nspectra):
        spectra_no = pb_peak_fwd.getSpectrum(i).getSpectrumNo()
        if spectra_no not in bad_fwd:
            # lead recoil
            t_pb_rec = find_lead_peak(pb_peak_fwd, i, PB_TOF, PB_WINDOWS)
            l1_fwd[i] = 1e-6 * v1_avg * (t_pb_rec - t0_fwd[i]) - l0 * rtheta_fwd[i]
            logger.notice("    Spectra {0}: t_pb_rec={1}, L1={2}".format(spectra_no, t_pb_rec, l1_fwd[i]))
        else:
            logger.warning("    Skipping spectra {0}: previously marked as bad".format(spectra_no))
    # endif

    return np.concatenate([l1_back, l1_fwd]), e1_avg, t0_back, bad_back


def calibrate_theta(pb_runs, bkgd_run, l0, l1, t0, e1, theta_cur, dspacings, bad_spectra=None):
    # Uses the N longest Bragg peaks in the lead calibration data to compute an angle based on
    # Bragg's law: 2dsin(theta) = lambda. The dspacings provided are used to calculate an estimate
    # of the peak positions in TOF using a guess for theta. Each peak is fitted with a Voigt function
    # to determine the peak centre and this used to compute the incident lambda and hence theta for each peak.
    # The final value is then the average over the N peaks
    logger.notice("Calibrating theta for all detectors")
    pb_dspace = load_and_normalise_sample(pb_runs, bkgd_run, (BACKWARD_SPECTRA[0], FORWARD_SPECTRA[1]),
                                          CROP_RANGE_PB, PARAMS_PB)

    debug_ws('pb_dspace', pb_dspace)

    nspectra = pb_dspace.getNumberHistograms()
    logger.notice("Fitting to Pb bragg peaks in all spectra")
    theta_calib = np.zeros(nspectra)
    for i in range(nspectra):
        spectra_no = pb_dspace.getSpectrum(i).getSpectrumNo()
        if spectra_no in bad_spectra:
            logger.warning("    Skipping spectra {0}: previously marked as bad".format(spectra_no))
            continue
        peak_estimates_tof = dspacing_to_tof(dspacings, l0, l1[i], theta_cur[i])
        t_bragg = find_bragg_peaks(pb_dspace, i, spectra_no, peak_estimates_tof)
        # True TOF should subtract delay time
        t_bragg -= t0[i]
        lam_incident = tof_to_wavelength(t_bragg, l0, l1[i])
        stheta = 0.5 * lam_incident / (dspacings * 1e-10)
        theta_all = (np.degrees(np.arcsin(stheta)))
        theta_calib[i] = 2 * np.mean(theta_all)
        logger.notice("    Spectra {0}: estimate={1}, found={2}. "
                      "theta(deg)={3}".format(spectra_no, peak_estimates_tof, t_bragg, theta_calib[i]))
    # endfor

    return theta_calib


def dspacing_to_tof(dspacings, l0, l1, theta):
    lam0 = 2 * dspacings * 1e-4 * np.sin(0.5 * np.radians(theta))
    return (l0+l1) * N_MASS * lam0 / PLANCK


def tof_to_wavelength(tof, l0, l1):
    return PLANCK * tof * 1e-06 / (N_MASS * (l0 + l1))


def load_and_normalise_sample(sample_run, bkgd_run, spectra, crop_range, params, invert_divide=False):
    logger.notice("Loading " + str(sample_run))
    sample = load_runs(sample_run, 'FoilOut', spectra, crop_range, params)
    logger.notice("Loading " + str(bkgd_run))
    bkgd = load_runs(bkgd_run, 'FoilOut', spectra, crop_range, params)

    if sample.blocksize() > bkgd.blocksize():
        logger.notice("Rebinning sample to match background")
        bkgd = execute_alg_return_wksp("Rebin",InputWorkspace=bkgd,Params=params)

        sample = execute_alg_return_wksp("RebinToWorkspace", WorkspaceToRebin=sample,
                                         WorkspaceToMatch=bkgd)
    else:
        logger.notice("Rebinning background to match sample")
        sample = execute_alg_return_wksp("Rebin",InputWorkspace=sample,Params=params)
        bkgd = execute_alg_return_wksp("RebinToWorkspace", WorkspaceToRebin=bkgd,
                                       WorkspaceToMatch=sample)
    # Normalise
    if invert_divide:
        logger.notice("Normalising background by sample")
        sample = execute_alg_return_wksp("Divide", LHSWorkspace=bkgd, RHSWorkspace=sample)
    else:
        logger.notice("Normalising sample by background")
        sample = execute_alg_return_wksp("Divide", LHSWorkspace=sample, RHSWorkspace=bkgd)
    return sample


# ----------------------------------------------------------------------------------------------------------------------
# Utility Functions
# ----------------------------------------------------------------------------------------------------------------------

def debug_ws(name, ws):
    if DEBUG:
        mantid.AnalysisDataService.addOrReplace(name, ws)


def execute_alg(name, **kwargs):
    alg = AlgorithmManager.createUnmanaged(name)
    alg.initialize()
    alg.setChild(True)
    try:
        alg.setProperty("OutputWorkspace", "_")
    except RuntimeError:
        pass
    for name, value in iteritems(kwargs):
        if value is not None:
            alg.setProperty(name, value)
    alg.execute()
    return alg


def execute_alg_return_wksp(name, **kwargs):
    alg = execute_alg(name, **kwargs)
    return alg.getProperty("OutputWorkspace").value


def load_runs(runs, mode, spectra, crop_range,params):
    # Load and sum all runs
    if type(runs) == str:
        runs = runs.split("-")
    ws = load_run(runs[0], mode, spectra, crop_range)
    for run in runs[1:]:
        temp = load_run(run.strip(), mode, spectra, crop_range)
        ws = execute_alg_return_wksp('Plus', LHSWorkspace=ws, RHSWorkspace=temp)
    ws = execute_alg_return_wksp("Rebin",InputWorkspace=ws,Params=params)
    return ws


def load_run(run, mode, spectra, crop_range):
    # The current LoadVesuvio algorithm cannot load the only single-period runs so we have to resort
    # to LoadRaw and manually redoing some of the work to match the bins
    spectra_list = "{0}-{1}".format(*spectra)
    if nperiods(run) > 1:
        ws = execute_alg_return_wksp('LoadVesuvio', Filename=run,
                                     Mode=mode, SpectrumList=spectra_list)
    else:
        ws = execute_alg_return_wksp('LoadRaw', Filename='EVS' + run + '.raw',
                                     SpectrumList=spectra_list)
    
    
    # Crop
    return execute_alg_return_wksp('CropWorkspace', InputWorkspace=ws,
                                   XMin=crop_range[0], XMax=crop_range[1])


def nperiods(run):
    raw_check = execute_alg('RawFileInfo', Filename='EVS' + run + '.raw')
    return raw_check.getProperty('PeriodCount').value


def find_uranium_resonances(ws, index, positions, windows):
    """
    Run FindPeaks to determine peak centre of Uranium resonance peaks
    :param ws: Workspace containing data
    :param index: Index of spectrum to fit
    :param positions: List of expected positions of the peaks
    :param windows: Search windows around the peaks
    :return: Numpy array of peak centre values
    """
    # find the 3 resonance peaks
    peak_finder = execute_alg('FindPeaks', InputWorkspace=ws, PeakFunction='Gaussian',
                              BackgroundType='Linear', PeakPositions=positions, FitWindows=windows,
                              PeaksList="_", WorkspaceIndex=index)
    peak_params = peak_finder.getProperty("PeaksList").value
    t_res = np.array(peak_params.column('centre'))
    if len(t_res) != 3:
        spectra_no = ws.getSpectrum(index).getSpectrumNo()
        raise RuntimeError("Expecting 3 Uranium resonance peaks, "
                           "found {0} = {1} in spectra {2}".format(len(t_res), t_res, spectra_no))
    return t_res


def find_lead_peak(ws, index, position, window):
    """
    Run FindPeaks to determine peak centre of Lead recoil peak
    :param ws: Workspace containing data
    :param index: Index of spectrum to fit
    :param position: Expected positions of the peak in TOF
    :param window: Search windows around the peak in TOF
    :return: Peak centre value
    """
    peak_finder = execute_alg('FindPeaks', InputWorkspace=ws, PeakFunction='Voigt',
                              BackgroundType='Linear', PeakPositions=position, FitWindows=window,
                              PeaksList="_", WorkspaceIndex=index)
    peak_params = peak_finder.getProperty("PeaksList").value
    t_res = peak_params.column('centre')
    if len(t_res) != 1:
        spectra_no = ws.getSpectrum(index).getSpectrumNo()
        raise RuntimeError("Expecting 1 Lead recoil peak, "
                           "found {0} = {1} in spectra {2}".format(len(t_res), t_res, spectra_no))
    return t_res[0]


def find_bragg_peaks(ws, index, spectra_no, positions):
    """
    Run FindPeaks to determine peak centres of bragg peaks in Lead sample
    :param ws: Workspace containing data
    :param index: Index of spectrum to fit
    :param position: Expected positions of the peak in TOF
    :param window: Search windows around the peak in TOF
    :return: Peak centre value
    """

    peak_finder_params = {'InputWorkspace': ws, 'PeakFunction': 'Gaussian',
                          'BackgroundType': 'Linear', 'RawPeakParameters': False,
                          'PeaksList': "_", 'WorkspaceIndex': index}
    if spectra_no < FORWARD_SPECTRA[0]:
        # back scattering
        peak_finder_params.update(
            {'PeakPositions': positions,
             'PeakPositionTolerance': BRAGG_POSITION_TOL_BACK,
             'FWHM': BRAGG_PEAK_FWHM_BACK})
    else:
        # forward scattering
        peak_finder_params.update(
            {'PeakPositions': positions,
             'PeakPositionTolerance': BRAGG_POSITION_TOL_FWD,
             'FWHM': BRAGG_PEAK_FWHM_FWD,'HighBackground':True})
    peak_finder = execute_alg('FindPeaks', **peak_finder_params)
    peak_params = peak_finder.getProperty("PeaksList").value
    debug_ws('peak_params', peak_params)
    centres = peak_params.column('centre')
    widths = peak_params.column('width')
    # Find peaks closest to given expected positions
    t_bragg = []
    for bragg in positions:
        separation = sys.float_info.max
        closest_match = None
        for i, (found_peak, width) in enumerate(zip(centres, widths)):
            test_separation = np.abs(bragg - found_peak)
            if width > 0.0 and test_separation < separation:
                separation = test_separation
                closest_match = found_peak
        # endfor
        if closest_match is not None:
            t_bragg.append(closest_match)
    # endfor

    if len(t_bragg) != len(positions):
        raise RuntimeError("Expecting {0} bragg peaks, "
                           "found {0} = {1} in spectra {2}.".format(len(t_bragg), t_bragg, spectra_no))
    return np.array(t_bragg)


def write_ip_file(filepath, theta, t0, l0, l1):
    file_header = b'\t'.join(['plik', 'det', 'theta', 't0', 'L0', 'L1']) + '\n'
    fmt = "%d  %d  %.4f  %.4f  %.3f  %.4f"
    det = range(BACKWARD_SPECTRA[0], FORWARD_SPECTRA[1]+1)
    l0_arr = np.empty_like(l1)
    l0_arr.fill(l0)

    # pad the start of the file with dummy data for the monitors
    file_data = np.asarray([[1, 1, 0, 0, 0, 0], [2, 2, 0, 0, 0, 0]])
    file_data = np.append(file_data, np.column_stack((det, det, theta, t0, l0_arr, l1)), axis=0)
    with open(filepath, 'wb') as f_handle:
        f_handle.write(file_header)
        np.savetxt(f_handle, file_data, fmt=fmt)


def linear_least_squares(xdata, ydata):
    """
    Compute least-square fit of data to linear function
    :param xdata: X axis
    :param ydata: Y axis
    :return: Computed parameter values
    """

    # Compute params with leastsq fit to linear function
    def func(params, x, yobs):
        return np.square(yobs - (x * params[0] + params[1]))

    out = leastsq(func, x0=[0.1, 0.1], args=(xdata, ydata),
                  full_output=True)
    if out[4] in (1, 2, 3, 4):
        return out[0]
    else:
        raise RuntimeError("Error computing leastsq for L0/t0: {0}".format(out[3]))



if __name__ == '__main__':
    # Parameters
    _run_range = "17083-17084"# 2mm Pb sample in the beam with no U foil in . At present, the load_run and nperiods function cannot find Vesuvvio raw data with VESUVIO000 prefixes such as"42209-42228"#
    _background = "17086"# Vanadium sampe in the beam with no U foil in. At present, the load_run and nperiods function cannot find Vesuvvio raw data with VESUVIO000 prefixes such as"41977-41997"#
    _input_parameter_file = "C:/Users/vesuvio/Desktop/Mantid Vesuvio Calibration 2015/uranium calibration and IP files/IP20182.par"
    #_input_parameter_file  = "K:/Neutron_computations/MANTID\Mantid Vesuvio Calibration 2015/uranium calibration and IP files/IP20182.par"
    # Mass of a lead in amu
    _mass = 207.19
    # d-spacings of a lead sample
    _d_spacings = np.array([1.489, 1.750, 2.475, 2.858])
    _calc_l0 = True
    _iterations = 2
    _output_ip_file = "C:/Users/vesuvio/Desktop/Mantid Vesuvio Calibration 2015/uranium calibration and IP files/IP20182_new_no_L0.par"
    #_output_ip_file = "K:/Neutron_computations/MANTID/Mantid Vesuvio Calibration 2015/uranium calibration and IP files/IP20182_new_calibrate_vesuvio.par"
    output_workspace = "lead_calibration"

    calibrate(_run_range, _background, _input_parameter_file, _mass,
              _d_spacings, _iterations, _output_ip_file)

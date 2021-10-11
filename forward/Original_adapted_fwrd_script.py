'''
This file was last edited by Giovanni Romanelli on 21/02/2019.
The structure of the script is as follows:
    First, a concise explanation of what the file contains;
        Second, a list of functions representing the core of the script, and written so as to be 
        general to all data reductions and analysis of VESUVIO data;
            Last, towards the end of the document, a user driven script where the details of each
            particular experiment should inserted.
            
The script has been tested to be run in Mantid in a Windows operative system.

PLEASE, DO NOT MODIFY THE "TECHNICAL SECTION" UNLESS YOU ARE AN 
EXPERT VESUVIO INSTRUMENT SCIENTIST.
'''

##########################################################
####        TECHNICAL SECTION - NOT FOR USERS
##########################################################
import numpy as np
import matplotlib.pyplot as plt
import mantid                          
from mantid.simpleapi import *    
from scipy import signal
from scipy.optimize import curve_fit
from scipy import optimize
from scipy.ndimage import convolve1d
import time
from pathlib import Path
repoPath = Path(__file__).absolute().parent  # Path to the repository
start_time = time.time()

# command for the formatting of the printed output
np.set_printoptions(suppress=True, precision=4, linewidth= 150 )

#
#   INITIALISING FUNCTIONS AND USEFUL FUNCTIONS
#
def fun_gaussian(x, sigma):
    gaussian = np.exp(-x**2/2/sigma**2)
    gaussian /= np.sqrt(2.*np.pi)*sigma
    return gaussian

def fun_lorentzian(x, gamma):
    lorentzian = gamma/np.pi / (x**2 + gamma**2)
    return lorentzian

def fun_pseudo_voigt(x, sigma, gamma): #input std gaussiana e hwhm lorentziana
    fg, fl = 2.*sigma*np.sqrt(2.*np.log(2.)), 2.*gamma #parameters transformed to gaussian and lorentzian FWHM
    f = 0.5346 * fl + np.sqrt(0.2166*fl**2 + fg**2 )
    eta = 1.36603 *fl/f - 0.47719 * (fl/f)**2 + 0.11116 *(fl/f)**3
    sigma_v, gamma_v = f/(2.*np.sqrt(2.*np.log(2.))), f /2.
    pseudo_voigt = eta * fun_lorentzian(x, gamma_v) + (1.-eta) * fun_gaussian(x, sigma_v)
    norm=np.sum(pseudo_voigt)*(x[1]-x[0])
    return pseudo_voigt#/np.abs(norm)

def fun_derivative3(x,fun): # Used to evaluate numerically the FSE term.
    derivative =[0.]*len(fun)
    for i in range(6,len(fun)-6):
        derivative[i] = -fun[i+6] +24.*fun[i+5] -192.*fun[i+4] +488.*fun[i+3] +387.*fun[i+2] -1584.*fun[i+1]
        derivative[i]+= fun[i-6]  -24.*fun[i-5]   +192.*fun[i-4]  -488.*fun[i-3]   -387.*fun[i-2]   +1584.*fun[i-1]
        derivative[i]/=(x[i+1]-x[i])**3
    derivative=np.array(derivative)/12**3
    return derivative

def fun_derivative4(x,fun): # not used at present. Can be used for the H4 polynomial in TOF fitting.
    derivative =[0.]*len(fun)
    for i in range(8,len(fun)-8):
        derivative[i] = fun[i-8]   -32.*fun[i-7]  +384*fun[i-6]  -2016.*fun[i-5]  +3324.*fun[i-4]  +6240.*fun[i-3]  -16768*fun[i-2]  -4192.*fun[i-1]  +26118.*fun[i]
        derivative[i]+=fun[i+8] -32.*fun[i+7] +384*fun[i+6] -2016.*fun[i+5] +3324.*fun[i+4] +6240.*fun[i+3] -16768*fun[i+2] -4192.*fun[i+1]
        derivative[i]/=(x[i+1]-x[i])**4
    derivative=np.array(derivative)/12**4
    return derivative

def load_ip_file(spectrum):
    #print "Loading parameters from file: ", namedtuple
    ipfile =  repoPath / 'ip2018_3.par'
    f = open(ipfile, 'r')
    data = f.read()
    lines = data.split('\n')
    for line in range(lines.__len__()):
        col = lines[line].split('\t')
        if col[0].isdigit() and int(col[0]) == spectrum:
            angle = float(col[2])
            T0 = float(col[3])
            L0 = float(col[4])
            L1 = float(col[5])
    f.close()
    return angle, T0, L0, L1

def load_resolution_parameters(spectrum): # TO BE COMPLETED
    if spectrum > 134: # resolution parameters for front scattering detectors, in case of single difference
        dE1= 73. # meV , gaussian standard deviation
        dTOF= 0.37 # us
        dTheta= 0.016 #rad
        dL0= 0.021 # meters
        dL1= 0.023 # meters
        lorentzian_res_width = 24. # meV , HFHM
    if spectrum < 135: # resolution parameters for back scattering detectors, in case of double difference
        dE1= 88.7 # meV , gaussian standard deviation
        dTOF= 0.37 # us
        dTheta= 0.016 #rad
        dL0= 0.021 # meters
        dL1= 0.023 # meters
        lorentzian_res_width = 40.3 # meV , HFHM
    return dE1, dTOF, dTheta, dL0, dL1, lorentzian_res_width
    
def load_constants():
    mN=1.008    #a.m.u.
    Ef=4906.         # meV
    en_to_vel = 4.3737 * 1.e-4
    vf=np.sqrt( Ef ) * en_to_vel        #m/us
    hbar = 2.0445
    return mN, Ef, en_to_vel, vf, hbar

def load_data(file): # Not used at present. Can be used to load spectra from file rather than workspace.
    file = open(file, 'r')
    data = file.read()
    file.close()
    x , y , e = [] , [] , [] 
    i = 0
    for line in data.split('\n'):
        vars = line.split()
        if len(vars) == 3:
            x.append(float(vars[0])) , y.append(float(vars[1])) , e.append(float(vars[2]))
            i += 1
    x, y, err = np.array(x), np.array(y), np.array(e)
    return x, y, e

def load_workspace(ws_name, spectrum):
    ws=mtd[str(ws_name)]
    ws_len, ws_spectra =ws.blocksize()-1, ws.getNumberHistograms()
    ws_x,ws_y, ws_e = [0.]*ws_len, [0.]*ws_len,[0.]*ws_len
    for spec in range(ws_spectra):
        if ws.getSpectrum(spec).getSpectrumNo() == spectrum :
            for i in range(ws_len):
                # converting the histogram into points
                ws_y[i] = ( ws.readY(spec)[i] / (ws.readX(spec)[i+1] - ws.readX(spec)[i] ) )
                ws_e[i] = ( ws.readE(spec)[i] / (ws.readX(spec)[i+1] - ws.readX(spec)[i] ) )
                ws_x[i] = ( 0.5 * (ws.readX(spec)[i+1] + ws.readX(spec)[i] ) )
    ws_x, ws_y, ws_e = np.array(ws_x), np.array(ws_y), np.array(ws_e)
    return ws_x, ws_y, ws_e

#
#   FITTING FUNCTIONS
#
def block_fit_ncp(par,first_spectrum,last_spectrum,masses,ws_name,fit_arguments, verbose):
    
    print("\n", "Fitting Workspace: ", str(ws_name))
        
    intensities=np.zeros((len(masses),last_spectrum-first_spectrum+1))
    widths=np.zeros((len(masses),last_spectrum-first_spectrum+1))
    positions=np.zeros((len(masses),last_spectrum-first_spectrum+1))
    spectra=np.zeros((last_spectrum-first_spectrum+1))

    ws=mtd[str(ws_name)]
    ws_len, ws_spectra =ws.blocksize()-1, ws.getNumberHistograms()
    ncp, ncp_m = [0.]*ws_len, [0.]*ws_len
    
    CloneWorkspace(InputWorkspace=str(ws_name),OutputWorkspace=str(ws_name)+'_tof_fitted_profiles')
    for m in range(len(masses)):
        CloneWorkspace(InputWorkspace=str(ws_name),OutputWorkspace=str(ws_name)+ '_tof_fitted_profile_' + str(m + 1))
        
    ######################
    par_chi = np.zeros((last_spectrum-first_spectrum+1, len(par)+3))
    ###########

    j=0
    
    print("Fitting parameters are given as: [Intensity Width Centre] for each NCP")
    
    for spectrum in range(first_spectrum,last_spectrum+1):
        
        spec_index = spectrum - first_spectrum
        
        data_x, data_y,data_e = load_workspace(ws_name , spectrum)
        
        if (data_y.all()==0):
            
            print(spectrum, " ... skipping ...")
            
            for m in range(len(masses)):      #introduced this loop from original
                intensities[m][j]=None
                widths[m][j]=None
                positions[m][j]=None
            
            #####
            par_chi[j, -1] = None
            par_chi[j, -2] = None
            par_chi[j, 1:-2] = np.full(len(par), None)
            #####
            
            tmp=mtd[str(ws_name)+'_tof_fitted_profiles']
            for bin in range(ws_len):
                tmp.dataY(spec_index)[bin] = 0.
                tmp.dataE(spec_index)[bin] = 0.
       
            for m in range(len(masses)):
                tmp=mtd[str(ws_name) + '_tof_fitted_profile_' + str(m + 1)]
                for bin in range(ws_len):
                    tmp.dataY(spec_index)[bin] = 0.
                    tmp.dataE(spec_index)[bin] = 0. 
                    
        else:
            
            ncp_fit, fitted_par, result = fit_ncp(par, spectrum, masses, data_x, data_y, data_e, fit_arguments)
            
            ncp = calculate_ncp(fitted_par, spectrum , masses, data_x)
            
            tmp=mtd[str(ws_name)+'_tof_fitted_profiles']
            for bin in range(ws_len):
                tmp.dataY(spec_index)[bin] =  ncp[bin]
                tmp.dataE(spec_index)[bin] = 0.
           
            for m in range(len(masses)):
                tmp=mtd[str(ws_name) + '_tof_fitted_profile_' + str(m + 1)]
                ncp_m = calculate_ncp_m(fitted_par, m, spectrum , masses, data_x)
                for bin in range(ws_len):
                    tmp.dataY(spec_index)[bin] = ncp_m[bin] 
                    tmp.dataE(spec_index)[bin] = 0. 
             
            reduced_chi2 = result["fun"]/(len(data_x) - len(par))    
           ####### 
            par_chi[j, -1] = result["nit"]
            par_chi[j, -2] = reduced_chi2
            par_chi[j, 1:-2] = fitted_par
            ######
            npars = int(len(par)/len(masses))
                        
            for m in range(len(masses)):
                    intensities[m][j]=float(fitted_par[npars*m])
                    widths[m][j]=float(fitted_par[npars*m+1])
                    positions[m][j]=float(fitted_par[npars*m+2])
                    
            print(spectrum, fitted_par, "%.4g" % reduced_chi2)
       
        par_chi[j, 0] = spectrum
        spectra[j]=spectrum
        j +=1
        
    return spectra, intensities, widths, positions, par_chi


def fit_ncp(par, spectrum, masses, data_x, data_y, data_e, fit_arguments):
    boundaries, constraints = fit_arguments[0], fit_arguments[1]
    result = optimize.minimize(errfunc, par[:], args=(spectrum, masses, data_x, data_y, data_e), method='SLSQP', bounds = boundaries, constraints=constraints)
    fitted_par = result["x"]
    ncp = calculate_ncp(fitted_par, spectrum , masses, data_x)
    return ncp, fitted_par, result

def errfunc( par ,  spectrum, masses, data_x,  data_y, data_e):
    # this function provides the scalar to be minimised, with meaning of the non-reduced chi2
    ncp = calculate_ncp(par, spectrum , masses, data_x)
    if (np.sum(data_e) > 0):
        chi2 =  ((ncp - data_y)**2)/(data_e)**2 # Chi square per degree of freedom
    else:
        chi2 =  (ncp - data_y)**2
    return chi2.sum()

def calculate_ncp(par, spectrum , masses, data_x):
    angle, T0, L0, L1 = load_ip_file(spectrum)
    mN, Ef, en_to_vel, vf, hbar = load_constants()
    ncp = 0. # initialising the function values
    # velocities in m/us, times in us, energies in meV
    v0, E0, delta_E, delta_Q = calculate_kinematics(data_x, angle, T0, L0, L1 )
    npars = int(len(par)/len(masses))
    for m in range(len(masses)):#   [parameter_index + number_of_mass * number_of_parameters_per_mass ]
        mass, hei , width, centre = masses[m] , par[m*npars], par[1+m*npars], par[2+m*npars]
        E_r = ( hbar * delta_Q )**2 / 2. / mass
        y = mass / hbar**2 /delta_Q * (delta_E - E_r) 
        joy = fun_gaussian(y-centre, 1.)
        pcb = np.where(joy == joy.max()) # this finds the peak-centre bin (pcb)
        gaussian_res_width, lorentzian_res_width = calculate_resolution(spectrum, data_x[pcb], mass)
        # definition of the experimental neutron compton profile
        gaussian_width = np.sqrt( width**2 + gaussian_res_width**2 )
        joy = fun_pseudo_voigt(y-centre, gaussian_width, lorentzian_res_width)
        FSE =  - fun_derivative3(y,joy)*width**4/delta_Q * 0.72 # 0.72 is an empirical coefficient. One can alternatively add a fitting parameter for this term.
        #H4  = some_missing_coefficient *  fun_derivative4(y,joy) /(4.*width**4) /32.
        ncp += hei * (joy + FSE ) * E0 * E0**(-0.92) * mass / delta_Q # Here -0.92 is a parameter describing the epithermal flux tail.
    return ncp

def calculate_ncp_m(par, m, spectrum , masses, data_x):
    angle, T0, L0, L1 = load_ip_file(spectrum)
    mN, Ef, en_to_vel, vf, hbar = load_constants()
    ncp_m = 0. # initialising the function values
    # velocities in m/us, times in us, energies in meV
    v0, E0, delta_E, delta_Q = calculate_kinematics(data_x, angle, T0, L0, L1 )
    npars = int(len(par)/len(masses))
    mass, hei , width, centre = masses[m] , par[m*npars], par[1+m*npars], par[2+m*npars]
    E_r = ( hbar * delta_Q )**2 / 2. / mass
    y = mass / hbar**2 /delta_Q * (delta_E - E_r) 
    joy = fun_gaussian(y-centre, 1.)
    pcb = np.where(joy == joy.max()) # this finds the peak-centre bin (pcb)
    gaussian_res_width, lorentzian_res_width = calculate_resolution(spectrum, data_x[pcb], mass)
    # definition of the experimental neutron compton profile
    gaussian_width = np.sqrt( width**2 + gaussian_res_width**2 )
    joy = fun_pseudo_voigt(y-centre, gaussian_width, lorentzian_res_width)
    FSE =  - fun_derivative3(y,joy)*width**4/delta_Q * 0.72 # 0.72 is an empirical coefficient. One can alternatively add a fitting parameter for this term.
    #H4  = some_missing_coefficient *  fun_derivative4(y,joy) /(4.*width**4) /32.
    ncp_m= hei * (joy + FSE ) * E0 * E0**(-0.92) * mass / delta_Q # Here -0.92 is a parameter describing the epithermal flux tail.
    return ncp_m

def calculate_kinematics(data_x, angle, T0, L0, L1 ):
    mN, Ef, en_to_vel, vf, hbar = load_constants()
    t_us = data_x - T0
    v0 = vf * L0 / ( vf * t_us - L1 )
    E0 =  ( v0 / en_to_vel )**2 
    delta_E = E0 -Ef  
    delta_Q2 = 2. * mN / hbar**2 * ( E0 + Ef - 2. * np.sqrt( E0*Ef) * np.cos(angle/180.*np.pi) )
    delta_Q = np.sqrt( delta_Q2 )
    return v0, E0, delta_E, delta_Q

def calculate_resolution(spectrum, data_x, mass):
    angle, T0, L0, L1 = load_ip_file(spectrum)
    mN, Ef, en_to_vel, vf, hbar = load_constants()
    # all standard deviations, apart from lorentzian hwhm
    dE1, dTOF, dTheta, dL0, dL1, lorentzian_res_width = load_resolution_parameters(spectrum)
    v0, E0, delta_E, delta_Q = calculate_kinematics(data_x, angle, T0, L0, L1 )
    # definition of the resolution components in meV:
    dEE = (1. + (E0 / Ef)**1.5 * ( L1 / L0 ) )**2 * dE1**2 + (2. * E0 * v0 / L0 )**2 * dTOF**2 
    dEE+= ( 2. * E0**1.5 / Ef**0.5 / L0 )**2 * dL1**2 + ( 2. * E0 / L0 )**2 * dL0**2
    dQQ =  (1. - (E0 / Ef )**1.5 *L1 / L0 - np.cos(angle/180.*np.pi) * ( ( E0 / Ef )**0.5 - L1 / L0 * E0 / Ef ))**2 * dE1**2
    dQQ+= ( ( 2. * E0 * v0 / L0 )**2 * dTOF**2 + (2. * E0**1.5 / L0 / Ef**0.5 )**2 *dL1**2 + ( 2. * E0 / L0 )**2 * dL0**2 ) * np.abs( Ef / E0 * np.cos(angle/180.*np.pi) -1.)
    dQQ+= ( 2. * np.sqrt( E0 * Ef )* np.sin(angle/180.*np.pi) )**2 * dTheta**2
    # conversion from meV^2 to A^-2
    dEE*= ( mass / hbar**2 /delta_Q )**2
    dQQ*= ( mN / hbar**2 /delta_Q )**2
    gaussian_res_width =   np.sqrt( dEE + dQQ ) # in A-1
    #lorentzian component in meV
    dEE_lor = (1. + (E0 / Ef)**1.5 * ( L1 / L0 ) )**2                                                       # is it - or +?
    dQQ_lor =  (1. - (E0 / Ef )**1.5 *L1 / L0 - np.cos(angle/180.*np.pi) * ( ( E0 / Ef )**0.5 + L1 / L0 * E0 / Ef )) **2
    # conversion from meV^2 to A^-2
    dEE_lor*= ( mass / hbar**2 /delta_Q )**2
    dQQ_lor*= ( mN / hbar**2 /delta_Q )**2
    lorentzian_res_width *= np.sqrt( dEE_lor + dQQ_lor ) # in A-1
    return gaussian_res_width, lorentzian_res_width # gaussian std dev, lorentzian hwhm 
#
#       CORRECTION FUNCTIONS
#
def calculate_mean_widths_and_intensities(masses,widths,intensities,spectra, verbose):
    better_widths, better_intensities =np.zeros((len(masses),len(widths[0]))),np.zeros((len(masses),len(widths[0])))
    mean_widths,widths_std,mean_intensity_ratios,mean_intensity_ratios_std=np.zeros((len(masses))),np.zeros((len(masses))),np.zeros((len(masses))),np.zeros((len(masses)))
    for m in range(len(masses)):
        mean_widths[m]=np.nanmean(widths[m])
        widths_std[m]=np.nanstd(widths[m])
        for index in range(len(widths[0])): # over all spectra
            if  abs( widths[m][index]-mean_widths[m] ) > widths_std[m]:
                better_widths[m][index],better_intensities[m][index]= None, None
            else:
                better_widths[m][index],better_intensities[m][index]= widths[m][index],intensities[m][index]
        mean_widths[m]=np.nanmean(better_widths[m])
        widths_std[m]=np.nanstd(better_widths[m])
    for spec in range(len(spectra)):
        normalisation = better_intensities[:,spec]
        better_intensities[:,spec]/=normalisation.sum()
    for m in range(len(masses)):
        mean_intensity_ratios[m] = np.nanmean(better_intensities[m])
        mean_intensity_ratios_std[m] = np.nanstd(better_intensities[m])
    for m in range(len(masses)):
        print("\n", "Mass: ", masses[m], " width: ", mean_widths[m], " \pm ", widths_std[m])
        print("\n", "Mass: ", masses[m], " mean_intensity_ratio: ", mean_intensity_ratios[m], " \pm ", mean_intensity_ratios_std[m])
    return mean_widths, mean_intensity_ratios

def calculate_sample_properties(masses,mean_widths,mean_intensity_ratios, mode, verbose):
    if mode == "GammaBackground":
        profiles=""
        for m in range(len(masses)):
            mass, width, intensity=str(masses[m]), str(mean_widths[m]),str(mean_intensity_ratios[m])
            profiles+= "ws_name=GaussianComptonProfile,Mass="+mass+",Width="+width+",Intensity="+intensity+';' 
        sample_properties = profiles
    elif mode == "MultipleScattering":
        MS_properties=[]
        for m in range(len(masses)):
            MS_properties.append(masses[m])
            MS_properties.append(mean_intensity_ratios[m])
            MS_properties.append(mean_widths[m])
        sample_properties = MS_properties        
    if verbose:
        print("\n", "The sample properties for ",mode," are: ", sample_properties)
    return sample_properties
        
def correct_for_gamma_background(ws_name):
    if verbose:
        print("Evaluating the Gamma Background Correction.")
    # Create an empty workspace for the gamma correction
    CloneWorkspace(InputWorkspace=ws_name,OutputWorkspace="gamma_background_correction")
    ws=mtd["gamma_background_correction"]
    for spec in range(ws.getNumberHistograms()):
        profiles=''
        for m in range(masses.__len__()):
            mass,width,intensity=str(masses[m]), str(mean_widths[m]),str(mean_intensity_ratios[m])
            profiles+= "name=GaussianComptonProfile,Mass="+mass+",Width="+width+",Intensity="+intensity+';'
        background, corrected = VesuvioCalculateGammaBackground(InputWorkspace=ws_name, 
                                                                        ComptonFunction=profiles, WorkspaceIndexList=spec)
        for bin in range(ws.blocksize()):
            ws.dataY(spec)[bin],ws.dataE(spec)[bin]=background.dataY(0)[bin],background.dataE(0)[bin]
    RenameWorkspace(InputWorkspace= "gamma_background_correction", OutputWorkspace = str(ws_name)+"_gamma_background")
    DeleteWorkspace("background")
    DeleteWorkspace("corrected")
    return

def create_slab_geometry(ws_name,vertical_width, horizontal_width, thickness):
        half_height, half_width, half_thick = 0.5*vertical_width, 0.5*horizontal_width, 0.5*thickness
        xml_str = \
        " <cuboid id=\"sample-shape\"> " \
        + "<left-front-bottom-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (half_width,-half_height,half_thick) \
        + "<left-front-top-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (half_width, half_height, half_thick) \
        + "<left-back-bottom-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (half_width, -half_height, -half_thick) \
        + "<right-front-bottom-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (-half_width, -half_height, half_thick) \
        + "</cuboid>"
        CreateSampleShape(ws_name, xml_str)
        return

def correct_for_multiple_scattering(ws_name,first_spectrum,last_spectrum, sample_properties, 
                                                        transmission_guess, multiple_scattering_order, number_of_events):
    if verbose:
        print("Evaluating the Multiple Scattering Correction.")
    dens, trans = VesuvioThickness(Masses=masses, Amplitudes=mean_intensity_ratios, TransmissionGuess=transmission_guess,Thickness=0.1)         
    _TotScattering, _MulScattering = VesuvioCalculateMS(ws_name, NoOfMasses=len(masses), SampleDensity=dens.cell(9,1), 
                                                                        AtomicProperties=sample_properties, BeamRadius=2.5,
                                                                        NumScatters=multiple_scattering_order, 
                                                                        NumEventsPerRun=int(number_of_events))
    data_normalisation = Integration(ws_name) 
    simulation_normalisation = Integration("_TotScattering")
    for workspace in ("_MulScattering","_TotScattering"):
        Divide(LHSWorkspace = workspace, RHSWorkspace = simulation_normalisation, OutputWorkspace = workspace)
        Multiply(LHSWorkspace = workspace, RHSWorkspace = data_normalisation, OutputWorkspace = workspace)
        RenameWorkspace(InputWorkspace = workspace, OutputWorkspace = str(ws_name)+workspace)
    DeleteWorkspace(data_normalisation)
    DeleteWorkspace(simulation_normalisation)
    DeleteWorkspace(trans)
    DeleteWorkspace(dens)
    return
############################
### functions to fit the NCP in the y space
############################
def subtract_other_masses(ws_last_iteration, intensities, widths, positions, spectra, masses):
    first_ws = CloneWorkspace(InputWorkspace=ws_last_iteration)
    ##----------------------
    all_ncp_m = np.zeros((len(masses)-1, first_ws.getNumberHistograms(), first_ws.blocksize()-1))
    ##--------------------------
    for index in range(len(spectra)):
        data_x, data_y, data_e = load_workspace(first_ws , spectra[index])
        if (data_y.all()==0):
            for bin in range(len(data_x)-1):
                first_ws.dataY(index)[bin] = 0
        else:
            for m in range(len(masses)-1):
                other_par = (intensities[m+1, index],widths[m+1, index],positions[m+1, index])
                ncp = calculate_ncp(other_par, spectra[index], [masses[m+1]], data_x)
                ##--------------
                all_ncp_m[m, index, :] = ncp
                ##------------------
                for bin in range(len(data_x)-1):
                    first_ws.dataY(index)[bin] -= ncp[bin]*(data_x[bin+1]-data_x[bin])

    return first_ws, all_ncp_m    #originally returns just the workspace

def convert_to_y_space_and_symmetrise(ws_name,mass):
    max_Y = np.ceil(2.5*mass+27)
    rebin_parameters = str(-max_Y)+","+str(2.*max_Y/120)+","+str(max_Y)
    ConvertToYSpace(InputWorkspace=ws_name,Mass=mass,OutputWorkspace=ws_name+"_JoY",QWorkspace=ws_name+"_Q")
    Rebin(InputWorkspace=ws_name+"_JoY", Params = rebin_parameters,FullBinsOnly=True, OutputWorkspace= ws_name+"_JoY")
    tmp=CloneWorkspace(InputWorkspace=ws_name+"_JoY")
    for j in range(tmp.getNumberHistograms()):
        for k in range(tmp.blocksize()):
            tmp.dataE(j)[k] =0.
            if (tmp.dataY(j)[k]!=0):
                tmp.dataY(j)[k] =1.
    tmp=SumSpectra('tmp')
    SumSpectra(InputWorkspace=ws_name+"_JoY",OutputWorkspace=ws_name+"_JoY")
    Divide(LHSWorkspace=ws_name+"_JoY", RHSWorkspace="tmp", OutputWorkspace =ws_name+"_JoY")
    ws=mtd[ws_name+"_JoY"]
    tmp=CloneWorkspace(InputWorkspace=ws_name+"_JoY")
    for k in range(tmp.blocksize()):
        tmp.dataE(0)[k] =(ws.dataE(0)[k]+ws.dataE(0)[ws.blocksize()-1-k])/2.
        tmp.dataY(0)[k] =(ws.dataY(0)[k]+ws.dataY(0)[ws.blocksize()-1-k])/2
    RenameWorkspace(InputWorkspace="tmp",OutputWorkspace=ws_name+"_JoY")
    normalise_workspace(ws_name+"_JoY")
    return max_Y

def calculate_mantid_resolutions(ws_name, mass):
    max_Y = np.ceil(2.5*mass+27)
    rebin_parameters = str(-max_Y)+","+str(2.*max_Y/240)+","+str(max_Y)
    ws= mtd[ws_name]
    for index in range(ws.getNumberHistograms()):
        VesuvioResolution(Workspace=ws,WorkspaceIndex=index,Mass=mass,OutputWorkspaceYSpace="tmp")
        tmp=Rebin("tmp",rebin_parameters)
        if index == 0:
            RenameWorkspace("tmp","resolution")
        else:
            AppendSpectra("resolution", "tmp", OutputWorkspace= "resolution")
    SumSpectra(InputWorkspace="resolution",OutputWorkspace="resolution")
    normalise_workspace("resolution")
    DeleteWorkspace("tmp")
    
def normalise_workspace(ws_name):
    tmp_norm = Integration(ws_name)
    Divide(LHSWorkspace=ws_name,RHSWorkspace="tmp_norm",OutputWorkspace=ws_name)
    DeleteWorkspace("tmp_norm")


################################################################################################
##################                                                                                                                                 ##################
##################                                                                                                                                 ##################
##################                                                                                                                                 ##################
##################                                                  HAVE FUN!                                                               ##################
##################                                                                                                                                 ##################
##################                                                                                                                                 ##################
##################                                                                                                                                 ##################
##################                                                                                                                                 ##################
################################################################################################
##########################################################
####        USER SECTION  -  FOR USERS 
##########################################################


'''
The user section is composed of an initialisation section, an iterative analysis/reduction section
of the spectra in the time-of-flight domain, and a final section where the analysis of the corrected
hydrogen neutron Compton profile is possible in the Y-space domain.

The fit procedure in the time-of-flight domain is  based on the scipy.minimize.optimize() tool,
used with the SLSQP minimizer, that can handle both boundaries and constraints for fitting parameters.

The Y-space analysis is, at present, performed on a single spectrum, being the result of
the sum of all the corrected spectra, subsequently symmetrised and unit-area normalised.

The Y-space fit is performed using the Mantid minimiser and average Mantid resolution function, using
a Gauss-Hermite expansion including H0 and H4 at present, while H3 (proportional to final-state effects)
is not needed as a result of the symmetrisation.
'''
load_data=True                             # If data have already been loaded, it can be put to Fasle to save time;
verbose=True                                 # If True, prints the value of the fitting parameters for each time-of-flight spectrum
plot_iterations = True                      # If True, plots all the time-of-flight spectra and fits in a single window for each iteration
number_of_iterations = 2              # This is the number of iterations for the reduction analysis in time-of-flight.
fit_in_Y_space = False      # If True, corrected time-of-flight spectra containing H only are transformed to Y-space and fitted.

ws_name="starch_80_RD_"
ws_name_raw="starch_80_RD_raw_"
ws_name_empty="starch_80_RD_empty_"
first_spectrum, last_spectrum = 144,147   #144, 182

masses = [1.0079,12,16,27]
abs_cross_sections = [] # This should be a vector of absorprion-to-scattering cross sections for each mass.

simple_gaussian_fit = True

if load_data:
    spectrum_list=str(first_spectrum)+'-'+str(last_spectrum)
    runs='43066-43076'
    ipfile = r'./ip2018_3.par'
    # LoadVesuvio(Filename=runs,SpectrumList=spectrum_list,Mode="SingleDifference",SumSpectra=False,InstrumentParFile=ipfile, OutputWorkspace=ws_name_raw)
    Load(Filename= r"./input_ws/starch_80_RD_raw.nxs", OutputWorkspace=ws_name_raw)
    Rebin(InputWorkspace=ws_name_raw,Params="110,1.,430",OutputWorkspace=ws_name_raw)
    SumSpectra(InputWorkspace=ws_name_raw, OutputWorkspace=ws_name_raw+'_sum')
    CloneWorkspace(InputWorkspace = ws_name_raw, OutputWorkspace = ws_name)

# Parameters for the multiple-scattering correction, including the shape of the sample.
transmission_guess = 0.8537 #  experimental value from VesuvioTransmission
multiple_scattering_order, number_of_events = 2, 1.e5
vertical_width, horizontal_width, thickness = 0.1, 0.1, 0.001 # expressed in meters
create_slab_geometry(ws_name,vertical_width, horizontal_width, thickness)

# parameters of the neutron Compton profiles to be fitted.
# masses = [1.0079,12,16,27]
# abs_cross_sections = [] # This should be a vector of absorprion-to-scattering cross sections for each mass.

# simple_gaussian_fit = True

## backscattering results
#Mass:  12  width:  12.71902583773011  \pm  3.4359310000812378

#Mass:  12  mean_intensity_ratio:  0.4664958744017039  \pm  0.12299478866963577

#Mass:  16  width:  8.768010990253744  \pm  1.2827275020662328

#Mass:  16  mean_intensity_ratio:  0.2073362848376038  \pm  0.1052121526142924

#Mass:  27  width:  13.897050850329862  \pm  0.3128325318724326

#Mass:  27  mean_intensity_ratio:  0.3261678407606922  \pm  0.05777670083343279

#parameters as:  Intensity,            NCP width,   NCP centre A-1
# first  element:       H                                                                                         
par           = (   1,                   4.7,                 0.           )
bounds     = (  (0, None),  ( 3,6),  (-3., 1.))
# first  element:       C                                                                                           
par           += (   1,                   12.71,                 0.           )
bounds     += (  (0, None),  (12.71,12.71),  (-3., 1.))  
# second  element:       0                                                                                           
par           += (   1,                    8.76,                 0.           )
bounds     += (  (0, None),  (8.76,8.76),  (-3., 1.))    
# third element:       Al                                                                                         
par           += (   1,                   13.897,                 0.           )
bounds     += (  (0, None),  (13.897,13.897),  (-3., 1.))


#constraints =  ({'type': 'eq', 'fun': lambda par:  par[6] -0.2963*par[3] })
constraints =  ()
fit_arguments = [bounds, constraints]
# spectra to be masked
#spectra_to_be_masked = "173,174,179"
spectra_to_be_masked = [173, 174, 179]
H_spectra_to_be_masked = ""

############### my edits #############
######################################

spectra_to_be_masked = np.array(spectra_to_be_masked)
spectra_to_be_masked = spectra_to_be_masked[(spectra_to_be_masked >= first_spectrum) & (spectra_to_be_masked <= last_spectrum)]   #detectors within spectrums

spec_offset = mtd[ws_name].getSpectrum(0).getSpectrumNo()  
first_idx, last_idx = first_spectrum - spec_offset, last_spectrum - spec_offset
CropWorkspace(InputWorkspace=ws_name, StartWorkspaceIndex = first_idx, EndWorkspaceIndex = last_idx, OutputWorkspace=ws_name) #for MS correction
##spec_offset = mtd[ws_name].getSpectrum(0).getSpectrumNo()

all_mean_widths, all_mean_intensities = np.zeros((number_of_iterations, len(masses))), np.zeros((number_of_iterations, len(masses)))
all_spec_best_par_chi_nit = np.zeros((number_of_iterations, last_spectrum-first_spectrum+1, len(masses)*3+3))
all_fit_workspaces = np.zeros((number_of_iterations, mtd[ws_name].getNumberHistograms(), mtd[ws_name].blocksize())) 

#######################################
#######################################

# Iterative analysis and correction of time-of-flight spectra.
for iteration in range(number_of_iterations):
    if iteration == 0:
        ws_to_be_fitted = CloneWorkspace(InputWorkspace = ws_name, OutputWorkspace = ws_name+str(iteration))
    ws_to_be_fitted = mtd[ws_name+str(iteration)]
    MaskDetectors(Workspace=ws_to_be_fitted,SpectraList=spectra_to_be_masked)

    ##-------------
    all_fit_workspaces[iteration] = ws_to_be_fitted.extractY()
    ##-------------


    # Fit and plot where the spectra for the current iteration
    spectra, intensities, widths, positions, par_chi = block_fit_ncp(par,first_spectrum,last_spectrum, masses,ws_to_be_fitted, fit_arguments, verbose)
    # Calculate mean widths and intensities
    mean_widths, mean_intensity_ratios = calculate_mean_widths_and_intensities(masses, widths, intensities, spectra, verbose) # at present is not multiplying for 0,9

    ##----------
    all_mean_widths[iteration] = np.array(mean_widths)
    all_mean_intensities[iteration] = np.array(mean_intensity_ratios)
    all_spec_best_par_chi_nit[iteration] = par_chi
    ##----------

    if (number_of_iterations - iteration -1 > 0):
        # evaluate gamma background correction ---------- This creates a background workspace with ws_name :  str(ws_name)+"_gamma_background"
        sample_properties = calculate_sample_properties(masses, mean_widths, mean_intensity_ratios, "GammaBackground", verbose)
        correct_for_gamma_background(ws_name)
        Scale(InputWorkspace = str(ws_name)+"_gamma_background", OutputWorkspace = str(ws_name)+"_gamma_background", Factor=0.9, Operation = "Multiply")
        # evaluate multiple scattering correction --------- This creates a background workspace with ws_name :  str(ws_name)+"_MulScattering"
        # skip the multiscattering correction for now
        # sample_properties = calculate_sample_properties(masses, mean_widths, mean_intensity_ratios, "MultipleScattering", verbose)
        # correct_for_multiple_scattering(ws_name, first_spectrum,last_spectrum, sample_properties, transmission_guess, multiple_scattering_order, number_of_events)
        # # Create corrected workspace
        Minus(LHSWorkspace= ws_name, RHSWorkspace = str(ws_name)+"_gamma_background", OutputWorkspace = ws_name+str(iteration+1))
        # Minus(LHSWorkspace= ws_name+str(iteration+1), RHSWorkspace = str(ws_name)+"_MulScattering", OutputWorkspace = ws_name+str(iteration+1))

if fit_in_Y_space:
    first_ws = subtract_other_masses(ws_name+str(number_of_iterations-1), intensities, widths, positions, spectra, masses)
    RenameWorkspace("first_ws", ws_name+'_H')
    Rebin(InputWorkspace=ws_name+'_H',Params="110,1.,430",OutputWorkspace=ws_name+'_H')
    MaskDetectors(Workspace=ws_name+'_H',SpectraList=H_spectra_to_be_masked)
    RemoveMaskedSpectra(InputWorkspace=ws_name+'_H', OutputWorkspace=ws_name+'_H')      
    
    # Conversion to hydrogen West-scaling variable
    rebin_params='-20,0.5,20'
    ConvertToYSpace(InputWorkspace=ws_name+'_H',Mass=1.0079,OutputWorkspace=ws_name+'joy',QWorkspace=ws_name+'q')
    Rebin(InputWorkspace=ws_name+'joy',Params=rebin_params,OutputWorkspace=ws_name+'joy')
    Rebin(InputWorkspace=ws_name+'q',Params=rebin_params,OutputWorkspace=ws_name+'q')
    tmp=Integration(InputWorkspace=ws_name+'joy',RangeLower='-20',RangeUpper='20')
    Divide(LHSWorkspace=ws_name+'joy',RHSWorkspace='tmp',OutputWorkspace=ws_name+'joy')
    
    # Symmetrisation to remove the FSEs
    ws=mtd[ws_name+'joy']
    for j in range(ws.getNumberHistograms()):
        for k in range(ws.blocksize()):
            if (ws.dataX(j)[k]<0):
                ws.dataY(j)[k] =ws.dataY(j)[ws.blocksize()-1-k]
                ws.dataE(j)[k] =ws.dataE(j)[ws.blocksize()-1-k]


    # Definition of a normalising workspace taking into consideration the kinematic constraints
    ws=CloneWorkspace(InputWorkspace=ws_name+'joy')
    for j in range(ws.getNumberHistograms()):
        for k in range(ws.blocksize()):
            ws.dataE(j)[k] =0.000001
            if (ws.dataY(j)[k]!=0):
                ws.dataY(j)[k] =1.
    ws=SumSpectra('ws')
    RenameWorkspace('ws',ws_name+'joy_sum_normalisation')


    # Definition of the sum of all spectra
    SumSpectra(ws_name+'joy',OutputWorkspace=ws_name+'joy_sum')
    Divide(LHSWorkspace=ws_name+'joy_sum',RHSWorkspace=ws_name+'joy_sum_normalisation',OutputWorkspace=ws_name+'joy_sum')

    # Definition of the resolution functions
    resolution=CloneWorkspace(InputWorkspace=ws_name+'joy')
    resolution=Rebin(InputWorkspace='resolution',Params='-20,0.5,20')
    for i in range(resolution.getNumberHistograms()):
        VesuvioResolution(Workspace=ws_name+str(iteration),WorkspaceIndex=str(i), Mass=1.0079, OutputWorkspaceYSpace='tmp')
        tmp=Rebin(InputWorkspace='tmp',Params='-20,0.5,20')
        for p in range (tmp.blocksize()):
            resolution.dataY(i)[p]=tmp.dataY(0)[p]

    # Definition of the sum of resolution functions
    resolution_sum=SumSpectra('resolution')
    tmp=Integration('resolution_sum')
    resolution_sum=Divide('resolution_sum','tmp')
    DeleteWorkspace('tmp')        


    ############################################################################
    ######
    ######              FIT OF THE SUM OF SPECTRA 
    ######
    ############################################################################
    print('\n','Fit on the sum of spectra in the West domain','\n')
    for minimizer_sum in ('Levenberg-Marquardt','Simplex'):
        CloneWorkspace(InputWorkspace = ws_name+'joy_sum', OutputWorkspace = ws_name+minimizer_sum+'_joy_sum_fitted')
        
        if (simple_gaussian_fit):
            function='''composite=Convolution,FixResolution=true,NumDeriv=true;
            ws_name=Resolution,Workspace=resolution_sum,WorkspaceIndex=0;
            ws_name=UserFunction,Formula=y0+A*exp( -(x-x0)^2/2/sigma^2)/(2*3.1415*sigma^2)^0.5,
            y0=0,A=1,x0=0,sigma=5,   ties=()'''
        else:
            function='''composite=Convolution,FixResolution=true,NumDeriv=true;
            ws_name=Resolution,Workspace=resolution_sum,WorkspaceIndex=0;
            ws_name=UserFunction,Formula=y0+A*exp( -(x-x0)^2/2/sigma1^2)/(2*3.1415*sigma1^2)^0.5
            +A*0.054*exp( -(x-x0)^2/2/sigma2^2)/(2*3.1415*sigma2^2)^0.5,
            y0=0,x0=0,A=0.7143,sigma1=4.76, sigma2=5,   ties=(sigma1=4.76)'''

        Fit(Function=function, InputWorkspace=ws_name+minimizer_sum+'_joy_sum_fitted', Output=ws_name+minimizer_sum+'_joy_sum_fitted',Minimizer=minimizer_sum)
        
        ws=mtd[ws_name+minimizer_sum+'_joy_sum_fitted_Parameters']
        print('Using the minimizer: ',minimizer_sum)
        print('Hydrogen standard deviation: ',ws.cell(3,1),' +/- ',ws.cell(3,2))


# ----------------- My edits ----------------------
all_tot_ncp = np.zeros((number_of_iterations, mtd[ws_name].getNumberHistograms(), mtd[ws_name].blocksize()))
all_indiv_ncp = np.zeros((number_of_iterations, len(masses), mtd[ws_name].getNumberHistograms(), mtd[ws_name].blocksize()))
                    
for i in range(number_of_iterations):
    ncp_ws_name = ws_name + str(i)
    ncp_tot = mtd[ncp_ws_name + "_tof_fitted_profiles"]
    ncp_tot_dataY = ncp_tot.extractY()
    all_tot_ncp[i] = ncp_tot_dataY
    
    for m in range(len(masses)):
        ncp_m = mtd[ncp_ws_name + "_tof_fitted_profile_" + str(m+1)]
        ncp_m_dataY = ncp_m.extractY()
        all_indiv_ncp[i, m] = ncp_m_dataY

##-------------------save results-------------------
savepath = repoPath / "tests" / "fixatures" / "original_adapted_run_144-182"

np.savez(savepath, all_fit_workspaces = all_fit_workspaces, \
                all_spec_best_par_chi_nit = all_spec_best_par_chi_nit, \
                all_mean_widths = all_mean_widths, all_mean_intensities = all_mean_intensities, \
                all_tot_ncp = all_tot_ncp, all_indiv_ncp = all_indiv_ncp)


#"C:\Users\guijo\Desktop\Work\My_edited_scripts\tests_data\original_4.2_no_mulscat\original_spec3-13_iter1"
end_time = time.time()
print("execution time: ", end_time - start_time, "seconds")
############################################################################
######
######              AND THEY ALL LIVED HAPPILY EVER AFTER
######
############################################################################


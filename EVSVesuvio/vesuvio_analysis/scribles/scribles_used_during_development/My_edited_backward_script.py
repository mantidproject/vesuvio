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

# command for the formatting of the printed output
np.set_printoptions(suppress=True, precision=4, linewidth= 150 )

#
#   INITIALISING FUNCTIONS AND USEFUL FUNCTIONS
#
def fun_gaussian(x, sigma):
    """Gaussian function centered at zero"""
    gaussian = np.exp(-x**2/2/sigma**2)
    gaussian /= np.sqrt(2.*np.pi)*sigma
    return gaussian

def fun_lorentzian(x, gamma):
    """Lorentzian centered at zero"""
    lorentzian = gamma/np.pi / (x**2 + gamma**2)
    return lorentzian

def fun_pseudo_voigt(x, sigma, gamma): 
    """Convolution between Gaussian with std sigma and Lorentzian with HWHM gamma"""
    fg, fl = 2.*sigma*np.sqrt(2.*np.log(2.)), 2.*gamma #parameters transformed to gaussian and lorentzian FWHM
    f = 0.5346 * fl + np.sqrt(0.2166*fl**2 + fg**2 )
    eta = 1.36603 *fl/f - 0.47719 * (fl/f)**2 + 0.11116 *(fl/f)**3
    sigma_v, gamma_v = f/(2.*np.sqrt(2.*np.log(2.))), f /2.
    pseudo_voigt = eta * fun_lorentzian(x, gamma_v) + (1.-eta) * fun_gaussian(x, sigma_v)
    norm=np.sum(pseudo_voigt)*(x[1]-x[0])
    return pseudo_voigt#/np.abs(norm)

def fun_derivative3(x, fun):
    """Numerical approximation for the third derivative"""   
    x, fun, derivative = np.array(x), np.array(fun), np.zeros(len(fun))
    derivative = - np.roll(fun,-6) + 24*np.roll(fun,-5) - 192*np.roll(fun,-4) + 488*np.roll(fun,-3) + 387*np.roll(fun,-2) - 1584*np.roll(fun,-1)  \
                 + np.roll(fun,+6) - 24*np.roll(fun,+5) + 192*np.roll(fun,+4) - 488*np.roll(fun,+3) - 387*np.roll(fun,+2) + 1584*np.roll(fun,+1)
    derivative /= np.power(np.roll(x,-1) - x, 3) 
    derivative /= 12**3
    derivative[:6], derivative[-6:] = np.zeros(6), np.zeros(6)  #need to correct for beggining and end of array because of the rolls
    return derivative
    
# def fun_derivative4(x,fun): # not used at present. Can be used for the H4 polynomial in TOF fitting.
#     derivative =[0.]*len(fun)
#     for i in range(8,len(fun)-8):
#         derivative[i] = fun[i-8]   -32.*fun[i-7]  +384*fun[i-6]  -2016.*fun[i-5]  +3324.*fun[i-4]  +6240.*fun[i-3]  -16768*fun[i-2]  -4192.*fun[i-1]  +26118.*fun[i]
#         derivative[i]+=fun[i+8] -32.*fun[i+7] +384*fun[i+6] -2016.*fun[i+5] +3324.*fun[i+4] +6240.*fun[i+3] -16768*fun[i+2] -4192.*fun[i+1]
#         derivative[i]/=(x[i+1]-x[i])**4
#     derivative=np.array(derivative)/12**4
#     return derivative

def load_ip_file(spectrum):
    """Instrument parameters of VESUVIO"""  
    
    ipfile = r'C:\Users\guijo\Desktop\Work\ip2018.par'
    f = open(ipfile, 'r')
    data = f.read()
    lines = data.split('\n')
    for line in lines:       
        col = line.split('\t')
        if col[0].isdigit() and int(col[0]) == spectrum:
            angle = float(col[2])
            T0 = float(col[3])
            L0 = float(col[4])
            L1 = float(col[5])
    f.close()
    return angle, T0, L0, L1    
    
def load_resolution_parameters(spectrum):   
    """Resolution of parameters to propagate into TOF resolution"""
    
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
        lorentzian_res_width = 40.3 # meV , HFHM    #I thought the Lorentzian component came from dE1?
    return dE1, dTOF, dTheta, dL0, dL1, lorentzian_res_width
    
def load_constants():
    """Output: the mass of the neutron, final energy of neutrons (selected by gold foil),
    factor to change energies into velocities, final velocity of neutron and hbar"""
    
    mN=1.008    #a.m.u.
    Ef=4906.         # meV
    en_to_vel = 4.3737 * 1.e-4
    vf=np.sqrt( Ef ) * en_to_vel        #m/us
    hbar = 2.0445
    return mN, Ef, en_to_vel, vf, hbar

# def load_data(file): # Not used at present. Can be used to load spectra from file rather than workspace.
#     file = open(file, 'r')
#     data = file.read()
#     file.close()
#     x , y , e = [] , [] , [] 
#     i = 0
#     for line in data.split('\n'):
#         vars = line.split()
#         if len(vars) == 3:
#             x.append(float(vars[0])) , y.append(float(vars[1])) , e.append(float(vars[2]))
#             i += 1
#     x, y, err = np.array(x), np.array(y), np.array(e)
#     return x, y, e

def load_workspace(ws, spectrum):   
    """Returns the data arrays for a given spectrum number for a given workspace ws"""
    
    spec_offset = ws.getSpectrum(0).getSpectrumNo()  
    spec_idx = spectrum - spec_offset

    ws_y, ws_x, ws_e = ws.readY(spec_idx), ws.readX(spec_idx), ws.readE(spec_idx)
    
    hist_widths = ws_x[1:] - ws_x[:-1]     #length decreases by one
    data_y = ws_y[:-1] / hist_widths
    data_e = ws_e[:-1] / hist_widths
    data_x = (ws_x[:-1] + ws_x[1:]) / 2    #compute mean point of bins, there is also a ConvertToPointData that does a similar thing
       
    return data_x, data_y, data_e    

#
#   FITTING FUNCTIONS
#
def block_fit_ncp(par, first_spectrum, last_spectrum, masses, ws, fit_arguments, verbose):
    
    ###  Still need to make a correction for the last column of the fitted workspaces
    
    """Builds Workspaces with the fitted synthetic C(t), both globally and each mass individually
       Output: array with spectra no, intensities, widths and centers for J(y) for each spectra"""
       
    print ("\n", "Fitting Workspace: ", ws.name())
    
    ws_len, ws_no_spectra = ws.blocksize(), ws.getNumberHistograms()
   
    intensities = np.empty((len(masses), ws_no_spectra))  #intensities of J(y)
    widths = np.empty((len(masses), ws_no_spectra))       #widths of J(y)
    positions = np.empty((len(masses), ws_no_spectra))    #center positions of J(y) 
   
    intensities[:,:], widths[:,:], positions[:,:] = np.nan, np.nan, np.nan #this line puts all the values to nan, good for testing just a few spectrums
    
    #Create Workspaces to pass calculated values of ncp
    CloneWorkspace(InputWorkspace=ws,OutputWorkspace=ws.name()+'_tof_fitted_profiles')
    for m in range(len(masses)):
        CloneWorkspace(InputWorkspace=ws,OutputWorkspace=ws.name()+ '_tof_fitted_profile_'+str(m+1))
    
    print ("\n Fitting parameters are given as: [Intensity Width Centre] for each NCP")
    
    spectra = range(first_spectrum, last_spectrum + 1)
    for spec_idx, spectrum in enumerate(spectra):
        
        data_x, data_y, data_e = load_workspace(ws, spectrum)      #load spectrum line into data arrays
        
        if np.all(data_y == 0):      #if all values in line are zero  
            print (spectrum, " ... skipping ...")
            
            ncp_indiv_m = np.zeros((len(masses), len(data_x)))  #skip fitting, all values are zero
            ncp = np.zeros(len(data_x))
            fitted_par = np.empty(len(par))
            fitted_par[:] = np.nan
            reduced_chi2 = np.nan
             
        else:
            ncp_indiv_m, ncp, fitted_par, result = fit_ncp(par, spectrum, masses, data_x, data_y, data_e, fit_arguments)
            reduced_chi2 = result["fun"]/(len(data_x) - len(par))       
          
        #write ncp data into the workspaces
        tmp = mtd[ws.name()+'_tof_fitted_profiles']           
        tmp.dataY(spec_idx)[:-1] = ncp               #ncp is calculated from data_x which is smaller than ws_len by one
        tmp.dataY(spec_idx)[-1] = np.nan             #last vale of data_y, workspaces are only used for plotting
        tmp.dataE(spec_idx)[:] = np.zeros(ws_len)

        npars = int(len(par)/len(masses))              
        for m in range(len(masses)):
            tmp = mtd[ws.name()+'_tof_fitted_profile_'+str(m+1)]   
            tmp.dataY(spec_idx)[:-1] = ncp_indiv_m[m]
            tmp.dataY(spec_idx)[-1] = np.nan
            tmp.dataE(spec_idx)[:] = np.zeros(ws_len)
            
            intensities[m, spec_idx] = float(fitted_par[npars*m])
            widths[m, spec_idx] = float(fitted_par[npars*m+1])
            positions[m, spec_idx] = float(fitted_par[npars*m+2])
            
        print (spectrum, fitted_par, "%.4g" % reduced_chi2)   
    #print(intensities[:,:5], widths[:,:5], positions[:,:5])
    return spectra, intensities, widths, positions


def fit_ncp(par, spectrum, masses, data_x, data_y, data_e, fit_arguments):   
    """Fits the synthetic C(t) to data_y in TOF space"""
    
    boundaries, constraints = fit_arguments[0], fit_arguments[1]  #boundaries set the bounds for the parameters, constraints sets equalities between parameters
    result = optimize.minimize(errfunc, par[:], args=(spectrum, masses, data_x, data_y, data_e), method='SLSQP', bounds = boundaries, constraints=constraints)
    
    fitted_par =  result["x"]    #previously: result.values()[5]
    ncp_indiv_m, ncp = calculate_ncp(fitted_par, spectrum , masses, data_x)
    return ncp_indiv_m, ncp, fitted_par, result

def errfunc(par, spectrum, masses, data_x,  data_y, data_e):
    """Function to be minimized, operates in TOF space"""
    
    # this function provides the scalar to be minimised, with meaning of the non-reduced chi2
    ncp_indiv_m, ncp = calculate_ncp(par, spectrum , masses, data_x)
    
    if (np.sum(data_e) > 0):
        chi2 =  ((ncp - data_y)**2)/(data_e)**2 # Chi square per degree of freedom  #What? Over the error?
    else:
        chi2 =  (ncp - data_y)**2
    return chi2.sum()

def calculate_ncp(par, spectrum , masses, data_x):
    
    """Creates a synthetic C(t) to be fitted to TOF values, from J(y) and resolution functions"""
    
    angle, T0, L0, L1 = load_ip_file(spectrum)
    mN, Ef, en_to_vel, vf, hbar = load_constants()
    v0, E0, delta_E, delta_Q = calculate_kinematics(data_x, angle, T0, L0, L1 )    # velocities in m/us, times in us, energies in meV

    #ncp = 0. # initialising the function values
    ncp_indiv_m = np.zeros((len(masses), len(data_x)))   
    npars = int(len(par)/len(masses))
    for m in range(len(masses)):      
        #width and centre are parameters of J(y)
        mass, hei , width, centre = masses[m] , par[m*npars], par[1+m*npars], par[2+m*npars]
        
        E_r = ( hbar * delta_Q )**2 / 2. / mass          #Energy of recoil
        y = mass / hbar**2 /delta_Q * (delta_E - E_r)    #y-scaling
        #print("y:\n", y[:10])
         
#         joy = fun_gaussian(y-centre, 1.)
#         pcb = np.where(joy == joy.max())     # this finds the peak-centre bin (pcb)
        pcb = np.argmin(np.abs(y-centre))      #finds bin closest to the center of ncp ie at the peak
        
        gaussian_res_width, lorentzian_res_width = calculate_resolution(spectrum, data_x[pcb], mass)
        #print("res:\n gauss:", gaussian_res_width, "lorz: ", lorentzian_res_width)
        # definition of the experimental neutron compton profile
        # Combine the errors from Gaussians in quadrature to obtain error on J(y)
        gaussian_width = np.sqrt( width**2 + gaussian_res_width**2 )
        
        #Convolution between J(Y) and R(t)
        #Technique used because Lorentzian std can not be added in quadrature
        joy = fun_pseudo_voigt(y-centre, gaussian_width, lorentzian_res_width)
        
        #Finite State events correction
        FSE =  - fun_derivative3(y,joy)*width**4/delta_Q * 0.72 # 0.72 is an empirical coefficient. One can alternatively add a fitting parameter for this term.
        #H4  = some_missing_coefficient *  fun_derivative4(y,joy) /(4.*width**4) /32.
        
        #synthetic C(t)
        ncp_indiv_m[m] = hei * (joy + FSE ) * E0 * E0**(-0.92) * mass / delta_Q # Here -0.92 is a parameter describing the epithermal flux tail.
    
    ncp = np.sum(ncp_indiv_m, axis=0)
    return ncp_indiv_m, ncp

def calculate_kinematics(data_x, angle, T0, L0, L1 ):   
    """kinematics with conservation of momentum and energy""" 
    
    mN, Ef, en_to_vel, vf, hbar = load_constants()
    t_us = data_x - T0                  #T0 is electronic delay due to instruments
    v0 = vf * L0 / ( vf * t_us - L1 )
    E0 =  ( v0 / en_to_vel )**2         #en_to_vel is a factor used to easily change velocity to energy and vice-versa
    
    delta_E = E0 -Ef  
    delta_Q2 = 2. * mN / hbar**2 * ( E0 + Ef - 2. * np.sqrt( E0*Ef) * np.cos(angle/180.*np.pi) )
    delta_Q = np.sqrt( delta_Q2 )
    return v0, E0, delta_E, delta_Q

def calculate_resolution(spectrum, data_x, mass): 
    
    """Calculates the resolution widths in y-space, from the individual resolutions in the following parameters:
       Gaussian dist (std): L0, theta, L1, TOF and E1
       Lorentzian dist (HWHM): E1
       input: spectrum, TOF of the peak of J(y), mass of element
       output: gaussian width and lorenztian width to be propagated through J(y)"""
       
    angle, T0, L0, L1 = load_ip_file(spectrum)
    mN, Ef, en_to_vel, vf, hbar = load_constants()
    v0, E0, delta_E, delta_Q = calculate_kinematics(data_x, angle, T0, L0, L1 )
    dE1, dTOF, dTheta, dL0, dL1, dE1_lor = load_resolution_parameters(spectrum)   #load resolution of indivisual parameters
    
    # Calculate dw^2 and dq^2 [meV] for Gaussian, derivatives included  during unit conversion
    dW2 = (1. + (E0 / Ef)**1.5 * ( L1 / L0 ) )**2 * dE1**2 + (2. * E0 * v0 / L0 )**2 * dTOF**2   \
           + ( 2. * E0**1.5 / Ef**0.5 / L0 )**2 * dL1**2 + ( 2. * E0 / L0 )**2 * dL0**2
    
    dQ2 =  (1. - (E0 / Ef )**1.5 *L1 / L0 - np.cos(angle/180.*np.pi) * ( ( E0 / Ef )**0.5 - L1 / L0 * E0 / Ef ))**2 * dE1**2    \
           + ( ( 2. * E0 * v0 / L0 )**2 * dTOF**2 + (2. * E0**1.5 / L0 / Ef**0.5 )**2 *dL1**2 + ( 2. * E0 / L0 )**2 * dL0**2 ) * np.abs( Ef / E0 * np.cos(angle/180.*np.pi) -1.)   \
           + ( 2. * np.sqrt( E0 * Ef )* np.sin(angle/180.*np.pi) )**2 * dTheta**2

    dW2 *= ( mass / hbar**2 /delta_Q )**2              # conversion from meV^2 to A^-2, dydW = (M/q)^2
    dQ2 *= ( mN / hbar**2 /delta_Q )**2
    
    gaussian_res_width =   np.sqrt( dW2 + dQ2 ) # in A-1    #same as dy^2 = (dy/dw)^2*dw^2 + (dy/dq)^2*dq^2
    
    #Same procedure for lorentzian component in meV
    dWdE1_lor = (1. + (E0 / Ef)**1.5 * ( L1 / L0 ) )**2         # is it - or +?
    dQdE1_lor =  (1. - (E0 / Ef )**1.5 *L1 / L0 - np.cos(angle/180.*np.pi) * ( ( E0 / Ef )**0.5 + L1 / L0 * E0 / Ef )) **2

    dWdE1_lor *= ( mass / hbar**2 /delta_Q )**2      # conversion from meV^2 to A^-2
    dQdE1_lor *= ( mN / hbar**2 /delta_Q )**2

    lorentzian_res_width = np.sqrt( dWdE1_lor + dQdE1_lor ) * dE1_lor   # in A-1     #same as dy^2 = (dw/dE1)^2*dE1^2 + (dq/dE1)^2*dE1^2
    return gaussian_res_width, lorentzian_res_width # gaussian std dev, lorentzian hwhm 
    
#
#       CORRECTION FUNCTIONS
#
def calculate_mean_widths_and_intensities(masses,widths,intensities,spectra, verbose): #spectra and verbose not used 
    """calculates the mean widths and intensities of the Compton profile J(y) for each mass """ 
    
    #widths, intensities = np.array(widths), np.array(intensities)   
    mean_widths = np.array([np.nanmean(widths, axis=1)])     #shape (1,4)
    widths_std = np.array([np.nanstd(widths, axis=1)])
    
    width_deviation = np.abs(widths - mean_widths.transpose())         #subtraction row by row 
    better_widths = np.where(width_deviation > widths_std.transpose(), np.nan, widths)   #where True, replace by nan
    better_intensities = np.where(width_deviation > widths_std.transpose(), np.nan, intensities)
    
    mean_widths = np.nanmean(better_widths, axis=1)   #shape (1,4)
    widths_std = np.nanstd(better_widths, axis=1)

    normalization_sum = np.sum(better_intensities, axis=0)        #not nansum(), to propagate nan
    better_intensities = better_intensities / normalization_sum
    
    mean_intensity_ratios = np.nanmean(better_intensities, axis=1)  
    mean_intensity_ratios_std = np.nanstd(better_intensities, axis=1)   
#     print("\n intensities after norm: \n", better_intensities[:, :5])
#     print("\n widths: \n", better_widths[:, :5])    
#     print("\n intensity ratios and std: \n", mean_intensity_ratios, mean_intensity_ratios_std)
#     print("\n mean widths and std: \n", mean_widths, widths_std)   
    for m in range(len(masses)):
        print ("\n", "Mass: ", masses[m], " width: ", mean_widths[m], " \pm ", widths_std[m])
        print ("\n", "Mass: ", masses[m], " mean_intensity_ratio: ", mean_intensity_ratios[m], " \pm ", mean_intensity_ratios_std[m])
    return mean_widths, mean_intensity_ratios


def calculate_sample_properties(masses,mean_widths,mean_intensity_ratios, mode, verbose):
    """returns the one of the inputs necessary for the VesuvioCalculateGammaBackground
    or VesuvioCalculateMS"""
    
    if mode == "GammaBackground":
        profiles = ""
        for m in range(len(masses)):
            mass, width, intensity=str(masses[m]), str(mean_widths[m]),str(mean_intensity_ratios[m])
            profiles += "name=GaussianComptonProfile,Mass="+mass+",Width="+width+",Intensity="+intensity+';' 
        sample_properties = profiles
        
    elif mode == "MultipleScattering":
        if hydrogen_peak:   #if hydrogen_peak is set to True
            # ADDITION OF THE HYDROGEN INTENSITY AS PROPORTIONAL TO A FITTED NCP (OXYGEN HERE)            
            masses = np.append(masses, 1.0079)
            mean_widths = np.append(mean_widths, 5.0)           
            mean_intensity_ratios = np.append(mean_intensity_ratios, hydrogen_to_mass0_ratio * mean_intensity_ratios[0])
            mean_intensity_ratios /= np.sum(mean_intensity_ratios)
            
        MS_properties = np.zeros(3*len(masses))
        MS_properties[::3] = masses
        MS_properties[1::3] = mean_intensity_ratios
        MS_properties[2::3] = mean_widths                    
        sample_properties = list(MS_properties)    
    else:
        print("\n Mode entered not valid")
    if verbose:
        print ("\n The sample properties for ", mode, " are: ", sample_properties)
    return sample_properties
        
# def correct_for_gamma_background(ws_name):  #not used for backscattering
#     if verbose:
#         print ("Evaluating the Gamma Background Correction.")
#     # Create an empty workspace for the gamma correction
#     CloneWorkspace(InputWorkspace=ws_name,OutputWorkspace="gamma_background_correction")
#     ws=mtd["gamma_background_correction"]
#     for spec in range(ws.getNumberHistograms()):
#         profiles=''
#         for m in range(masses.__len__()):
#             mass,width,intensity=str(masses[m]), str(mean_widths[m]),str(mean_intensity_ratios[m])
#             profiles+= "name=GaussianComptonProfile,Mass="+mass+",Width="+width+",Intensity="+intensity+';'
#         background, corrected = VesuvioCalculateGammaBackground(InputWorkspace=ws_name, 
#                                                                         ComptonFunction=profiles, WorkspaceIndexList=spec)
#         for bin in range(ws.blocksize()):
#             ws.dataY(spec)[bin],ws.dataE(spec)[bin]=background.dataY(0)[bin],background.dataE(0)[bin]
#     RenameWorkspace(InputWorkspace= "gamma_background_correction", OutputWorkspace = str(ws_name)+"_gamma_background")
#     DeleteWorkspace("background")
#     DeleteWorkspace("corrected")
#     return

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

def correct_for_multiple_scattering(ws_name,first_spectrum,last_spectrum, sample_properties, transmission_guess, multiple_scattering_order, number_of_events):
    
    MS_masses = sample_properties[::3]     #selects only the masses, every 3 numbers
    MS_amplitudes = sample_properties[1::3]   #same as above, but starts at first intensity
    if verbose:
        print ("Evaluating the Multiple Scattering Correction.")
        
    dens, trans = VesuvioThickness(Masses=MS_masses, Amplitudes=MS_amplitudes, TransmissionGuess=transmission_guess,Thickness=0.1)         
    _TotScattering, _MulScattering = VesuvioCalculateMS(ws_name, NoOfMasses=len(MS_masses), SampleDensity=dens.cell(9,1), AtomicProperties=sample_properties, \
                                                        BeamRadius=2.5, NumScatters=multiple_scattering_order, NumEventsPerRun=int(number_of_events))
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
    #the only remaining workspaces are the _MulScattering and _TotScattering
    return
        
############################
### functions to fit the NCP in the y space
############################

def subtract_other_masses(ws_last_iteration, intensities, widths, positions, spectra, masses):
    #haven't tested this one yet
    #what does this do? subtracts NCP of masses from each other?
    first_ws = CloneWorkspace(InputWorkspace=ws_last_iteration)
    for index in range(len(spectra)):
        data_x, data_y, data_e = load_workspace(first_ws , spectra[index])
        if np.all(data_y==0):    
            first_ws.dataY(index)[:] = 0   # assigning it to zero if its already zero?
#             for bin in range(len(data_x)-1):
#                 first_ws.dataY(index)[bin] = 0
        else:
            for m in range(len(masses)-1):
                other_par = (intensities[m+1, index],widths[m+1, index],positions[m+1, index])
                ncp = calculate_ncp(other_par, spectra[index], [masses[m+1]], data_x)
                first_ws.dataY(index)[:-1] -= ncp*(data_x[1:]-data_x[:-1])
#                 for bin in range(len(data_x)-1):
#                     first_ws.dataY(index)[bin] -= ncp[bin]*(data_x[bin+1]-data_x[bin])
    return first_ws

def convert_to_y_space_and_symmetrise(ws_name,mass):  
    """input: TOF workspace
       output: workspace in y-space for given mass with dataY symetrised"""
          
    ws_y, ws_q = ConvertToYSpace(InputWorkspace=ws_name,Mass=mass,OutputWorkspace=ws_name+"_JoY",QWorkspace=ws_name+"_Q")
    max_Y = np.ceil(2.5*mass+27)    #where from
    rebin_parameters = str(-max_Y)+","+str(2.*max_Y/120)+","+str(max_Y)   #first bin boundary, width, last bin boundary, so 120 bins over range
    ws_y = Rebin(InputWorkspace=ws_y, Params = rebin_parameters, FullBinsOnly=True, OutputWorkspace=ws_name+"_JoY")
   
    matrix_Y = np.zeros((ws_y.getNumberHistograms(), ws_y.blocksize()))
    for spec_idx in range(len(matrix_Y)):                 #pass the y-data onto an array to easily manipulate
        matrix_Y[spec_idx, :] = ws_y.readY(spec_idx)        
    matrix_Y[matrix_Y != 0] = 1
    sum_Y = np.nansum(matrix_Y, axis=0)   
    
    ws_y = SumSpectra(InputWorkspace=ws_y, OutputWorkspace=ws_name+"_JoY")
    tmp=CloneWorkspace(InputWorkspace=ws_y)
    tmp.dataY(0)[:] = sum_Y
    tmp.dataE(0)[:] = np.zeros(tmp.blocksize())
    
    ws = Divide(LHSWorkspace=ws_y, RHSWorkspace=tmp, OutputWorkspace =ws_name+"_JoY")
    ws.dataY(0)[:] = (ws.readY(0)[:] + np.flip(ws.readY(0)[:])) / 2           #symetrise dataY
    ws.dataE(0)[:] = (ws.readE(0)[:] + np.flip(ws.readE(0)[:])) / 2           #sumetrise dataE
    normalise_workspace(ws)
    return max_Y 

def calculate_mantid_resolutions(ws_name, mass):
    #uses a for loop because the fuction VesuvioResolution takes in one spectra at a time
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

verbose=True                                  # If True, prints the value of the fitting parameters for each time-of-flight spectrum
plot_iterations = True                        # If True, plots all the time-of-flight spectra and fits in a single window for each iteration
number_of_iterations = 1     #4               # This is the number of iterations for the reduction analysis in time-of-flight.
name='CePtGe12_100K_DD_'  
runs='44462-44463'         # 100K             # The numbers of the runs to be analysed
empty_runs='43868-43911'   # 100K             # The numbers of the empty runs to be subtracted
spectra='3-134'                               # Spectra to be analysed
first_spectrum,last_spectrum = 3, 134
tof_binning='275.,1.,420'                             # Binning of ToF spectra
mode='DoubleDifference'
ipfile='ip2018.par'                                   # Optional instrument parameter file
detectors_masked=[18,34,42,43,59,60,62,118,119,133]   # Optional spectra to be masked

#------------include case where workspace is cropped
detectors_masked = np.array(detectors_masked)
detectors_masked = detectors_masked[detectors_masked < last_spectrum]
#-------------------------------------

########since I do not have the path for Vesuvio, load files from my path
if (name+'raw' not in mtd):
    print ('\n', 'Loading the sample runs: ', runs, '\n')
    Load(Filename= r"C:/Users/guijo/Desktop/Work/CePtGe12_backward_100K_scipy/CePtGe12_100K_DD_raw.nxs", OutputWorkspace="CePtGe12_100K_DD_raw")
    #LoadVesuvio(Filename=runs, SpectrumList=spectra, Mode=mode, InstrumentParFile=ipfile,OutputWorkspace=name+'raw')    
    Rebin(InputWorkspace=name+'raw',Params=tof_binning,OutputWorkspace=name+'raw') 
    SumSpectra(InputWorkspace=name+'raw', OutputWorkspace=name+'raw'+'_sum')

if (name+'empty' not in mtd):
    Load(Filename= r"C:/Users/guijo/Desktop/Work/CePtGe12_backward_100K_scipy/CePtGe12_100K_DD_empty.nxs", OutputWorkspace="CePtGe12_100K_DD_empty")
    print ('\n', 'Loading the empty runs: ', empty_runs, '\n')
    #LoadVesuvio(Filename=empty_runs, SpectrumList=spectra, Mode=mode, InstrumentParFile=ipfile,OutputWorkspace=name+'empty') 
    Rebin(InputWorkspace=name+'empty',Params=tof_binning,OutputWorkspace=name+'empty')     
    Minus(LHSWorkspace=name+'raw', RHSWorkspace=name+'empty', OutputWorkspace=name)
    
# Parameters for the multiple-scattering correction, including the shape of the sample.
transmission_guess = 0.98                               #experimental value from VesuvioTransmission
multiple_scattering_order, number_of_events = 2, 1.e5
hydrogen_peak=False                                     # hydrogen multiple scattering
hydrogen_to_mass0_ratio = 0                             # hydrogen-to-mass[0] ratio obtaiend from the preliminary fit of forward scattering  0.77/0.02 =38.5
vertical_width, horizontal_width, thickness = 0.1, 0.1, 0.001 # expressed in meters
create_slab_geometry(name,vertical_width, horizontal_width, thickness)

masses=[140.1,195.1,72.6,27]             # Array of masses to be fitted

abs_cross_sections = []                  # This should be a vector of absorprion-to-scattering cross sections for each mass.

simple_gaussian_fit = True

#Parameters as: Intensity, NCP width, NCP centre A-1
## Cerium
#Classical value of Standard deviation of the momentum distribution:  16.9966  inv A
# Debye value of Standard deviation of the momentum distribution:   18.22 inv A 
par     = ( 1,           18.22,       0.       )
bounds  = ((0, None),   (17,20),    (-30., 30.))    

## Platinum
#Classical value of Standard deviation of the momentum distribution:  20.0573  inv A
# Debye value of Standard deviation of the momentum distribution:   22.5 inv A 
par    += ( 1,           22.5,        0.       )
bounds += ((0, None),   (20,25),    (-30., 30.))

## Germanium
#Classical value of Standard deviation of the momentum distribution:  12.2352  inv A
# Debye value of Standard deviation of the momentum distribution:   15.4 inv A 
par    += ( 1,           15.4,        0.       )
bounds += ((0, None),   (12.2,18),  (-10., 10.))

## Aluminium
#Classical value of Standard deviation of the momentum distribution:  7.4615  inv A
# Debye value of Standard deviation of the momentum distribution:  9.93 inv A
par    += ( 1,           9.93,        0.       )
bounds += ((0, None),   (9.8,10),   (-10., 10.))

# Intensity constraints 
			
# CePt4Ge12 in Al can
#  Ce cross section * stoichiometry = 2.94*1 = 2.94	 barn  
#  Pt cross section * stoichiometry = 11.71*4 = 46.84	 barn  
#  Ge cross section * stoichiometry = 8.6*12 = 103.2	 barn


constraints =  ({'type': 'eq', 'fun': lambda par:  par[0] -2.94/46.84*par[3] },{'type': 'eq', 'fun': lambda par:  par[0] -2.94/103.2*par[6] })
fit_arguments = [bounds, constraints]

#--------------crop original workspace for runs of only a few spectrums
spec_offset = mtd[name].getSpectrum(0).getSpectrumNo()  
first_idx, last_idx = first_spectrum - spec_offset, last_spectrum - spec_offset
CropWorkspace(InputWorkspace=name, StartWorkspaceIndex = first_idx, EndWorkspaceIndex = last_idx, OutputWorkspace=name)
#---------------------------------------------------

# Iterative analysis and correction of time-of-flight spectra.
# This is done so that each iteration corrects for Multiple Scattering 
for iteration in range(number_of_iterations):
    if iteration == 0:
        ws_to_be_fitted = CloneWorkspace(InputWorkspace = name, OutputWorkspace = name+str(iteration))
    ws_to_be_fitted = mtd[name+str(iteration)]
    MaskDetectors(Workspace=ws_to_be_fitted,SpectraList=detectors_masked)

    #Obtain the best fit parameters for all spectrums
    spectra, intensities, widths, positions = block_fit_ncp(par,first_spectrum,last_spectrum, masses,ws_to_be_fitted, fit_arguments, verbose)
    print("\n intensities: \n", intensities[:,:5], "\n widths \n", widths[:,:5], "\n positions \n", positions[:,:5])
    # Calculate mean widths and intensities per mass
    mean_widths, mean_intensity_ratios = calculate_mean_widths_and_intensities(masses, widths, intensities, spectra, verbose) # at present is not multiplying for 0,9

    if (number_of_iterations - iteration -1 > 0):   #ie if not at the last iteration
        # evaluate multiple scattering correction --------- This creates a background workspace with name :  str(ws_name)+"_MulScattering"
        sample_properties = calculate_sample_properties(masses, mean_widths, mean_intensity_ratios, "MultipleScattering", verbose)
        correct_for_multiple_scattering(name, first_spectrum,last_spectrum, sample_properties, transmission_guess, multiple_scattering_order, number_of_events)
        # Create corrected workspace to be used in the subsquent iteration
        Minus(LHSWorkspace= name, RHSWorkspace = str(name)+"_MulScattering", OutputWorkspace = name+str(iteration+1))
        
############################################################################
######
######              AND THEY ALL LIVED HAPPILY EVER AFTER
######
############################################################################


import numpy as np
import matplotlib.pyplot as plt
import mantid                          
from mantid.simpleapi import *
from scipy import optimize
import time

start_time=time.time()  #to show the computational time 

# command for formatting of the printed output
np.set_printoptions(suppress=True, precision=4, linewidth= 150 )

#-------------------------------------------------------Mathematical functions---------------------------------------------------------------

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
    
    k6 = ( - fun[:, 12:  ] + fun[:,  :-12] ) * 1
    k5 = ( + fun[:, 11:-1] - fun[:, 1:-11] ) * 24
    k4 = ( - fun[:, 10:-2] + fun[:, 2:-10] ) * 192
    k3 = ( + fun[:,  9:-3] - fun[:, 3:-9 ] ) * 488
    k2 = ( + fun[:,  8:-4] - fun[:, 4:-8 ] ) * 387
    k1 = ( - fun[:,  7:-5] + fun[:, 5:-7 ] ) * 1584
    
    dev = k1 + k2 + k3 + k4 + k5 + k6
    dev /= np.power( x[:, 7:-5] - x[:, 6:-6], 3)
    dev /= 12**3
    
    derivative = np.zeros(fun.shape)
    derivative[:, 6:-6] = dev                   #need to pad with zeros left and right to return array with same shape
    return derivative
  

#------------------------------------------Creating Matrix Arrays Outside Fitting--------------------------------------------

def load_workspace_into_array(ws):   
    """Input: Workspace to extract data from
       Output: dataY, dataX and dataE as arrays and converted to point data"""

    dataY = ws.extractY()
    dataE = ws.extractE()
    dataX = ws.extractX()
    
    hist_widths = dataX[:, 1:] - dataX[:, :-1]
    dataY = dataY[:, :-1] / hist_widths
    dataE = dataE[:, :-1] / hist_widths
    dataX = (dataX[:, 1:] + dataX[:, :-1]) / 2
    
    return dataY, dataX, dataE   

def load_constants():
    """Output: the mass of the neutron, final energy of neutrons (selected by gold foil),
    factor to change energies into velocities, final velocity of neutron and hbar"""
    
    mN=1.008    #a.m.u.
    Ef=4906.         # meV
    en_to_vel = 4.3737 * 1.e-4
    vf=np.sqrt( Ef ) * en_to_vel        #m/us
    hbar = 2.0445
    return mN, Ef, en_to_vel, vf, hbar 

def load_ip_file_into_array(ip_path, first_spec, last_spec):  #there is got to be a more elegant way of doing this
    """Loads instrument parameters into array, from the file in the specified path
       ATTENTION: path needs to be a raw string"""
    
    data = np.loadtxt(ip_path, dtype=str)[1:].astype(float)
    #This array is for all 196 spectrums, need to slice it to be the same length as dataX
    spectra = data[:, 0]
    select_rows = np.where((spectra >= first_spec) & (spectra <= last_spec))
    data_ip = data[select_rows]
    print("Data_ip first column: \n", data_ip[:,0])             #spectrums selected    
    return data_ip

def calculate_kinematics_arrays(dataX, data_ip):          
    """kinematics quantities calculated from TOF data
       Input: dataX matrix array
       Output: kinematics arrays whith same shape as dataX"""
    
    mN, Ef, en_to_vel, vf, hbar = load_constants()    
    det, plick, angle, T0, L0, L1 = np.hsplit(data_ip, 6)     #each is of len(dataX)
    t_us = dataX - T0                                         #T0 is electronic delay due to instruments
    v0 = vf * L0 / ( vf * t_us - L1 )
    E0 =  np.square( v0 / en_to_vel )            #en_to_vel is a factor used to easily change velocity to energy and vice-versa
    
    delta_E = E0 - Ef  
    delta_Q2 = 2. * mN / hbar**2 * ( E0 + Ef - 2. * np.sqrt(E0*Ef) * np.cos(angle/180.*np.pi) )
    delta_Q = np.sqrt( delta_Q2 )
    return v0, E0, delta_E, delta_Q              #shape(no_spectrums, len_spec)

def convert_to_y_space(dataX, masses, delta_Q, delta_E):    #inputs have shape (no_specs, len_spec)
    """Input: TOF matrix array, masses array, momentum and energy transfer array
       Output: y space matrix array for all masses, with shape (len(masses), dataX.shape)"""
    
    dataX, delta_Q, delta_E = dataX[np.newaxis, :, :],  delta_Q[np.newaxis, :, :], delta_E[np.newaxis, :, :]   #prepare arrays to broadcast
    mN, Ef, en_to_vel, vf, hbar = load_constants()
    
    E_r = np.square( hbar * delta_Q ) / 2. / masses               #Energy of recoil
    all_y_spaces = masses / hbar**2 /delta_Q * (delta_E - E_r)    #y-scaling  
    
    reshaped_yspaces = reshape_yspace(all_y_spaces)    #reshape from (4, 134, 144) to (134, 4, 144)
    return reshaped_yspaces

def reshape_yspace(A):
    """Exchanges the first two indices of A, used to rearrange yspace for map in main fitting procedure"""
    return np.stack(np.split(A, len(A), axis=0), axis=2)[0]

def load_resolution_parameters(data_ip):   
    """Resolution of parameters to propagate into TOF resolution
       Output: matrix with each parameter in each column"""
    
    spectrums = data_ip[:, 0] 
    L = len(spectrums)
    #for spec no below 135, back scattering detectors, in case of double difference
    #for spec no 135 or above, front scattering detectors, in case of single difference
    dE1 = np.where(spectrums < 135, 88.7, 73)   #meV, STD
    dE1_lorz = np.where(spectrums < 135, 40.3, 24)  #meV, HFHM
    dTOF = np.repeat(0.37, L)      #us
    dTheta = np.repeat(0.016, L)   #rad
    dL0 = np.repeat(0.021, L)      #meters
    dL1 = np.repeat(0.023, L)      #meters
    
    res_pars = np.vstack((dE1, dTOF, dTheta, dL0, dL1, dE1_lorz)).transpose()  #store all parameters in a matrix
    return res_pars      
                                               
#--------------------------------------------------------Fitting procedure----------------------------------------------------------

def fit_ncp(datax, datay, datae, yspace, res_pars, data_ip, v0, E0, delta_e, delta_q):
    """Fits the NCP and returns the best fit parameters for each row of data"""
    
    if np.all(datay == 0):           #if all zeros, then parameters are all nan, so they are ignored later down the line
        return np.full(len(par)+2, np.nan)
    
    result = optimize.minimize(err_func, par[:], args=(masses, datax, datay, datae, yspace, res_pars, data_ip, v0, E0, delta_e, delta_q), \
                               method='SLSQP', bounds = bounds, constraints=constraints) 
    return np.append(result["x"], [result["fun"]/(len(datax) - len(par)), result["nit"]])   #fitted parameters + chi2 + no iterations  

def err_func(par, masses, datax, datay, datae, yspace, res_pars, data_ip, v0, E0, delta_e, delta_q):
    """Error function to be minimized, operates in TOF space"""
    
    ncp_all_m, ncp = calculate_ncp(par, masses, datax, yspace, res_pars, data_ip, v0, E0, delta_e, delta_q)     
    if (np.sum(datae) > 0):
        chi2 =  ((ncp - datay)**2)/(datae)**2    #weighted fit
    else:
        chi2 =  (ncp - datay)**2
    return np.sum(chi2)

def calculate_ncp(par, masses, datax, yspace, res_pars, data_ip, v0, E0, delta_e, delta_q):    #yspace, res have shape (4, 144) and (4,2)
    """Creates a synthetic C(t) to be fitted to TOF values, from J(y) and resolution functions
       shapes: par (1, 12), masses (4,1,1), datax (1, n), yspace (4, n), res (4, 2), delta_q (1, n), E0 (1,n)"""
    
    masses = masses.reshape(len(masses),1)    
    intensity = par[::3].reshape(masses.shape)
    width = par[1::3].reshape(masses.shape)
    position = par[2::3].reshape(masses.shape)   
    
    gauss_res, lorz_res = calculate_resolution_all_masses(res_pars, position, masses, yspace, data_ip, v0, E0, delta_e, delta_q) #shapes (4,1)
    
    tot_gauss_width = np.sqrt( width**2 + gauss_res**2 )                 #shape(4,1)  
    
    joy = fun_pseudo_voigt(yspace-position, tot_gauss_width, lorz_res)   #shape(4, 144)
    
    FSE =  - fun_derivative3(yspace,joy) * width**4 / delta_q * 0.72     #fun_derivative needs to be changed to axis=1
    
    ncp_all_m = intensity * (joy + FSE ) * E0 * E0**(-0.92) * masses / delta_q     #shape(4, 144)
    ncp = np.sum(ncp_all_m, axis=0)
    return ncp_all_m, ncp

def calculate_resolution_all_masses(res_pars, centers, masses, yspace, data_ip, v0, E0, delta_e, delta_q):    
    """Calculates the resolution widths in y-space, from the individual resolutions in the following parameters:
       Gaussian dist (std): L0, theta, L1, TOF and E1
       Lorentzian dist (HWHM): E1
       input: for each spectrum, centers and yspace have shape (4,1), the rest are row for each spectrum
       output: gaussian and lorenztian widths for each mass, shape (4,1), to be propagated through J(y)"""
    
    det, plick, angle, T0, L0, L1 = data_ip           #each is of len(dataX) 
    dE1, dTOF, dTheta, dL0, dL1, dE1_lorz = res_pars
    mN, Ef, en_to_vel, vf, hbar = load_constants()
    
    #resolution is evaluated at the peak of J(y) i.e. when y=center
    ypeaks = np.abs(yspace - centers).min(axis=1).reshape(len(yspace), 1)    #Find minimums of each row and reshape to broadcast
    ypeaks_mask = np.abs(yspace - centers)==ypeaks            #bolean matrix that selects the bins of the peaks, need to compare absolute values!
    #expand kinematics arrays to be the same shape as ypeaks_mask
    v0, E0, delta_E, delta_Q = v0*np.ones(ypeaks.shape), E0*np.ones(ypeaks.shape), delta_e*np.ones(ypeaks.shape), delta_q*np.ones(ypeaks.shape)
    v0, E0, delta_E, delta_Q = v0[ypeaks_mask], E0[ypeaks_mask], delta_E[ypeaks_mask], delta_Q[ypeaks_mask] 
    v0, E0, delta_E, delta_Q = v0.reshape(ypeaks.shape), E0.reshape(ypeaks.shape), delta_E.reshape(ypeaks.shape), delta_Q.reshape(ypeaks.shape)
    
    dW2 = (1. + (E0 / Ef)**1.5 * (L1 / L0))**2 * dE1**2 + (2.*E0*v0 / L0)**2 * dTOF**2   \
           + (2. * E0**1.5 / Ef**0.5 / L0)**2 * dL1**2 + (2. * E0 / L0)**2 * dL0**2
    dQ2 =  (1. - (E0 / Ef)**1.5 * L1/L0 - np.cos(angle/180.*np.pi) * ((E0 / Ef )**0.5 - L1/L0 * E0/Ef))**2 * dE1**2    \
           + ((2.*E0 * v0/L0 )**2 * dTOF**2 + (2.*E0**1.5 / L0 / Ef**0.5)**2 *dL1**2 + (2.*E0 / L0)**2 * dL0**2) * np.abs(Ef/E0 * np.cos(angle/180.*np.pi)-1) \
           + (2. * np.sqrt(E0 * Ef)* np.sin(angle/180.*np.pi))**2 * dTheta**2

    dW2 *= ( masses / hbar**2 /delta_Q )**2              # conversion from meV^2 to A^-2, dydW = (M/q)^2
    dQ2 *= ( mN / hbar**2 /delta_Q )**2
    gaussian_res_width =   np.sqrt( dW2 + dQ2 ) # in A-1    #same as dy^2 = (dy/dw)^2*dw^2 + (dy/dq)^2*dq^2

    #Same procedure for lorentzian component in meV
    dWdE1_lor = (1. + (E0/Ef)**1.5 * (L1/L0))**2         # is it - or +?
    dQdE1_lor =  (1. - (E0/Ef)**1.5 * L1/L0 - np.cos(angle/180.*np.pi) * ((E0/Ef)**0.5 + L1/L0 * E0/Ef)) **2

    dWdE1_lor *= ( masses / hbar**2 /delta_Q )**2      # conversion from meV^2 to A^-2
    dQdE1_lor *= ( mN / hbar**2 /delta_Q )**2
    lorentzian_res_width = np.sqrt( dWdE1_lor + dQdE1_lor ) * dE1_lorz   # in A-1     #same as dy^2 = (dw/dE1)^2*dE1^2 + (dq/dE1)^2*dE1^2

    return gaussian_res_width, lorentzian_res_width  #shape (4,1) for each
    
#-----------------------------------Extract data from best fit params and create matrices and workspaces----------------------------

def build_ncp_matrices(par, datax, yspace, res_pars, data_ip, v0, E0, delta_e, delta_q):
    """input: all row shape
       output: row shape with the ncp for each mass"""
    
    if np.all(np.isnan(par)):
        return np.full(yspace.shape, np.nan)
    
    ncp_m, ncp = calculate_ncp(par, masses, datax, yspace, res_pars, data_ip, v0, E0, delta_e, delta_q)        
    return ncp_m

def create_ncp_workspaces(ncp_all_m_reshaped, ws):
    """Transforms the data straight from the map and creates matrices of the fitted ncp and respective workspaces"""
    
    ncp_total = np.sum(ncp_all_m_reshaped, axis=1)     #shape(no of spec, len of spec)
    ncp_all_m = reshape_yspace(ncp_all_m_reshaped)     #same operation as we did for y spaces ie exchange of first two indices
    dataX = ws.extractX()                              
    
    hist_widths = dataX[:, 1:] - dataX[:, :-1]
    dataX = dataX[:, :-1]                              #cut last column to match ncp length
    dataY = ncp_total * hist_widths

    
    #spec_nos = str(range(first_spec, last_spec+1))    #correct numbers of spectrums, currently not working
    CreateWorkspace(DataX=dataX.flatten(), DataY=dataY.flatten(), Nspec=len(dataX), OutputWorkspace=ws.name()+"_tof_fitted_profiles")
    for i, ncp_m in enumerate(ncp_all_m):
        CreateWorkspace(DataX=dataX.flatten(), DataY=ncp_m.flatten(), Nspec=len(dataX),\
                        OutputWorkspace=ws.name()+"_tof_fitted_profile_"+str(i+1))
    return ncp_all_m, ncp_total
        

def calculate_mean_widths_and_intensities(widths, intensities): #spectra and verbose not used 
    """calculates the mean widths and intensities of the Compton profile J(y) for each mass """ 
    
    #widths, intensities = np.array(widths), np.array(intensities)   
    mean_widths = np.nanmean(widths, axis=1).reshape(4,1)    #shape (4,1) for broadcasting
    widths_std = np.nanstd(widths, axis=1).reshape(4,1)
    
    width_deviation = np.abs(widths - mean_widths)         #subtraction row by row, shape (4, n)
    better_widths = np.where(width_deviation > widths_std, np.nan, widths)   #where True, replace by nan
    better_intensities = np.where(width_deviation > widths_std, np.nan, intensities)
    
    mean_widths = np.nanmean(better_widths, axis=1)   #shape (1,4)
    widths_std = np.nanstd(better_widths, axis=1)

    normalization_sum = np.sum(better_intensities, axis=0)        #not nansum(), to propagate nan
    better_intensities /= normalization_sum
    
    mean_intensity_ratios = np.nanmean(better_intensities, axis=1)  
    mean_intensity_ratios_std = np.nanstd(better_intensities, axis=1)   

    print("\nMasses: ", masses.reshape(1,4)[:], "\nMean Widths: ", mean_widths[:], "\nMean Intensity Ratios: ", mean_intensity_ratios[:])
    return mean_widths, mean_intensity_ratios

#------------------------------------------------Main procedure to fit spectrums---------------------------------------------

def block_fit_ncp(ws):     #Need to change main procedure
    """Runs the main procedure for the fitting of the input workspace ws"""
    ws_dataY = ws.extractY()
    
    #--------------Prepare all matrices before fitting-------------------
    dataY, dataX, dataE = load_workspace_into_array(ws)                      #shape(134, 144)
    data_ip = load_ip_file_into_array(ip_path, first_spec, last_spec)        #shape(134,-)
    v0, E0, delta_E, delta_Q = calculate_kinematics_arrays(dataX, data_ip)   #shape(134, 144)
    all_yspaces = convert_to_y_space(dataX, masses, delta_Q, delta_E)        #shape(134, 4, 144)
    res_pars = load_resolution_parameters(data_ip)                           #shape(134,-)
    
    scaling_factor = 1
    if scaleDataY:
        scaling_factor = 100 / np.sum(dataY, axis=1).reshape(len(dataY), 1)      #no justification for factor of 100, currently arbitrary
        
    dataY *= scaling_factor
    #print(np.sum(dataY, axis=1).reshape(len(dataY), 1))   #to confirm that it is normalizing to a hundred
    #--------------------------Fitting----------------------------------
    par_chi_nit = list(map(fit_ncp, dataX, dataY, dataE, all_yspaces, res_pars, data_ip, v0, E0, delta_E, delta_Q))
    par_chi_nit = np.array(par_chi_nit)    
    
    spec = data_ip[:, 0, np.newaxis]   #shape (no of specs, 1)
    spec_par_chi_nit = np.append(spec, par_chi_nit, axis=1)    #ie includes the chi2 and nit values as an additional column
    
    spec_par_chi_nit[:, 1:-2 :3] /= scaling_factor 
    print("[spec_no ------------------------best fit par-----------------------chi2 nit]:\n\n", spec_par_chi_nit)
        
    all_best_par = np.array(spec_par_chi_nit)[:, 1:-2] 

    #-----------------second map to build ncp data----------------------
    #print(v0.shape, E0.shape, delta_E.shape, delta_Q.shape)
    ncp_all_m = list(map(build_ncp_matrices, all_best_par, dataX, all_yspaces, res_pars, data_ip, v0, E0, delta_E, delta_Q))
    ncp_all_m = np.array(ncp_all_m)    #reminder that ncp_all_m comes in shape (134, 4, 144)
    
    ncp_all_m, ncp_total = create_ncp_workspaces(ncp_all_m, ws)   #ncp_all_m now has shape (4, 134, 144)
    #ncp_total /= scaling_factor

    #---------------from best fit parameters, build intensities, mass and width arrays, fitted profiles-----------------
    intensities, widths, positions = all_best_par[:, 0::3].T, all_best_par[:, 1::3].T, all_best_par[:, 2::3].T     #shape (4,134)
    mean_widths, mean_intensity_ratios = calculate_mean_widths_and_intensities(widths, intensities)
    return [mean_widths, mean_intensity_ratios, spec_par_chi_nit, ncp_total, ws_dataY]

#---------------------------------------------Correct for Multiple Scattering---------------------------------------------------

def create_slab_geometry(ws_name,vertical_width, horizontal_width, thickness):  #Don't know what it does
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

def calculate_sample_properties(masses, mean_widths, mean_intensity_ratios, mode):
    """returns the one of the inputs necessary for the VesuvioCalculateGammaBackground
    or VesuvioCalculateMS"""
    masses = masses.reshape(4)
    
    if mode == "GammaBackground":      #Not used for backscattering
        profiles = ""
        for m, mass in enumerate(masses):
            width, intensity = str(mean_widths[m]), str(mean_intensity_ratios[m])
            profiles += "name=GaussianComptonProfile,Mass=" + str(mass) + ",Width=" + width + ",Intensity=" + intensity + ';' 
        sample_properties = profiles
        
    elif mode == "MultipleScattering":
        if hydrogen_peak:   
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
    print ("\n The sample properties for ", mode, " are: ", sample_properties)
    return sample_properties

def correct_for_multiple_scattering(ws_name, sample_properties, transmission_guess, multiple_scattering_order, number_of_events):
    """Uses the Mantid algorithm for the MS correction to create two Workspaces _TotScattering and _MulScattering"""
     
    print("Evaluating the Multiple Scattering Correction.")    
    MS_masses = sample_properties[::3]        #selects only the masses, every 3 numbers
    MS_amplitudes = sample_properties[1::3]   #same as above, but starts at first intensity
    
    dens, trans = VesuvioThickness(Masses=MS_masses, Amplitudes=MS_amplitudes, TransmissionGuess=transmission_guess,Thickness=0.1)

    _TotScattering, _MulScattering = VesuvioCalculateMS(ws_name, NoOfMasses=len(MS_masses), SampleDensity=dens.cell(9,1), \
                                                                        AtomicProperties=sample_properties, BeamRadius=2.5, \
                                                                        NumScatters=multiple_scattering_order, \
                                                                        NumEventsPerRun=int(number_of_events))
    data_normalisation = Integration(ws_name) 
    simulation_normalisation = Integration("_TotScattering")
    for workspace in ("_MulScattering","_TotScattering"):
        Divide(LHSWorkspace = workspace, RHSWorkspace = simulation_normalisation, OutputWorkspace = workspace)
        Multiply(LHSWorkspace = workspace, RHSWorkspace = data_normalisation, OutputWorkspace = workspace)
        RenameWorkspace(InputWorkspace = workspace, OutputWorkspace = str(ws_name)+workspace)
    DeleteWorkspaces([data_normalisation, simulation_normalisation, trans, dens])
    return     #the only remaining workspaces are the _MulScattering and _TotScattering

#--------------------------------------------------Functions that act in y space---------------------------------------------

def subtract_other_masses(ws, ncp_all_m):   #I tested this function but not throughouly, so could have missed something
    """Input: workspace from last iteration, ncp for all masses
       Output: workspace with all the ncp subtracted except for the first mass"""

    ncp_all_m = ncp_all_m[1:, :, :]       #select all masses other than the first one
    ncp_tot = np.sum(ncp_all_m, axis=0)   #sum the ncp for remaining masses   
    dataY, dataX, dataE = ws.extractY(), ws.extractX(), ws.extractE()
    
    dataY[:, :-1] -= ncp_tot * (dataX[:, 1:] - dataX[:, :-1])    #the original uses data_x, ie the mean points of the histograms, not dataX!
    #but this makes more sense to calculate histogram widths, we can preserve one more data point 
    first_ws = CreateWorkspace(DataX=dataX.flatten(), DataY=dataY.flatten(), DataE=dataE.flatten(), Nspec=len(dataX))
    return first_ws

def convert_to_yspace_and_symetrise(ws_name, mass):  
    """input: TOF workspace
       output: workspace in y-space for given mass with dataY symetrised"""
          
    ws_y, ws_q = ConvertToYSpace(InputWorkspace=ws_name,Mass=mass,OutputWorkspace=ws_name+"_JoY",QWorkspace=ws_name+"_Q")
    max_Y = np.ceil(2.5*mass+27)    #where from
    rebin_parameters = str(-max_Y)+","+str(2.*max_Y/120)+","+str(max_Y)   #first bin boundary, width, last bin boundary, so 120 bins over range
    ws_y = Rebin(InputWorkspace=ws_y, Params = rebin_parameters, FullBinsOnly=True, OutputWorkspace=ws_name+"_JoY")
    
    matrix_Y = ws_y.extractY()        
    matrix_Y[(matrix_Y != 0) & (matrix_Y != np.nan)] = 1       #safeguarding against nans as well
    no_y = np.nansum(matrix_Y, axis=0)   
    
    ws_y = SumSpectra(InputWorkspace=ws_y, OutputWorkspace=ws_name+"_JoY")      
    tmp = CloneWorkspace(InputWorkspace=ws_y)
    tmp.dataY(0)[:] = no_y
    tmp.dataE(0)[:] = np.zeros(tmp.blocksize())
    
    ws = Divide(LHSWorkspace=ws_y, RHSWorkspace=tmp, OutputWorkspace =ws_name+"_JoY")   #use of Divide and not nanmean, err are prop automatically
    ws.dataY(0)[:] = (ws.readY(0)[:] + np.flip(ws.readY(0)[:])) / 2                     #symetrise dataY
    ws.dataE(0)[:] = (ws.readE(0)[:] + np.flip(ws.readE(0)[:])) / 2                     #symetrise dataE
    normalise_workspace(ws)
    return max_Y

def calculate_mantid_resolutions(ws_name, mass):
    #only for loop in this script because the fuction VesuvioResolution takes in one spectra at a time
    #haven't really tested this one becuase it's not modified
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
    
#---------------------------------------------------------Other functions------------------------------------------------------
def normalise_workspace(ws_name):
    tmp_norm = Integration(ws_name)
    Divide(LHSWorkspace=ws_name,RHSWorkspace="tmp_norm",OutputWorkspace=ws_name)
    DeleteWorkspace("tmp_norm")
    
def loadRawAndEmptyWsFromUserPath(userRawPath, userEmptyPath):
    """Loads the raw and empty ws from either a user specified path"""
    
    tof_binning='275.,1.,420'                             # Binning of ToF spectra
    runs='44462-44463'         # 100K             # The numbers of the runs to be analysed
    empty_runs='43868-43911'   # 100K             # The numbers of the empty runs to be subtracted

    print ('\n', 'Loading the sample runs: ', runs, '\n')
    Load(Filename = userRawPath, OutputWorkspace= name+"raw")
    Rebin(InputWorkspace=name+'raw',Params=tof_binning,OutputWorkspace=name+'raw') 
    SumSpectra(InputWorkspace=name+'raw', OutputWorkspace=name+'raw'+'_sum')
    
    print ('\n', 'Loading the empty runs: ', empty_runs, '\n')
    Load(Filename = userEmptyPath, OutputWorkspace= name+"empty")
    Rebin(InputWorkspace=name+'empty',Params=tof_binning,OutputWorkspace=name+'empty')     
    Minus(LHSWorkspace=name+'raw', RHSWorkspace=name+'empty', OutputWorkspace=name)
    
    
def LoadRawAndEmptyWsVesuvio():
    """Loads the raw and empty workspaces with Vesuvio parameters"""
    
    runs='44462-44463'         # 100K             # The numbers of the runs to be analysed
    empty_runs='43868-43911'   # 100K             # The numbers of the empty runs to be subtracted
    spectra='3-134'                               # Spectra to be analysed
    tof_binning='275.,1.,420'                             # Binning of ToF spectra
    mode='DoubleDifference'
    ipfile='ip2018.par'
    
    print ('\n', 'Loading the sample runs: ', runs, '\n')
    LoadVesuvio(Filename=runs, SpectrumList=spectra, Mode=mode, InstrumentParFile=ipfile, OutputWorkspace=name+'raw')  
    Rebin(InputWorkspace=name+'raw',Params=tof_binning,OutputWorkspace=name+'raw') 
    SumSpectra(InputWorkspace=name+'raw', OutputWorkspace=name+'raw'+'_sum')
    
    print ('\n', 'Loading the empty runs: ', empty_runs, '\n')
    LoadVesuvio(Filename=empty_runs, SpectrumList=spectra, Mode=mode, InstrumentParFile=ipfile, OutputWorkspace=name+'empty') 
    Rebin(InputWorkspace=name+'empty',Params=tof_binning,OutputWorkspace=name+'empty')     
    Minus(LHSWorkspace=name+'raw', RHSWorkspace=name+'empty', OutputWorkspace=name)

def loadSyntheticNcpWorkspace(syntheticResultsPath):
    """Loads the synthetic ncp workspace from previous fit results"""    
    
    results = np.load(syntheticResultsPath)
    dataY = results["all_tot_ncp"][0, first_idx : last_idx+1]               #Now the data to be fitted will be the ncp of first iteration
    dataX = mtd[name].extractX()[first_idx : last_idx+1, :-1]                                #cut last collumn to match ncp length 
    
    ws_to_be_fitted = CreateWorkspace(DataX=dataX.flatten(), DataY=dataY.flatten(), Nspec=len(dataX), OutputWorkspace=name+"0")
    print(dataY.shape, dataX.shape)
    return ws_to_be_fitted

def cropAndCloneMainWorkspace():
    """Crops the main workspace and returns a copy with changed name"""   
    CropWorkspace(InputWorkspace=name, StartWorkspaceIndex = first_idx, EndWorkspaceIndex = last_idx, OutputWorkspace=name)
    ws_to_be_fitted = CloneWorkspace(InputWorkspace = name, OutputWorkspace = name+"0")  
    return ws_to_be_fitted
    
def convertFirstAndLastSpecToIdx(first_spec, last_spec):
    """Used because idexes remain consistent between different workspaces, which might not be the case for spec numbers"""
    spec_offset = mtd[name].getSpectrum(0).getSpectrumNo()      #use the main ws as the reference point
    first_idx = first_spec - spec_offset
    last_idx = last_spec - spec_offset
    return first_idx, last_idx
    
class resultsObject:
    def __init__(self):
        """Initialized all zeros arrays to be used for storing fitting results"""
        
        noOfSpec = ws_to_be_fitted.getNumberHistograms()
        lenOfSpec = ws_to_be_fitted.blocksize()
        noOfMasses = len(masses)

        all_fit_workspaces = np.zeros((number_of_iterations, noOfSpec, lenOfSpec))
        all_spec_best_par_chi_nit = np.zeros((number_of_iterations, noOfSpec, noOfMasses*3+3))
        all_tot_ncp = np.zeros((number_of_iterations, noOfSpec, lenOfSpec - 1))
        all_mean_widths = np.zeros((number_of_iterations, noOfMasses))
        all_mean_intensities = np.zeros(all_mean_widths.shape)
        
        resultsList = [all_mean_widths, all_mean_intensities, all_spec_best_par_chi_nit, all_tot_ncp, all_fit_workspaces]
        self.resultsList = resultsList
        
    def append(self, mulscatIter, resultsToAppend):
        for i, resultArray in enumerate(resultsToAppend):
            self.resultsList[i][mulscatIter] = resultsToAppend[i]
    
    def save(self, savePath):
        all_mean_widths, all_mean_intensities, all_spec_best_par_chi_nit, all_tot_ncp, all_fit_workspaces = self.resultsList
        np.savez(savePath,
                 all_fit_workspaces = all_fit_workspaces,
                 all_spec_best_par_chi_nit = all_spec_best_par_chi_nit,
                 all_mean_widths = all_mean_widths, 
                 all_mean_intensities = all_mean_intensities,
                 all_tot_ncp = all_tot_ncp)
                 

######################################################################################################################################
######################################################                      ##########################################################
######################################################     USER SECTION     ##########################################################
######################################################                      ##########################################################
######################################################################################################################################

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

# #----------------------------------------------Initial fitting parameters and constraints---------------------------------------------------
# Elements                                                             Cerium     Platinum     Germanium    Aluminium
# Classical value of Standard deviation of the momentum distribution:  16.9966    20.0573      12.2352      7.4615         inv A
# Debye value of Standard deviation of the momentum distribution:      18.22      22.5         15.4         9.93           inv A 

#Parameters:   Intensity,   NCP Width,    NCP centre
par     =      ( 1,           18.22,       0.       )     #Cerium
bounds  =      ((0, None),   (17,20),    (-30., 30.))     
par    +=      ( 1,           22.5,        0.       )     #Platinum
bounds +=      ((0, None),   (20,25),    (-30., 30.))
par    +=      ( 1,           15.4,        0.       )
bounds +=      ((0, None),   (12.2,18),  (-10., 10.))     #Germanium
par    +=      ( 1,           9.93,        0.       )
bounds +=      ((0, None),   (9.8,10),   (-10., 10.))     #Aluminium

# Intensity Constraints
# CePt4Ge12 in Al can
#  Ce cross section * stoichiometry = 2.94 * 1 = 2.94    barn   
#  Pt cross section * stoichiometry = 11.71 * 4 = 46.84  barn  
#  Ge cross section * stoichiometry = 8.6 * 12 = 103.2   barn

constraints =  ({'type': 'eq', 'fun': lambda par:  par[0] -2.94/46.84*par[3] }, {'type': 'eq', 'fun': lambda par:  par[0] -2.94/103.2*par[6] })

#------------------------------------------------------------- Inputs ---------------------------------------------------------------------
name='CePtGe12_100K_DD_'  
masses=np.array([140.1, 195.1, 72.6, 27]).reshape(4, 1, 1)
ip_path = r'C:\Users\guijo\Desktop\Work\ip2018.par'   #needs to be raw string


#----------------------- Load main unaltered workspaces -------------------------
loadVesuvioWs = False
if loadVesuvioWs:
    loadRawAndEmptyWsVesuvio()
else:
    userRawPath = r"C:/Users/guijo/Desktop/Work/CePtGe12_backward_100K_scipy/CePtGe12_100K_DD_raw.nxs"
    userEmptyPath = r"C:/Users/guijo/Desktop/Work/CePtGe12_backward_100K_scipy/CePtGe12_100K_DD_empty.nxs"
    loadRawAndEmptyWsFromUserPath(userRawPath, userEmptyPath)

#---------------------- Select Spectra to fit --------------------

number_of_iterations = 2                      # This is the number of iterations for the reduction analysis in time-of-flight.
first_spec, last_spec = 3, 5               #3, 134
first_idx, last_idx = convertFirstAndLastSpecToIdx(first_spec, last_spec)


detectors_masked = np.array([18,34,42,43,59,60,62,118,119,133])   # Optional spectra to be masked
detectors_masked = detectors_masked[(detectors_masked >= first_spec) & (detectors_masked <= last_spec)]   #detectors within spectrums
  
    
# #-------------------------Parameters for the multiple-scattering correction, including the shape of the sample----------------------

transmission_guess = 0.98                     #experimental value from VesuvioTransmission
multiple_scattering_order, number_of_events = 2, 1.e5
hydrogen_peak = False                          # hydrogen multiple scattering
hydrogen_to_mass0_ratio = 0             # hydrogen-to-mass[0] ratio obtaiend from the preliminary fit of forward scattering  0.77/0.02 =38.5

vertical_width, horizontal_width, thickness = 0.1, 0.1, 0.001 # expressed in meters
create_slab_geometry(name, vertical_width, horizontal_width, thickness)

#-------------------- Choose and prepare main workspace for fitting-------------------------
synthetic_workspace = False

if synthetic_workspace:
    syntheticResultsPath = r"C:\Users\guijo\Desktop\work_repos\scatt_scripts\backward\runs_data\opt_spec3-134_iter4_ncp_nightlybuild.npz"
    ws_to_be_fitted = loadSyntheticNcpWorkspace(syntheticResultsPath)
else:
    ws_to_be_fitted = cropAndCloneMainWorkspace()

scaleDataY = False

# --------Generate arrays where we are going to store the data for each iteration-------------       
            
thisScriptResults = resultsObject()

# #-------------------------------------------------- Main iterative procedure ---------------------------------------------------------
for iteration in range(number_of_iterations):
    
    ws_to_be_fitted = mtd[name+str(iteration)]                                    #picks up workspace from previous iteration
    MaskDetectors(Workspace = ws_to_be_fitted, SpectraList = detectors_masked)    #this line is probably not necessary
    
    fittedNcpResults = block_fit_ncp(ws_to_be_fitted)     #main fit
    
    thisScriptResults.append(iteration, fittedNcpResults)
    mean_widths, mean_intensity_ratios = fittedNcpResults[:2]

    if (iteration < number_of_iterations - 1):   #if not at the last iteration, evaluate multiple scattering correction
        sample_properties = calculate_sample_properties(masses, mean_widths, mean_intensity_ratios, "MultipleScattering")
        correct_for_multiple_scattering(name, sample_properties, transmission_guess, multiple_scattering_order, number_of_events)    
        #create corrected workspace to be used in the subsquent iteration
        Minus(LHSWorkspace= name, RHSWorkspace = name+"_MulScattering", OutputWorkspace = name+str(iteration+1))

#----------------------------------------test for sub masses function----------------------------------------------
# ws_sub_m = subtract_other_masses(mtd[name+str(number_of_iterations-1)], ncp_all_m)
# dataX_sub_m, dataY_sub_m, dataE_sub_m = ws_sub_m.extractX(), ws_sub_m.extractY(), ws_sub_m.extractE() 

#-------------------------------------------test for ncp workspaces-----------------------------------------------
#  Records the data of the ncp workspaces
# all_tot_ncp = np.zeros((number_of_iterations, mtd[name].getNumberHistograms(), mtd[name].blocksize()-1))
# all_indiv_ncp = np.zeros((number_of_iterations, len(masses), mtd[name].getNumberHistograms(), mtd[name].blocksize()-1))
#                        
# for i in range(number_of_iterations):
#     ncp_ws_name = name + str(i)
#     ncp_tot = mtd[ncp_ws_name + "_tof_fitted_profiles"]
#     ncp_tot_dataY = ncp_tot.extractY()
#     all_tot_ncp[i] = ncp_tot_dataY
#     
#     for m in range(len(masses)):
#         ncp_m = mtd[ncp_ws_name + "_tof_fitted_profile_" + str(m+1)]
#         ncp_m_dataY = ncp_m.extractY()
#         all_indiv_ncp[i, m] = ncp_m_dataY
#                 
                      
#----------------------------------------------------store data for testing---------------------------------------------------------
#savepath = r"C:\Users\guijo\Desktop\work_repos\scatt_scripts\backward\runs_data\opt_spec3-134_iter4_ncp_nightlybuild_synthetic_fit"
savePath = r"C:\Users\guijo\Desktop\work_repos\scatt_scripts\backward\runs_data\opt_spec3-134_iter4_ncp_nightlybuild_clean"

thisScriptResults.save(savePath)

end_time = time.time()
print("running time: ", end_time-start_time, " seconds")
import numpy as np
import matplotlib.pyplot as plt
import mantid                          
from mantid.simpleapi import *    
#from scipy import signal
#from scipy.optimize import curve_fit
from scipy import optimize

# command for the formatting of the printed output
np.set_printoptions(suppress=True, precision=4, linewidth= 150 )

#----------------------------------Mathematical functions----------------------------

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
    derivative = - np.roll(fun,-6) + 24*np.roll(fun,-5) - 192*np.roll(fun,-4) + 488*np.roll(fun,-3) + 387*np.roll(fun,-2) - 1584*np.roll(fun,-1) \
                 + np.roll(fun,+6) - 24*np.roll(fun,+5) + 192*np.roll(fun,+4) - 488*np.roll(fun,+3) - 387*np.roll(fun,+2) + 1584*np.roll(fun,+1)
    derivative /= np.power(np.roll(x,-1) - x, 3) 
    derivative /= 12**3
    derivative[:6], derivative[-6:] = np.zeros(6), np.zeros(6)  #need to correct for beggining and end of array because of the rolls
    return derivative

#-----------------------------------Creating Matrix Arrays Outside Fitting-----------------------------------

def load_workspace_into_array(ws, first_spec, last_spec, spec_offset):   
    """Input: Workspace to extract data from
       Output: dataY, dataX and dataE as arrays and converted to point data"""

    dataY = ws.extractY()
    dataE = ws.extractE()
    dataX = ws.extractX()
    
    hist_widths = dataX[:, 1:] - dataX[:, :-1]
    dataY = dataY[:, :-1] / hist_widths
    dataE = dataE[:, :-1] / hist_widths
    dataX = (dataX[:, 1:] + dataX[:, :-1]) / 2
    
    start = first_spec - spec_offset
    end = last_spec - spec_offset + 1 
    return dataY[start:end], dataX[start:end], dataE[start:end]

def load_constants():
    """Output: the mass of the neutron, final energy of neutrons (selected by gold foil),
    factor to change energies into velocities, final velocity of neutron and hbar"""
    
    mN=1.008    #a.m.u.
    Ef=4906.         # meV
    en_to_vel = 4.3737 * 1.e-4
    vf=np.sqrt( Ef ) * en_to_vel        #m/us
    hbar = 2.0445
    return mN, Ef, en_to_vel, vf, hbar 

def load_ip_file_into_array(ip_path, dataX, first_spec, last_spec):  
    """Loads instrument parameters into array, from the file in the specified path
       ATTENTION: path needs to be a raw string"""
    
    f = open(ip_path, 'r')
    data = f.read()
    f.close()
    lines = data.split("\n")[1:-1]  #take out the first line of non numbers and the last empty line
    data_str = [line.split("\t") for line in lines]
    data = [[float(i) for i in line] for line in data_str]    #convert all str types into float
    data = np.array(data)                                     #convert to array for slicing
    
    #This array is for all 196 spectrums, need to slice it to be the same length as dataX
    spectra = data[:, 0]
    select_rows = np.where((spectra >= first_spec) & (spectra <= last_spec))
    data = data[select_rows]
    print("Data_ip first column: \n", data[:,0])
    
    return data

def calculate_kinematics_arrays(dataX, data_ip):   
    """kinematics quantities calculated from TOF data
       Input: dataX matrix array
       Output: kinematics arrays for all spectrums"""
    
    mN, Ef, en_to_vel, vf, hbar = load_constants()    
    det, plick, angle, T0, L0, L1 = np.hsplit(data_ip, 6)           #each is of len(dataX)
    t_us = dataX - T0                                   #T0 is electronic delay due to instruments
    v0 = vf * L0 / ( vf * t_us - L1 )
    E0 =  np.square( v0 / en_to_vel )            #en_to_vel is a factor used to easily change velocity to energy and vice-versa
    
    delta_E = E0 - Ef  
    delta_Q2 = 2. * mN / hbar**2 * ( E0 + Ef - 2. * np.sqrt(E0*Ef) * np.cos(angle/180.*np.pi) )
    delta_Q = np.sqrt( delta_Q2 )
    return v0, E0, delta_E, delta_Q

def convert_to_y_space(dataX, masses, delta_Q, delta_E):
    """Input: TOF matrix array, masses array, momentum and energy transfer array
       Output: y space matrix array for each mass, position of y=0 array"""
    
    mN, Ef, en_to_vel, vf, hbar = load_constants()
    all_y_spaces = np.zeros((len(masses), len(dataX), len(dataX[0])))
    for m, mass in enumerate(masses):
        E_r = np.square( hbar * delta_Q ) / 2. / mass          #Energy of recoil
        y = mass / hbar**2 /delta_Q * (delta_E - E_r)    #y-scaling       
        all_y_spaces[m, :, :] = y
    return all_y_spaces

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

def calculate_resolution_all_masses(res_pars, masses, all_y_spaces, data_ip, v0, E0, delta_E, delta_Q):    
    """Calculates the resolution widths in y-space, from the individual resolutions in the following parameters:
       Gaussian dist (std): L0, theta, L1, TOF and E1
       Lorentzian dist (HWHM): E1
       input: TOF matrix array of the peak of J(y), mass of element
       output: gaussian width and lorenztian width to be propagated through J(y)"""
    
    #print(v0.shape)
    det, plick, angle, T0, L0, L1 = np.hsplit(data_ip, 6)           #each is of len(dataX) 
    #print(det.shape)
    dE1, dTOF, dTheta, dL0, dL1, dE1_lorz = np.hsplit(res_pars, 6) 
    #print(dE1.shape)
    mN, Ef, en_to_vel, vf, hbar = load_constants()
    all_v0, all_E0, all_delta_E, all_delta_Q = v0, E0, delta_E, delta_Q    #store original arrays in new variables to be used in the loop
    res_all_masses = np.zeros((len(masses), 2, len(res_pars)))  
    for m, mass in enumerate(masses): 
        
        #First need to find the kinematics for y=0 ie select bin with y closest to zero)
        y = all_y_spaces[m, :, :]      #extract y-space for individual mass
        #print(y.shape)
        yzeros = np.array([np.min(np.abs(y), axis=1)]).transpose()      #Find minimums of each row and reshape to broadcast
        #print(yzeros.shape)
        #print(f"yzeros for mass {mass}: \n", yzeros[:10])
        yzeros_idx = abs(y)==yzeros     #bolean matrix that selects the bins for which y=0, need to compare absolute values!
        #print(yzeros_idx.shape)
        v0, E0, delta_E, delta_Q = all_v0[yzeros_idx], all_E0[yzeros_idx], all_delta_E[yzeros_idx], all_delta_Q[yzeros_idx] 
        #change shape to colums to broadcast
        v0, E0 = v0.reshape((len(v0),1)), E0.reshape((len(E0),1))
        delta_E, delta_Q = delta_E.reshape((len(delta_E),1)), delta_Q.reshape((len(delta_Q),1))
        #print(v0.shape)
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
        lorentzian_res_width = np.sqrt( dWdE1_lor + dQdE1_lor ) * dE1_lorz   # in A-1     #same as dy^2 = (dw/dE1)^2*dE1^2 + (dq/dE1)^2*dE1^2
        
        res_all_masses[m, 0, :] = gaussian_res_width[:, 0]          #selects the only column
        res_all_masses[m, 1, :] = lorentzian_res_width[:, 0]
    return res_all_masses # gaussian std dev, lorentzian hwhm 

#----------------------------------------Prepare data that depends on mass-------------------------------

def reshape_yspace(all_y_spaces):
    A = all_y_spaces
    reshaped = np.zeros( (len(A[0]), len(A), len(A[0,0])) )
    for n in range(len(A[0])):
        for i in range(len(A)):
            reshaped[n, i, :] = A[i, n, :]
    return reshaped
                         
def reshape_resolution(res_all_masses):   
    A = res_all_masses
    reshaped = np.zeros((len(A[0,0]), len(A), 2))
    for n in range(len(A[0,0])):
        for i in range(len(A)):
            for j in range(2):
                reshaped[n, i, j] = A[i, j, n]
    return reshaped                        

#---------------------------------------Fitting procedure-------------------------------------

def fit_ncp(datax, datay, datae, v0, E0, delta_e, delta_q, yspace, res):
    """Fits the NCP and returns the best fit parameters for each row of data"""
    
    if np.all(datay == 0):           #if all zeros, then parameters are all nan, so they are ignored later down the line
        empty = np.empty(len(par))
        empty[:] = np.nan
        return list(empty)
    
    result = optimize.minimize(err_func, par[:], args=(masses, datax, datay, datae, yspace, res, delta_q, E0), method='SLSQP', \
                               bounds = bounds, constraints=constraints) 
    return list(result["x"])   #fitted parameters       #reduced_chi2 = result["fun"]/(len(datax) - len(par))  

def err_func(par, masses, datax, datay, datae, yspace, res, delta_q, E0):
    """Error function to be minimized, operates in TOF space"""
    
    ncp_all_m, ncp = calculate_ncp(par, masses, datax, yspace, res, delta_q, E0)     
    if (np.sum(datae) > 0):
        chi2 =  ((ncp - datay)**2)/(datae)**2    #weighted fit
    else:
        chi2 =  (ncp - datay)**2
    return np.sum(chi2)

def calculate_ncp(par, masses, datax, yspace, res, delta_q, E0):    
    """Creates a synthetic C(t) to be fitted to TOF values, from J(y) and resolution functions"""
    
    ncp_all_m = np.zeros((len(masses), len(datax)))   
    for m, mass in enumerate(masses):      
        #height, width and centre are parameters of J(y)
        hei, width, centre = par[3*m : 3*m+3]        
        y = yspace[m]
        gaussian_res_width, lorentzian_res_width = res[m]   #resolutions agree up to third decimal place
        # Combine the errors from Gaussians in quadrature to obtain error on J(y)
        gaussian_width = np.sqrt( width**2 + gaussian_res_width**2 ) 
        joy = fun_pseudo_voigt(y-centre, gaussian_width, lorentzian_res_width)       
        #Finite State events correction
        FSE =  - fun_derivative3(y,joy) * width**4 / delta_q * 0.72 # 0.72 is an empirical coefficient.        
        #synthetic C(t)
        ncp_all_m[m] = hei * (joy + FSE ) * E0 * E0**(-0.92) * mass / delta_q # Here -0.92 is a parameter describing the epithermal flux tail    
    ncp = np.sum(ncp_all_m, axis=0)
    return ncp_all_m, ncp
    
#-----------------------------Extract data from best fit params and create workspaces------------------------

def build_ncp_matrices(par, datax, yspace, res, delta_q, E0):
    """input: all row parameters
       output: row with the ncp for each mass"""
    
    ncp_m, ncp = calculate_ncp(par, masses, datax, yspace, res, delta_q, E0)        
    return ncp_m

def create_ncp_workspaces(ncp_all_m_reshaped, dataX, ws):
    """Transforms the data straight from the map and creates matrices of the ncp fits and respective workspaces"""
    
    ncp_total = np.sum(ncp_all_m_reshaped, axis=1)
    ncp_all_m = reshape_yspace(ncp_all_m_reshaped)     #same operation as we did for y spaces ie exchange of first two indices
    
    CreateWorkspace(DataX=dataX, DataY=ncp_total, DataE=np.zeros(dataX.shape), OutputWorkspace=ws.name()+"_tof_fitted_profiles")
    for i, ncp_m in enumerate(ncp_all_m):
        CreateWorkspace(DataX=dataX, DataY=ncp_m, DataE=np.zeros(dataX.shape), OutputWorkspace=ws.name()+"_tof_fitted_profile_"+str(i+1))
    return ncp_all_m, ncp_total
        
def build_fit_data(all_best_par):
    """"Seperates quantities in different arrays"""
    
    intensities = all_best_par[:, 0::3].T
    widths = all_best_par[:, 1::3].T
    centers = all_best_par[:, 2::3].T
    return intensities, widths, centers    

def calculate_mean_widths_and_intensities(masses, widths, intensities): #spectra and verbose not used 
    """calculates the mean widths and intensities of the Compton profile J(y) for each mass """ 
    
    #widths, intensities = np.array(widths), np.array(intensities)   
    mean_widths = np.array([np.nanmean(widths, axis=1)]).transpose()     #shape (4,1)
    widths_std = np.array([np.nanstd(widths, axis=1)]).transpose()
    
    width_deviation = np.abs(widths - mean_widths)         #subtraction row by row 
    better_widths = np.where(width_deviation > widths_std, np.nan, widths)   #where True, replace by nan
    better_intensities = np.where(width_deviation > widths_std, np.nan, intensities)
    
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
    for m, mass in enumerate(masses):
        print ("\nMass: ", mass, "\nmean width: ", mean_widths[m], " \pm ", widths_std[m])
        print ("mean intensity ratio: ", mean_intensity_ratios[m], " \pm ", mean_intensity_ratios_std[m])
    return mean_widths, mean_intensity_ratios

#-----------------------------Correct for Multiple Scattering-----------------------

def calculate_sample_properties(masses,mean_widths,mean_intensity_ratios, mode):
    """returns the one of the inputs necessary for the VesuvioCalculateGammaBackground
    or VesuvioCalculateMS"""
    
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

def correct_for_multiple_scattering(main_ws, sample_properties, transmission_guess, \
                                    multiple_scattering_order, number_of_events):
    """Uses the Mantid algorithm for the MS correction to create a Workspace for the MS"""
    
    print ("Evaluating the Multiple Scattering Correction.")    
    MS_masses = sample_properties[::3]        #selects only the masses, every 3 numbers
    MS_amplitudes = sample_properties[1::3]   #same as above, but starts at first intensity
        
    dens, trans = VesuvioThickness(Masses=MS_masses, Amplitudes=MS_amplitudes, TransmissionGuess=transmission_guess,Thickness=0.1)   
    
    totscat, mulscat = VesuvioCalculateMS(main_ws, NoOfMasses=len(MS_masses), SampleDensity=dens.cell(9,1),\
                                           AtomicProperties=sample_properties, BeamRadius=2.5, \
                                           NumScatters=multiple_scattering_order, NumEventsPerRun=int(number_of_events))
    
    data_normalisation = Integration(main_ws) 
    simulation_normalisation = Integration(totscat)
    for ws, ws_name in zip((mulscat, totscat), (main_ws.name()+"_MulScattering", main_ws.name()+"_TotScattering")):
        ws = Divide(LHSWorkspace = ws, RHSWorkspace = simulation_normalisation)
        ws = Multiply(LHSWorkspace = ws, RHSWorkspace = data_normalisation)
        RenameWorkspace(InputWorkspace = ws, OutputWorkspace = ws_name)
    DeleteWorkspaces([data_normalisation, simulation_normalisation, trans, dens])
    #the only remaining workspaces are the _MulScattering and _TotScattering
    return
    
#-----------------------------testing functions------------------------

def select_spectra(idx, dataX, dataY, dataE, v0, E0, delta_E, delta_Q, reshaped_y_spaces, reshaped_res, best_par):
    return dataX[idx], dataY[idx], dataE[idx], v0[idx], E0[idx], delta_E[idx], delta_Q[idx], reshaped_y_spaces[idx], reshaped_res[idx], best_par[idx]

#----------------------------Multiple scattering input---------------
transmission_guess = 0.98                               #experimental value from VesuvioTransmission
multiple_scattering_order, number_of_events = 2, 1.e5
hydrogen_peak=False                                     # hydrogen multiple scattering
hydrogen_to_mass0_ratio = 0

#----------------------------Input parameters------------------

masses=[140.1,195.1,72.6,27] 

par     = ( 1,           18.22,       0.       )
bounds  = ((0, None),   (17,20),    (-30., 30.))     
par    += ( 1,           22.5,        0.       )
bounds += ((0, None),   (20,25),    (-30., 30.))
par    += ( 1,           15.4,        0.       )
bounds += ((0, None),   (12.2,18),  (-10., 10.))
par    += ( 1,           9.93,        0.       )
bounds += ((0, None),   (9.8,10),   (-10., 10.))
constraints =  ({'type': 'eq', 'fun': lambda par:  par[0] -2.94/46.84*par[3] }, {'type': 'eq', 'fun': lambda par:  par[0] -2.94/103.2*par[6] })


import numpy as np
import matplotlib.pyplot as plt
import mantid                          
from mantid.simpleapi import *
from scipy import optimize
import time

start_time=time.time()

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
    return dataY[np.newaxis, start:end, :], dataX[np.newaxis, start:end, :], dataE[np.newaxis, start:end, :]  #even if ws is cropped, still works

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
    print("Data_ip first column: \n", data[:,0])             #spectrums selected    
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
    return v0, E0, delta_E, delta_Q                   #shape(no_spectrums, len_spec)

def convert_to_y_space(dataX, masses, delta_Q, delta_E):
    """Input: TOF matrix array, masses array, momentum and energy transfer array
       Output: y space matrix array for each mass, position of y=0 array"""
    
    mN, Ef, en_to_vel, vf, hbar = load_constants()
    E_r = np.square( hbar * delta_Q ) / 2. / masses   #Energy of recoil
    all_y_spaces = masses / hbar**2 /delta_Q * (delta_E - E_r)    #y-scaling  
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
    
    det, plick, angle, T0, L0, L1 = np.hsplit(data_ip, 6)           #each is of len(dataX) 
    dE1, dTOF, dTheta, dL0, dL1, dE1_lorz = np.hsplit(res_pars, 6) 
    mN, Ef, en_to_vel, vf, hbar = load_constants()
     
    yzeros = np.abs(all_y_spaces).min(axis=2).reshape(len(masses), len(all_y_spaces[0]), 1)    #Find minimums of each row and reshape to broadcast

    yzeros_mask = abs(all_y_spaces)==yzeros     #bolean matrix that selects the bins for which y=0, need to compare absolute values!
    #expand kinematics arrays to be the same shape as yzeros_mask
    v0, E0, delta_E, delta_Q = v0*np.ones(masses.shape), E0*np.ones(masses.shape), delta_E*np.ones(masses.shape), delta_Q*np.ones(masses.shape)
    v0, E0, delta_E, delta_Q = v0[yzeros_mask], E0[yzeros_mask], delta_E[yzeros_mask], delta_Q[yzeros_mask] 
    v0, E0, delta_E, delta_Q = v0.reshape(yzeros.shape), E0.reshape(yzeros.shape), delta_E.reshape(yzeros.shape), delta_Q.reshape(yzeros.shape)
    
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

    return np.concatenate((gaussian_res_width, lorentzian_res_width), axis=2)


#----------------------------------------Prepare data that depends on mass-------------------------------

def reshape_yspace(all_y_spaces):     #find more elegant way of doing this later
    A = all_y_spaces
    reshaped = np.zeros( (len(A[0]), len(A), len(A[0,0])) )
    for n in range(len(A[0])):
        for i in range(len(A)):
            reshaped[n, i, :] = A[i, n, :]
    return reshaped
                                               
#---------------------------------------Fitting procedure-------------------------------------

def fit_ncp(datax, datay, datae, v0, E0, delta_e, delta_q, yspace, res):
    """Fits the NCP and returns the best fit parameters for each row of data"""
    
    if np.all(datay == 0):           #if all zeros, then parameters are all nan, so they are ignored later down the line
        return np.full(len(par)+2, np.nan)
    
    result = optimize.minimize(err_func, par[:], args=(masses, datax, datay, datae, yspace, res, delta_q, E0), method='SLSQP', \
                               bounds = bounds, constraints=constraints) 
    return np.append(result["x"], [result["fun"]/(len(datax) - len(par)), result["nit"]])   #fitted parameters    

def err_func(par, masses, datax, datay, datae, yspace, res, delta_q, E0):
    """Error function to be minimized, operates in TOF space"""
    
    ncp_all_m, ncp = calculate_ncp(par, masses, datax, yspace, res, delta_q, E0)     
    if (np.sum(datae) > 0):
        chi2 =  ((ncp - datay)**2)/(datae)**2    #weighted fit
    else:
        chi2 =  (ncp - datay)**2
    return np.sum(chi2)

def calculate_ncp(par, masses, datax, yspace, res, delta_q, E0):    #yspace, res have shape (4, 144) and (4,2)
    """Creates a synthetic C(t) to be fitted to TOF values, from J(y) and resolution functions
       shapes: par (1, 12), masses (4,1,1), datax (1, n), yspace (4, n), res (4, 2), delta_q (1, n), E0 (1,n)"""
    
    gaussian_res_width, lorentzian_res_width = np.split(res, 2, axis=1)     #shape(4,1)
    
    masses = masses.reshape(4,1)
    intensity = par[::3].reshape(masses.shape)
    width = par[1::3].reshape(masses.shape)
    position = par[2::3].reshape(masses.shape)
    
    gaussian_width = np.sqrt( width**2 + gaussian_res_width**2 )     #shape(4,1)
    
    joy = fun_pseudo_voigt(yspace-position, gaussian_width, lorentzian_res_width)   #shape(4, n)
    
    FSE =  - fun_derivative3(yspace,joy) * width**4 / delta_q * 0.72   #fun_derivative needs to be changed to axis=1
    
    ncp_all_m = intensity * (joy + FSE ) * E0 * E0**(-0.92) * masses / delta_q     #shape(4,n)
    ncp = np.sum(ncp_all_m, axis=0)
    return ncp_all_m, ncp
    
#-----------------------------Extract data from best fit params and create workspaces------------------------

def build_ncp_matrices(par, datax, yspace, res, delta_q, E0):
    """input: all row parameters
       output: row with the ncp for each mass"""
    
    ncp_m, ncp = calculate_ncp(par, masses, datax, yspace, res, delta_q, E0)        
    return ncp_m

def create_ncp_workspaces(ncp_all_m_reshaped, dataX, ws, first_spec, last_spec):
    """Transforms the data straight from the map and creates matrices of the ncp fits and respective workspaces"""
    
    ncp_total = np.sum(ncp_all_m_reshaped, axis=1)     #shape(no of spec, len of spec)
    ncp_all_m = reshape_yspace(ncp_all_m_reshaped)     #same operation as we did for y spaces ie exchange of first two indices
    
    #spec_nos = str(range(first_spec, last_spec+1))    #correct numbers of spectrums, currently not working
    CreateWorkspace(DataX=dataX.flatten(), DataY=ncp_total.flatten(), Nspec=len(dataX[0]), OutputWorkspace=ws.name()+"_tof_fitted_profiles")
    for i, ncp_m in enumerate(ncp_all_m):
        CreateWorkspace(DataX=dataX.flatten(), DataY=ncp_m.flatten(), Nspec=len(dataX[0]), OutputWorkspace=ws.name()+"_tof_fitted_profile_"+str(i+1))
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

#-----------------------------Main procedure to fit spectrums-----------------------

def block_fit_ncp(ws):     #Need to change main procedure
    """Runs the main procedure for the fitting of the input workspace ws"""
    
    #--------------Prepare all matrices before fitting-------------------
    dataY, dataX, dataE = load_workspace_into_array(ws, first_spec, last_spec, spec_offset)    #shapes(1,-,-)
    data_ip = load_ip_file_into_array(ip_path, first_spec, last_spec)
    v0, E0, delta_E, delta_Q = calculate_kinematics_arrays(dataX, data_ip)
    all_y_spaces = convert_to_y_space(dataX, masses, delta_Q, delta_E)
    res_pars = load_resolution_parameters(data_ip)
    res_all_masses = calculate_resolution_all_masses(res_pars, masses, all_y_spaces, data_ip, v0, E0, delta_E, delta_Q)
    reshaped_yspaces = reshape_yspace(all_y_spaces)
    reshaped_resolution = reshape_yspace(res_all_masses)   #gaussian is first element of pair
    
    #--------------------------Fitting----------------------------------
    par_chi_nit = list(map(fit_ncp, dataX[0], dataY[0], dataE[0], v0[0], E0[0], delta_E[0], delta_Q[0], reshaped_yspaces, reshaped_resolution))
    par_chi_nit = np.array(par_chi_nit)
    
    spec = data_ip[:, 0, np.newaxis]   #shape (no of specs, 1)
    spec_par_chi_nit = np.append(spec, par_chi_nit, axis=1)
    print("best fit parameters: [spec no -------------------best fit par----------------------chi2 nit]\n\n", spec_par_chi_nit)
        
    all_best_par = np.array(spec_par_chi_nit)[:, 1:-2]

    #-----------------second map to build ncp data----------------------
    ncp_all_m_reshaped = list(map(build_ncp_matrices, all_best_par, dataX[0], reshaped_yspaces, reshaped_resolution, delta_Q[0], E0[0]))
    ncp_all_m_reshaped = np.array(ncp_all_m_reshaped)
    #print("ncp_m: ", ncp_all_m_reshaped.shape)
    ncp_all_m, ncp_total = create_ncp_workspaces(ncp_all_m_reshaped, dataX, ws, first_spec, last_spec)

    #---------------from best fit parameters, build intensities, mass and width arrays, fitted profiles-----------------
    intensities, widths, positions = all_best_par[:, 0::3].T, all_best_par[:, 1::3].T, all_best_par[:, 2::3].T     #shape (4,n)
    mean_widths, mean_intensity_ratios = calculate_mean_widths_and_intensities(widths, intensities)
    return mean_widths, mean_intensity_ratios, intensities, widths, positions, spec_par_chi_nit

#-----------------------------Correct for Multiple Scattering-----------------------

# def create_slab_geometry(ws_name,vertical_width, horizontal_width, thickness):  #Don't know what it does
#         half_height, half_width, half_thick = 0.5*vertical_width, 0.5*horizontal_width, 0.5*thickness
#         xml_str = \
#         " <cuboid id=\"sample-shape\"> " \
#         + "<left-front-bottom-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (half_width,-half_height,half_thick) \
#         + "<left-front-top-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (half_width, half_height, half_thick) \
#         + "<left-back-bottom-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (half_width, -half_height, -half_thick) \
#         + "<right-front-bottom-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (-half_width, -half_height, half_thick) \
#         + "</cuboid>"
#         CreateSampleShape(ws_name, xml_str)
#         return

# def calculate_sample_properties(masses,mean_widths,mean_intensity_ratios, mode):
#     """returns the one of the inputs necessary for the VesuvioCalculateGammaBackground
#     or VesuvioCalculateMS"""
    
#     if mode == "GammaBackground":      #Not used for backscattering
#         profiles = ""
#         for m, mass in enumerate(masses):
#             width, intensity = str(mean_widths[m]), str(mean_intensity_ratios[m])
#             profiles += "name=GaussianComptonProfile,Mass=" + str(mass) + ",Width=" + width + ",Intensity=" + intensity + ';' 
#         sample_properties = profiles
        
#     elif mode == "MultipleScattering":
#         if hydrogen_peak:   
#             # ADDITION OF THE HYDROGEN INTENSITY AS PROPORTIONAL TO A FITTED NCP (OXYGEN HERE)            
#             masses = np.append(masses, 1.0079)
#             mean_widths = np.append(mean_widths, 5.0)           
#             mean_intensity_ratios = np.append(mean_intensity_ratios, hydrogen_to_mass0_ratio * mean_intensity_ratios[0])
#             mean_intensity_ratios /= np.sum(mean_intensity_ratios)
            
#         MS_properties = np.zeros(3*len(masses))
#         MS_properties[::3] = masses
#         MS_properties[1::3] = mean_intensity_ratios
#         MS_properties[2::3] = mean_widths                    
#         sample_properties = list(MS_properties)    
#     else:
#         print("\n Mode entered not valid")
#     print ("\n The sample properties for ", mode, " are: ", sample_properties)
#     return sample_properties

# def correct_for_multiple_scattering(name, sample_properties, transmission_guess, \
#                                     multiple_scattering_order, number_of_events):
#     """Uses the Mantid algorithm for MS to create two workspaces: _MulScattering and _TotScattering"""
    
#     print ("Evaluating the Multiple Scattering Correction.") 
#     MS_masses = sample_properties[::3]        #selects only the masses, every 3 numbers
#     MS_amplitudes = sample_properties[1::3]   #same as above, but starts at first intensity
        
#     dens, trans = VesuvioThickness(Masses=MS_masses, Amplitudes=MS_amplitudes, TransmissionGuess=transmission_guess,Thickness=0.1)   
    
#     totscat, mulscat = VesuvioCalculateMS(name, NoOfMasses=len(MS_masses), SampleDensity=dens.cell(9,1),\
#                                            AtomicProperties=sample_properties, BeamRadius=2.5, \
#                                            NumScatters=multiple_scattering_order, NumEventsPerRun=int(number_of_events))
    
#     data_normalisation = Integration(name)            #changed from original 
#     simulation_normalisation = Integration(totscat)
#     for ws, ws_name in zip((mulscat, totscat), (name+"MulScattering", name+"TotScattering")):
#         ws = Divide(LHSWorkspace = ws, RHSWorkspace = simulation_normalisation)
#         ws = Multiply(LHSWorkspace = ws, RHSWorkspace = data_normalisation)
#         RenameWorkspace(InputWorkspace = ws, OutputWorkspace = ws_name)
#     DeleteWorkspaces([data_normalisation, simulation_normalisation, trans, dens, mulscat, totscat])
#     return     #the only remaining workspaces are the _MulScattering and _TotScattering

# #----------------------------Other functions that act in the y space---------------------------
# #Still need to go over these and improve them

# def subtract_other_masses(ws_last_iteration, intensities, widths, positions, spectra, masses):
#     #haven't tested this one yet
#     #what does this do? subtracts NCP of masses from each other?
#     first_ws = CloneWorkspace(InputWorkspace=ws_last_iteration)
#     for index in range(len(spectra)):
#         data_x, data_y, data_e = load_workspace(first_ws , spectra[index])
#         if np.all(data_y==0):    
#             first_ws.dataY(index)[:] = 0   # assigning it to zero if its already zero?
# #             for bin in range(len(data_x)-1):
# #                 first_ws.dataY(index)[bin] = 0
#         else:
#             for m in range(len(masses)-1):
#                 other_par = (intensities[m+1, index],widths[m+1, index],positions[m+1, index])
#                 ncp = calculate_ncp(other_par, spectra[index], [masses[m+1]], data_x)
#                 first_ws.dataY(index)[:-1] -= ncp*(data_x[1:]-data_x[:-1])
# #                 for bin in range(len(data_x)-1):
# #                     first_ws.dataY(index)[bin] -= ncp[bin]*(data_x[bin+1]-data_x[bin])
#     return first_ws

# def convert_to_y_space_and_symmetrise(ws_name,mass):  
#     """input: TOF workspace
#        output: workspace in y-space for given mass with dataY symetrised"""
          
#     ws_y, ws_q = ConvertToYSpace(InputWorkspace=ws_name,Mass=mass,OutputWorkspace=ws_name+"_JoY",QWorkspace=ws_name+"_Q")
#     max_Y = np.ceil(2.5*mass+27)    #where from
#     rebin_parameters = str(-max_Y)+","+str(2.*max_Y/120)+","+str(max_Y)   #first bin boundary, width, last bin boundary, so 120 bins over range
#     ws_y = Rebin(InputWorkspace=ws_y, Params = rebin_parameters, FullBinsOnly=True, OutputWorkspace=ws_name+"_JoY")
   
#     matrix_Y = np.zeros((ws_y.getNumberHistograms(), ws_y.blocksize()))
#     for spec_idx in range(len(matrix_Y)):                 #pass the y-data onto an array to easily manipulate
#         matrix_Y[spec_idx, :] = ws_y.readY(spec_idx)        
#     matrix_Y[matrix_Y != 0] = 1
#     sum_Y = np.nansum(matrix_Y, axis=0)   
    
#     ws_y = SumSpectra(InputWorkspace=ws_y, OutputWorkspace=ws_name+"_JoY")
#     tmp=CloneWorkspace(InputWorkspace=ws_y)
#     tmp.dataY(0)[:] = sum_Y
#     tmp.dataE(0)[:] = np.zeros(tmp.blocksize())
    
#     ws = Divide(LHSWorkspace=ws_y, RHSWorkspace=tmp, OutputWorkspace =ws_name+"_JoY")
#     ws.dataY(0)[:] = (ws.readY(0)[:] + np.flip(ws.readY(0)[:])) / 2           #symetrise dataY
#     ws.dataE(0)[:] = (ws.readE(0)[:] + np.flip(ws.readE(0)[:])) / 2           #sumetrise dataE
#     normalise_workspace(ws)
#     return max_Y 

# def calculate_mantid_resolutions(ws_name, mass):
#     #uses a for loop because the fuction VesuvioResolution takes in one spectra at a time
#     max_Y = np.ceil(2.5*mass+27)
#     rebin_parameters = str(-max_Y)+","+str(2.*max_Y/240)+","+str(max_Y)
#     ws= mtd[ws_name]
#     for index in range(ws.getNumberHistograms()):
#         VesuvioResolution(Workspace=ws,WorkspaceIndex=index,Mass=mass,OutputWorkspaceYSpace="tmp")
#         tmp=Rebin("tmp",rebin_parameters)
#         if index == 0:
#             RenameWorkspace("tmp","resolution")
#         else:
#             AppendSpectra("resolution", "tmp", OutputWorkspace= "resolution")
#     SumSpectra(InputWorkspace="resolution",OutputWorkspace="resolution")
#     normalise_workspace("resolution")
#     DeleteWorkspace("tmp")
    
# def normalise_workspace(ws_name):
#     tmp_norm = Integration(ws_name)
#     Divide(LHSWorkspace=ws_name,RHSWorkspace="tmp_norm",OutputWorkspace=ws_name)
#     DeleteWorkspace("tmp_norm")


# #----------------------------Input parameters----------------------------

par     = ( 1,           18.22,       0.       )
bounds  = ((0, None),   (17,20),    (-30., 30.))     
par    += ( 1,           22.5,        0.       )
bounds += ((0, None),   (20,25),    (-30., 30.))
par    += ( 1,           15.4,        0.       )
bounds += ((0, None),   (12.2,18),  (-10., 10.))
par    += ( 1,           9.93,        0.       )
bounds += ((0, None),   (9.8,10),   (-10., 10.))

constraints =  ({'type': 'eq', 'fun': lambda par:  par[0] -2.94/46.84*par[3] }, {'type': 'eq', 'fun': lambda par:  par[0] -2.94/103.2*par[6] })

masses=np.array([140.1,195.1,72.6,27]).reshape(4, 1, 1)
ip_path = r'C:\Users\guijo\Desktop\Work\ip2018.par'   #needs to be raw string

number_of_iterations = 1                     # This is the number of iterations for the reduction analysis in time-of-flight.
runs='44462-44463'         # 100K             # The numbers of the runs to be analysed
empty_runs='43868-43911'   # 100K             # The numbers of the empty runs to be subtracted
spectra='3-134'                               # Spectra to be analysed
first_spec, last_spec = 3, 134                #3, 134
tof_binning='275.,1.,420'                             # Binning of ToF spectra
mode='DoubleDifference'
ipfile='ip2018.par'                                   # Optional instrument parameter file
detectors_masked=[18,34,42,43,59,60,62,118,119,133]   # Optional spectra to be masked

detectors_masked = np.array(detectors_masked)
detectors_masked = detectors_masked[(detectors_masked >= first_spec) & (detectors_masked <= last_spec)]   #detectors within spectrums

name='CePtGe12_100K_DD_'  

# #--------------Load workspace - since I do not have the path for Vesuvio, load files from my path----------------

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
    
# #-------------Parameters for the multiple-scattering correction, including the shape of the sample-----------
# transmission_guess = 0.98                               #experimental value from VesuvioTransmission
# multiple_scattering_order, number_of_events = 2, 1.e5
# hydrogen_peak=False                                     # hydrogen multiple scattering
# hydrogen_to_mass0_ratio = 0                             # hydrogen-to-mass[0] ratio obtaiend from the preliminary fit of forward scattering  0.77/0.02 =38.5
# vertical_width, horizontal_width, thickness = 0.1, 0.1, 0.001 # expressed in meters
# create_slab_geometry(name,vertical_width, horizontal_width, thickness)

# #--------------------------------------------------Run the code---------------------------------------------------
spec_offset = mtd[name].getSpectrum(0).getSpectrumNo()  
first_idx, last_idx = first_spec - spec_offset, last_spec - spec_offset
CropWorkspace(InputWorkspace=name, StartWorkspaceIndex = first_idx, EndWorkspaceIndex = last_idx, OutputWorkspace=name) #for MS correction
spec_offset = mtd[name].getSpectrum(0).getSpectrumNo()  

# ws_to_be_fitted = CloneWorkspace(InputWorkspace = name, OutputWorkspace = name+"0")  #initialize ws for the first fit
# for iteration in range(number_of_iterations):
    
#     ws_to_be_fitted = mtd[name+str(iteration)]                              #picks up workspace from previous iteration
#     MaskDetectors(Workspace=ws_to_be_fitted,SpectraList=detectors_masked)
    
#     mean_widths, mean_intensity_ratios = block_fit_ncp(ws_to_be_fitted)

#     if (iteration < number_of_iterations - 1):   #if not at the last iteration, evaluate multiple scattering correction
#         sample_properties = calculate_sample_properties(masses, mean_widths, mean_intensity_ratios, "MultipleScattering")
#         #next step creates _Mulscattering and _TotScattering workspaces
#         correct_for_multiple_scattering(name, sample_properties, transmission_guess, multiple_scattering_order, number_of_events)
#         #create corrected workspace to be used in the subsquent iteration
#         Minus(LHSWorkspace= name, RHSWorkspace = name+"MulScattering", OutputWorkspace = name+str(iteration+1))
        

#---------------------testing----------------------
ws = mtd[name]
mean_widths, mean_intensity_ratios, intensities, widths, positions, spec_best_par_chi = block_fit_ncp(ws)

savepath = r"C:\Users\guijo\Desktop\Work\My_edited_scripts\tests_data\optimized_6.0_no_mulscat\no_for_loops_spec3-134_iter1_par_chi_nit"
# np.savez(savepath, intensities=intensities, widths=widths, positions=positions, \
#          mean_widths=mean_widths, mean_intensity_ratios=mean_intensity_ratios, spec_best_par_chi = spec_best_par_chi)
# #np.savez(savepath, spec_best_par_chi = spec_best_par_chi)
end_time = time.time()
print("running time: ", end_time-start_time, " seconds")
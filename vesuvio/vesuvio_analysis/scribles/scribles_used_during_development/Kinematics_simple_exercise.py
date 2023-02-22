# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
from scipy import optimize

#--------------functions utilized during fitting---------------

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

def fun_derivative3(x, fun):
    """Numerical approximation for the third derivative"""   
    x, fun, derivative = np.array(x), np.array(fun), np.zeros(len(fun))
    derivative += - np.roll(fun,-6) + 24*np.roll(fun,-5) - 192*np.roll(fun,-4) + 488*np.roll(fun,-3) + 387*np.roll(fun,-2) - 1584*np.roll(fun,-1)
    derivative += + np.roll(fun,+6) - 24*np.roll(fun,+5) + 192*np.roll(fun,+4) - 488*np.roll(fun,+3) - 387*np.roll(fun,+2) + 1584*np.roll(fun,+1)
    derivative /= np.power(np.roll(x,-1) - x, 3)
    derivative /= 12**3
    derivative[:6], derivative[-6:] = np.zeros(6), np.zeros(6)  #need to correct for beggining and end of array
    return derivative

#---------------loading functions------------------------

def load_ip_file(spectrum):
    #print "Loading parameters from file: ", namedtuple
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
    return angle, T0, L0, L1    #if a spectrum is not found, will yield local variable error


def load_constants():
    """Output: the mass of the neutron, final energy of neutrons (selected by gold foil),
    factor to change energies into velocities, final velocity of neutron and hbar"""
    
    mN=1.008    #a.m.u.
    Ef=4906.         # meV
    en_to_vel = 4.3737 * 1.e-4
    vf=np.sqrt( Ef ) * en_to_vel        #m/us
    hbar = 2.0445
    return mN, Ef, en_to_vel, vf, hbar
    
def load_workspace(ws, spectrum):
    
    """Returns the data arrays for a given spectrum number for a given workspace ws"""
    
    spec_offset = ws.getSpectrum(0).getSpectrumNo()  
    spec_idx = spectrum - spec_offset

    ws_y, ws_x, ws_e = ws.readY(spec_idx), ws.readX(spec_idx), ws.readE(spec_idx)
    
    hist_widths = ws_x[1:] - ws_x[:-1]     #length decreases by one
    data_y = ws_y[:-1] / hist_widths
    data_e = ws_e[:-1] / hist_widths
    data_x = (ws_x[:-1] + ws_x[1:]) / 2    #compute mean point of bins
       
    return data_x, data_y, data_e    


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
 
#-------------------------used in the fitting function below-----------------
def load_resolution_parameters(spectrum):   
    """Resolution of parameters to propagate into TOF values"""
    
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

#-------------------Fitting the Sythetic NCP -------------------------------

def block_fit_ncp(par, first_spectrum, last_spectrum, masses, ws, fit_arguments, verbose):
    
    ###  Still need to make a correction for the last column of the fitted workspaces
    
    """Builds Workspaces with the fitted sythetic C(t), both globally and each mass individually
       Output: array with spectra no, intensities, widths and centers for J(y) for each spectra"""
       
    print ("\n", "Fitting Workspace: ", ws.name())
    
    ws_len, ws_no_spectra = ws.blocksize(), ws.getNumberHistograms()
   
    intensities = np.empty((len(masses), ws_no_spectra))  #intensities of J(y)
    widths = np.empty((len(masses), ws_no_spectra))       #widths of J(y)
    positions = np.empty((len(masses), ws_no_spectra))    #center positions of J(y) 
   
    intensities[:,:], widths[:,:], positions[:,:] = np.nan, np.nan, np.nan #this line puts all the values to nan, good for testing just a few iterations
    
    #Create Workspaces to pass calculated values
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
    
    boundaries, constraints = fit_arguments[0], fit_arguments[1]  #what are these used for?
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
    # velocities in m/us, times in us, energies in meV
    v0, E0, delta_E, delta_Q = calculate_kinematics(data_x, angle, T0, L0, L1 )
     
    ncp = 0. # initialising the function values
    ncp_indiv_m = np.zeros((len(masses), len(data_x)))   
    npars = int(len(par)/len(masses))
    for m in range(len(masses)):      
        #width and centre are parameters of J(y)
        mass, hei , width, centre = masses[m] , par[m*npars], par[1+m*npars], par[2+m*npars]
        
        E_r = ( hbar * delta_Q )**2 / 2. / mass          #Energy of recoil
        y = mass / hbar**2 /delta_Q * (delta_E - E_r)    #y-scaling
         
        joy = fun_gaussian(y-centre, 1.)     #Why just not use np.where(y == centre)?
        #Maybe because no value of y is exactly equal to centre, so need to find the closest bin to center
        #Do it by using the peak of the function
        pcb = np.where(joy == joy.max())     # this finds the peak-centre bin (pcb)
        
        gaussian_res_width, lorentzian_res_width = calculate_resolution(spectrum, data_x[pcb], mass)
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


def build_kinematic_workspaces(ws, masses, first_spec, last_spec):

    for m in range(len(masses)):
        CloneWorkspace(InputWorkspace=ws,OutputWorkspace=str(ws)+"_y_"+str(m))
    ws_w = CloneWorkspace(InputWorkspace=ws,OutputWorkspace=str(ws)+"_w")
    ws_q = CloneWorkspace(InputWorkspace=ws,OutputWorkspace=str(ws)+"_q")
    
    spec_offset = ws.getSpectrum(0).getSpectrumNo()  

    for spec_no in range(first_spec, last_spec):   #later change to range(min_spectrum, max_spectrum), ie this is spec no
        data_x, data_y, data_e = load_workspace(ws, spec_no)  
        angle, T0, L0, L1 = load_ip_file(spec_no)   #this function takes the spectrum number, not index!       
        mN, Ef, en_to_vel, vf, hbar = load_constants()
        v0, E0, delta_E, delta_Q = calculate_kinematics(data_x, angle, T0, L0, L1)
        
        spec_idx = spec_no - spec_offset
        
        for i, m in enumerate(masses):
            E_r = ( hbar * delta_Q )**2 / 2. / m
            y = m / hbar**2 /delta_Q * (delta_E - E_r)
            mtd[str(ws)+"_y_"+str(i)].dataX(spec_idx)[:-1] = y
            mtd[str(ws)+"_y_"+str(i)].dataX(spec_idx)[-1] = np.nan

        ws_w.dataX(spec_idx)[:-1] = delta_E
        ws_w.dataX(spec_idx)[-1] = np.nan
        ws_q.dataX(spec_idx)[:-1] = delta_Q
        ws_q.dataX(spec_idx)[-1] = np.nan
 

def plot_kinematics(ws, ws_ncp, spectrum, masses): #ws is original workspace
    mN, Ef, en_to_vel, vf, hbar = load_constants()
    spec_offset = ws.getSpectrum(0).getSpectrumNo()  
    spec_idx = spectrum - spec_offset

    ncp = ws_ncp.readY(spec_idx)   
    data_w = mtd[str(ws)+"_w"].readX(spec_idx)
    data_q = mtd[str(ws)+"_q"].readX(spec_idx)
    
    plt.figure()
    ax = plt.axes(projection='3d')
    #Q, W = np.meshgrid(data_q, data_w)  
    ax.plot3D(data_q, data_w, ncp)
    ax.plot3D(data_q, data_w, 0, label="instrument trajectory")
    
    for i, mass in enumerate(masses):        
        ax.plot3D(data_q, data_q**2 * hbar**2 /2 / mass, label = r"$ \omega = \frac{q^2}{2 M_%i} $" % i)
        ws_ncp_m = mtd[ws.name()+'_tof_fitted_profile_'+str(i+1)]   
        ncp_m = ws_ncp_m.readY(spec_idx)
        ax.plot3D(data_q, data_w, ncp_m)
        
    ax.set_xlabel("q")
    ax.set_ylabel(r"$\omega$")
    ax.set_ylim(-500, 1700)
    ax.set_zlabel("Counts")
    plt.legend()
    plt.show()

def plot_indiv_graphs(ws, spectrum, masses):
    
    spec_offset = ws.getSpectrum(0).getSpectrumNo()  
    spec_idx = spectrum - spec_offset
 
    plt.figure()
    for m in range(len(masses)):
        ws_ncp_m = mtd[ws.name()+'_tof_fitted_profile_'+str(m+1)]   
        ncp_m = ws_ncp_m.readY(spec_idx)
        y_m = mtd[ws.name()+"_y_"+str(m)].readX(spec_idx)
        plt.plot(y_m, ncp_m, label="ncp of mass %i" % masses[m])
    plt.title("NCP of individual masses")
    plt.legend()
    plt.show()
        
#-------------------------------------------end of functions-----------------------------------------    
    
ws = Load(Filename = "C:/Users/guijo/Desktop/Work/CePtGe12_backward_100K_scipy/CePtGe12_100K_DD_.nxs", OutputWorkspace = "CePtGe12_100K_DD_") 

par        = (    1,          18.22,    0.        )
bounds     = (  (0, None),   (17,20),  (-30., 30.))    
# second element:       Pt                                                                                        
par        += (   1,          22.5,     0.        )
bounds     += (  (0, None),  (20,25),  (-30., 30.))
# third element:       Ge                                                                                        
par        += (   1,          15.4,     0.         )
bounds     += (  (0, None),  (12.2,18),  (-10., 10.))
# fourth element:       Al                                                                                        
par        += (   1,          9.93,     0.         )
bounds     += (  (0, None),  (9.8,10),  (-10., 10.))


constraints =  ({'type': 'eq', 'fun': lambda par:  par[0] -2.94/46.84*par[3] },{'type': 'eq', 'fun': lambda par:  par[0] -2.94/103.2*par[6] })
fit_arguments = [bounds, constraints]

masses=[140.1,195.1,72.6,27] 
first_spec, last_spec = 3, 7
verbose = True

block_fit_ncp(par, first_spec, last_spec, masses, ws, fit_arguments, verbose)

ws_ncp = mtd[ws.name()+"_tof_fitted_profiles"]
build_kinematic_workspaces(ws, masses, 3, 7)
plot_kinematics(ws, ws_ncp, 3, masses)   #independent of the masses in the sample 
plot_indiv_graphs(ws, 3, masses)

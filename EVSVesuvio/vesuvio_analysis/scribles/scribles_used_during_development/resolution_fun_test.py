# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

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
    """Calculates the resolution widths in y-space, from the individual resolutions in the following parameters:
       Gaussian dist (std): L0, theta, L1, TOF and E1
       Lorentzian dist (HWHM): E1"""
       
    angle, T0, L0, L1 = load_ip_file(spectrum)
    mN, Ef, en_to_vel, vf, hbar = load_constants()
    v0, E0, delta_E, delta_Q = calculate_kinematics(data_x, angle, T0, L0, L1 )
    #load resolution of indivisual parameters
    dE1, dTOF, dTheta, dL0, dL1, lorentzian_res_width = load_resolution_parameters(spectrum)
    # Calculate dw^2 and dq^2 [meV]
    dEE = (1. + (E0 / Ef)**1.5 * ( L1 / L0 ) )**2 * dE1**2 + (2. * E0 * v0 / L0 )**2 * dTOF**2 
    dEE+= ( 2. * E0**1.5 / Ef**0.5 / L0 )**2 * dL1**2 + ( 2. * E0 / L0 )**2 * dL0**2
    dQQ =  (1. - (E0 / Ef )**1.5 *L1 / L0 - np.cos(angle/180.*np.pi) * ( ( E0 / Ef )**0.5 - L1 / L0 * E0 / Ef ))**2 * dE1**2
    dQQ+= ( ( 2. * E0 * v0 / L0 )**2 * dTOF**2 + (2. * E0**1.5 / L0 / Ef**0.5 )**2 *dL1**2 + ( 2. * E0 / L0 )**2 * dL0**2 ) * np.abs( Ef / E0 * np.cos(angle/180.*np.pi) -1.)
    dQQ+= ( 2. * np.sqrt( E0 * Ef )* np.sin(angle/180.*np.pi) )**2 * dTheta**2
    # conversion from meV^2 to A^-2
    dEE*= ( mass / hbar**2 /delta_Q )**2
    dQQ*= ( mN / hbar**2 /delta_Q )**2
    #dy^2 = dw^2 + dq^2                              #why not dy^2 = (dy/dw)^2*dw^2 + (dy/dq)^2*dq^2 ????
    gaussian_res_width =   np.sqrt( dEE + dQQ ) # in A-1
    #lorentzian component in meV
    dEE_lor = (1. + (E0 / Ef)**1.5 * ( L1 / L0 ) )**2         # is it - or +?
    dQQ_lor =  (1. - (E0 / Ef )**1.5 *L1 / L0 - np.cos(angle/180.*np.pi) * ( ( E0 / Ef )**0.5 + L1 / L0 * E0 / Ef )) **2
    # conversion from meV^2 to A^-2
    dEE_lor*= ( mass / hbar**2 /delta_Q )**2
    dQQ_lor*= ( mN / hbar**2 /delta_Q )**2
    #next step is dy^2 = (dw/dE1)^2*dE1^2 + (dq/dE1)^2*dE1^2
    lorentzian_res_width *= np.sqrt( dEE_lor + dQQ_lor ) # in A-1
    return gaussian_res_width, lorentzian_res_width # gaussian std dev, lorentzian hwhm 
    
def calculate_resolution_improved(spectrum, data_x, mass): 
    """Calculates the resolution widths in y-space, from the individual resolutions in the following parameters:
       Gaussian dist (std): L0, theta, L1, TOF and E1
       Lorentzian dist (HWHM): E1
       input: spectrum, TOF of the peak of J(y), mass of element
       output: gaussian width and lorenztian width to be propagated through J(y)"""
       
    angle, T0, L0, L1 = load_ip_file(spectrum)
    mN, Ef, en_to_vel, vf, hbar = load_constants()
    v0, E0, delta_E, delta_Q = calculate_kinematics(data_x, angle, T0, L0, L1 )
    #load resolution of indivisual parameters
    dE1, dTOF, dTheta, dL0, dL1, dE1_lor = load_resolution_parameters(spectrum)
    # Calculate dw^2 and dq^2 [meV]
    dW2 = (1. + (E0 / Ef)**1.5 * ( L1 / L0 ) )**2 * dE1**2 + (2. * E0 * v0 / L0 )**2 * dTOF**2   \
           + ( 2. * E0**1.5 / Ef**0.5 / L0 )**2 * dL1**2 + ( 2. * E0 / L0 )**2 * dL0**2
    
    dQ2 =  (1. - (E0 / Ef )**1.5 *L1 / L0 - np.cos(angle/180.*np.pi) * ( ( E0 / Ef )**0.5 - L1 / L0 * E0 / Ef ))**2 * dE1**2    \
           + ( ( 2. * E0 * v0 / L0 )**2 * dTOF**2 + (2. * E0**1.5 / L0 / Ef**0.5 )**2 *dL1**2 + ( 2. * E0 / L0 )**2 * dL0**2 ) * np.abs( Ef / E0 * np.cos(angle/180.*np.pi) -1.)   \
           + ( 2. * np.sqrt( E0 * Ef )* np.sin(angle/180.*np.pi) )**2 * dTheta**2
    # conversion from meV^2 to A^-2
    dW2 *= ( mass / hbar**2 /delta_Q )**2
    dQ2 *= ( mN / hbar**2 /delta_Q )**2
    #why not dy^2 = (dy/dw)^2*dw^2 + (dy/dq)^2*dq^2 ????
    gaussian_res_width =   np.sqrt( dW2 + dQ2 ) # in A-1
    #lorentzian component in meV
    dWdE1_lor = (1. + (E0 / Ef)**1.5 * ( L1 / L0 ) )**2         # is it - or +?
    dQdE1_lor =  (1. - (E0 / Ef )**1.5 *L1 / L0 - np.cos(angle/180.*np.pi) * ( ( E0 / Ef )**0.5 + L1 / L0 * E0 / Ef )) **2
    # conversion from meV^2 to A^-2
    dWdE1_lor *= ( mass / hbar**2 /delta_Q )**2
    dQdE1_lor *= ( mN / hbar**2 /delta_Q )**2
    #next step is dy^2 = (dw/dE1)^2*dE1^2 + (dq/dE1)^2*dE1^2
    lorentzian_res_width = np.sqrt( dWdE1_lor + dQdE1_lor ) * dE1_lor   # in A-1
    return gaussian_res_width, lorentzian_res_width # gaussian std dev, lorentzian hwhm 
    
    
print(calculate_resolution_improved(3, 100, 140))
print(calculate_resolution(3, 100, 140))

print(calculate_resolution_improved(5, 200, 27))
print(calculate_resolution(5, 200, 27))
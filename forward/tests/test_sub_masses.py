import unittest
import numpy as np
import numpy.testing as nptest
import matplotlib.pyplot as plt
from mantid.simpleapi import *    

#optimized function to test
target = __import__("../Optimized_fwd_script.py")
opt_sub_mass = target.subtractAllMassesExceptFirst

# Original function
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

def fun_pseudo_voigt(x, sigma, gamma): #input std gaussiana e hwhm lorentziana
    fg, fl = 2.*sigma*np.sqrt(2.*np.log(2.)), 2.*gamma #parameters transformed to gaussian and lorentzian FWHM
    f = 0.5346 * fl + np.sqrt(0.2166*fl**2 + fg**2 )
    eta = 1.36603 *fl/f - 0.47719 * (fl/f)**2 + 0.11116 *(fl/f)**3
    sigma_v, gamma_v = f/(2.*np.sqrt(2.*np.log(2.))), f /2.
    pseudo_voigt = eta * fun_lorentzian(x, gamma_v) + (1.-eta) * fun_gaussian(x, sigma_v)
    norm=np.sum(pseudo_voigt)*(x[1]-x[0])
    return pseudo_voigt#/np.abs(norm)

def fun_gaussian(x, sigma):
    gaussian = np.exp(-x**2/2/sigma**2)
    gaussian /= np.sqrt(2.*np.pi)*sigma
    return gaussian

def fun_lorentzian(x, gamma):
    lorentzian = gamma/np.pi / (x**2 + gamma**2)
    return lorentzian

def fun_derivative3(x,fun): # Used to evaluate numerically the FSE term.
    derivative =[0.]*len(fun)
    for i in range(6,len(fun)-6):
        derivative[i] = -fun[i+6] +24.*fun[i+5] -192.*fun[i+4] +488.*fun[i+3] +387.*fun[i+2] -1584.*fun[i+1]
        derivative[i]+= fun[i-6]  -24.*fun[i-5]   +192.*fun[i-4]  -488.*fun[i-3]   -387.*fun[i-2]   +1584.*fun[i-1]
        derivative[i]/=(x[i+1]-x[i])**3
    derivative=np.array(derivative)/12**3
    return derivative

# Load example workspace 
wsExample = Load(Filename=r"../input_ws/starch_80_RD_raw.nxs")

# Same initial conditions
masses = [1.0079,12,16,27]
spectra = 

ncpForEachMass = 

class TestSubMasses(unittest.TestCase):


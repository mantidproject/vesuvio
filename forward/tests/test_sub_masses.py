import unittest
import numpy as np
import numpy.testing as nptest
import matplotlib.pyplot as plt
from mantid.simpleapi import *    
from pathlib import Path
currentPath = Path(__file__).absolute().parent  # Path to the repository

#optimized function to test
def subtractAllMassesExceptFirst(ws, ncpForEachMass):
    """Input: workspace from last iteration, ncpTotal for each mass
       Output: workspace with all the ncpTotal subtracted except for the first mass"""

    ncpForEachMass = switchFirstTwoAxis(ncpForEachMass)
    # Select all masses other than the first one
    ncpForEachMass = ncpForEachMass[1:, :, :]
    # Sum the ncpTotal for remaining masses
    ncpTotal = np.sum(ncpForEachMass, axis=0)
    dataY, dataX, dataE = ws.extractY(), ws.extractX(), ws.extractE()

    # The original uses the mean points of the histograms, not dataX!
    dataY[:, :-1] -= ncpTotal * (dataX[:, 1:] - dataX[:, :-1])
    # But this makes more sense to calculate histogram widths, we can preserve one more data point
    # Last column fo data reamains unaltered, so is faulty
    # Need to decide if I keep this column or not
    wsSubMass = CreateWorkspace(DataX=dataX.flatten(), DataY=dataY.flatten(), DataE=dataE.flatten(), Nspec=len(dataX))
    return wsSubMass

def switchFirstTwoAxis(A):
    """Exchanges the first two indices of an array A,
    rearranges matrices per spectrum for iteration of main fitting procedure
    """
    return np.stack(np.split(A, len(A), axis=0), axis=2)[0]

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
    return first_ws  #, all_ncp_m    #originally returns just the workspace

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

def load_ip_file(spectrum):
    #print "Loading parameters from file: ", namedtuple
    ipfile =  currentPath / ".." / 'ip2018_3.par'
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
masses = [1.0079, 12, 16, 27]
dataFilePath = currentPath / "fixatures" / "data_to_test_func_sub_mass.npz"

class TestSubMasses(unittest.TestCase):
    def setUp(self):
        dataFile = np.load(dataFilePath)

        ncpForEachMass = dataFile["all_ncp_for_each_mass"][0]
        parameters = dataFile["all_spec_best_par_chi_nit"][0]
        spectra = parameters[:, 0]
        parameters = parameters[:, 1:-2]
        intensities = parameters[:, ::3].T
        widths = parameters[:, 1::3].T
        centers = parameters[:, 2::3].T

        self.wsOri =  subtract_other_masses(
            wsExample, 
            intensities, widths, centers, 
            spectra, masses
            )
        self.wsOpt = subtractAllMassesExceptFirst(wsExample, ncpForEachMass)

        self.rtol = 0.0001
        self.equal_nan = True

    def test_dataY(self):
        oriDataY = self.wsOri.extractY()#[:, :-2]   # Last two columns are the problem!
        optDataY = self.wsOpt.extractY()#[:, :-2]
        optDataY = np.where(np.isnan(optDataY), 0, optDataY)

        totalMask = np.isclose(
            optDataY, oriDataY, rtol=self.rtol, equal_nan=self.equal_nan
            )
        totalDiffMask = ~ totalMask
        noDiff = np.sum(totalDiffMask)
        maskSize = totalDiffMask.size
        print("\nNo of different dataY points:\n",
            noDiff, " out of ", maskSize,
            f"ie {100*noDiff/maskSize:.1f} %")
        
        plt.figure()
        plt.imshow(totalMask, aspect="auto", cmap=plt.cm.RdYlGn, 
                    interpolation="nearest", norm=None)
        plt.title("Comparison between optDataY and oriDataY")
        plt.xlabel("TOF")
        plt.ylabel("Spectra")
        plt.show()
        nptest.assert_almost_equal(
            oriDataY, optDataY, decimal=10
        )

if __name__ == "__main__":
    unittest.main()
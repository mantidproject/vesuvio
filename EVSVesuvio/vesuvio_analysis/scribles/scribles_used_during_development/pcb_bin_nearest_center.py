# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

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

def load_constants():
    """Output: the mass of the neutron, final energy of neutrons (selected by gold foil),
    factor to change energies into velocities, final velocity of neutron and hbar"""
    
    mN=1.008    #a.m.u.
    Ef=4906.         # meV
    en_to_vel = 4.3737 * 1.e-4
    vf=np.sqrt( Ef ) * en_to_vel        #m/us
    hbar = 2.0445
    return mN, Ef, en_to_vel, vf, hbar

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

def fun_gaussian(x, sigma):
    """Gaussian function centered at zero"""
    gaussian = np.exp(-x**2/2/sigma**2)
    gaussian /= np.sqrt(2.*np.pi)*sigma
    return gaussian

ws = Load(Filename="C:/Users/guijo/Desktop/Work/CePtGe12_backward_100K_scipy/CePtGe12_100K_DD_.nxs", OutputWorkspace="CePtGe12_100K_DD_.nxs")

mass = 140
centre = 100
data_x = ws.readX(0)
angle, T0, L0, L1 = load_ip_file(3)
mN, Ef, en_to_vel, vf, hbar = load_constants()
v0, E0, delta_E, delta_Q = calculate_kinematics(data_x, angle, T0, L0, L1 )    # velocities in m/us, times in us, energies in meV

E_r = ( hbar * delta_Q )**2 / 2. / mass          #Energy of recoil
y = mass / hbar**2 /delta_Q * (delta_E - E_r)    #y-scaling
 
joy = fun_gaussian(y-centre, 1.)     #Why just not use np.where(y == centre)?
#Maybe because no value of y is exactly equal to centre, so need to find the closest bin to center
#Do it by using the peak of the function
pcb = np.where(joy == joy.max())     # this finds the peak-centre bin (pcb)
print(data_x[pcb])

pcb = np.argmin(np.abs(y-centre))
print(data_x[pcb])


#Original at line 257
#         joy = fun_gaussian(y-centre, 1.)     #Why just not use np.where(y == centre)?
#         #Maybe because no value of y is exactly equal to centre, so need to find the closest bin to center
#         #Do it by using the peak of the function
#         pcb = np.where(joy == joy.max())     # this finds the peak-centre bin (pcb)
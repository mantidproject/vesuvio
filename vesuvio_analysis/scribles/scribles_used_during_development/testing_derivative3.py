# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

def fun_derivative3(x,fun): # Used to evaluate numerically the FSE term.
    derivative =[0.]*len(fun)
    for i in range(6,len(fun)-6):
        derivative[i] = -fun[i+6] +24.*fun[i+5] -192.*fun[i+4] +488.*fun[i+3] +387.*fun[i+2] -1584.*fun[i+1]
        derivative[i]+= fun[i-6]  -24.*fun[i-5]   +192.*fun[i-4]  -488.*fun[i-3]   -387.*fun[i-2]   +1584.*fun[i-1]
        derivative[i]/=(x[i+1]-x[i])**3
    derivative=np.array(derivative)/12**3
    return derivative

def edited_fun_derivative3(x, fun):
    """Numerical approximation for the third derivative"""   
    x, fun, derivative = np.array(x), np.array(fun), np.zeros(len(fun))
    derivative += - np.roll(fun,-6) + 24*np.roll(fun,-5) - 192*np.roll(fun,-4) + 488*np.roll(fun,-3) + 387*np.roll(fun,-2) - 1584*np.roll(fun,-1)
    derivative += + np.roll(fun,+6) - 24*np.roll(fun,+5) + 192*np.roll(fun,+4) - 488*np.roll(fun,+3) - 387*np.roll(fun,+2) + 1584*np.roll(fun,+1)
    derivative /= np.power(np.roll(x,-1) - x, 3)
    derivative /= 12**3
    derivative[:6], derivative[-6:] = np.zeros(6), np.zeros(6)  #need to correct for beggining and end of array
    return derivative
    
def fun(x):
    return np.power(x, 3) #+ 2*np.power(x, 2) + 3*np.power(x, 4)
    
x = range(-10, 10)
fun = fun(x)
print(fun_derivative3(x, fun))
print(edited_fun_derivative3(x, fun))
# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

# from jupyterthemes import jtplot
# jtplot.style(theme='monokai', context='notebook', ticks=True, grid=True)
# 
resolution = Load(r"fixatures/yspace_fit/resolution.nxs", OutputWorkspace="resolution_sum")
H_JoY_Sym = Load(r"fixatures/yspace_fit/H_JoY_Sym.nxs")
name = H_JoY_Sym.name()
resName = resolution.name()

rebinPars = "-20,0.5,20"
Rebin(InputWorkspace=name, Params=rebinPars, OutputWorkspace=name)
Rebin(InputWorkspace=resName, Params=rebinPars, OutputWorkspace=resName)


function='''composite=Convolution,FixResolution=true,NumDeriv=true;
name=Resolution,Workspace=resolution_sum,WorkspaceIndex=0;
name=UserFunction,Formula=y0+A*exp( -(x-x0)^2/2/sigma^2)/(2*3.1415*sigma^2)^0.5,
y0=0,A=1,x0=0,sigma=5,   ties=()'''

Fit(
    Function=function, 
    InputWorkspace=name, 
    Output=name+'_joy_sum_fitted',
    Minimizer="Simplex"
    )
    
ws=mtd[name+'_joy_sum_fitted_Parameters']

popt = [ws.cell(0,1), ws.cell(2,1), ws.cell(1,1), ws.cell(3,1)]
pcov = [ws.cell(0,2), ws.cell(2,2), ws.cell(1,2), ws.cell(3,2)]
# 
# def convolvedGaussian(x, pars, res):
#     gaussFunc = gaussian(x, pars)
#     # Mode=same guarantees that convolved signal remains centered 
#     convGauss = signal.convolve(gaussFunc, res, mode="same")
#     return convGauss

def gaussian(x, pars):
    y0, x0, A, sigma = pars
    """Gaussian centered at zero"""
    return y0 + A / (2*np.pi)**0.5 / sigma * np.exp(-(x-x0)**2/2/sigma**2)

dataX = mtd[name].extractX()[0]
res = mtd["resolution_sum"].extractY()[0]
resX = mtd["resolution_sum"].extractX()[0]

# Several ways of computing Integrals
histWidths = resX[1:] - resX[:-1]
integral = np.sum(histWidths * res[:-1])
print(integral)       
print(np.sum(res*0.5))
mantid_int = Integration(resolution)

# res = np.roll(res, -1)
yfitGaussian = gaussian(dataX, popt)
print("norm gauss:", np.sum(yfitGaussian*0.5))
yfit = signal.convolve(yfitGaussian, res, mode="same") * 0.5
print("norm convolved gauss:", np.sum(yfit*0.5))

plt.figure()
plt.plot(dataX, yfitGaussian, label="yfitGaussian")
plt.plot(dataX, res, label="resolution")
plt.plot(dataX, yfit, label="convolvedGaussian")
plt.legend()
plt.show()

myFit = CloneWorkspace(name)
myFit.dataY(0)[:] = yfit
myFit.dataX(0)[:] = dataX

myFit_int = Integration(myFit)

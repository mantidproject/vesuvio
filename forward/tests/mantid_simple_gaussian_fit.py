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

dataX_len = len(dataX)
print(dataX_len)
halfPoint = int(np.round(dataX_len/2))
# Shift not due to symetric resolution
#res[:halfPoint] = np.flip(res)[:halfPoint]
#print(res[halfPoint-5 : halfPoint])
print(dataX[halfPoint-2 : halfPoint+2])
print(res[halfPoint-2 : halfPoint+2])

# # Resolution has to be re-centered
# # Currently only works with even lenght of array
# print("centering resolution ...")
# firstZeroIdx = np.argmin(np.abs(dataX))
# print(dataX[firstZeroIdx-2 : firstZeroIdx+2]) 
# print(dataX[firstZeroIdx])
# A = dataX[:2*firstZeroIdx+2]  #+2 is for even zero points
# print(A[0], A[-1])
# 
# # Crop resolution
# res = res[:2*firstZeroIdx+2]
# 

# The shift is due to the fact that the convolution returns
# an array that has 3 points at the peak instead of 1
# solution: use mode='full', remove the maximum point from
# array, crop the full array to match the size of original func,
# and need to align the two peaks


yfitGaussian = gaussian(dataX, popt)

# Add a zero data point to function being convolved
# Output of numerical convolution will have this lenght
# yfitGaussian0 = np.pad(yfitGaussian, (0, 1), 'constant', constant_values=(0, 0))
# print(len(yfitGaussian0))
#print(yfitGaussian[halfPoint-2 : halfPoint+2])
print("norm gauss:", np.sum(yfitGaussian*0.5))

yfit = signal.convolve(yfitGaussian, res, mode="same") * 0.5
print(len(yfit))
print(yfit[:4], yfit[-4:])
#print(yfit[halfPoint-2 : halfPoint+2])
print("norm convolved gauss:", np.sum(yfit*0.5))

#take out the maximum point, lenght is now the original before padding with zero
# yfitMaxMask = yfit==np.max(yfit)
# yfit = yfit[~yfitMaxMask]

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

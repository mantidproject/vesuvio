
from os import error
import numpy as np
import numpy.testing as nptest
import matplotlib.pyplot as plt
from mantid.simpleapi import *    
from pathlib import Path
currentPath = Path(__file__).absolute().parent  # Path to the repository

from scipy import signal
from scipy import ndimage
from scipy import optimize

from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=True)

resolution = Load(r"fixatures/yspace_fit/resolution.nxs", OutputWorkspace="resolution_sum")
H_JoY_Sym = Load(r"fixatures/yspace_fit/H_JoY_Sym.nxs")
name = H_JoY_Sym.name()
resName = resolution.name()

rebinPars = "-20,0.5,20"
Rebin(InputWorkspace=name, Params=rebinPars, OutputWorkspace=name)
Rebin(InputWorkspace=resName, Params=rebinPars, OutputWorkspace=resName)


def fitProfileInYSpace(wsYSpaceSym, wsRes):
    for minimizer_sum in ('Levenberg-Marquardt','Simplex'):
        CloneWorkspace(InputWorkspace = wsYSpaceSym, OutputWorkspace = name+minimizer_sum+'_joy_sum_fitted')
        
        function='''composite=Convolution,FixResolution=true,NumDeriv=true;
        name=Resolution,Workspace=resolution_sum,WorkspaceIndex=0;
        name=UserFunction,Formula=y0+A*exp( -(x-x0)^2/2/sigma1^2)/(2*3.1415*sigma1^2)^0.5
        +A*0.054*exp( -(x-x0)^2/2/sigma2^2)/(2*3.1415*sigma2^2)^0.5,
        y0=0,x0=0,A=0.7143,sigma1=4.76, sigma2=5,   ties=(sigma1=4.76)'''

        Fit(
            Function=function, 
            InputWorkspace=name+minimizer_sum+'_joy_sum_fitted', 
            Output=name+minimizer_sum+'_joy_sum_fitted',
            Minimizer=minimizer_sum
            )
        
        ws=mtd[name+minimizer_sum+'_joy_sum_fitted_Parameters']
        print('Using the minimizer: ',minimizer_sum)
        print('Hydrogen standard deviation: ',ws.cell(4,1),' +/- ',ws.cell(4,2))

def gaussian(x, y0, x0, A, sigma):
    return y0 + A / (2*np.pi)**0.5 / sigma * np.exp(-(x-x0)**2/2/sigma**2)


def optimizedFitInYSpace(wsYSpaceSym, wsRes):
    res = wsRes.extractY()[0]
    dataY = wsYSpaceSym.extractY()[0]
    dataX = wsYSpaceSym.extractX()[0]
    dataE = wsYSpaceSym.extractE()[0]

    def convolvedGaussian(x, y0, x0, A, sigma):
        histWidths = x[1:] - x[:-1]
        gaussFunc = gaussian(x, y0, x0, A, 4.76) + gaussian(x, 0, x0, 0.054*A, sigma)
        convGauss = ndimage.convolve1d(gaussFunc, res, mode="constant") * histWidths[0]
        return convGauss
        
    p0 = [0, 0, 0.7143, 5]
    popt, pcov = optimize.curve_fit(
        convolvedGaussian, dataX, dataY, p0=p0,
        sigma=dataE
    )
    yfit = convolvedGaussian(dataX, *popt)

    print("Best fit pars from curve_fit:\n", popt)
    return popt, pcov

def fitWithMinimize(ws, wsRes):
    res = wsRes.extractY()[0]
    resX = wsRes.extractX()[0]
    resXNew = np.arange(-20, 20, 0.5)
    res = np.interp(resXNew, resX, res)
    
    wsNewRes = CloneWorkspace(wsRes)
    wsNewRes.dataY(0)[:] = res
    wsNewRes.dataX(0)[:] = resXNew
    
    dataY = ws.extractY()[0]
    dataX = ws.extractX()[0]
    dataE = ws.extractE()[0]
 
    def errorFunc(pars, res, dataX, dataY, dataE):
        yFit = convolvedGaussian(dataX, pars, res)
        chi2 = (dataY - yFit)**2 / dataE**2
        return np.sum(chi2)

    def convolvedGaussian(x, pars, res):
        histWidths = x[1:] - x[:-1]
        y0, x0, A, sigma = pars
        gaussFunc = gaussian(x, y0, x0, A, 4.76) + gaussian(x, 0, x0, 0.054*A, sigma)
        convGauss = ndimage.convolve1d(gaussFunc, res, mode="constant") * histWidths[0]
        return convGauss
    
    pars=np.array([0,0,1,5])
    result = optimize.minimize(
        errorFunc,
        pars,
        args=(res, dataX, dataY, dataE),
        method="SLSQP",
        options={"disp":True}
    )
    print("Best fit pars using scipy's minimize: \n",
            result["x"])
    fitPars = result["x"]
    yfit = convolvedGaussian(dataX, fitPars, res)
    return fitPars, yfit

ws = mtd[name]
wsRes = mtd[resName]

fitProfileInYSpace(ws, wsRes)
popt, pcov = optimizedFitInYSpace(ws, wsRes)
fitPars, yfit = fitWithMinimize(ws, wsRes)

wsYFit = CloneWorkspace(wsRes, OutputWorkspace="MyFit")
wsYFit.dataY(0)[:] = yfit
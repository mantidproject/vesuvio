
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

simple_gaussian_fit = True

def fitProfileInYSpace(wsYSpaceSym, wsRes):
    for minimizer_sum in ('Levenberg-Marquardt','Simplex'):
        CloneWorkspace(InputWorkspace = wsYSpaceSym, OutputWorkspace = name+minimizer_sum+'_joy_sum_fitted')
        
        if (simple_gaussian_fit):
            function='''composite=Convolution,FixResolution=true,NumDeriv=true;
            name=Resolution,Workspace=resolution_sum,WorkspaceIndex=0;
            name=UserFunction,Formula=y0+A*exp( -(x-x0)^2/2/sigma^2)/(2*3.1415*sigma^2)^0.5,
            y0=0,A=1,x0=0,sigma=5,   ties=()'''

        Fit(
            Function=function, 
            InputWorkspace=name+minimizer_sum+'_joy_sum_fitted', 
            Output=name+minimizer_sum+'_joy_sum_fitted',
            Minimizer=minimizer_sum
            )
        
        ws=mtd[name+minimizer_sum+'_joy_sum_fitted_Parameters']
        print('Using the minimizer: ',minimizer_sum)
        print('Hydrogen standard deviation: ',ws.cell(3,1),' +/- ',ws.cell(3,2))

    popt = [ws.cell(0,1), ws.cell(2,1), ws.cell(1,1), ws.cell(3,1)]
    pcov = [ws.cell(0,2), ws.cell(2,2), ws.cell(1,2), ws.cell(3,2)]
    return popt, pcov


def gaussian(x, y0, x0, A, sigma):
    """Gaussian centered at zero"""
    return y0 + A / (2*np.pi)**0.5 / sigma * np.exp(-(x-x0)**2/2/sigma**2)



def optimizedFitInYSpace(wsYSpaceSym, wsRes):
    res = wsRes.extractY()[0]
    resX = wsRes. extractX()[0]
    resNewX = np.arange(-20, 20, 0.5)
    res = np.interp(resNewX, resX, res)

    dataY = wsYSpaceSym.extractY()[0]
    dataX = wsYSpaceSym.extractX()[0]
    dataE = wsYSpaceSym.extractE()[0]

    def convolvedGaussian(x, y0, x0, A, sigma):

        histWidths = x[1:] - x[:-1]
        if ~ (np.max(histWidths)==np.min(histWidths)):
            raise AssertionError("The histograms widhts need to be the same for the discrete convolution to work!")

        gaussFunc = gaussian(x, y0, x0, A, sigma)
        convGauss = ndimage.convolve1d(gaussFunc, res, mode="constant") * histWidths[0]
        return convGauss
        
    p0 = [0, 0, 1, 5]
    popt, pcov = optimize.curve_fit(
        convolvedGaussian, dataX, dataY, p0=p0,
        sigma=dataE
    )
    yfit = convolvedGaussian(dataX, *popt)
    Residuals = dataY - yfit
    
    # Create Workspace with the fit results
    CreateWorkspace(DataX=np.concatenate((dataX, dataX, dataX)), 
                    DataY=np.concatenate((dataY, yfit, Residuals)), NSpec=3,
                    OutputWorkspace="CurveFitResults")
    return popt, pcov, yfit


def fitWithMinimize(ws, wsRes):
    res = wsRes.extractY()[0]
    resX = wsRes. extractX()[0]
    resNewX = np.arange(-20, 20, 0.5)
    res = np.interp(resNewX, resX, res)

    dataY = ws.extractY()[0]
    dataX = ws.extractX()[0]
    dataE = ws.extractE()[0]
 
    def errorFunc(pars, res, dataX, dataY, dataE):
        yFit = convolvedGaussian(dataX, pars, res)
        chi2 = (dataY - yFit)**2 / dataE**2
        return np.sum(chi2)

    def convolvedGaussian(x, popt, res):
        y0, x0, A, sigma = popt

        histWidths = x[1:] - x[:-1]
        if ~ (np.max(histWidths)==np.min(histWidths)):
            raise AssertionError("The histograms widhts need to be the same for the discrete convolution to work!")

        gaussFunc = gaussian(x, y0, x0, A, sigma)
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
    fitPars = result["x"]
    yfit = convolvedGaussian(dataX, fitPars, res)

    return fitPars, yfit


ws = mtd[name]
wsRes = mtd[resName]

poptOri, pcovOri = fitProfileInYSpace(ws, wsRes)
popt, pcov, yfit = optimizedFitInYSpace(ws, wsRes)
fitPars, yfitMinimize = fitWithMinimize(ws, wsRes)

# ---- somethjing dirty down below

def convolvedGaussian(x, popt, res):
    y0, x0, A, sigma = popt

    histWidths = x[1:] - x[:-1]
    if ~ (np.max(histWidths)==np.min(histWidths)):
        raise AssertionError("The histograms widhts need to be the same for the discrete convolution to work!")

    gaussFunc = gaussian(x, y0, x0, A, sigma)
    convGauss = ndimage.convolve1d(gaussFunc, res, mode="constant") * histWidths[0]
    return convGauss

res = wsRes.extractY()[0]
resX = wsRes. extractX()[0]
resNewX = np.arange(-20, 20, 0.5)
res = np.interp(resNewX, resX, res)

dataY = ws.extractY()[0]
dataX = ws.extractX()[0]
dataE = ws.extractE()[0]

print("\nOriginal parameters:\n",poptOri, 
    "\nOptimized parameters curve_fit: \n", popt,
    "\nOptimized parameters minimize_fit: \n", fitPars)

yfitOpt = convolvedGaussian(dataX, popt, res)
yfitOri = convolvedGaussian(dataX, poptOri, res)
yfitMinimize = convolvedGaussian(dataX, fitPars, res)

plt.figure()
plt.errorbar(dataX, dataY, yerr=dataE, fmt="none", label="Data")
plt.plot(dataX, yfitOpt, "-^", label="curve_fit")
plt.plot(dataX, yfitOri, "-v", label="mantid_Fit")
plt.plot(dataX, yfitMinimize, "->", label="minimize_fit")
plt.legend()
plt.show()

wsYFit = CloneWorkspace(wsRes, OutputWorkspace="MyFit")
wsYFit.dataY(0)[:] = yfitMinimize

# savepath = currentPath / ".." / "figures_poster" / "make_plots" / "data_H_NCP_fit"
# np.savez(savepath,
#         dataX = dataX, 
#         dataY = dataY,
#         fit = yfitOpt,
#         dataE = dataE,
#         )
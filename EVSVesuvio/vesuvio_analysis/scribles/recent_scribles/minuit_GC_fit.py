import numpy as np
from scipy import signal, stats
import matplotlib.pyplot as plt
from pathlib import Path
repoPath = Path(__file__).absolute().parent
from mantid.simpleapi import Load
from scipy import ndimage
import time
from iminuit import cost, Minuit


def main():
    resPath = repoPath / "wsResSmall.nxs"
    joyPath = repoPath / "wsJOYsmall.nxs"
    wsRes = Load(str(resPath), OutputWorkspace="wsRes")
    wsJOY = Load(str(joyPath), OutputWorkspace="wsJOY")

    fitProfileMinuit(wsJOY, wsRes)


def oddConvolution(x, y, res):
    # assert np.min(x) == -np.max(x), "Resolution needs to be in symetric range!"
    # assert x.size == res.size, " Resolution needs to have the same no of points as spectrum!"
    
    if x.size % 2 == 0:
        rangeRes = x.size+1  # If even change to odd
    else:
        rangeRes = x.size    # If odd, keep being odd

    xInterp = np.linspace(np.min(x), np.max(x), rangeRes)
    xDelta = xInterp[1] - xInterp[0]
    resInterp = np.interp(xInterp, x, res)

    conv = signal.convolve(y, resInterp, mode="same") * xDelta
    return conv 


def fitProfileMinuit(wsYSpaceSym, wsRes):

    dataY = wsYSpaceSym.extractY()[0]
    dataX = wsYSpaceSym.extractX()[0]
    dataE = wsYSpaceSym.extractE()[0]


    resY = wsRes.extractY()[0]
    resX = wsRes. extractX()[0]

    assert np.min(resX) == -np.max(resX), "Resolution needs to be in symetric range!"
    assert resX == dataX, "Resolution range needs to be equal to dataX"


    def convolvedModel(x, A, x0, sigma1, c4, c6):
        gramCharlier =  A * np.exp(-(x-x0)**2/2/sigma1**2) / (np.sqrt(2*3.1415*sigma1**2)) \
                *(1 + c4/32*(16*((x-x0)/np.sqrt(2)/sigma1)**4 \
                -48*((x-x0)/np.sqrt(2)/sigma1)**2+12) \
                +c6/384*(64*((x-x0)/np.sqrt(2)/sigma1)**6 \
                -480*((x-x0)/np.sqrt(2)/sigma1)**4 + 720*((x-x0)/np.sqrt(2)/sigma1)**2 - 120))
        return oddConvolution(x, gramCharlier, resY)
        # return ndimage.convolve1d(gramCharlier, resolution, mode="constant") * histWidths0

    def constrFunc(*pars):
        return convolvedModel(dataX, *pars)

    # Fit with Minuit
    costFun = cost.LeastSquares(dataX, dataY, dataE, convolvedModel)
    m = Minuit(costFun, A=1, x0=0, sigma1=4, c4=0, c6=0)
    m.limits["A"] = (0, None)
    m.limits["c4"] = (0, None)      # c4 always positive
    m.simplex()
    m.scipy(constraints=optimize.NonlinearConstraint(constrFunc, 0, np.inf))

    # Explicit calculation of Hessian after the fit
    m.hesse()
    m.minos()
    print(m)
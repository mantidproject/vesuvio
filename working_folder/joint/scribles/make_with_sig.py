import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit, cost, util
from iminuit.util import make_with_signature, describe
from scipy import signal, stats
from mantid.simpleapi import Load
from pathlib import Path
repoPath = Path(__file__).absolute().parent

def fun(x, y0, A, x0, sigma):
            return y0 + A / (2*np.pi)**0.5 / sigma * np.exp(-(x-x0)**2/2/sigma**2)


def oddEvenApproach(x, y, res):
    assert np.min(x) == -np.max(x), "Resolution needs to be in symetric range!"
    if x.size % 2 == 0:
        rangeRes = x.size+1  # If even change to odd
    else:
        rangeRes = x.size    # If odd, keep being odd

    xNew = np.linspace(np.min(x), np.max(x), rangeRes)
    xNew0 = xNew[1] - xNew[0]
    resNew = np.interp(xNew, x, res)

    yResSig = signal.convolve(y, resNew, mode="same") * xNew0
    return yResSig 


def modelOdd(x, y0, A, x0, sigma):
    gauss = y0 + A / (2*np.pi)**0.5 / sigma * np.exp(-(x-x0)**2/2/sigma**2)
    return oddEvenApproach(x, gauss, res)


resPath = repoPath / "wsResSmall.nxs"
joyPath = repoPath / "wsJOYsmall.nxs"
wsRes = Load(str(resPath), OutputWorkspace="wsRes")
wsJOY = Load(str(joyPath), OutputWorkspace="wsJOY")

dataRes = wsRes.extractY()
dataX = wsJOY.extractX()
dataY = wsJOY.extractY()
dataE = wsJOY.extractE()

# dataX = wsRes.dataX(0)
# dataE = 0.05 * np.random.random(dataX.shape)
# dataY = fun(dataX, 0, 1, 0, 5) + dataE * np.random.random(dataX.shape)

defaultPars = {
    "y0" : 0,
    "A" : 1,
    "x0" : 0,
    "sigma" : 5           
}

sharedPars = ["sigma"]

unsharedPars = [key for key in defaultPars if key not in sharedPars]
unsharedArgs = {}
totCost = 0
for i, (x, y, yerr, res) in enumerate(zip(dataX, dataY, dataE, dataRes)):

    for upar in unsharedPars:
        unsharedArgs[upar] = upar+str(i)

    def modelOdd(x, y0, A, x0, sigma):
        gauss = y0 + A / (2*np.pi)**0.5 / sigma * np.exp(-(x-x0)**2/2/sigma**2)
        return oddEvenApproach(x, gauss, res)

    costFunInterp = cost.LeastSquares(
        x, y, yerr, make_with_signature(modelOdd, **unsharedArgs)
        )
    totCost += costFunInterp

    initPars = {}
    # Add shared parameters
    for spar in sharedPars:
        initPars[spar] = defaultPars[spar] 

    # Add unshared parameters 
    for i in range(len(dataY)):
        for upar in unsharedPars:
            initPars[upar+str(i)] = defaultPars[upar]

m = Minuit(totCost, **initPars)

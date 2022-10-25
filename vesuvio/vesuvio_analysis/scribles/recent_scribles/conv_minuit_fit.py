import numpy as np
from scipy import signal, stats
import matplotlib.pyplot as plt
from pathlib import Path
repoPath = Path(__file__).absolute().parent
from mantid.simpleapi import Load
from scipy import ndimage
import time
from iminuit import cost, Minuit


def fun(x, y0, A, x0, sigma):
            return y0 + A / (2*np.pi)**0.5 / sigma * np.exp(-(x-x0)**2/2/sigma**2)

def interpApproach(x, y, res):
    dens = 10000

    xInterp = np.linspace(np.min(x), np.max(x), dens)
    resInterp = np.interp(xInterp, x, res)
    yInterp = np.interp(xInterp, x, y)
    xDelta = xInterp[1] - xInterp[0]

    convInterp = signal.convolve(yInterp, resInterp, mode="same") * xDelta

    convFinal = np.interp(x, xInterp, convInterp)
    return convFinal


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

def modelInterp(x, y0, A, x0, sigma):
    return interpApproach(x, fun(x, y0, A, x0, sigma), res)


def modelOdd(x, y0, A, x0, sigma):
    return oddEvenApproach(x, fun(x, y0, A, x0, sigma), res)

resPath = repoPath / "wsResSmall.nxs"
joyPath = repoPath / "wsJOYsmall.nxs"
wsRes = Load(str(resPath), OutputWorkspace="wsRes")
wsJOY = Load(str(joyPath), OutputWorkspace="wsJOY")

res = wsRes.dataY(0)
# dataX = wsJOY.dataX(0)
# dataY = wsJOY.dataY(0)
# dataE = wsJOY.dataE(0)

dataX = wsRes.dataX(0)
dataE = 0.05 * np.random.random(dataX.shape)
dataY = fun(dataX, 0, 1, 0, 5) + dataE * np.random.random(dataX.shape)

# Fit with Minuit
costFunInterp = cost.LeastSquares(dataX, dataY, dataE, modelInterp)
costFunOdd = cost.LeastSquares(dataX, dataY, dataE, modelOdd)

def doFit(costFun):
    m = Minuit(costFun, y0=0, A=1, x0=0, sigma=5)
    m.limits["A"] = (0, None)
    m.simplex()
    m.migrad()
    return m.values

t0 = time.time()
valuesInterp = doFit(costFunInterp)
t1 = time.time()
timeInterp = t1-t0

t0 = time.time()
valuesOdd = doFit(costFunOdd)
t1 = time.time()
timeOdd = t1-t0

print("Ratio interp / odd: ", timeInterp / timeOdd)

plt.errorbar(dataX, dataY, dataE, fmt=".", label="Data")
plt.plot(dataX, res, label="Resolution")
plt.plot(dataX, modelInterp(dataX, *valuesInterp), label="Conv Fit Interp")
plt.plot(dataX, modelOdd(dataX, *valuesOdd), label="Conv Fit Odd")
plt.vlines(0, 0, 0.3, color="k", ls="--")
plt.legend()
plt.show()

print("Interp values: ", valuesInterp)
print("Odd values: ", valuesOdd)
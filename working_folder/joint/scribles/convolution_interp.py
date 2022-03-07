import numpy as np
from scipy import signal, stats
import matplotlib.pyplot as plt
from pathlib import Path
repoPath = Path(__file__).absolute().parent
from mantid.simpleapi import Load
from scipy import ndimage
import time


def gauss(x, y0, A, x0, sigma):
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


resPath = repoPath / "wsResSmall.nxs"
wsRes = Load(str(resPath), OutputWorkspace="wsRes")

res = wsRes.dataY(0)
x = wsRes.dataX(0)
y = gauss(x, 0, 1, 0, 5)

t0 = time.time()
for i in range(100):
    convInterp = interpApproach(x, y ,res)
t1 = time.time()
timeInterp = t1-t0

t0 = time.time()
for i in range(100):
    convOdd = oddEvenApproach(x, y ,res)
t1 = time.time()
timeOdd = t1-t0

print("Ratio interp / odd: ", timeInterp / timeOdd)

plt.plot(x, y, label="Data")
plt.plot(x, res, label="Resolution")
plt.plot(x, convInterp, label="Conv Interp")
plt.plot(x, convOdd, label="Conv Odd")
plt.vlines(0, 0, 0.3, color="k", ls="--")
plt.legend()
plt.show()
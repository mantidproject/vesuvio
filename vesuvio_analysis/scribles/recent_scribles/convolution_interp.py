import numpy as np
from scipy import signal, stats
import matplotlib.pyplot as plt
from pathlib import Path
repoPath = Path(__file__).absolute().parent
from mantid.simpleapi import Load
from scipy import ndimage
import time


def model(x, y0, A, x0, sigma):
            return y0 + A / (2*np.pi)**0.5 / sigma * np.exp(-(x-x0)**2/2/sigma**2)

# Contains possible approaces for the convolution
def chooseApprch(x, res, denseConv=False, yDense=True):
    """
    Two approches: High densitiy grid or grid with odd number (peak at center)
    Choose whether the model is in grid or no grid
    """

    assert np.min(x) == -np.max(x), "Resolution needs to be in symetric range!"
    assert x.size == res.size, "x and res need to be the same size!"

    if denseConv:
        dens = 10000
    else:
        if res.size % 2 == 0:
            dens = res.size+1  # If even change to odd
        else:
            dens = res.size    # If odd, keep being odd)

    xDense = np.linspace(np.min(x), np.max(x), dens)
    xDelta = xDense[1] - xDense[0]
    resDense = np.interp(xDense, x, res)

    yModel = model(xDense, *pars) if yDense else model(x, *pars)

    convDense = signal.convolve(yModel, resDense, mode="same") * xDelta

    conv = np.interp(x, xDense, convDense) if yDense else convDense 
    return conv


resPath = repoPath / "wsResSmall.nxs"
wsRes = Load(str(resPath), OutputWorkspace="wsRes")

res = wsRes.dataY(0)
x = wsRes.dataX(0)
pars = (0.1, 1, 5, 5)

plt.plot(x, model(x, *pars), label="Data")
plt.plot(x, chooseApprch(x, res, denseConv=False, yDense=False), label="Conv Odd yDense=False")
plt.plot(x, chooseApprch(x, res, denseConv=False, yDense=True), label="Conv Odd yDense=True")
plt.plot(x, chooseApprch(x, res, denseConv=True, yDense=True), label="Conv Dense yDense=True")
plt.plot(x, chooseApprch(x, res, denseConv=True, yDense=False), label="Conv Dense yDense=False")
plt.legend()
plt.show()


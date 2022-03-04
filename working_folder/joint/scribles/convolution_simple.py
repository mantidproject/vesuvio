import numpy as np
from scipy import signal, stats
import matplotlib.pyplot as plt
from pathlib import Path
repoPath = Path(__file__).absolute().parent
from mantid.simpleapi import Load
from scipy import ndimage



def gauss(x, y0, A, x0, sigma):
            return y0 + A / (2*np.pi)**0.5 / sigma * np.exp(-(x-x0)**2/2/sigma**2)

resPath = repoPath / "wsResSmall.nxs"
wsRes = Load(str(resPath), OutputWorkspace="wsRes")

res = wsRes.dataY(0)
x = wsRes.dataX(0)
y = gauss(x, 0, 1, 0, 5)

assert np.min(x) == -np.max(x), "Resolution needs to be in symetric range!"
if x.size % 2 == 0:
    rangeRes = x.size+1  # If even change to odd
else:
    rangeRes = x.size    # If odd, keep being odd
    
xNew = np.linspace(np.min(x), np.max(x), rangeRes)

resNew = np.interp(xNew, x, res)

yResIm = ndimage.convolve1d(y, resNew, mode="constant")
yResSig = signal.convolve(y, resNew, mode="same")


plt.plot(x, y, label="Data")
plt.plot(xNew, resNew, label="Resolution")
plt.plot(x, yResIm, label="Res Image")
plt.plot(x, yResSig, label="Res Signal")
plt.vlines(0, 0, 0.3, color="k", ls="--")
plt.legend()
plt.show()


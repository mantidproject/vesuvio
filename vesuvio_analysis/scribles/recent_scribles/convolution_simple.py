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

rangeRes = x.size
xNew = np.linspace(np.min(x), np.max(x), rangeRes)
xNew0 = xNew[1] - xNew[0]
x0 = x[1] - x[0]
resNew = np.interp(xNew, x, res)

yResIm = ndimage.convolve1d(y, resNew, mode="constant") * xNew0
yResSig = signal.convolve(y, resNew, mode="same") * x0


plt.plot(x, y, label="Data")
plt.plot(xNew, resNew, label="Resolution")
plt.plot(x, yResIm, label="Res Image")
plt.plot(x, yResSig, label="Res Signal")

rangeRes = x.size+1
xNew = np.linspace(np.min(x), np.max(x), rangeRes)
x0 = xNew[1] - xNew[0]
resNew = np.interp(xNew, x, res)
yResSig = signal.convolve(y, resNew, mode="same") * x0
plt.plot(x, yResSig, "--", linewidth=3, label="Res Signal +1")

# rangeRes = x.size-1
# xNew = np.linspace(np.min(x), np.max(x), rangeRes)
# x0 = xNew[1] - xNew[0]

# resNew = np.interp(xNew, x, res)
# yResSig = signal.convolve(y, resNew, mode="same") * x0
# plt.plot(x, yResSig, label="Res Signal -1")

plt.vlines(0, 0, 0.3, color="k", ls="--")
plt.legend()
plt.show()


fig, ax = plt.subplots(1)
ax.plot(x, res, label="Original resolution")
ax.plot(xNew, resNew, label="Interpolated resolution")
plt.legend()
plt.show()
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
from mantid.simpleapi import Load
from pathlib import Path
repoPath = Path(__file__).absolute().parent

def model(x, y0, A, x0, sigma):
            return y0 + A / (2*np.pi)**0.5 / sigma * np.exp(-(x-x0)**2/2/sigma**2)

resPath = repoPath / "wsResSmall.nxs"
wsRes = Load(str(resPath), OutputWorkspace="wsRes")

res = wsRes.dataY(0)
x = wsRes.dataX(0)
pars = [0, 1, 1, 5]
# y = model(x, *pars)

dens = 10001

xDense = np.linspace(np.min(x), np.max(x), dens)
xDelta = xDense[1] - xDense[0]
resDense = np.interp(xDense, x, res)

convDenseSig = signal.convolve(model(xDense, *pars), resDense, mode="same") * xDelta
convSig = np.interp(x, xDense, convDenseSig)

convDenseNp = np.convolve(model(xDense, *pars), resDense) * xDelta
convDenseNp = convDenseNp[int(np.floor(dens/2)-1):-int(np.floor(dens/2))]
print(convDenseNp.size)
convNp = np.interp(x, xDense, convDenseNp)

plt.plot(x, convSig, label="signal")
plt.plot(x, convNp, label="np")
plt.legend()
plt.show()

np.testing.assert_allclose(convSig, convNp)
print("Arrays Equal!")

import numpy as np
import numpy.testing as nptest
import matplotlib.pyplot as plt

from scipy import signal
from scipy import ndimage

from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=True)

def gaussian(x, y0, x0, A, sigma):
    """Gaussian centered at zero"""
    return y0 + A / (2*np.pi)**0.5 / sigma * np.exp(-(x-x0)**2/2/sigma**2)
    
# Generate gaussian

x = np.arange(-50.5, 50.5)
y = gaussian(x, 0, 10, 1, 10)
print("Integral of function: ", np.sum(y))

res = gaussian(x, 0, 0, 1, 3)
print("Integral of resolution before interpolation: ", np.sum(res))
xNew = x + 0.5
resNew = np.interp(xNew, x, res)
res = resNew
print("Integral of resolution after interpolation: ", np.sum(res))

conv_signal = signal.convolve(y, res, mode="same")
conv_ndimage = ndimage.convolve1d(y, res, mode="constant")

plt.plot(x, y, "o", label="Function")
plt.plot(x, res, label="Resolution")
plt.plot(x, conv_ndimage, "o", label="ndimage.concolve1d")
plt.plot(x, conv_signal, "^", label="signal.convolve")
plt.legend()
plt.show()
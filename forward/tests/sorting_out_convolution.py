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

x = np.arange(-50, 50)
y = gaussian(x, 0, 10, 1, 10)
print("Integral: ", np.sum(y))

res = gaussian(x, 0, -10, 1, 3)

# convolution = signal.convolve(y, res, mode="same")
convolution = ndimage.convolve1d(y, res, mode="constant")
print(convolution.shape)

plt.plot(x, y, "o", label="Function")
plt.plot(x, res, label="Resolution")
plt.plot(x, convolution, "o", label="Convolution")
plt.legend()
plt.show()
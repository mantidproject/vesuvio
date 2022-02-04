
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

# Write a function that performs the numerical convolution between two functions
def myNumConvolution(f, h):
    h = np.flip(h)
    convLen = len(f) + len(h)

    h = np.pad(h, (len(f),0), 'constant', constant_values=(0,0))
    f = np.pad(f, (0, len(h)), 'constant', constant_values=(0,0))
    
    result = np.zeros(convLen)
    for n in range(1, convLen):    # Can not start at zero
        result[n-1] = np.sum(h[-n:]*f[:n])
    return result

h = np.ones(5)/5   # Normalized square signal
x = np.arange(100)

def gaussian(x, x0, A, sigma):
    """Gaussian centered at zero"""
    return A / (2*np.pi)**0.5 / sigma * np.exp(-(x-x0)**2/2/sigma**2)

f = gaussian(x, 30, 1, 3)
# f /= np.sum(f)
# print(np.sum(f))

myConv = myNumConvolution(f, h)
npConv = np.convolve(f, h)
scipyConv = signal.convolve(f, h)
# Test my function to the numpy equivalent
np.testing.assert_almost_equal(myConv[:-1], npConv, decimal=15)
np.testing.assert_almost_equal(scipyConv, npConv, decimal=15)
print("Discrete Convolution, mode='full':\n",
        "My convolution gives same results as the numpy algorithm")

scipyConv = signal.convolve(f, h, mode="same")
print("Sum of conv: ", np.sum(scipyConv))
plt.plot(x, f)
#plt.plot(np.arange(-10, -5), h)
plt.plot(x, scipyConv)


#%%
import numpy as np
import matplotlib.pyplot as plt

# Write a function that performs the numerical convolution between two functions
def myNumConvolution(f, h):
    h = np.flip(h)
    convLen = len(f) + len(h)
    print(convLen)

    h = np.pad(h, (len(f),0), 'constant', constant_values=(0,0))
    f = np.pad(f, (0, len(h)), 'constant', constant_values=(0,0))
    print(h.shape, f.shape)
    result = np.zeros(convLen)
    for n in range(1, convLen):    # Can not start at zero
        result[n-1] = np.sum(h[-n:]*f[:n])
    return result

h = np.ones(5)
fx = np.arange(15)
f = np.exp(-fx*0.1)

myConv = myNumConvolution(f, h)
npConv = np.convolve(f, h)

# Test my function to the numpy equivalent
np.testing.assert_almost_equal(myConv[:-1], npConv, decimal=15)

#%%
plt.plot(fx, f)
plt.plot(np.arange(-10, -5), h)
plt.plot(np.arange(19), npConv)
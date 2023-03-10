import numpy as np
import matplotlib.pyplot as plt


def gaussian(x, sigma):
    """Gaussian function centered at zero"""
    gaussian = np.exp(-x**2/2/sigma**2)
    gaussian /= np.sqrt(2.*np.pi)*sigma
    return gaussian


def lorentizian(x, gamma):
    """Lorentzian centered at zero"""
    lorentzian = gamma/np.pi / (x**2 + gamma**2)
    return lorentzian


def pseudoVoigt(x, sigma, gamma):
    """Convolution between Gaussian with std sigma and Lorentzian with HWHM gamma"""
    fg, fl = 2.*sigma*np.sqrt(2.*np.log(2.)), 2.*gamma
    f = 0.5346 * fl + np.sqrt(0.2166*fl**2 + fg**2)
    eta = 1.36603 * fl/f - 0.47719 * (fl/f)**2 + 0.11116 * (fl/f)**3
    sigma_v, gamma_v = f/(2.*np.sqrt(2.*np.log(2.))), f / 2.
    pseudo_voigt = eta * lorentizian(x, gamma_v) + (1.-eta) * gaussian(x, sigma_v)
    return pseudo_voigt 
        


x = np.linspace(-20, 20, 100)
voigt = pseudoVoigt(x, 7, 2)

norm1 = np.abs(np.sum(voigt) * (x[1] - x[0]))
norm2 = np.abs(np.trapz(voigt, x))

plt.plot(x, voigt, label="voigt")
plt.plot(x, voigt/norm1, label="voigt norm sum")
plt.plot(x, voigt/norm2, label="voigt norm trapz")
plt.legend()
plt.show()
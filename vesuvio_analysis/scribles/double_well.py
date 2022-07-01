from cProfile import label
from distutils import errors
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit, cost


def model(x, A, d, R, sig1, sig2):

    h = 2.04
    theta = np.linspace(0, np.pi, 300)[:, np.newaxis]
    y = x[np.newaxis, :]

    sigTH = np.sqrt( sig1**2*np.cos(theta)**2 + sig2**2*np.sin(theta)**2 )

    alpha = 2*( d*sig2*sig1*np.sin(theta) / sigTH )**2
    beta = ( 2*sig1**2*d*np.cos(theta) / sigTH**2 ) * y

    denom = 2.506628 * sigTH * (1 + R**2 + 2*R*np.exp(-2*d**2*sig1**2))
    jp = np.exp( -y**2/(2*sigTH**2)) * (1 + R**2 + 2*R*np.exp(-alpha)*np.cos(beta)) / denom
    jp *= np.sin(theta)

    JBest = np.trapz(jp, x=theta, axis=0)
    JBest /= np.abs(np.trapz(JBest, x=y))
    JBest *= A
    return JBest

defaultPars = {
    "A" : 1,
    "d" : 1,
    "sig1" : 3,
    "sig2" : 5,
    "R" : 1
}

y = np.linspace(-20, 20, 100)
result = model(y, **defaultPars)
noise = np.random.random(len(y)) - 0.5  # Half positive and half negative
data = result + noise * 0.02
error = noise * 0.04
print(y, "\n", data, "\n", error)

costFun = cost.LeastSquares(y, data, error, model)
m = Minuit(costFun, **defaultPars)
m.simplex()
m.migrad()

print(m.covariance.correlation())

leg = ""
for p, v, e in zip(defaultPars, m.values, m.errors):
    leg += f"{p}: {v:.2f} +/- {e:.2f}\n"

plt.errorbar(y, data, error, fmt=".")
plt.plot(y, model(y, *m.values), label=leg)
plt.legend()
plt.show()

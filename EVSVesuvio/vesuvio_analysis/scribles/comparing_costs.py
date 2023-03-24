from iminuit import Minuit, describe, cost
import numpy as np
import matplotlib.pyplot as plt
from iminuit.util import make_func_code


# Comparison between making my own cost function to use on Minuit and 
# Minuits own cost function
# Gives same result, which is comforting

class LeastSquares:   # This is a great way to generalize making least squares functions
    """
    Generic least-squares cost function with error.
    """

    errordef = Minuit.LEAST_SQUARES # for Minuit to compute errors correctly
    
    def __init__(self, model, x, y, err):
        self.model = model  # model predicts y for given x
        self.x = np.asarray(x)
        self.y = np.asarray(y)
        self.err = np.asarray(err)
        self.func_code = make_func_code(describe(model)[1:])

    def __call__(self, *par):  # we accept a variable number of model parameters
        ym = self.model(self.x, *par)
        return np.sum((self.y - ym) ** 2 / self.err ** 2)

    @property
    def ndata(self):
        return len(self.x)


# Make some fake data

def model(x, A, x0, sigma):
    return  A / (2*np.pi)**0.5 / sigma * np.exp(-(x-x0)**2/2/sigma**2)

x = np.linspace(-20, 20, 100)
err = np.random.normal(0, 0.1, len(x))
y = model(x, 15, 2, 5) + err
defaultPars  = {"A":1, "x0":0, "sigma":2}

minlq = cost.LeastSquares(x, y, err, model)
mylq = LeastSquares(model, x, y ,err)

m = Minuit(minlq, **defaultPars)
m.migrad()
mlabel = "Cost LQ"
for p, v, e in zip(m.parameters, m.values, m.errors):
    mlabel+=f"\n{p}={v:.2f} pm {e:.2f}"


my = Minuit(mylq, **defaultPars)
my.migrad()
mylabel = "My LQ"
for p, v, e in zip(my.parameters, my.values, my.errors):
    mylabel+=f"\n{p}={v:.2f} pm {e:.2f}"


plt.errorbar(x, y, err, fmt="o")
plt.plot(x, model(x, *m.values), label=mlabel)
plt.plot(x, model(x, *my.values), label=mylabel)
plt.legend()
plt.show()




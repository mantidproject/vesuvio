# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# everything in iminuit is done through the Minuit object, so we import it
from iminuit import Minuit, cost, util

# we also need a cost function to fit and import the LeastSquares function
from iminuit.cost import LeastSquares

# display iminuit version
import iminuit
print("iminuit version:", iminuit.__version__)
from iminuit.util import describe, make_func_code
import numba as nb

#%%
@nb.njit()
def HermitePolynomial(x, A, x0, sigma1, c4, c6):
    func = A * np.exp(-(x-x0)**2/2/sigma1**2) / (np.sqrt(2*3.1415*sigma1**2)) \
            *(1 + c4/32*(16*((x-x0)/np.sqrt(2)/sigma1)**4 \
            -48*((x-x0)/np.sqrt(2)/sigma1)**2+12) \
            +c6/384*(64*((x-x0)/np.sqrt(2)/sigma1)**6 \
            -480*((x-x0)/np.sqrt(2)/sigma1)**4 + 720*((x-x0)/np.sqrt(2)/sigma1)**2 - 120)) 
    return func

def gaussian(x, y0, A, x0, sigma):
        return y0 + A / (2*np.pi)**0.5 / sigma * np.exp(-(x-x0)**2/2/sigma**2)


np.random.seed(1)
x = np.linspace(-20, 20, 100)
yerr = np.random.rand(x.size) * 0.03
dataY = HermitePolynomial(x, 1, 0, 5, 0.1, 0.2) + yerr * np.random.randn(x.size)

# @nb.njit()
# def myCost(*pars):
#     return np.sum(np.square((dataY-HermitePolynomial(x, *pars)/yerr)))

# myCost.func_code = make_func_code(["A", "x0", "sigma1", "c4", "c6"])

# m = Minuit(myCost, A=1, x0=0, sigma1=4, c4=0, c6=0)
# m.errordef = m.LEAST_SQUARES
# m.limits["A"] = (0, None)


HermitePolynomial(x, 1, 0, 5, 0.1, 0.2)
costFun = cost.LeastSquares(x, dataY, yerr, HermitePolynomial)
m = Minuit(costFun, A=1, x0=0, sigma1=4, c4=0, c6=0)
m.limits["A"] = (0, None)

costFun(1, 0, 5, 0.1, 0.2)

# Run some minuit functions for JIT compilation
m.fcn([1, 0, 5, 0.1, 0.2])
m.grad
m.hesse()

values = list(m.values)
print(values)


#%%
%%timeit -r 5 -n 1
m.simplex()
#%%
%%timeit -r 5 -n 1
m.migrad()
#%%
%%timeit -r 5 -n 1
constraints = optimize.NonlinearConstraint(lambda *pars: HermitePolynomial(x, *pars), 0, np.inf)
m.scipy()
m.scipy(constraints=constraints)
#%%
%%timeit -r 5 -n 1
m.hesse()
# %%
print(values)

# %%

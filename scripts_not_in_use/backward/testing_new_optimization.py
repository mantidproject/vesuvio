
#%%
#Test the partitioned least squares method

import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
plt.style.use('dark_background')

noise = np.random.randn(100)
x = np.linspace(-1, 1, 100)
y = 2 - 1 * np.exp(-1.5*x) + noise * 0.1

plt.plot(x, y, ".")

def f(par, x):
    return par[0] + par[1]*np.exp(par[2]*x)

def errFunc(par, x, y):
    return np.sum(np.square(y - f(par, x)))

initPars = [1, 2, -3]
fitResult = optimize.minimize(
    errFunc, 
    initPars, 
    args=(x, y), 
    options={"disp":True}
    )
fitc = fitResult["x"]
print(fitc)

fitY = f(fitc, x)
plt.plot(x, fitY)

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import linalg 
plt.style.use('dark_background')

noise = np.random.randn(100)
x = np.linspace(-1, 1, 100)
y = 2 - 1 * np.exp(-1.5*x) + noise * 0.1

plt.plot(x, y, ".")

def f(linPars, c, x):
    return linPars[0] + linPars[1]*np.exp(c*x)

def errFunc(c, x, y):

    # Now that we fixed c, calculate linear parameters
    p, res, rnk, s = fitLinPars(c, x, y)

    return np.sum(np.square(y - f(p, c, x)))

def fitLinPars(c, x, y):
    M = x[:, np.newaxis] * np.ones((1,2))
    M[:, 0] = M[:, 0] ** 0
    M[:, 1] = np.exp(c * M[:, 1]) 

    return linalg.lstsq(M, y)

initc = 1
fitResult = optimize.minimize(
    errFunc, 
    initc, 
    args=(x, y), 
    options={"disp":True}
    )
fitc = fitResult["x"]
print(fitc)

fitp, res, rnk, s = fitLinPars(fitc, x, y)

fitY = f(fitp, fitc, x)
plt.plot(x, fitY)

#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy import linalg 
plt.style.use('dark_background')

import cvxopt as co

noise = np.random.randn(100)
x = np.linspace(-1, 1, 100)
y = 2 - 1 * np.exp(-1.5*x) + noise * 0.1

plt.plot(x, y, ".")

def f(linPars, c, x):
    return linPars[0] + linPars[1]*np.exp(c*x)

def errFunc(c, x, y):

    # Now that we fixed c, calculate linear parameters
    p = fitLinPars(c, x, y)
    p = np.array(p)

    return np.sum(np.square(y - f(p, c, x)))

def fitLinPars(c, x, y):
    M = x[:, np.newaxis] * np.ones((1,2))
    M[:, 0] = M[:, 0] ** 0
    M[:, 1] = np.exp(c * M[:, 1]) 

    #Least squares problem can be written as a quadratic program
    P = co.matrix(M.T @ M, tc="d")
    q = co.matrix(-y.T @ M, tc = "d")

    sol = co.solvers.qp(P,q)
    return sol["x"]

initc = 1
fitResult = optimize.minimize(
    errFunc, 
    initc, 
    args=(x, y), 
    options={"disp":True}
    )
fitc = fitResult["x"]
print(fitc)

fitp = fitLinPars(fitc, x, y)

fitY = f(fitp, fitc, x)
plt.plot(x, fitY)

# %%

from pydoc import describe
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit, cost, util
from iminuit.util import describe
from scipy import optimize


def runManualMinos(minuitObj, constrFunc, var:str, bound:int):

    varVal = minuitObj.values[var]
    varErr = minuitObj.errors[var]

    # Before any modification to minuitObj, store fval of best fit
    fValsMin = minuitObj.fval      # Used to calculate error bands at the end

    # Initiate arrays
    varSpace = np.linspace(varVal - bound*varErr, varVal + bound*varErr, 30)
    fValsScipy = np.zeros(varSpace.shape)
    fValsMigrad = np.zeros(varSpace.shape)

    minuitObj.fixed[var] = True        
    for i in range(fValsScipy.size):
        minuitObj.values[var] = varSpace[i]  

        minuitObj.migrad()        # Unconstrained fit
        fValsMigrad[i] = minuitObj.fval

        # Constrained fit
        minuitObj.scipy(constraints=optimize.NonlinearConstraint(constrFunc, 0, np.inf))
        fValsScipy[i] = minuitObj.fval
    
    # Use intenpolation to create dense array of fmin values 
    denseVarSpace = np.linspace(np.min(varSpace), np.max(varSpace), 1000)
    denseFVals = np.interp(denseVarSpace, varSpace, fValsScipy)

    # Calculate points of intersection with line delta fmin val = 1
    idxErr = np.argwhere(np.diff(np.sign(denseFVals - fValsMin - 1)))
    lerr, uerr = denseVarSpace[idxErr]

    # Plot results
    fig, ax = plt.subplots(1)
    ax.plot(denseVarSpace, denseFVals, label="fVals Constr")
    ax.plot(varSpace, fValsMigrad, label="fVals Unconstr")

    ax.hlines(fValsMin+1, np.min(varSpace), np.max(varSpace))
    ax.hlines(fValsMin, np.min(varSpace), np.max(varSpace))

    ax.axvspan(lerr, uerr, alpha=0.2, color="red", label="Manual Minos error")
    ax.axvspan(varVal-varErr, varVal+varErr, alpha=0.2, color="grey", label="Hessian Std error")
    
    ax.vlines(varVal, np.min(fValsMigrad), np.max(fValsScipy), "k", "--")
    plt.legend()
    plt.show()

    return denseVarSpace[idxErr] - varVal  # Array with lower and upper asymetric errors



def HermitePolynomial(x, A, x0, sigma1, c4, c6):
    func = A * np.exp(-(x-x0)**2/2/sigma1**2) / (np.sqrt(2*3.1415*sigma1**2)) \
            *(1 + c4/32*(16*((x-x0)/np.sqrt(2)/sigma1)**4 \
            -48*((x-x0)/np.sqrt(2)/sigma1)**2+12) \
            +c6/384*(64*((x-x0)/np.sqrt(2)/sigma1)**6 \
            -480*((x-x0)/np.sqrt(2)/sigma1)**4 + 720*((x-x0)/np.sqrt(2)/sigma1)**2 - 120)) 
    return func

def constrFunc(*pars):
    return HermitePolynomial(x, *pars)

np.random.seed(1)
x = np.linspace(-20, 20, 100)
yerr = np.random.rand(x.size) * 0.01
dataY = HermitePolynomial(x, 1, 0, 5, 1, 1) + yerr * np.random.randn(x.size)

costFun = cost.LeastSquares(x, dataY, yerr, HermitePolynomial)
constraints = optimize.NonlinearConstraint(constrFunc, 0, np.inf)

m = Minuit(costFun, A=1, x0=0, sigma1=6, c4=0, c6=0)
m.limits["A"] = (0, None)
m.simplex()
m.scipy(constraints=constraints)
m.hesse()

# try:
#     m.minos()

# except (RuntimeError, TypeError):
print("\nAutomatic MINOS failed because constr result away from minima")
print("Running Manual implementation of MINOS ...\n")

runManualMinos(m, constrFunc, "sigma1", 2)


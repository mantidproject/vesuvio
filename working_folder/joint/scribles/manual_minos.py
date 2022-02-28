from pydoc import describe
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit, cost, util
from iminuit.util import describe
from scipy import optimize


def runManualMinos(minuitObj, costFun, initPars:dict, constraints, posPars:list, var:str, bound:int):

    varVal = minuitObj.values[var]
    varErr = minuitObj.errors[var]

    print("\nValue ", var, ": ", varVal)
    print("Err ", var, ": ", varErr, "\n\n")

    varSpace = np.linspace(varVal - bound*varErr, varVal + bound*varErr, 30)
    # Initialize array to store the values of minimum chi2
    fVals = np.zeros(varSpace.shape)

    for i in range(fVals.size):
        # Change initial parameter in dictionary
        # ----- Create dictionary from best fit parameters of minuitObj
        initPars[var] = varSpace[i]  

        m = Minuit(costFun, **initPars)
        m.limits[posPars] = (0, None)    # Copy limits from minuitObj
        m.fixed[var] = True        
        # print(m.parameters)
        # print(m.values)
        # print(describe(costFun))

        m.simplex()
        # m.migrad()
        m.scipy(constraints=constraints)
        fVals[i] = m.fval
    
    # Use intenpolation to create dense array of fmin values 
    denseVarSpace = np.linspace(np.min(varSpace), np.max(varSpace), 1000)
    denseFVals = np.interp(denseVarSpace, varSpace, fVals)

    # Calculate points of intersection with line delta fmin val = 1
    fValsMin = minuitObj.fval    
    idxErr = np.argwhere(np.diff(np.sign(denseFVals - fValsMin - 1)))

    # Plot results
    lerr, uerr = denseFVals[idxErr]
    fig, ax = plt.subplots(1)
    ax.plot(denseVarSpace, denseFVals, label="fVals")
    ax.axvspan(lerr, uerr, alpha=0.2, color="red", label="Manual Minos error")
    ax.vlines(varVal, np.min(fVals), np.max(fVals), "k", "--")
    plt.show()

    return denseFVals[idxErr]   # Array with lower and upper asymetric errors



def HermitePolynomial(x, A, x0, sigma1, c4, c6):
    func = A * np.exp(-(x-x0)**2/2/sigma1**2) / (np.sqrt(2*3.1415*sigma1**2)) \
            *(1 + c4/32*(16*((x-x0)/np.sqrt(2)/sigma1)**4 \
            -48*((x-x0)/np.sqrt(2)/sigma1)**2+12) \
            +c6/384*(64*((x-x0)/np.sqrt(2)/sigma1)**6 \
            -480*((x-x0)/np.sqrt(2)/sigma1)**4 + 720*((x-x0)/np.sqrt(2)/sigma1)**2 - 120)) 
    return func


np.random.seed(1)
x = np.linspace(-20, 20, 100)
yerr = np.random.rand(x.size) * 0.01
dataY = HermitePolynomial(x, 1, 0, 5, 1, 1) + yerr * np.random.randn(x.size)


costFun = cost.LeastSquares(x, dataY, yerr, HermitePolynomial)
constraints = optimize.NonlinearConstraint(lambda *pars: HermitePolynomial(x, *pars), 0, np.inf)
kwd = {
    "A" : 1,
    "x0" : 0,
    "sigma1" : 6,
    "c4" : 0,
    "c6" : 0
}
posPars = ["A"]


m = Minuit(costFun, **kwd)
m.limits[posPars] = (0, None)
m.simplex()
m.scipy(constraints=constraints)
m.hesse()

# try:
#     m.minos()

# except (RuntimeError, TypeError):
print("\nAutomatic MINOS failed because constr result away from minima")
print("Running Manual implementation of MINOS ...\n")

runManualMinos(m, costFun, kwd, constraints, posPars, "sigma1", 2)


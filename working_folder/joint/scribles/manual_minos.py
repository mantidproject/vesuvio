from pydoc import describe
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit, cost, util
from iminuit.util import describe
from scipy import optimize
from mantid import plots

def runAndPlotManualMinos(minuitObj):
    # Set format of subplots
    height = 2
    width = int(np.ceil(len(minuitObj.parameters)/2))
    figsize = (12, 7)
    # Output plot to Mantid
    fig, axs = plt.subplots(height, width, tight_layout=True, figsize=figsize)  #subplot_kw={'projection':'mantid'}
    fig.canvas.set_window_title("Manual Implementation of Minos algorithm")

    merrors = {}
    for p, ax in zip(minuitObj.parameters, axs.flat):
        lerr, uerr = runMinosForPar(m, constrFunc, p, 2, ax)
        merrors[p] = np.array([lerr, uerr])

    # Hide plots not in use:
    for ax in axs.flat:
        if not ax.lines:   # If empty list
            ax.set_visible(False)

    # ALl axes share same legend, so set figure legend to first axis
    handle, label = axs[0, 0].get_legend_handles_labels()
    fig.legend(handle, label, loc='lower right')
    plt.show()


def runMinosForPar(minuitObj, constrFunc, var:str, bound:int, ax):

    minuitObj.migrad()
    minuitObj.scipy(constraints=optimize.NonlinearConstraint(constrFunc, 0, np.inf))
    minuitObj.hesse()

    varVal = minuitObj.values[var]
    varErr = minuitObj.errors[var]
    # Store fval of best fit
    fValsMin = minuitObj.fval      # Used to calculate error bands at the end

    # Initiate arrays
    varSpace = np.linspace(varVal - bound*varErr, varVal + bound*varErr, 30)
    fValsScipy = np.zeros(varSpace.shape)
    fValsMigrad = np.zeros(varSpace.shape)

    # Run Minos algorithm
    minuitObj.fixed[var] = True        # Variable to be fixed at each iteration
    for i in range(fValsScipy.size):
        minuitObj.values[var] = varSpace[i]      # Fix variable

        minuitObj.migrad()        # Unconstrained fit
        fValsMigrad[i] = minuitObj.fval

        # Constrained fit
        minuitObj.scipy(constraints=optimize.NonlinearConstraint(constrFunc, 0, np.inf))
        fValsScipy[i] = minuitObj.fval
    
    minuitObj.fixed[var] = False    # Release variable       

    # Use intenpolation to create dense array of fmin values 
    varSpaceDense = np.linspace(np.min(varSpace), np.max(varSpace), 1000)
    fValsScipyDense = np.interp(varSpaceDense, varSpace, fValsScipy)
    # Calculate points of intersection with line delta fmin val = 1
    idxErr = np.argwhere(np.diff(np.sign(fValsScipyDense - fValsMin - 1)))
    
    if idxErr.size != 2:    # Intersections not found, there is an error somewhere
        lerr, uerr = 0, 0   
    else:
        lerr, uerr = varSpaceDense[idxErr].flatten() - varVal

    ax.plot(varSpaceDense, fValsScipyDense, label="fVals Constr Scipy")
    plotProfile(ax, var, varSpace, fValsMigrad, lerr, uerr, fValsMin, varVal, varErr)
  
    return lerr, uerr


def plotAutoMinos(minuitObj):
    # Set format of subplots
    height = 2
    width = int(np.ceil(len(minuitObj.parameters)/2))
    figsize = (12, 7)
    # Output plot to Mantid
    fig, axs = plt.subplots(height, width, tight_layout=True, figsize=figsize)  #subplot_kw={'projection':'mantid'}
    fig.canvas.set_window_title("Plot of automatic Minos algorithm")

    for p, ax in zip(minuitObj.parameters, axs.flat):
        loc, fvals, status = minuitObj.mnprofile(p, bound=2)
        

        minfval = minuitObj.fval
        minp = minuitObj.values[p]
        hessp = minuitObj.errors[p]
        lerr = m.merrors[p].lower
        uerr = m.merrors[p].upper
        plotProfile(ax, p, loc, fvals, lerr, uerr, minfval, minp, hessp)

    # Hide plots not in use:
    for ax in axs.flat:
        if not ax.lines:   # If empty list
            ax.set_visible(False)

    # ALl axes share same legend, so set figure legend to first axis
    handle, label = axs[0, 0].get_legend_handles_labels()
    fig.legend(handle, label, loc='lower right')
    plt.show()   


def plotProfile(ax, var, varSpace, fValsMigrad, lerr, uerr, fValsMin, varVal, varErr):
    """
    Plots likelihood profilef for the Migrad fvals.
    x: varSpace;
    y: fValsMigrad
    """

    ax.set_title(var+f" = {varVal:.3f} {lerr:.3f} {uerr:+.3f}")

    ax.plot(varSpace, fValsMigrad, label="fVals Migrad")

    ax.axvspan(lerr+varVal, uerr+varVal, alpha=0.2, color="red", label="Manual Minos error")
    ax.axvspan(varVal-varErr, varVal+varErr, alpha=0.2, color="grey", label="Hessian Std error")
    
    ax.axvline(varVal, 0.03, 0.97, color="k", ls="--")
    ax.axhline(fValsMin+1, 0.03, 0.97, color="k")
    ax.axhline(fValsMin, 0.03, 0.97, color="k")
   


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
# m.migrad()
m.hesse()

try:
    m.minos()
    plotAutoMinos(m)

except RuntimeError:
    print("\nAutomatic MINOS failed because constr result away from minima")
    print("Running Manual implementation of MINOS ...\n")
    
    runAndPlotManualMinos(m)

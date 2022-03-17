from email.policy import default
import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit, cost, util
from iminuit.util import make_with_signature, describe, make_func_code
from pathlib import Path
from mantid.simpleapi import Load, CropWorkspace
from scipy import optimize
from scipy import ndimage, signal
import time
import numba as nb

repoPath = Path(__file__).absolute().parent
nbkwds = {
    "parallel" : False,
    "fastmath" : True
}

def main():

    # joyPath = repoPath / "wsJOYsmall.nxs"
    # resPath = repoPath / "wsResSmall.nxs"
    joyPath = repoPath / "wsDHMTjoy.nxs"
    resPath = repoPath / "wsDHMTres.nxs"
    
    wsJoY = Load(str(joyPath), OutputWorkspace="wsJoY")
    wsRes = Load(str(resPath), OutputWorkspace="wsRes")
    print("No of spec res: ", wsRes.getNumberHistograms())

    wsJoY = CropWorkspace(InputWorkspace="wsJoY", OutputWorkspace="wsJoY", StartWorkspaceIndex=0, EndWorkspaceIndex=10)
    wsRes = CropWorkspace(InputWorkspace="wsRes", OutputWorkspace="wsRes", StartWorkspaceIndex=0, EndWorkspaceIndex=10)
    print("No of spec res: ", wsRes.getNumberHistograms())

    fitGlobalFit(wsJoY, wsRes, False)
    plt.show()


def fitGlobalFit(ws, wsRes, gaussFlag):
    axs = plotData(ws)
    
    dataY = ws.extractY()
    dataE = ws.extractE()
    dataX = ws.extractX()
    dataRes = wsRes.extractY()

    # Removed masked spectra so that zeros are not fitted
    zerosRowMask = np.all(dataY==0, axis=1)
    dataY = dataY[~zerosRowMask]
    dataE = dataE[~zerosRowMask]
    dataX = dataX[~zerosRowMask]
    dataRes = dataRes[~zerosRowMask]

    # TODO: Possible symetrisation goes here

    model, defaultPars, sharedPars = selectModelAndPars(gaussFlag)   
 
    totCost = 0
    for i, (x, y, yerr, res) in enumerate(zip(dataX, dataY, dataE, dataRes)):
        totCost += calcCostFun(model, i, x, y, yerr, res, sharedPars)
    
    print("\n", describe(totCost))

    initPars = {}
    # Populate with initial shared parameters
    for sp in sharedPars:
        initPars[sp] = defaultPars[sp]
    # Add initial unshared parameters
    unsharedPars = [key for key in defaultPars if key not in sharedPars]
    for up in unsharedPars:
        for i in range(len(dataY)):
            initPars[up+str(i)] = defaultPars[up]

    m = Minuit(totCost, **initPars)

    for i in range(len(dataY)):     # Limit for both Gauss and Gram Charlier
        m.limits["A"+str(i)] = (0, np.inf)

    t0 = time.time()
    if gaussFlag:

        #TODO Ask about the limits on y0
        for i in range(len(dataY)):
            m.limits["y0"+str(i)] = (0, np.inf)

        m.simplex()
        m.migrad() 

    else:

        x = dataX[0]
        def constr(*pars):
            """Constraint for positivity of Gram Carlier.
            Input: All parameters defined in original function.
            Format *pars to work with Minuit.
            x is defined outside function.
            Builds array with all constraints from individual functions."""

            sharedPars = pars[:3]    # sigma1, c4, c6
            joinedGC = np.zeros(int((len(pars)-3)/2) * x.size)
            for i, (A, x0) in enumerate(zip(pars[3::2], pars[4::2])):
                joinedGC[i*x.size : (i+1)*x.size] = model(x, *sharedPars, A, x0)
            
            # if np.any(joinedGC==0):
            #     raise ValueError("Args where zero: ", np.argwhere(joinedGC==0))
            
            return joinedGC

        m.simplex()
        m.scipy(constraints=optimize.NonlinearConstraint(constr, 0, np.inf))
    
    m.hesse()
    t1 = time.time()
    print(f"Running time of fitting: {t1-t0:.2f} seconds")
    print("Value of minimum: ", m.fval, "\n")
    for p, v, e in zip(m.parameters, m.values, m.errors):
        print(f"{p:7s} = {v:7.3f} +/- {e:7.3f}")
    plotFit(ws.extractX(), totCost, m, axs)
    return 


def selectModelAndPars(gaussFlag):
    if gaussFlag:
        def model(x, sigma, y0, A, x0):
            gauss = y0 + A / (2*np.pi)**0.5 / sigma * np.exp(-(x-x0)**2/2/sigma**2)
            return gauss 

        defaultPars = {
            "sigma" : 5,
            "y0" : 0,
            "A" : 1,
            "x0" : 0,         
        }

        sharedPars = ["sigma"]

    else:
        def model(x, sigma1, c4, c6, A, x0):
            return A * np.exp(-(x-x0)**2/2/sigma1**2) / (np.sqrt(2*3.1415*sigma1**2)) \
                    *(1 + c4/32*(16*((x-x0)/np.sqrt(2)/sigma1)**4 \
                    -48*((x-x0)/np.sqrt(2)/sigma1)**2+12) \
                    +c6/384*(64*((x-x0)/np.sqrt(2)/sigma1)**6 \
                    -480*((x-x0)/np.sqrt(2)/sigma1)**4 + 720*((x-x0)/np.sqrt(2)/sigma1)**2 - 120)) 
        
        defaultPars = {
            "sigma1" : 6,
            "c4" : 0,
            "c6" : 0,
            "A" : 1,
            "x0" : 0          
        }

        sharedPars = ["sigma1", "c4", "c6"]  

    assert all(isinstance(item, str) for item in sharedPars), "Parameters in list must be strings."

    return model, defaultPars, sharedPars


def calcCostFun(model, i, x, y, yerr, res, sharedPars):
    "Returns cost function for one spectrum i to be summed to total cost function"
   
    xDense, xDelta, resDense = chooseXDense(x, res, False)
    def convolvedModel(xrange, *pars):
        """Performs convolution first on high density grid and interpolates to desired x range"""
        convDense = signal.convolve(model(xDense, *pars), resDense, mode="same") * xDelta
        return np.interp(xrange, xDense, convDense)

    costSig = [key if key in sharedPars else key+str(i) for key in describe(model)]
    convolvedModel.func_code = make_func_code(costSig)
    print(describe(convolvedModel))

    # Data without cut-offs
    nonZeros = y != 0
    xNZ = x[nonZeros]
    yNZ = y[nonZeros]
    yerrNZ = yerr[nonZeros]

    costFun = cost.LeastSquares(xNZ, yNZ, yerrNZ, convolvedModel)
    return costFun


def chooseXDense(x, res, flag):
    """Make high density symmetric grid for convolution"""

    assert np.min(x) == -np.max(x), "Resolution needs to be in symetric range!"
    assert x.size == res.size, "x and res need to be the same size!"

    if flag:
        if x.size % 2 == 0:
            dens = x.size+1  # If even change to odd
        else:
            dens = x.size    # If odd, keep being odd)
    else:
        dens = 1000

    xDense = np.linspace(np.min(x), np.max(x), dens)
    xDelta = xDense[1] - xDense[0]
    resDense = np.interp(xDense, x, res)
    return xDense, xDelta, resDense


def plotData(ws):
    dataY = ws.extractY()
    dataE = ws.extractE()
    dataX = ws.extractX()

    mask = np.all(dataY==0, axis=1)
    dataY = dataY[~mask]
    dataE = dataE[~mask]
    dataX = dataX[~mask]

    rows = 2

    fig, axs = plt.subplots(rows, int(np.ceil(len(dataY)/rows)), 
        figsize=(15, 8), 
        tight_layout=True
        )

    for x, y, yerr, ax in zip(dataX, dataY, dataE, axs.flat):
        ax.errorbar(x, y, yerr, fmt="k.", label="Data")
    return axs


def plotFit(dataX, totCost, minuit, axs):
    for x, costFun, ax in zip(dataX, totCost, axs.flat):
        signature = describe(costFun)

        values = minuit.values[signature]
        errors = minuit.errors[signature]

        yfit = costFun.model(x, *values)

        # Build a decent legend
        leg = []
        for p, v, e in zip(signature, values, errors):
            leg.append(f"${p} = {v:.3f} \pm {e:.3f}$")

        ax.fill_between(x, yfit, label="\n".join(leg), alpha=0.4)
        ax.legend()

main()
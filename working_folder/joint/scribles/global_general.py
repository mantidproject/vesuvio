import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit, cost, util
from iminuit.util import make_with_signature, describe
from pathlib import Path
from mantid.simpleapi import Load, CropWorkspace
from scipy import optimize
from scipy import ndimage, signal

repoPath = Path(__file__).absolute().parent

def main():

    joyPath = repoPath / "wsJOYsmall.nxs"
    resPath = repoPath / "wsResSmall.nxs"
    wsJoY = Load(str(joyPath), OutputWorkspace="wsJoY")
    wsRes = Load(str(resPath), OutputWorkspace="wsRes")
    print("No of spec res: ", wsRes.getNumberHistograms())

    wsJoY = CropWorkspace(InputWorkspace="wsJoY", OutputWorkspace="wsJoY", EndWorkspaceIndex=3)
    wsRes = CropWorkspace(InputWorkspace="wsRes", OutputWorkspace="wsRes", EndWorkspaceIndex=3)
    print("No of spec res: ", wsRes.getNumberHistograms())

    axs = plotData(wsJoY)

    m, totCost = fitGlobalFit(wsJoY, wsRes, True)
    print("Sigma: ", m.values["sigma"])

    plotFit(wsJoY.extractX(), totCost, m, axs)
    plt.show()




def fitGlobalFit(ws, wsRes, gaussFlag):
    
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

    totCost, kwargs = totalCostFun(dataX, dataY, dataE, dataRes, gaussFlag)
    print(describe(totCost))
    print(kwargs)

    m = Minuit(totCost, **kwargs)

    for i in range(len(dataY)):     # Limit for both Gauss and Gram Charlier
        m.limits["A"+str(i)] = (0, np.inf)

    # print(m.params)
    if gaussFlag:
        #TODO Ask about the limits on y0

        for i in range(len(dataY)):
            m.limits["y0"+str(i)] = (0, np.inf)
        m.simplex()
        m.migrad()
        # print(m.params)
        

    else:
        m.limits["c4"] = (0, np.inf)

        x = dataX[0]
        def constr(*pars):
            """Constraint for positivity of Gram Carlier.
            Input: All parameters defined in original function.
            Format *pars to work with Minuit.
            x is defined outside function, one constraint per value of x"""

            sigma1, c4, c6 = pars[:3]
            return (1 + c4/32*(16*(x/np.sqrt(2)/sigma1)**4 \
                    -48*(x/np.sqrt(2)/sigma1)**2+12) \
                    +c6/384*(64*(x/np.sqrt(2)/sigma1)**6 \
                    -480*(x/np.sqrt(2)/sigma1)**4 + 720*(x/np.sqrt(2)/sigma1)**2 - 120))

        m.simplex()
        m.scipy(constraints=optimize.NonlinearConstraint(constr, 0, np.inf))
    
    m.hesse()
    return m, totCost


def totalCostFun(dataX, dataY, dataE, dataRes, gaussFlag):
    
    if gaussFlag:
        def fun(x, y0, A, x0, sigma):
            return y0 + A / (2*np.pi)**0.5 / sigma * np.exp(-(x-x0)**2/2/sigma**2)

        defaultPars = {
            "y0" : 0,
            "A" : 1,
            "x0" : 0,
            "sigma" : 5           
        }

        sharedPars = ["sigma"]

    else:
        def fun(x, sigma1, c4, c6, A, x0):
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

    totCost, initPars = buildCostFun(dataX, dataY, dataE, dataRes, fun, defaultPars, sharedPars)
    return totCost, initPars


def oddConvolution(x, y, res):
    assert np.min(x) == -np.max(x), "Resolution needs to be in symetric range!"
    if x.size % 2 == 0:
        rangeRes = x.size+1  # If even change to odd
    else:
        rangeRes = x.size    # If odd, keep being odd

    xInterp = np.linspace(np.min(x), np.max(x), rangeRes)
    xDelta = xInterp[1] - xInterp[0]
    resInterp = np.interp(xInterp, x, res)

    conv = signal.convolve(y, resInterp, mode="same") * xDelta
    return conv


def buildCostFun(dataX, dataY, dataE, dataRes, fun, defaultPars, sharedPars:list):
    """Shared parameters are specified in a list of strings"""

    assert all(isinstance(item, str) for item in sharedPars), "Parameters in list must be strings."
    
    unsharedPars = [key for key in defaultPars if key not in sharedPars]

    totCost = 0
    print(dataRes.shape)
    for i, (x, y, yerr, res) in enumerate(zip(dataX, dataY, dataE, dataRes)):

        # Interpolate resolution

        def convolvedModel(x, y0, A, x0, sigma):
            gauss = y0 + A / (2*np.pi)**0.5 / sigma * np.exp(-(x-x0)**2/2/sigma**2)
            return oddConvolution(x, gauss, res) #ndimage.convolve1d(gauss, res, mode="constant")

        unsharedArgs = {}
        for upar in unsharedPars:
            unsharedArgs[upar] = upar+str(i)

        costFun = cost.LeastSquares(
            x, y, yerr, make_with_signature(convolvedModel, **unsharedArgs)
            )

        totCost += costFun

    initPars = {}
    # Add shared parameters
    for spar in sharedPars:
        initPars[spar] = defaultPars[spar] 

    # Add unshared parameters 
    for i in range(len(dataY)):
        for upar in unsharedPars:
            initPars[upar+str(i)] = defaultPars[upar]
    
    return totCost, initPars


def plotData(ws):
    dataY = ws.extractY()
    dataE = ws.extractE()
    dataX = ws.extractX()

    mask = np.all(dataY==0, axis=1)
    dataY = dataY[~mask]
    dataE = dataE[~mask]
    dataX = dataX[~mask]

    fig, axs = plt.subplots(
        2, int(np.ceil(len(dataY)/2)), 
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
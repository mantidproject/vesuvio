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

repoPath = Path(__file__).absolute().parent

def main():

    joyPath = repoPath / "wsJOYsmall.nxs"
    resPath = repoPath / "wsResSmall.nxs"
    # joyPath = repoPath / "wsDHMTjoy.nxs"
    # resPath = repoPath / "wsDHMTres.nxs"
    
    wsJoY = Load(str(joyPath), OutputWorkspace="wsJoY")
    wsRes = Load(str(resPath), OutputWorkspace="wsRes")
    print("No of spec res: ", wsRes.getNumberHistograms())

    wsJoY = CropWorkspace(InputWorkspace="wsJoY", OutputWorkspace="wsJoY", StartWorkspaceIndex=4, EndWorkspaceIndex=11)
    wsRes = CropWorkspace(InputWorkspace="wsRes", OutputWorkspace="wsRes", StartWorkspaceIndex=4, EndWorkspaceIndex=11)
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
    for key in defaultPars:
        if key in sharedPars:
            initPars[key] = defaultPars[key]
        else:
            for i in range(len(dataY)):
                initPars[key+str(i)] = defaultPars[key]

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
            x is defined outside function, one constraint per value of x"""

            # sigma1, c4, c6 = pars[:3]
            # return (1 + c4/32*(16*(x/np.sqrt(2)/sigma1)**4 \
            #         -48*(x/np.sqrt(2)/sigma1)**2+12) \
            #         +c6/384*(64*(x/np.sqrt(2)/sigma1)**6 \
            #         -480*(x/np.sqrt(2)/sigma1)**4 + 720*(x/np.sqrt(2)/sigma1)**2 - 120))

            sharedPars = pars[:3]    # sigma1, c4, c6
            joinedGC = np.zeros(int((len(pars)-3)/2) * x.size)
            for i, (A, x0) in enumerate(zip(pars[3::2], pars[4::2])):
                joinedGC[i*x.size : (i+1)*x.size] = model(x, *sharedPars, A, x0)
            
            if np.any(joinedGC==0):
                raise ValueError("Args where zero: ", np.argwhere(joinedGC==0))
            
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
        def model(x, y0, A, x0, sigma):
            gauss = y0 + A / (2*np.pi)**0.5 / sigma * np.exp(-(x-x0)**2/2/sigma**2)
            return gauss 


        defaultPars = {
            "y0" : 0,
            "A" : 1,
            "x0" : 0,
            "sigma" : 5           
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
   
    def convolvedModel(x, *pars):
        return oddConvolution(x, model(x, *pars), res)

    costSig = [key if key in sharedPars else key+str(i) for key in describe(model)]
    convolvedModel.func_code = make_func_code(costSig)
    print(describe(convolvedModel))

    # Make fit ignore cut-off points, assign infinite error
    yerr = np.where(yerr==0, np.inf, yerr)

    costFun = cost.LeastSquares(x, y, yerr, convolvedModel)
    return costFun


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
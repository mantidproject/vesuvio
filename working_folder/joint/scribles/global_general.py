import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit, cost, util
from iminuit.util import make_with_signature, describe
from pathlib import Path
from mantid.simpleapi import Load
from scipy import optimize
from scipy import ndimage

repoPath = Path(__file__).absolute().parent

# def fitGlobalFit(ws, wsRes, fun, constr):
    
#     dataY = ws.extractY()
#     dataE = ws.extractE()
#     dataX = ws.extractX()

#     dataRes = wsRes.extractY()

#     totCost, kwargs = buildTotalCostFun(dataX, dataY, dataE, dataRes, fun)
#     print(describe(totCost))

#     m = Minuit(totCost, **kwargs)
#     return


# def totalCostFunAndArgs(dataX, dataY, dataE, dataRes, fun):
#     totCost = 0
#     kwargs = {}
#     for i, (x, y, yerr, res) in enumerate(zip(dataX, dataY, dataE, dataRes)):

#         def convolvedModel(x, *pars):
#             return ndimage.convolve1d(fun(x, *pars), res, mode="constant")

#         totCost += cost.LeastSquares(x, y, yerr, make_with_signature(convolvedModel, A="A"+str(i), x0="x0"+str(i)))
    
#     # Set initial parameters
#     # Non shared pars
#     for i in range(len(dataY)):
#         kwargs["A"+str(i)] = 1
#         kwargs["x0"+str(i)] = 0

#     # Shared pars
#     kwargs["sigma1"] = 4
#     kwargs["c4"] = 0
#     kwargs["c6"] = 0
#     return totCost, kwargs


def fitGlobalFit(ws, wsRes, gaussFlag):
    
    dataY = ws.extractY()
    dataE = ws.extractE()
    dataX = ws.extractX()

    dataRes = wsRes.extractY()

    totCost, kwargs = totalCostFun(dataX, dataY, dataE, dataRes, gaussFlag)
    print(describe(totCost))

    m = Minuit(totCost, **kwargs)

    for i in range(len(dataY)):     # Limit for both Gauss and Gram Charlier
        m.limits["A"+str(i)] = (0, np.inf)

    if gaussFlag:
        m.simplex()
        m.migrad()

    else:
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



    return


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

    totCost, initPars = totalCostFun(dataX, dataY, dataE, dataRes, fun, defaultPars, sharedPars)
    return totCost, initPars


def totalCostGaussian(dataX, dataY, dataE, dataRes, fun, defaultPars, sharedPars:list):
    """Shared parameters are specified in a list of strings"""

    assert all(isinstance(item, str) for item in sharedPars), "Parameters in list must be strings."
    
    unsharedPars = [key for key in defaultPars not in sharedPars]

    totCost = 0
    for i, (x, y, yerr, res) in enumerate(zip(dataX, dataY, dataE, dataRes)):

        def convolvedModel(x, *pars):    # Convolution at each spectra
            return ndimage.convolve1d(fun(x, *pars), res, mode="constant")

        unsharedArgs = {}
        for upar in unsharedPars:
            unsharedArgs[upar] = upar+str(i)

        totCost += cost.LeastSquares(
            x, y, yerr, make_with_signature(convolvedModel, **unsharedArgs)
            )

    initPars = {}
    # Add shared parameters
    for spar in sharedPars:
        initPars[spar] = defaultPars[spar] 

    # Add unshared parameters 
    for i in range(len(dataY)):
        for upar in unsharedPars:
            initPars[upar+str(i)] = defaultPars[upar]
    
    return totCost, initPars
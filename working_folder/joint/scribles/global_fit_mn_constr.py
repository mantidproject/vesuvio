import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit, cost, util
from iminuit.util import make_with_signature, describe
from pathlib import Path
from mantid.simpleapi import Load
from scipy import optimize
from scipy import ndimage

repoPath = Path(__file__).absolute().parent

# joyPath = repoPath / "wsJoY.nxs"
# wsJoY = Load(str(joyPath), OutputWorkspace="wsJoY")


def HermitePolynomial(x, sigma1, c4, c6, A, x0):
    func = A * np.exp(-(x-x0)**2/2/sigma1**2) / (np.sqrt(2*3.1415*sigma1**2)) \
            *(1 + c4/32*(16*((x-x0)/np.sqrt(2)/sigma1)**4 \
            -48*((x-x0)/np.sqrt(2)/sigma1)**2+12) \
            +c6/384*(64*((x-x0)/np.sqrt(2)/sigma1)**6 \
            -480*((x-x0)/np.sqrt(2)/sigma1)**4 + 720*((x-x0)/np.sqrt(2)/sigma1)**2 - 120)) 
    return func


def globalConstr(x, *pars):
    sigma1, c4, c6 = pars[:3]
    return (1 + c4/32*(16*(x/np.sqrt(2)/sigma1)**4 \
            -48*(x/np.sqrt(2)/sigma1)**2+12) \
            +c6/384*(64*(x/np.sqrt(2)/sigma1)**6 \
            -480*(x/np.sqrt(2)/sigma1)**4 + 720*(x/np.sqrt(2)/sigma1)**2 - 120))


def constructTotalCostFun(dataX, dataY, dataE, funToFit):
    """Builds cost function to fit all 64 forward spectra"""

    totCost = 0
    kwargs = {}
    for i, (x, y, yerr) in enumerate(zip(dataX, dataY, dataE)):
        totCost += cost.LeastSquares(x, y, yerr, make_with_signature(funToFit, A="A"+str(i), x0="x0"+str(i)))
        kwargs["A"+str(i)] = 1
        kwargs["x0"+str(i)] = 0
    return totCost, kwargs


def plotSingle(x, costFun, minuit, ax):
    """Plots single Fit with shared parameters"""

    signature = describe(costFun)

    values = minuit.values[signature]
    errors = minuit.errors[signature]

    yfit = HermitePolynomial(x, *values)

    # Build a decent legend
    leg = []
    for p, v, e in zip(signature, values, errors):
        leg.append(f"${p} = {v:.3f} \pm {e:.3f}$")

    ax.fill_between(x, yfit, label="\n".join(leg), alpha=0.4)


def plotData(x, y, yerr, ax):
    ax.errorbar(x, y, yerr, fmt="k.", label="Data")



def fitGlobalFit(ws, wsRes, fun, constr):
    
    dataY = ws.extractY()
    dataE = ws.extractE()
    dataX = ws.extractX()

    dataRes = wsRes.extractY()

    totCost, kwargs = buildTotalCostFun(dataX, dataY, dataE, dataRes, fun)
    print(describe(totCost))

    m = Minuit(totCost, **kwargs, sigma1=4, c4=0, c6=0)
    return


def buildTotalCostFun(dataX, dataY, dataE, dataRes, fun):
    totCost = 0
    kwargs = {}
    for i, (x, y, yerr, res) in enumerate(zip(dataX, dataY, dataE, dataRes)):

        def convolvedModel(*pars):
            return ndimage.convolve1d(fun(x, *pars), res, mode="constant")

        totCost += cost.LeastSquares(x, y, yerr, make_with_signature(convolvedModel, A="A"+str(i), x0="x0"+str(i)))
        kwargs["A"+str(i)] = 1
        kwargs["x0"+str(i)] = 0
    return totCost, kwargs


# Build random data with negativities
np.random.seed(1)
x = np.linspace(-20, 20, 100)

n = 5
dataY = np.zeros((n, x.size))
dataE = dataY.copy()
dataX = dataY.copy()

for i in range(n):
    dataX[i] = x
    dataE[i] = np.random.rand(x.size) * 0.01
    x0 = 0.7 * i #np.random.random()*1
    dataY[i] =  HermitePolynomial(x, 5, 1, -1, 1, x0) + dataE[i] * np.random.randn(x.size) 

# Pepare figure
noOfSpec = len(dataY)
noAxsWid = int(np.floor(np.sqrt(noOfSpec)))
noAxsLen = int(np.round(noOfSpec / noAxsWid))

fig, axs = plt.subplots(noAxsWid, noAxsLen, figsize=(15, 8), tight_layout=True)
# Plot original data
for x, y, yerr, ax in zip(dataX, dataY, dataE, axs.flat):
    plotData(x, y, yerr, ax)

#Build total cost function
totCost, kwargs = constructTotalCostFun(dataX, dataY, dataE, HermitePolynomial)
print(describe(totCost))

m = Minuit(totCost, **kwargs, sigma1=4, c4=0, c6=0)

#Aply bounds
m.limits["c4"] = (0, np.inf)
for i in range(len(dataY)):
    m.limits["A"+str(i)] = (0, np.inf)

# m.simplex()
m.migrad()
# Plot fitted data
for x, costF, ax in zip(dataX, totCost, axs.flat):
    plotSingle(x, costF, m, ax)
    # ax.legend()

constraints = optimize.NonlinearConstraint(lambda *pars: globalConstr(dataX[0], *pars), 0, np.inf)
# constraints = ()
m.scipy(constraints=constraints)
# Plot fitted data
for x, costF, ax in zip(dataX, totCost, axs.flat):
    plotSingle(x, costF, m, ax)
    # ax.legend()


"""
Constraint without shift parameter x0 seems to work very well, even for varying x0.
Might be violated at extremities of axis.
"""


plt.show()
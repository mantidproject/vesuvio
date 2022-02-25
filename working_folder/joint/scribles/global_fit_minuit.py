import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit, cost, util
from iminuit.util import make_with_signature, describe
from pathlib import Path
from mantid.simpleapi import Load
repoPath = Path(__file__).absolute().parent

joyPath = repoPath / "wsJoY.nxs"
wsJoY = Load(str(joyPath), OutputWorkspace="wsJoY")


def HermitePolynomial(x, A, x0, sigma1, c4, c6):
    func = A * np.exp(-(x-x0)**2/2/sigma1**2) / (np.sqrt(2*3.1415*sigma1**2)) \
            *(1 + c4/32*(16*((x-x0)/np.sqrt(2)/sigma1)**4 \
            -48*((x-x0)/np.sqrt(2)/sigma1)**2+12) \
            +c6/384*(64*((x-x0)/np.sqrt(2)/sigma1)**6 \
            -480*((x-x0)/np.sqrt(2)/sigma1)**4 + 720*((x-x0)/np.sqrt(2)/sigma1)**2 - 120)) 
    return func


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

    ax.plot(x, yfit, label="\n".join(leg))
    ax.set_ylim(-0.02, 0.1)

def plotData(x, y, yerr, ax):
    ax.errorbar(x, y, yerr, fmt=".", label="Data")

# Extract data
dataY = wsJoY.extractY()[:10]
dataX = wsJoY.extractX()[:10]
dataE = wsJoY.extractE()[:10]

# Remove zeros
# dataY = dataY[np.any(dataY!=0, axis=1)]
# dataX = dataY[np.any(dataX!=0, axis=1)]
# dataE = dataE[np.any(dataE!=0, axis=1)]

# Pepare figure
noOfSpec = len(dataY)
noAxsWid = int(np.floor(np.sqrt(noOfSpec)))
noAxsLen = int(np.round(noOfSpec / noAxsWid))

fig, axs = plt.subplots(noAxsWid, noAxsLen, figsize=(15, 8))
# Plot original data
for x, y, yerr, ax in zip(dataX, dataY, dataE, axs.flat):
    plotData(x, y, yerr, ax)


totCost, kwargs = constructTotalCostFun(dataX, dataY, dataE, HermitePolynomial)

m = Minuit(totCost, **kwargs, sigma1=4, c4=0, c6=0)

#Aply bounds
for i in range(len(dataY)):
    m.limits["A"+str(i)] = (0, np.inf)

m.simplex()
m.migrad()

# The constrained fit is a bit of a nightmare though?
# One constraint for each spectrum, maybe that is too much

# Plot fitted data
for x, costF, ax in zip(dataX, totCost, axs.flat):
    plotSingle(x, costF, m, ax)
    ax.legend()
plt.show()
    
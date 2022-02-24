import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit, cost, util
from iminuit.util import make_with_signature, describe


def HermitePolynomial(x, A, x0, sigma1, c4, c6):
    func = A * np.exp(-(x-x0)**2/2/sigma1**2) / (np.sqrt(2*3.1415*sigma1**2)) \
            *(1 + c4/32*(16*((x-x0)/np.sqrt(2)/sigma1)**4 \
            -48*((x-x0)/np.sqrt(2)/sigma1)**2+12) \
            +c6/384*(64*((x-x0)/np.sqrt(2)/sigma1)**6 \
            -480*((x-x0)/np.sqrt(2)/sigma1)**4 + 720*((x-x0)/np.sqrt(2)/sigma1)**2 - 120)) 
    return func


np.random.seed(1)
x = np.linspace(-20, 20, 100)
yerr1 = np.random.rand(x.size) * 0.01
yerr2 = np.random.rand(x.size) * 0.01
dataY1 = HermitePolynomial(x, 1, 0, 5, 1, 1) + yerr1 * np.random.randn(x.size)
dataY2 = HermitePolynomial(x, 0.5, -5, 5, 1, 1) + yerr2 * np.random.randn(x.size)

costFun1 = cost.LeastSquares(x, dataY1, yerr1, make_with_signature(HermitePolynomial, A="A1", x0="x01"))
costFun2 = cost.LeastSquares(x, dataY2, yerr2, make_with_signature(HermitePolynomial, A="A2", x0="x02"))

totCostFun = costFun1 + costFun2
print(f"{describe(totCostFun)=}")

m = Minuit(totCostFun, A1=1, A2=1, x01=0, x02=0, sigma1=4, c4=0, c6=0)

m.simplex()
m.migrad()


def plotSingle(costFun, minuit, ax):
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

fig, axs = plt.subplots(1, 2, figsize=(14, 5))
# Add original data
axs[0].errorbar(x, dataY1, yerr1, fmt=".", label="Original Dataset 1")
axs[1].errorbar(x, dataY2, yerr2, fmt=".", label="Original Dataset 2")

for costF, ax in zip(totCostFun, axs):
    plotSingle(costF, m, ax)
    ax.legend()
plt.show()
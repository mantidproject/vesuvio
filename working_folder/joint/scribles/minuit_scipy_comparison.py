import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from iminuit import Minuit, cost
from iminuit.cost import LeastSquares


def HermitePolynomial(x, pars):
    A, x0, sigma1, c4, c6 = pars
    func = A * np.exp(-(x-x0)**2/2/sigma1**2) / (np.sqrt(2*3.1415*sigma1**2)) \
            *(1 + c4/32*(16*((x-x0)/np.sqrt(2)/sigma1)**4 \
            -48*((x-x0)/np.sqrt(2)/sigma1)**2+12) \
            +c6/384*(64*((x-x0)/np.sqrt(2)/sigma1)**6 \
            -480*((x-x0)/np.sqrt(2)/sigma1)**4 + 720*((x-x0)/np.sqrt(2)/sigma1)**2 - 120)) 
    return func


# for i in range(10):
i = 5    # Gives difficult fits
print("iteration ", i)
np.random.seed(i)
x = np.linspace(-20, 20, 100)
yerr = np.random.rand(x.size) * 0.03
c4 = 1 # np.random.uniform(-1, 1)
c6 = -3 #np.random.uniform(-2, 2)
pOri = [1, 0, 5, c4, c6]
dataY = HermitePolynomial(x, pOri) + yerr * np.random.randn(x.size)
p0 = [1, 0, 4, 0, 0]

costFun = cost.LeastSquares(x, dataY, yerr, HermitePolynomial)
m = Minuit(costFun, p0)
m.limits = [(0, None), (None, None), (None, None), (None, None), (None, None)] 

m.simplex()
# m.migrad()
constraints = optimize.NonlinearConstraint(lambda pars: HermitePolynomial(x, pars), 0, np.inf)
m.scipy(constraints=constraints)
m.hesse()
print("Chi2 from iminuit: ", m.fval)

# Scipy only fit
# Sort out cost fun
def myCost(p0):
    fun = HermitePolynomial(x, p0)
    return np.sum((dataY - fun)**2 / yerr**2)

result = optimize.minimize(myCost, p0, method="SLSQP", constraints=constraints)
popt3 = result["x"]
print("Chi2 from minimize(): ", result["fun"])

try:
    np.testing.assert_allclose(m.fval, result["fun"], rtol=1e-2)
except AssertionError:
    print("relative diff: ", m.fval / result["fun"])

plt.errorbar(x, dataY, yerr, fmt="o")
plt.fill_between(x, HermitePolynomial(x, m.values), color="red", alpha=0.2, label="Minuit with Scipy")
plt.fill_between(x, HermitePolynomial(x, popt3), color="green", alpha=0.2, label="Minimize[SLSQP]")
plt.legend()
plt.show()

# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# everything in iminuit is done through the Minuit object, so we import it
from iminuit import Minuit, cost, util

# we also need a cost function to fit and import the LeastSquares function
from iminuit.cost import LeastSquares

# display iminuit version
import iminuit
print("iminuit version:", iminuit.__version__)


#%%
def HermitePolynomial(x, A, x0, sigma1, c4, c6):
    func = A * np.exp(-(x-x0)**2/2/sigma1**2) / (np.sqrt(2*3.1415*sigma1**2)) \
            *(1 + c4/32*(16*((x-x0)/np.sqrt(2)/sigma1)**4 \
            -48*((x-x0)/np.sqrt(2)/sigma1)**2+12) \
            +c6/384*(64*((x-x0)/np.sqrt(2)/sigma1)**6 \
            -480*((x-x0)/np.sqrt(2)/sigma1)**4 + 720*((x-x0)/np.sqrt(2)/sigma1)**2 - 120)) 
    return func

def gaussian(x, y0, A, x0, sigma):
        return y0 + A / (2*np.pi)**0.5 / sigma * np.exp(-(x-x0)**2/2/sigma**2)


np.random.seed(1)
x = np.linspace(-20, 20, 100)
yerr = np.random.rand(x.size) * 0.03
dataY = HermitePolynomial(x, 1, 0, 5, 0.1, 0.2) + yerr * np.random.randn(x.size)


costFun = cost.LeastSquares(x, dataY, yerr, HermitePolynomial)
m = Minuit(costFun, A=1, x0=0, sigma1=4, c4=0, c6=0)
m.limits["A"] = (0, None)
m.simplex()
constraints = optimize.NonlinearConstraint(lambda *pars: HermitePolynomial(x, *pars), 0, np.inf)
m.scipy(constraints=constraints)
m.hesse()

cov = m.covariance
# print("\nCovariance Matrix:\n", cov)
corr = np.zeros(cov.shape)
for i in range(len(corr)):
    for j in range(len(corr[0])):
        corr[i, j] =  cov[i, j] / np.sqrt(cov[i, i] * cov[j, j])
corr *= 100
# print("\nCorrelation Matrix:\n", corr)


dataYFit, dataYCov = util.propagate(lambda pars: HermitePolynomial(x, *pars), m.values, m.covariance)
dataYSigma = np.sqrt(np.diag(dataYCov)) * m.fval / (len(x)-m.nfit)

# Fit Gaussian to compare
costFun1 = cost.LeastSquares(x, dataY, yerr, gaussian)
m1 = Minuit(costFun1, y0=0, A=1, x0=0, sigma=5)
m1.limits["A"] = (0, None)
m1.simplex()
m1.migrad()
m1.hesse()
yfit, ycov =  util.propagate(lambda pars: gaussian(x, *pars), m1.values, m1.covariance)
ci = np.sqrt(np.diag(ycov)) * m1.fval / (len(x)-m1.nfit)

#%%
plt.errorbar(x, dataY, yerr, fmt="o")
plt.plot(x, HermitePolynomial(x, *m.values), color="red", label="Hermite Poly")
plt.fill_between(x, dataYFit-dataYSigma, dataYFit+dataYSigma, color="red", alpha=0.2, label="CI Herm")
plt.plot(x, gaussian(x, *m1.values), "k--", label="Gaussian")
plt.fill_between(x, yfit-ci, yfit+ci, color="k", alpha=0.2, label="CI gauss")

# display legend with some fit info
fit_info = [
    f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {m.fval:.1f} / {x.size - m.nfit}",
]
for p, v, e in zip(m.parameters, m.values, m.errors):
    fit_info.append(f"{p} = ${v:.3f} \\pm {e:.3f}$")

plt.legend(title="\n".join(fit_info))

plt.show()

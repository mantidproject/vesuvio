# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize

# everything in iminuit is done through the Minuit object, so we import it
from iminuit import Minuit, cost

# we also need a cost function to fit and import the LeastSquares function
from iminuit.cost import LeastSquares

# display iminuit version
import iminuit
print("iminuit version:", iminuit.__version__)



def HermitePolynomial(x, A, x0, sigma1, c4, c6):
    func = A * np.exp(-(x-x0)**2/2/sigma1**2) / (np.sqrt(2*3.1415*sigma1**2)) \
            *(1 + c4/32*(16*((x-x0)/np.sqrt(2)/sigma1)**4 \
            -48*((x-x0)/np.sqrt(2)/sigma1)**2+12) \
            +c6/384*(64*((x-x0)/np.sqrt(2)/sigma1)**6 \
            -480*((x-x0)/np.sqrt(2)/sigma1)**4 + 720*((x-x0)/np.sqrt(2)/sigma1)**2 - 120)) 
    return func


# def constr(x, A, x0, sigma1, c4, c6):
#     # x0, sigma1, c4, c6 = p0[1:]
#     return (1 + c4/32*(16*((x-x0)/np.sqrt(2)/sigma1)**4 \
#                 -48*((x-x0)/np.sqrt(2)/sigma1)**2+12) \
#                 +c6/384*(64*((x-x0)/np.sqrt(2)/sigma1)**6 \
#                 -480*((x-x0)/np.sqrt(2)/sigma1)**4 + 720*((x-x0)/np.sqrt(2)/sigma1)**2 - 120))



np.random.seed(1)
x = np.linspace(-20, 20, 100)
yerr = np.random.rand(x.size) * 0.01
dataY = HermitePolynomial(x, 1, 0, 5, 1, -3) + yerr * np.random.randn(x.size)

p0 = [1, 0, 4, 0, 0]

costFun = cost.LeastSquares(x, dataY, yerr, HermitePolynomial)

m = Minuit(costFun, A=1, x0=0, sigma1=4, c4=0, c6=0)
m.limits["A"] = (0, None)

m.simplex()
constraints = optimize.NonlinearConstraint(lambda *pars: HermitePolynomial(x, *pars), 0, np.inf)
m.scipy(constraints=constraints)
m.hesse()

# %%

# try it with scipy 
def myCost(p0, x, dataY, yerr):
    fun = HermitePolynomial(x, *p0)
    return np.sum((dataY - fun)**2 / yerr**2)

def constr(p0):
    A, x0, sigma1, c4, c6 = p0
    return A * np.exp(-(x-x0)**2/2/sigma1**2) / (np.sqrt(2*3.1415*sigma1**2)) \
            *(1 + c4/32*(16*((x-x0)/np.sqrt(2)/sigma1)**4 \
            -48*((x-x0)/np.sqrt(2)/sigma1)**2+12) \
            +c6/384*(64*((x-x0)/np.sqrt(2)/sigma1)**6 \
            -480*((x-x0)/np.sqrt(2)/sigma1)**4 + 720*((x-x0)/np.sqrt(2)/sigma1)**2 - 120))
    
constraints = ({'type': 'ineq', 
                'fun': constr})

p0 = [1, 0, 4, 0, 0]
result = optimize.minimize(myCost, p0, args=(x, dataY, yerr), method="SLSQP", constraints=constraints)
popt3 = result["x"]
print("Chi2 from minimize(): ", result["fun"])

plt.errorbar(x, dataY, yerr, fmt="o")
plt.fill_between(x, HermitePolynomial(x, *m.values), color="red", alpha=0.2, label="Minuit with Scipy")
plt.fill_between(x, HermitePolynomial(x, *popt3), color="green", alpha=0.2, label="Minimize[SLSQP]")

# display legend with some fit info
fit_info = [
    f"$\\chi^2$ / $n_\\mathrm{{dof}}$ = {m.fval:.1f} / {x.size - m.nfit}",
]
for p, v, e in zip(m.parameters, m.values, m.errors):
    fit_info.append(f"{p} = ${v:.3f} \\pm {e:.3f}$")

plt.legend(title="\n".join(fit_info))

plt.show()

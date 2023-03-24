import numpy as np
import matplotlib.pyplot as plt
import numba as nb
import time
from pathlib import Path
scriblePath = Path(__file__).absolute().parent
from scipy.optimize import minimize
plt.style.use("seaborn-poster")

filePath = scriblePath / "GCGrid.npz"

GCPos = np.load(filePath)["GCGrid"]

c4 = np.linspace(-1, 1, 1000)
c6 = np.linspace(-1, 1, 1000)

boundary = np.nonzero(np.diff(GCPos, axis=0))

C4 = c4[:, np.newaxis] * np.ones((1, c6.size))
C6 = c4[np.newaxis, :] * np.ones((c4.size, 1))

xRaw = C6[boundary]
yRaw = C4[boundary]

idx = np.argsort(xRaw)
x = xRaw[idx]
y = yRaw[idx]

def fun(x, pars):
    return np.piecewise(
        x, [x<pars[0]], [lambda x: pars[1] + pars[2]*x, lambda x: pars[3]*np.exp(pars[4]*(x-pars[5])) ]#lambda x: pars[3] + pars[4]*(x-pars[5])**2]
    )
    # return pars[0] + pars[1]*(x-pars[2]) + pars[3]*(x**2-pars[4]) + pars[5]*(x**3-pars[6]) 

p0 = np.zeros(8)

def cost(pars):
    return np.sum((y-fun(x, pars))**2)

res = minimize(cost, x0=p0)
print(x.shape)
plt.plot(x, y, "k.", label="Boundary")

leg = []
for i, p in enumerate(res["x"]):
    leg.append(f"p{i} = {p:.2f} \n")

plt.plot(x, fun(x, res["x"]), "r-", label="".join(leg))
plt.xlabel("c6")
plt.ylabel("c4")
plt.legend()
plt.show()


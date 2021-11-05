import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
currentPath = Path(__file__).absolute().parent 

plt.style.use('seaborn-poster')
plt.rcParams['axes.facecolor'] = (0.9, 0.9, 0.9)
plt.rcParams.update({"axes.grid" : True, "grid.color": "white"})

loadpath = currentPath / "data_H_NCP_fit.npz"
data = np.load(loadpath)
dataX = data["dataX"]
dataY = data["dataY"]
dataE = data["dataE"]
fit = data["fit"]

fig, ax = plt.subplots(figsize=(5, 8))
ax.errorbar(dataX, dataY, yerr=dataE,
            fmt="none", linewidth=1.5, color="black",
            capsize=2, capthick=1.5)
ax.plot(dataX, dataY, "o", label=" Avg Counts", 
        linewidth=1, color="black", markersize=5)
ax.plot(dataX, fit, label="Gaussian Fit",
            linestyle="-", linewidth=1.25, color="red" )

plt.xlabel(r"y-space [$A^{-1}$]")
plt.ylabel(r"J(y) [$A$]")
# plt.ylim((-0.005, 0.06))
plt.xlim((-18, 18))
plt.legend(loc="upper left")
plt.legend()
plt.savefig("./H_NCP_Fit.pdf", bbox_inches="tight")

plt.show()

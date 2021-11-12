
import numpy as np
import matplotlib.pyplot as plt
from mantid.simpleapi import *    
from pathlib import Path
currentPath = Path(__file__).absolute().parent  # Path to the repository
plt.style.use('seaborn-poster')

firstPath = currentPath / "data_for_plots_cvxopt_fse.npz"
results = np.load(firstPath)

kMins = results["all_kMins"][0]
x = np.arange(len(kMins))

kMins = kMins.T
print(np.sum(kMins, axis=1))
mean = np.nanmean(kMins, axis=1)[:, np.newaxis]
std = np.nanstd(kMins, axis=1)[:, np.newaxis]
print("means:\n", mean, "\nstd:\n", std)

kMins[np.abs(kMins-mean)>std] = np.nan

fig, ax = plt.subplots()
ax.scatter(x, kMins[1], alpha=0.5, label=f"H")

# for i, kMass in enumerate(kMins):
#     ax.scatter(x, kMass, alpha=0.5, label=f"{i}")
#ax.set_yscale("log")
plt.legend()
plt.show()
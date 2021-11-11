
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
std = np.std(kMins, axis=1)[:, np.newaxis]
mean = np.mean(kMins, axis=1)[:, np.newaxis]
print(std, mean)

kMins[np.abs(kMins-mean)>std] = np.nan

fig, ax = plt.subplots()
for kMass in kMins:
    ax.hist(kMass, alpha=0.5)
plt.show()
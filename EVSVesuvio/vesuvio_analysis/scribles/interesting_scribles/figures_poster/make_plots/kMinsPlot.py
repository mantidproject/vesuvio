
import numpy as np
import matplotlib.pyplot as plt
from mantid.simpleapi import *    
from pathlib import Path
currentPath = Path(__file__).absolute().parent  # Path to the repository
plt.style.use('seaborn-poster')

firstPath = currentPath / "data_for_plots_cvxopt_fse.npz"

def plot_kvalues(path, massIdx):
    results = np.load(path)

    kMins = results["all_kMins"][0]
    MWidth = results["all_mean_widths"][0, massIdx]
    for i, ks in enumerate(kMins):
        if ks[0] <0:
            print(i)

    x = np.arange(len(kMins))

    kMins = kMins.T
    print(np.sum(kMins, axis=1))
    mean = np.nanmean(kMins, axis=1)[:, np.newaxis]
    std = np.nanstd(kMins, axis=1)[:, np.newaxis]
    print("means:\n", mean, "\nstd:\n", std)

    kMins[np.abs(kMins-mean)>std] = np.nan

    fig, ax = plt.subplots()
    ax.scatter(x, kMins[massIdx], alpha=0.5, label=f"k value, H profile")
    ax.plot(x, np.full(x.shape, MWidth)**4/3, label=r"$k=\sigma^4/3$")
    ax.plot(x, np.zeros(x.shape))
    # for i, kMass in enumerate(kMins):
    #     ax.scatter(x, kMass, alpha=0.5, label=f"{i}")
    #ax.set_yscale("log")
    plt.legend()
    plt.show()

plot_kvalues(firstPath, 0)
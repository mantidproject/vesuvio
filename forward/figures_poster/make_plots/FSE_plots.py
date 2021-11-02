import unittest
from jupyterthemes.jtplot import figsize
import numpy as np
import matplotlib.pyplot as plt
from mantid.simpleapi import *    
from pathlib import Path
currentPath = Path(__file__).absolute().parent  # Path to the repository
plt.style.use('seaborn-poster')
# plt.rcParams['axes.facecolor'] = (0.8, 0.8, 0.8)

posFSEPath = currentPath / "data_for_plots_positive_fse.npz"
negFSEPath = currentPath / "data_for_plots_negative_fse.npz"
zeroFSEPath = currentPath / "data_for_plots_no_fse.npz"
paths = [posFSEPath, negFSEPath, zeroFSEPath]
plt.figure()

for path, sign, color in zip(paths, [">", "<", "="], ["tab:blue", "tab:orange", "tab:purple"]):
    results = np.load(path)
    # masses = results["masses"]
    # hbar = 2.0445
    spec = 15
    # dataY = results["all_dataY"][0, spec]    # In the order of the script
    dataX = results["all_dataX"][0, spec]
    # dataE = results["all_dataE"][0, spec]
    # deltaQ = results["all_deltaQ"][0, spec]
    # deltaE = results["all_deltaE"][0, spec]
    # yspaces_for_each_mass = results["all_yspaces_for_each_mass"][0, spec]
    # spec_best_par_chi_nit = results["all_spec_best_par_chi_nit"][0, spec]
    # mean_widths = results["all_mean_widths"][0]
    # mean_intensities = results["all_mean_intensities"][0]
    # tot_ncp = results["all_tot_ncp"][0, spec]
    ncp_for_each_mass = results["all_ncp_for_each_mass"][0, spec]

    # plt.errorbar(dataX, dataY, yerr=dataE,
    #                 fmt="none", linewidth=0.5, color="black")
    # plt.plot(dataX, dataY, ".", label="Count Data", linewidth=1, color="black")

    ncp_m = ncp_for_each_mass[0]
    plt.fill_between(dataX, ncp_m, 
                label=f"H NCP, FSE"+sign+"0", color=color, alpha=0.6)
    plt.plot(dataX, ncp_m, linewidth=3,
                label=f"H NCP, FSE"+sign+"0", color=color)
    xmax = dataX[ncp_m==np.max(ncp_m)]
    plt.vlines(xmax, 0, np.max(ncp_m), color=color, linewidth=3)
    # for i, ncp_m in enumerate(ncp_for_each_mass):
    #     plt.fill_between(dataX, ncp_m, 
    #                     label=f"NCP, M={masses[i]}", alpha=0.5)
    
    # plt.plot(dataX, tot_ncp, label="Total NCP Fit",
    #             linestyle="-", linewidth=3, color="red" )

#plt.plot(dataX, np.zeros(dataX.shape), "--r", label="Zero line")
plt.xlabel(r"TOF [$\mu s$]")
plt.ylabel(r"C(t) [$\mu s^{-1}$]")
# plt.ylim((-0.005, 0.06))
# plt.xlim((120, 420))
plt.legend()
plt.show()
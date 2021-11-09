import unittest
from jupyterthemes.jtplot import figsize
import numpy as np
import matplotlib.pyplot as plt
from mantid.simpleapi import *    
from pathlib import Path
currentPath = Path(__file__).absolute().parent  # Path to the repository
plt.style.use('seaborn-poster')
# plt.rcParams['axes.facecolor'] = (0.8, 0.8, 0.8)

# Use generator to see if any ncp reaches zero
correctedFSEPath = currentPath / "data_for_plots_double_fit_fse_avg_widths.npz"
results = np.load(correctedFSEPath)
ncp_for_each_mass = results["all_ncp_for_each_mass"][0]

# ncp_m = [row[0] for row in ncp_for_each_mass if (np.min(row[0])<0)]
# print("No of ncp with negative value: ", len(ncp_m))

for i, row in enumerate(ncp_for_each_mass):
    if np.min(row[0]) < 0:
        print(i)




def plotFSE(loadPaths, signs, colors, lines):
    fig, ax = plt.subplots(figsize=(10, 10))
    axins = ax.inset_axes([0.59, 0.75, 0.40, 0.24])

    for path, sign, color, line in zip(loadPaths, signs, colors, lines):
        results = np.load(path)
        spec = 22
        try:
            x = results["all_dataX"][0, spec]
            ncp_for_each_mass = results["all_ncp_for_each_mass"][0, spec]
            FSE_for_each_mass = results["all_FSE"][0, spec]
            yspaces_for_each_mass = results["all_yspaces_for_each_mass"][0, spec]
        except KeyError:
            pass
            ncp_for_each_mass = results["all_indiv_ncp"][0, :, spec, :-1]
            FSE_for_each_mass = np.zeros(ncp_for_each_mass.shape)

        ncp_m = ncp_for_each_mass[0]
        FSE_m = FSE_for_each_mass[0]
        yspace_m = yspaces_for_each_mass[0]
        x = yspace_m
        
        ax.fill_between(x, ncp_m, label=f"H NCP, FSE "+sign,  color=color, alpha=0.6)
        axins.fill_between(x, ncp_m, color=color, alpha=0.6)
        
        # ax.plot(x, ncp_m, linewidth=3, linestyle=line,
        #             label=f"H NCP, FSE "+sign, color=color)
        # axins.plot(x, ncp_m, linewidth=3, linestyle=line,
        #             label=f"H NCP, FSE "+sign, color=color)

        ax.plot(x, FSE_m, linewidth=2, linestyle=line,
                    label=f"FSE "+sign, color=color)
        axins.plot(x, FSE_m, linewidth=2, linestyle=line,
                    label=f"FSE "+sign, color=color)

        xmax = x[ncp_m==np.max(ncp_m)]
        ax.vlines(xmax, 0, np.max(ncp_m), color=color, linewidth=2)

    x1, x2, y1, y2 = -26, -12, -0.001, 0.0005
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels('')
    axins.set_yticklabels('')
    ax.indicate_inset_zoom(axins, edgecolor="black", linewidth=1.5, label="Tails")

    ax.set_xlabel(r"y-space [$A^{-1}$]")
    ax.set_ylabel(r"C(t) [$A$]")
    #ax.set_xlim(-20, 20)
    ax.legend(loc="upper left")
    plt.show()

plotSignsOfFSE = False
plotQOfFSE = False
plotConstrFSE = False
plotFinalFSE = True

if plotSignsOfFSE:
    posFSEPath = currentPath / "data_for_plots_positive_fse.npz"
    negFSEPath = currentPath / "data_for_plots_negative_fse.npz"
    zeroFSEPath = currentPath / "data_for_plots_no_fse.npz"
    paths = [posFSEPath, negFSEPath, zeroFSEPath]
    signs = [">0", "<0", "=0"]
    colors = ["tab:blue", "tab:orange", "tab:purple"]
    linestyles = ["solid", "dashed", "dotted"]
    plotFSE(paths, signs, colors, linestyles)
elif plotQOfFSE:
    QtofPath = currentPath / "data_for_plots_negative_fse.npz"
    QpeakPath = currentPath / "data_for_plots_neg_fse_Q_at_peak.npz"
    originalPath =currentPath / "original_144-182_1iter.npz"
    paths = [QpeakPath, QtofPath, originalPath]
    labels = ["Q at peak", "Q(TOF)", "Q(TOF) Ori"]
    colors = ["tab:blue", "tab:orange", "tab:purple"]
    linestyles = ["solid", "dashed", "dotted"]
    plotFSE(paths, labels, colors, linestyles)
elif plotConstrFSE:
    negFSEPath = currentPath / "data_for_plots_negative_fse.npz"
    constrFSEPath = currentPath / "data_for_plots_H3_fse.npz"
    paths = [negFSEPath, constrFSEPath]
    labels = ["ori", "H3"]
    colors = ["tab:blue", "tab:orange", "tab:purple"]
    linestyles = ["solid", "dashed", "dotted"]
    plotFSE(paths, labels, colors, linestyles)
elif plotFinalFSE:
    firstPath = currentPath / "data_for_plots_double_fit_fse.npz"
    secondPath = currentPath / "data_for_plots_negative_fse_factor_third.npz"
    #thirdPath = currentPath / "data_for_plots_no_fse.npz"
    thirdPath = currentPath / "data_for_plots_double_fit_fse_avg_widths.npz"
    paths = [firstPath, secondPath, thirdPath]
    labels = ["fit X2", "*1/3", "fit X2 avg withs"]
    colors = ["tab:blue", "tab:orange", "tab:purple"]
    linestyles = ["solid", "dashed", "dotted"]
    plotFSE(paths, labels, colors, linestyles)


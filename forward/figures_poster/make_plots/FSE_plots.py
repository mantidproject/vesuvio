import unittest
from jupyterthemes.jtplot import figsize
import numpy as np
import matplotlib.pyplot as plt
from mantid.simpleapi import *    
from pathlib import Path
currentPath = Path(__file__).absolute().parent  # Path to the repository
plt.style.use('seaborn-poster')
# plt.rcParams['axes.facecolor'] = (0.8, 0.8, 0.8)

def plotFSE(loadPaths, signs, colors, lines):
    fig, ax = plt.subplots(figsize=(10, 10))
    axins = ax.inset_axes([0.59, 0.75, 0.40, 0.24])

    for path, sign, color, line in zip(loadPaths, signs, colors, lines):
        results = np.load(path)
        spec = 15
        try:
            dataX = results["all_dataX"][0, spec]
            ncp_for_each_mass = results["all_ncp_for_each_mass"][0, spec]
        except KeyError:
            pass
            ncp_for_each_mass = results["all_indiv_ncp"][0, :, spec, :-1]
            print(ncp_for_each_mass.shape)
            #break
        ncp_m = ncp_for_each_mass[0]
        
        ax.fill_between(dataX, ncp_m, color=color, alpha=0.6)
        axins.fill_between(dataX, ncp_m, color=color, alpha=0.6)
        
        ax.plot(dataX, ncp_m, linewidth=3, linestyle=line,
                    label=f"H NCP, FSE "+sign, color=color)
        axins.plot(dataX, ncp_m, linewidth=3, linestyle=line,
                    label=f"H NCP, FSE "+sign, color=color)
        
        xmax = dataX[ncp_m==np.max(ncp_m)]
        ax.vlines(xmax, 0, np.max(ncp_m), color=color, linewidth=3)

    x1, x2, y1, y2 = 350, 400, -0.0005, 0.0005
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels('')
    axins.set_yticklabels('')
    ax.indicate_inset_zoom(axins, edgecolor="black", linewidth=1.5, label="Tails")

    ax.set_xlabel(r"TOF [$\mu s$]")
    ax.set_ylabel(r"C(t) [$\mu s^{-1}$]")
    ax.set_xlim(200, 400)
    ax.legend(loc="upper left")
    plt.show()

plotSignsOfFSE = False
if plotSignsOfFSE:
    posFSEPath = currentPath / "data_for_plots_positive_fse.npz"
    negFSEPath = currentPath / "data_for_plots_negative_fse.npz"
    zeroFSEPath = currentPath / "data_for_plots_no_fse.npz"
    paths = [posFSEPath, negFSEPath, zeroFSEPath]
    signs = [">0", "<0", "=0"]
    colors = ["tab:blue", "tab:orange", "tab:purple"]
    linestyles = ["solid", "dashed", "dotted"]
    plotFSE(paths, signs, colors, linestyles)
else:
    QtofPath = currentPath / "data_for_plots_negative_fse.npz"
    QpeakPath = currentPath / "data_for_plots_neg_fse_Q_at_peak.npz"
    originalPath =currentPath / "original_144-182_1iter.npz"
    paths = [QpeakPath, QtofPath, originalPath]
    labels = ["Q at peak", "Q(TOF)", "Q(TOF) Ori"]
    colors = ["tab:blue", "tab:orange", "tab:purple"]
    linestyles = ["solid", "dashed", "dotted"]
    plotFSE(paths, labels, colors, linestyles)

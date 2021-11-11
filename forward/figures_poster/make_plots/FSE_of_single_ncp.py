import numpy as np
import matplotlib.pyplot as plt
from mantid.simpleapi import *    
from pathlib import Path

from scipy.sparse import data
currentPath = Path(__file__).absolute().parent  # Path to the repository
plt.style.use('seaborn-poster')
# plt.rcParams['axes.facecolor'] = (0.8, 0.8, 0.8)

def numericalThirdDerivative(x, fun):
    k6 = (- fun[:, 12:] + fun[:, :-12]) * 1
    k5 = (+ fun[:, 11:-1] - fun[:, 1:-11]) * 24
    k4 = (- fun[:, 10:-2] + fun[:, 2:-10]) * 192
    k3 = (+ fun[:,  9:-3] - fun[:, 3:-9]) * 488
    k2 = (+ fun[:,  8:-4] - fun[:, 4:-8]) * 387
    k1 = (- fun[:,  7:-5] + fun[:, 5:-7]) * 1584

    dev = k1 + k2 + k3 + k4 + k5 + k6
    dev /= np.power(x[:, 7:-5] - x[:, 6:-6], 3)
    dev /= 12**3

    derivative = np.zeros(fun.shape)
    # need to pad with zeros left and right to return array with same shape
    derivative[:, 6:-6] = dev
    return derivative

def H3(x):
    return 8 * np.power(x, 3) - 12 * x


noFSEPath = currentPath / "data_for_plots_no_fse.npz"
results = np.load(noFSEPath)
spec = 13

deltaQ = results["all_deltaQ"][0, spec, :]
ySpacesForEachMass = results["all_yspaces_for_each_mass"][0, spec, :, :]
JOfY = results["all_ncp_for_each_mass"][0, spec, :, :]
widths = results["all_spec_best_par_chi_nit"][0, spec, 2:-2:3].reshape(4, 1)
centers = results["all_spec_best_par_chi_nit"][0, spec, 3:-2:3].reshape(4, 1)

FSEderivative = - numericalThirdDerivative(ySpacesForEachMass, JOfY) * widths**4 / deltaQ * 1/3

fig, ax = plt.subplots()
ax.fill_between(ySpacesForEachMass[0], JOfY[0], label="J(y)", alpha=0.5)
xmax = ySpacesForEachMass[0][JOfY[0]==np.max(JOfY[0])]
ax.vlines(xmax, 0, np.max(JOfY[0]), linewidth=2)

ax.plot(ySpacesForEachMass[0], FSEderivative[0], label="FSE d3")
ax.fill_between(ySpacesForEachMass[0], JOfY[0] + FSEderivative[0], label="Corrected J(y) d3", alpha=0.4)

FSEH3 = widths / deltaQ /3 /2**(3/2) * H3((ySpacesForEachMass-centers)/widths/2**0.5) * JOfY
ax.plot(ySpacesForEachMass[0], FSEH3[0], label="FSE H3")
ax.fill_between(ySpacesForEachMass[0], JOfY[0] + FSEH3[0], label="Corrected J(y) H3", alpha=0.4)

plt.legend()
plt.show()
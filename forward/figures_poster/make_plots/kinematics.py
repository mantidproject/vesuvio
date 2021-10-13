import unittest
import numpy as np
import numpy.testing as nptest
import matplotlib.pyplot as plt
from mantid.simpleapi import *    
from pathlib import Path
currentPath = Path(__file__).absolute().parent  # Path to the repository

from jupyterthemes import jtplot
jtplot.style()

dataPath = currentPath / "data_for_plots.npz"

class TestPlots(unittest.TestCase):
    def setUp(self):
        results = np.load(dataPath)
        self.dataY = results["all_dataY"][0],    # In the order of the script
        self.dataX = results["all_dataX"][0],
        self.dataE = results["all_dataE"][0],
        self.deltaQ = results["all_deltaQ"][0],
        self.deltaE = results["all_deltaE"][0],
        self.yspaces_for_each_mass = results["all_yspaces_for_each_mass"][0],
        self.spec_best_par_chi_nit = results["all_spec_best_par_chi_nit"][0],
        self.mean_widths = results["all_mean_widths"][0],
        self.mean_intensities = results["all_mean_intensities"][0],
        self.tot_ncp = results["all_tot_ncp"][0],
        self.ncp_for_each_mass = results["all_ncp_for_each_mass"][0]

        self.spec = 0

    def test_structure_factor_plot(self):
        fig = plt.figure()
        ax = plt.axes(projection='3d')
        ax.plot3D(self.deltaQ[self.spec], self.deltaE[self.spec], self.dataY[self.spec], label="Data")
        plt.show()

if __name__ == "__main__":
    unittest.main()
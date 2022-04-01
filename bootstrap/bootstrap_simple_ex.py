import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
currentPath = Path(__file__).absolute().parent

resultsPath = currentPath / "current_forward.npz"
results = np.load(resultsPath)

dataY = results["all_fit_workspaces"][-1]
totNcp = results["all_tot_ncp"][-1]

print(dataY.shape)
print(totNcp.shape)

nSamples = 10
noOfMasses = 4
# Implement Bootsrap of residuals on already corrected data

residuals  = dataY - totNcp    # y = g(x) + res

bootSamples = np.zeros((nSamples, len(dataY), 3*noOfMasses))
# for i in range(ic.nSamples):


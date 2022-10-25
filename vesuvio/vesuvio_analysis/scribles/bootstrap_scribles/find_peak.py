from mantid.simpleapi import Load
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

currPath = Path(__file__).parent.absolute()

n1 = "BaH2_25C_FORWARD_0_Sum.nxs"
n2 = "BaH2_725C_FORWARD_0_Sum.nxs"
n3 = "BaH2_400C_FORWARD_0_Sum.nxs"
n4 = "BaH2_500C_FORWARD_0_Sum.nxs"

def relDiff(x0, x1):
    return np.abs((x0 - x1))

NWidth = 5
fig, axs = plt.subplots(4, 2)
for i, name in enumerate([n1, n2, n3, n4]):
    ws = Load(str(currPath/name))
    dataY = ws.dataY(0).copy()
    axs[i, 0].plot(range(len(dataY)), dataY)
    
    for j in range(1, NWidth):
        diffs = relDiff(dataY[:-j], dataY[j:])
        axs[i, 1].plot(range(j, len(diffs)+j), diffs, label=str(j))
    axs[i, 1].legend()
plt.show()



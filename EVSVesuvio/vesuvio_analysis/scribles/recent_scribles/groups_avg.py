import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit, cost, util
from iminuit.util import make_with_signature, describe, make_func_code
from pathlib import Path
from mantid.simpleapi import Load, CropWorkspace
from scipy import optimize
from scipy import ndimage, signal
import time
import numba as nb
repoPath = Path(__file__).absolute().parent


# def weightedAvg(wsYSpace):
#     """Returns ws with weighted avg of input ws"""

#     dataY = wsYSpace.extractY()
#     dataE = wsYSpace.extractE()

#     # TODO: Revise this, some zeros might not be cut offs

#     dataY[dataY==0] = np.nan
#     dataE[dataE==0] = np.nan

#     meanY, meanE = weightedAvgArr(dataY, dataE)
 
#     tempWs = SumSpectra(wsYSpace)
#     newWs = CloneWorkspace(tempWs, OutputWorkspace=wsYSpace.name()+"_Weighted_Avg")
#     newWs.dataY(0)[:] = meanY
#     newWs.dataE(0)[:] = meanE
#     DeleteWorkspace(tempWs)
#     return newWs


def weightedAvgArr(dataY, dataE):
    meanY = np.nansum(dataY/np.square(dataE), axis=0) / np.nansum(1/np.square(dataE), axis=0)
    meanE = np.sqrt(1 / np.nansum(1/np.square(dataE), axis=0))
    return meanY, meanE


joyPath = repoPath / "wsDHMTjoy.nxs"
resPath = repoPath / "wsDHMTres.nxs"

wsJoY = Load(str(joyPath), OutputWorkspace="wsJoY")
wsRes = Load(str(resPath), OutputWorkspace="wsRes")


dataY = wsJoY.extractY()
dataE = wsJoY.extractE()
dataX = wsJoY.extractX()
dataRes = wsRes.extractY()


idxList = [[3, 5, 6], [0, 2], [1]]

# Test that weighted functino works for one spectrum
y = dataY[0]
e = dataE[0]

meany, meane = weightedAvgArr(y, e)

# Careful to not weight single spectrum
# np.testing.assert_array_equal(y, meany)
# np.testing.assert_array_equal(e, meane)

# Assign masked spectra to nan before running this function

def avgWeightDetGroups(dataX, dataY, dataE, dataRes, idxList):
    wDataX = np.zeros((len(idxList), len(dataY[0])))
    wDataY = np.zeros(wDataX.shape)
    wDataE = np.zeros(wDataX.shape)
    wDataRes = np.zeros(wDataX.shape)

    for i, idxs in enumerate(idxList):
        groupE = dataE[idxs, :]
        groupY = dataY[idxs, :]
        groupX = dataX[idxs, :]
        groupRes = dataRes[idxs, :]
        
        # Deal with masked spectra here

        if len(groupY) == 1:
            meanY, meanE = groupY, groupE
            meanRes = groupRes
            print("Used unaltered spec")

        else:
            meanY, meanE = weightedAvgArr(groupY, groupE)
            meanRes = np.nanmean(groupRes, axis=0)
            print("Used wighted mean")

        np.testing.assert_array_equal(groupX[0], np.nanmean(groupX, axis=0))
        wDataX[i] = groupX[0]
        wDataY[i] = meanY
        wDataE[i] = meanE
        wDataRes[i] = meanRes 
    return wDataX, wDataY, wDataE, wDataRes


# fig, axs = plt.subplots(len(idxList))
# for x, y, e, ax in zip(wDataX, wDataY, wDataE, axs.flat):
#     ax.errorbar(x, y, e, fmt="k.")

plt.show()
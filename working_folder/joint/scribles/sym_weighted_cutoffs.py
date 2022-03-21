import numpy as np


def symmetrizeArr(dataY, dataE):
    """Performs Inverse variance weighting between two oposite points."""

    #TODO: Deal with cut-offs by mirroring opposite data
    dataY = dataY.copy()  # Copy arrays not to risk changing original data
    dataE = dataE.copy()

    cutOffMask = dataE==0
    dataY[cutOffMask] = np.flip(dataY, axis=1)[cutOffMask]
    dataE[cutOffMask] = np.flip(dataE, axis=1)[cutOffMask]
    # End of treatment

    yFlip = np.flip(dataY, axis=1)
    eFlip = np.flip(dataE, axis=1)

    # Inverse variance weighting
    dataYSym = (dataY/dataE**2 + yFlip/eFlip**2) / (1/dataE**2 + 1/eFlip**2)
    dataESym = 1 / np.sqrt(1/dataE**2 + 1/eFlip**2)

    
    assert np.all(dataYSym == np.flip(dataYSym, axis=1))
    assert np.all(dataESym == np.flip(dataESym, axis=1))

    # The errors are do not stay the same between two points with 
    # the same values and errors

    # assert np.all(dataYSym[cutOffMask] == dataY[cutOffMask])
    # assert np.all(dataESym[cutOffMask] == dataE[cutOffMask])

    return dataYSym, dataESym


Y = np.arange(15)
E = np.arange(15)*0.1

Y = Y[np.newaxis, :]
E = E[np.newaxis, :]

Y[:, -2:] = 0
E[:, -2:] = 0

print(Y)
print(E)

ySym, eSym = symmetrizeArr(Y, E)

print(ySym)
print(eSym)

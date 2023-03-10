import numpy as np


def symmetrizeArr(dataYOri, dataEOri):
    """
    Performs Inverse variance weighting between two oposite points.
    When one of the points is a cut-off and the other is a valid point, 
    the final value will be the valid point.
    """
    assert len(dataYOri.shape) == 2, "Symmetrization is written for 2D arrays."
    dataY = dataYOri.copy()  # Copy arrays not to risk changing original data
    dataE = dataEOri.copy()

    cutOffMask = dataE==0
    # Change values of yerr to leave cut-offs unchanged during symmetrisation
    dataE[cutOffMask] = np.full(np.sum(cutOffMask), np.inf)


    yFlip = np.flip(dataY, axis=1)
    eFlip = np.flip(dataE, axis=1)

    # Inverse variance weighting
    dataYSym = (dataY/dataE**2 + yFlip/eFlip**2) / (1/dataE**2 + 1/eFlip**2)
    dataESym = 1 / np.sqrt(1/dataE**2 + 1/eFlip**2)


    # Deal with effects from previously changing dataE=np.inf
    nanInfMask = dataESym==np.inf
    dataYSym[nanInfMask] = 0
    dataESym[nanInfMask] = 0

    # Test that arrays are symmetrised
    np.testing.assert_array_equal(dataYSym, np.flip(dataYSym, axis=1)), f"Symmetrisation failed in {np.argwhere(dataYSym!=np.flip(dataYSym))}"
    np.testing.assert_array_equal(dataESym, np.flip(dataESym, axis=1)), f"Symmetrisation failed in {np.argwhere(dataESym!=np.flip(dataESym))}"

    # Test that cut-offs were not included in the symmetrisation
    np.testing.assert_allclose(dataYSym[cutOffMask], np.flip(dataYOri, axis=1)[cutOffMask])
    np.testing.assert_allclose(dataESym[cutOffMask], np.flip(dataEOri, axis=1)[cutOffMask])

    return dataYSym, dataESym


Y = np.arange(50).reshape((5,10))
E = np.arange(50).reshape((5, 10))*0.1

Y = Y.astype(float)
E = E.astype(float)

zerosMask = Y!=Y
zerosMask[:, :2] = 1
zerosMask[3, 5:] = 1
zerosMask[:, -3:] = 1

Y[zerosMask] = 0
E[zerosMask] = 0

ySym, eSym = symmetrizeArr(Y, E)

np.set_printoptions(1)
# print(Y)
# print(E)


# print(ySym)
# print(eSym)



def weightedAvgArr(dataYOri, dataEOri):
    """Weighted average over 2D arrays."""

    dataY = dataYOri.copy()  # Copy arrays not to change original data
    dataE = dataEOri.copy()

    # Ignore invalid data by changing zeros to nans
    zerosMask = dataE==0
    dataY[zerosMask] = np.nan  
    dataE[zerosMask] = np.nan

    meanY = np.nansum(dataY/np.square(dataE), axis=0) / np.nansum(1/np.square(dataE), axis=0)
    meanE = np.sqrt(1 / np.nansum(1/np.square(dataE), axis=0))

    # Change invalid data back to original format with zeros
    nanInfMask = meanE==np.inf
    meanY[nanInfMask] = 0
    meanE[nanInfMask] = 0

    # Test that columns of zeros are left unchanged
    np.testing.assert_allclose((np.sum(dataYOri, axis=0)==0), (meanY==0)), "Collumns of zeros are not being ignored."
    np.testing.assert_allclose((np.sum(dataEOri, axis=0)==0), (meanE==0)), "Collumns of zeros are not being ignored."
    
    return meanY, meanE


meanY, meanE = weightedAvgArr(Y, E)
print(Y)
print(E)


print(meanY)
print(meanE)



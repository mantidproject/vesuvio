
from mantid.simpleapi import Load, Rebin, Scale, SumSpectra, Minus, CropWorkspace, \
                            CloneWorkspace, MaskDetectors, CreateWorkspace
import numpy as np
import numbers

from mvesuvio.analysis_fitting import passDataIntoWS


def loadRawAndEmptyWsFromUserPath(userWsRawPath, userWsEmptyPath, 
                                  tofBinning, name, scaleRaw, scaleEmpty, subEmptyFromRaw):
    print("\nLoading local workspaces ...\n")
    Load(Filename=str(userWsRawPath), OutputWorkspace=name + "raw")
    Rebin(
        InputWorkspace=name + "raw",
        Params=tofBinning,
        OutputWorkspace=name + "raw",
    )

    assert (isinstance(scaleRaw, numbers.Real)), "Scaling factor of raw ws needs to be float or int."
    Scale(
        InputWorkspace=name + "raw",
        OutputWorkspace=name + "raw",
        Factor=str(scaleRaw),
    )

    SumSpectra(InputWorkspace=name + "raw", OutputWorkspace=name + "raw" + "_sum")
    wsToBeFitted = CloneWorkspace(
        InputWorkspace=name + "raw", OutputWorkspace=name + "uncropped_unmasked"
    )

    # if mode=="DoubleDifference":
    if subEmptyFromRaw:
        Load(Filename=str(userWsEmptyPath), OutputWorkspace=name + "empty")
        Rebin(
            InputWorkspace=name + "empty",
            Params=tofBinning,
            OutputWorkspace=name + "empty",
        )

        assert (isinstance(scaleEmpty, float)) | (
            isinstance(scaleEmpty, int)
        ), "Scaling factor of empty ws needs to be float or int"
        Scale(
            InputWorkspace=name + "empty",
            OutputWorkspace=name + "empty",
            Factor=str(scaleEmpty),
        )

        SumSpectra(
            InputWorkspace=name + "empty", OutputWorkspace=name + "empty" + "_sum"
        )

        wsToBeFitted = Minus(
            LHSWorkspace=name + "raw",
            RHSWorkspace=name + "empty",
            OutputWorkspace=name + "uncropped_unmasked",
        )
    return wsToBeFitted


def cropAndMaskWorkspace(ws, firstSpec, lastSpec, maskedDetectors, maskTOFRange):
    """Returns cloned and cropped workspace with modified name"""
    # Read initial Spectrum number
    wsFirstSpec = ws.getSpectrumNumbers()[0]
    assert (
        firstSpec >= wsFirstSpec
    ), "Can't crop workspace, firstSpec < first spectrum in workspace."

    initialIdx = firstSpec - wsFirstSpec
    lastIdx = lastSpec - wsFirstSpec

    newWsName = ws.name().split("uncropped")[0]  # Retrieve original name
    wsCrop = CropWorkspace(
        InputWorkspace=ws,
        StartWorkspaceIndex=initialIdx,
        EndWorkspaceIndex=lastIdx,
        OutputWorkspace=newWsName,
    )

    maskBinsWithZeros(wsCrop, maskTOFRange)  # Used to mask resonance peaks

    MaskDetectors(Workspace=wsCrop, SpectraList=maskedDetectors)
    return wsCrop


def maskBinsWithZeros(ws, maskTOFRange):
    """
    Masks a given TOF range on ws with zeros on dataY.
    Leaves errors dataE unchanged, as they are used by later treatments.
    Used to mask resonance peaks.
    """

    if maskTOFRange is None:
        return

    dataX, dataY, dataE = extractWS(ws)
    start, end = [int(s) for s in maskTOFRange.split(",")]
    assert (
        start <= end
    ), "Start value for masking needs to be smaller or equal than end."
    mask = (dataX >= start) & (dataX <= end)  # TOF region to mask

    dataY[mask] = 0

    passDataIntoWS(dataX, dataY, dataE, ws)
    return


def extractWS(ws):
    """Directly extracts data from workspace into arrays"""
    return ws.extractX(), ws.extractY(), ws.extractE()


def histToPointData(dataY, dataX, dataE):
    """
    Used only when comparing with original results.
    Sets each dataY point to the center of bins.
    Last column of data is removed.
    Removed original scaling by bin widths
    """

    histWidths = dataX[:, 1:] - dataX[:, :-1]
    assert np.min(histWidths) == np.max(
        histWidths
    ), "Histogram widths need to be the same length"

    dataYp = dataY[:, :-1]
    dataEp = dataE[:, :-1]
    dataXp = dataX[:, :-1] + histWidths[0, 0] / 2
    return dataYp, dataXp, dataEp


def loadConstants():
    """Output: the mass of the neutron, final energy of neutrons (selected by gold foil),
    factor to change energies into velocities, final velocity of neutron and hbar"""
    mN = 1.008  # a.m.u.
    Ef = 4906.0  # meV
    en_to_vel = 4.3737 * 1.0e-4
    vf = np.sqrt(Ef) * en_to_vel  # m/us
    hbar = 2.0445
    constants = (mN, Ef, en_to_vel, vf, hbar)
    return constants


def gaussian(x, sigma):
    """Gaussian centered at zero"""
    gauss = np.exp(-(x**2) / 2 / sigma**2) 
    gauss /= np.sqrt(2.0 * np.pi) * sigma
    return gauss


def lorentzian(x, gamma):
    """Lorentzian centered at zero"""
    return gamma / np.pi / (x**2 + gamma**2)


def numericalThirdDerivative(x, y):
    k6 = (- y[:, 12:] + y[:, :-12]) * 1
    k5 = (+ y[:, 11:-1] - y[:, 1:-11]) * 24
    k4 = (- y[:, 10:-2] + y[:, 2:-10]) * 192
    k3 = (+ y[:, 9:-3] - y[:, 3:-9]) * 488
    k2 = (+ y[:, 8:-4] - y[:, 4:-8]) * 387
    k1 = (- y[:, 7:-5] + y[:, 5:-7]) * 1584

    dev = k1 + k2 + k3 + k4 + k5 + k6
    dev /= np.power(x[:, 7:-5] - x[:, 6:-6], 3)
    dev /= 12**3

    derivative = np.zeros_like(y)
    # Padded with zeros left and right to return array with same shape
    derivative[:, 6:-6] = dev
    return derivative


def createWS(dataX, dataY, dataE, wsName, parentWorkspace=None):
    ws = CreateWorkspace(
        DataX=dataX.flatten(),
        DataY=dataY.flatten(),
        DataE=dataE.flatten(),
        Nspec=len(dataY),
        OutputWorkspace=wsName,
        ParentWorkspace=parentWorkspace
    )
    return ws

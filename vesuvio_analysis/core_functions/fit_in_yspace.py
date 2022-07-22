from fileinput import filename
from inspect import signature
from multiprocessing.sharedctypes import Value
from tracemalloc import start
import matplotlib.pyplot as plt
import numpy as np
from mantid.simpleapi import *
from scipy import optimize
from scipy import ndimage, signal, interpolate
from pathlib import Path
from iminuit import Minuit, cost, util
from iminuit.util import make_func_code, describe
import time

repoPath = Path(__file__).absolute().parent  # Path to the repository



def fitInYSpaceProcedure(yFitIC, IC, wsFinal):

    ncpForEachMass = extractNCPFromWorkspaces(wsFinal, IC)
    wsResSum, wsRes = calculateMantidResolutionFirstMass(IC, yFitIC, wsFinal)

    wsSubMass = subtractAllMassesExceptFirst(IC, wsFinal, ncpForEachMass)
    if yFitIC.maskTOFRange != None:     # Mask resonance peak
        wsSubMass = maskResonancePeak(yFitIC, wsSubMass, ncpForEachMass[:, 0, :])  # Mask with ncp from first mass

    wsYSpace, wsQ = convertToYSpace(yFitIC.rebinParametersForYSpaceFit, wsSubMass, IC.masses[0]) 
    wsYSpace = putAllSpecInSameRange(wsYSpace, yFitIC)
    wsYSpaceAvg = reduceToWeightedAverage(wsYSpace, yFitIC)
    
    if yFitIC.symmetrisationFlag:
        wsYSpaceAvg = symmetrizeWs(wsYSpaceAvg)

    fitProfileMinuit(yFitIC, wsYSpaceAvg, wsResSum)
    fitProfileMantidFit(yFitIC, wsYSpaceAvg, wsResSum)
    
    printYSpaceFitResults(wsYSpaceAvg.name())

    yfitResults = ResultsYFitObject(IC, yFitIC, wsFinal.name(), wsYSpaceAvg.name())
    yfitResults.save()
    
    if yFitIC.globalFit:
        runGlobalFit(wsYSpace, wsRes, IC, yFitIC) 
    return yfitResults


def extractNCPFromWorkspaces(wsFinal, ic):
    """Extra function to extract ncps from loaded ws in mantid."""

    ncpForEachMass = mtd[wsFinal.name()+"_TOF_Fitted_Profile_0"].extractY()[np.newaxis, :, :]
    for i in range(1, ic.noOfMasses):
        ncpToAppend = mtd[wsFinal.name()+"_TOF_Fitted_Profile_" + str(i)].extractY()[np.newaxis, :, :]
        ncpForEachMass = np.append(ncpForEachMass, ncpToAppend, axis=0)    

    assert ncpForEachMass.shape == (ic.noOfMasses, wsFinal.getNumberHistograms(), wsFinal.blocksize()-1), "Extracted NCP not in correct shape."
    
    ncpForEachMass = switchFirstTwoAxis(ncpForEachMass)  # Organizes ncp by spectra
    print(f"\nExtracted NCP profiles from workspaces.\n")
    return ncpForEachMass


def maskResonancePeak(yFitIC, ws, ncp):
    """Masks a given TOF range on input ws. Currently acting on isolated mass ws."""

    start, end = [int(s) for s in yFitIC.maskTOFRange.split(",")]
    assert start <= end, "Start value for masking needs to be smaller or equal than end."
    dataY = ws.extractY()[:, :-1]
    dataX = ws.extractX()[:, :-1]

    # Mask dataY with NCP in given TOF region
    mask = (dataX >= start) & (dataX <= end)

    flag = yFitIC.maskTypeProcedure

    if flag=="NCP_&_REBIN":
        dataY[mask] = ncp[mask]   # Replace values by best fit NCP

    elif (flag=="NAN_&_INTERP") | (flag=="NAN_&_BIN"):
        dataY[mask] = 0    # Zeros are preserved during ConvertToYSpace

    else:
        raise ValueError ("Mask type not recognized, options: 'NCP_&_REBIN', 'NAN_&_INTERP', 'NAN_&_BIN'")


    wsMasked = CloneWorkspace(ws, OutputWorkspace=ws.name()+"_Masked")
    for i in range(wsMasked.getNumberHistograms()):
        wsMasked.dataY(i)[:-1] = dataY[i, :]
    SumSpectra(wsMasked, OutputWorkspace=wsMasked.name()+"_Sum")
    return wsMasked
    

def calculateMantidResolutionFirstMass(IC, yFitIC, ws):
    mass = IC.masses[0]

    resName = ws.name()+"_Resolution"
    for index in range(ws.getNumberHistograms()):
        VesuvioResolution(Workspace=ws,WorkspaceIndex=index,Mass=mass,OutputWorkspaceYSpace="tmp")
        Rebin(InputWorkspace="tmp", Params=yFitIC.rebinParametersForYSpaceFit, OutputWorkspace="tmp")

        if index == 0:   # Ensures that workspace has desired units
            RenameWorkspace("tmp",  resName)
        else:
            AppendSpectra(resName, "tmp", OutputWorkspace=resName)
   
    MaskDetectors(resName, WorkspaceIndexList=IC.maskedDetectorIdx)
    wsResSum = SumSpectra(InputWorkspace=resName, OutputWorkspace=resName+"_Sum")
 
    normalise_workspace(wsResSum)
    DeleteWorkspace("tmp")
    return wsResSum, mtd[resName]

    
def normalise_workspace(ws_name):
    tmp_norm = Integration(ws_name)
    Divide(LHSWorkspace=ws_name,RHSWorkspace=tmp_norm,OutputWorkspace=ws_name)
    DeleteWorkspace("tmp_norm")


def subtractAllMassesExceptFirst(ic, ws, ncpForEachMass):
    """Input: workspace from last iteration, ncpTotal for each mass
       Output: workspace with all the ncpTotal subtracted except for the first mass"""

    ncpForEachMass = switchFirstTwoAxis(ncpForEachMass)
    # Select all masses other than the first one
    ncpForEachMassExceptFirst = ncpForEachMass[1:, :, :]
    # Sum the ncpTotal for remaining masses
    ncpTotalExceptFirst = np.sum(ncpForEachMassExceptFirst, axis=0)

    wsSubMass = CloneWorkspace(InputWorkspace=ws, OutputWorkspace=ws.name()+"_Mass0")
    for j in range(wsSubMass.getNumberHistograms()):
        if wsSubMass.spectrumInfo().isMasked(j):
            continue

        # Due to different sizes, last value of original ws remains untouched
        binWidths = wsSubMass.dataX(j)[1:] - wsSubMass.dataX(j)[:-1]
        wsSubMass.dataY(j)[:-1] -= ncpTotalExceptFirst[j] * binWidths

     # Mask spectra again, to be seen as masked from Mantid's perspective
    MaskDetectors(Workspace=wsSubMass, WorkspaceIndexList=ic.maskedDetectorIdx)  

    SumSpectra(InputWorkspace=wsSubMass.name(), OutputWorkspace=wsSubMass.name()+"_Sum")

    if np.any(np.isnan(wsSubMass.extractY())):
        raise ValueError("The workspace for the isolated first mass countains NaNs in non-masked spectra, might cause problems!")
    return wsSubMass


def switchFirstTwoAxis(A):
    """Exchanges the first two indices of an array A,
    rearranges matrices per spectrum for iteration of main fitting procedure
    """
    return np.stack(np.split(A, len(A), axis=0), axis=2)[0]


def convertToYSpace(rebinPars, ws0, mass):
    wsJoY, wsQ = ConvertToYSpace(
        InputWorkspace=ws0, Mass=mass, 
    OutputWorkspace=ws0.name()+"_JoY", QWorkspace=ws0.name()+"_Q"
        )
    # wsJoY = Rebin(
    #     InputWorkspace=wsJoY, Params=rebinPars, 
    #     FullBinsOnly=True, OutputWorkspace=ws0.name()+"_JoY"
    #     )
    # wsQ = Rebin(
    #     InputWorkspace=wsQ, Params=rebinPars, 
    #     FullBinsOnly=True, OutputWorkspace=ws0.name()+"_Q"
    #     )
    
    # If workspace has nans present, normalization will put zeros on the full spectrum
    assert np.any(np.isnan(wsJoY.extractY()))==False, "Nans present before normalization."
    
    # normalise_workspace(wsJoY)
    return wsJoY, wsQ

def buildXRangeFromRebinPars(yFitIC):
    # Range used in case mask is set to NAN
    first, step, last = [float(s) for s in yFitIC.rebinParametersForYSpaceFit.split(",")]
    xp = np.arange(first, last, step) + step/2   # Correction to match Mantid range
    return xp



def putAllSpecInSameRange(wsJoY, yFitIC):
    rebinPars = yFitIC.rebinParametersForYSpaceFit

    # In case where no masking is present, use Mantid Rebin
    if yFitIC.maskTOFRange==None:
        wsJoYR = Rebin( InputWorkspace=wsJoY, Params=rebinPars, FullBinsOnly=True, OutputWorkspace=wsJoY.name()+"_Rebinned")
        # normalise_workspace(wsJoYR)
        return wsJoYR

    # Else use one of the three available procedures
    maskProc = yFitIC.maskTypeProcedure
    # Range used in case of interpolation or special binning
    xp = buildXRangeFromRebinPars(yFitIC)

    if maskProc=="NCP_&_REBIN":
        assert ~np.any(np.all(wsJoY.extractY()==0), axis=0), "Rebin cannot operate on JoY ws with masked values."
        wsJoYR = Rebin( InputWorkspace=wsJoY, Params=rebinPars, FullBinsOnly=True, OutputWorkspace=wsJoY.name()+"_Rebinned")
        normalise_workspace(wsJoYR)
    
    elif maskProc=="NAN_&_INTERP":
        wsJoYR = interpYSpace(wsJoY, xp)   # Interpolates onto range xp
        normalise_workspace(wsJoYR)

    elif maskProc=="NAN_&_BIN":
        wsJoYR = dataXBining(wsJoY, xp)    # xp range is used as centers of bins
        # In this case, wsJoYR is not yet reduced for suitable normalisation, so do that after averaging.

    else:
        raise ValueError("yFitIC.maskTypeProcedure not recognized.")

    return wsJoYR



def interpYSpace(ws, xp):
    dataX, dataY, dataE = extractWS(ws)

    # Change zeros to nans, to make sure they are ignored during interpolation
    mask = (dataY==0) | (dataE==0)
    for data in [dataY, dataE]:
        data[mask] = np.nan

    # New interpolated dimensions    
    dataXP = np.zeros((len(dataX), len(xp)))
    dataYP = dataXP.copy()
    dataEP = dataXP.copy()

    for i, (x, y, e) in enumerate(zip(dataX, dataY, dataE)):

        if x[0] > xp[0]:    # Correct for interpolated range
            x = np.hstack(([xp[0]], x))
            y = np.hstack(([np.nan], y))
            e = np.hstack(([np.nan], e))

        if x[-1] < xp[-1]:    # Correct for interpolated range
            x = np.hstack((x, [xp[-1]]))
            y = np.hstack((y, [np.nan]))
            e = np.hstack((e, [np.nan]))

        yp, ep = interpSpec(xp, x, y, e)

        dataXP[i] = xp
        dataYP[i] = yp
        dataEP[i] = ep

    # # Change NaNs to zeros to match masking of Rebin()
    # assert np.all((dataYP==np.nan)==(dataEP==np.nan)), "Masked values with nans should be the same on dataY and dataE."
    # nanMask = dataYP==np.nan
    # dataYP[nanMask] = 0
    # dataEP[nanMask] = 0

    wsInterp = CreateWorkspace(DataX=dataXP.flatten(), DataY=dataYP.flatten(), DataE=dataEP.flatten(), NSpec=len(dataXP), OutputWorkspace=ws.name()+"_Interp")
    return wsInterp


def interpSpec(xp, x, y, e):
    f = interpolate.interp1d(x, y)
    yp = f(xp)
    # Calculate errors on interpolated values
    fPlus = interpolate.interp1d(x, y+e)
    fMinus = interpolate.interp1d(x, y-e)
    ep = (fPlus(xp) - fMinus(xp)) / 2
    return yp, ep


def extractWS(ws):
    """Directly exctracts data from workspace into arrays"""
    return ws.extractX(), ws.extractY(), ws.extractE()


def passDataIntoWS(dataX, dataY, dataE, ws):
    "Modifies ws data to input data"
    for i in range(ws.getNumberHistograms()):
        ws.dataX(i)[:] = dataX[i, :]
        ws.dataY(i)[:] = dataY[i, :]
        ws.dataE(i)[:] = dataE[i, :]
    return ws


def dataXBining(ws, xp):

    assert np.min(xp[:-1]-xp[1:]) == np.max(xp[:-1]-xp[1:]), "Bin widths need to be the same."
    step = xp[1] - xp[0]   # Calculate step from first two numbers
    # Form bins with xp being the centers
    bins = np.append(xp, [xp[-1]+step]) - step/2

    dataX, dataY, dataE = extractWS(ws)
    # Loop below changes only the values of DataX
    for i, x in enumerate(dataX):

        # Select only valid range xr
        mask = (x<np.min(bins)) | (x>np.max(bins))
        xr = x[~mask]

        idxs = np.digitize(xr, bins)
        newXR = np.array([xp[idx] for idx in idxs-1])  # Bin idx 1 refers to first bin ie idx 0 of centers

        # Pad unvalid values with nans
        newX = x
        newX[mask] = np.nan
        newX[~mask] = newXR
        dataX[i] = newX       # Update DataX

    # Mask zeros with nans 
    mask = dataY==0
    dataY[mask] = np.nan
    dataE[mask] = np.nan

    wsXBins = CloneWorkspace(ws, OutputWorkspace=ws.name()+"_XBinned")
    wsXBins = passDataIntoWS(dataX, dataY, dataE, wsXBins)
    return wsXBins


def reduceToWeightedAverage(wsJoYR, yFitIC):

    if yFitIC.maskTOFRange==None: 
        wsYSpaceAvg = weightedAvgCols(wsJoYR)
        # normalise_workspace(wsYSpaceAvg)
        return wsYSpaceAvg

    maskProc = yFitIC.maskTypeProcedure

    if (maskProc=="NCP_&_REBIN") | (maskProc=="NAN_&_INTERP"):
        wsYSpaceAvg = weightedAvgCols(wsJoYR)

    elif maskProc=="NAN_&_BIN":
        xp = buildXRangeFromRebinPars(yFitIC)
        wsYSpaceAvg = weightedAvgXBins(wsJoYR, xp)
        normalise_workspace(wsYSpaceAvg)

    else:
        raise ValueError("yFitIC.maskTypeProcedure not recognized.")

    return wsYSpaceAvg


def weightedAvgXBins(wsXBins, xp):
    dataX, dataY, dataE = extractWS(wsXBins)

    meansY, meansE = weightedAvgXBinsArr(dataX, dataY, dataE, xp)

    wsYSpaceAvg = CreateWorkspace(DataX=xp, DataY=meansY, DataE=meansE, NSpec=1, OutputWorkspace=wsXBins.name()+"_WeightedAvg")
    return wsYSpaceAvg


def weightedAvgXBinsArr(dataX, dataY, dataE, xp):
    meansY = np.zeros(len(xp))
    meansE = np.zeros(len(xp))
    for i in range(len(xp)):
        # Perform weighted average over all dataY and dataE values with the same xp[i]
        # Change shape to column to match weighted average function
        allY = dataY[dataX==xp[i]][:, np.newaxis]
        allE = dataE[dataX==xp[i]][:, np.newaxis]
        assert allY.shape==allE.shape, "Selection of points Y and E with same X should be the same."

        if (allY.size==0):   # If no points were found for a given abcissae
            mY, mE = 0, 0  # Mask with zeros
        elif (allY.size==1):   # If one point was found, set to that point
            mY, mE = allY[0, 0], allE[0, 0]
        else:
            # Weighted avg over all spectra and several points per spectra
            mY, mE = weightedAvgArr(allY, allE)    # Outputs masks with zeros

        meansY[i] = mY
        meansE[i] = mE
    
    return meansY, meansE


def weightedAvgCols(wsYSpace):
    """Returns ws with weighted avg of input ws"""
    
    dataX, dataY, dataE = extractWS(wsYSpace)
    # dataY = wsYSpace.extractY()
    # dataE = wsYSpace.extractE()

    meanY, meanE = weightedAvgArr(dataY, dataE)

    wsYSpaceAvg = CreateWorkspace(DataX=dataX[0, :], DataY=meanY, DataE=meanE, NSpec=1, OutputWorkspace=wsYSpace.name()+"_WeightedAvg")
    # tempWs = SumSpectra(wsYSpace)
    # newWs = CloneWorkspace(tempWs, OutputWorkspace=wsYSpace.name()+"_Weighted_Avg")
    # newWs.dataY(0)[:] = meanY
    # newWs.dataE(0)[:] = meanE
    # DeleteWorkspace(tempWs)

    return wsYSpaceAvg


def weightedAvgArr(dataYOri, dataEOri):
    """Weighted average over columns of 2D arrays."""

    # Run some tests
    assert dataYOri.shape==dataEOri.shape, "Y and E arrays should have same shape for weighted average."
    assert np.all((dataYOri==0)==(dataEOri==0)), "Masked zeros should match in DataY and DataE."
    assert np.all(np.isnan(dataYOri)==np.isnan(dataEOri)), "Masked nans should match in DataY and DataE."
    assert len(dataYOri) > 1, "Weighted average needs more than one element to be performed."

    dataY = dataYOri.copy()  # Copy arrays not to change original data
    dataE = dataEOri.copy()

    # Ignore invalid data by changing zeros to nans
    # If data is already masked with nans, it remains unaltered
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
    assert np.all((meanY==0)==(meanE==0)), "Weighted avg output should have masks in the same DataY and DataE."
    assert np.all((np.all(dataYOri==0, axis=0) | np.all(np.isnan(dataYOri), axis=0)) == (meanY==0)), "Masked cols should be ignored."
    
    return meanY, meanE


def symmetrizeWs(avgYSpace):
    """Symmetrizes workspace after weighted average,
       Needs to have symmetric binning"""

    # dataX = avgYSpace.extractX()
    # dataY = avgYSpace.extractY()
    # dataE = avgYSpace.extractE()
    dataX, dataY, dataE = extractWS(avgYSpace)

    dataYSym, dataESym = symmetrizeArr(dataY, dataE)

    wsSym = CloneWorkspace(avgYSpace, OutputWorkspace=avgYSpace.name()+"_Symmetrised")
    wsSym = passDataIntoWS(dataX, dataYSym, dataESym, wsSym)
    # for i in range(Sym.getNumberHistograms()):
    #     Sym.dataY(i)[:] = dataYSym[i]
    #     Sym.dataE(i)[:] = dataESym[i] 
    return wsSym


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


def fitProfileMinuit(yFitIC, wsYSpaceSym, wsRes):

    dataX, dataY, dataE = extractFirstSpectra(wsYSpaceSym)
    resX, resY, resE = extractFirstSpectra(wsRes)
    assert np.all(dataX==resX), "Resolution should operate on the same range as DataX"

    model, defaultPars, sharedPars = selectModelAndPars(yFitIC.fitModel)

    xDelta, resDense = oddPointsRes(resX, resY)
    def convolvedModel(x, y0, *pars):
        return y0 + signal.convolve(model(x, *pars), resDense, mode="same") * xDelta

    signature = describe(model)[:]      # Build signature of convolved function
    signature[1:1] = ["y0"]     # Add intercept as first fitting parameter after range 'x'

    convolvedModel.func_code = make_func_code(signature)    
    defaultPars["y0"] = 0    # Add initialization of parameter to dictionary

    # Fit only valid values, ignore cut-offs 
    dataXNZ, dataYNZ, dataENZ = selectNonZeros(dataX, dataY, dataE)

    # Fit with Minuit
    costFun = cost.LeastSquares(dataXNZ, dataYNZ, dataENZ, convolvedModel)
    m = Minuit(costFun, **defaultPars)

    m.limits["A"] = (0, None)
    if yFitIC.fitModel=="DOUBLE_WELL":
        m.limits["d"] = (0, None)
        m.limits["R"] = (0, None)

    if yFitIC.fitModel=="SINGLE_GAUSSIAN":
        m.simplex()
        m.migrad()

        def constrFunc()->None:  # No constraint function for gaussian profile
            return
    else:
        def constrFunc(*pars):   # Constrain physical model before convolution
            return model(dataXNZ, *pars[1:])   # First parameter is intercept, not part of model()
        
        m.simplex()
        m.scipy(constraints=optimize.NonlinearConstraint(constrFunc, 0, np.inf))

    # Explicit calculation of Hessian after the fit
    m.hesse()

    # Weighted Chi2
    chi2 = m.fval / (len(dataXNZ)-m.nfit)

    # Best fit and confidence band
    # Calculated for the whole range of dataX, including where zero
    dataYFit, dataYCov = util.propagate(lambda pars: convolvedModel(dataX, *pars), m.values, m.covariance)
    dataYSigma = np.sqrt(np.diag(dataYCov))
    dataYSigma *= chi2        # Weight the confidence band
    Residuals = dataY - dataYFit

    # Create workspace to store best fit curve and errors on the fit
    wsMinFit = createFitResultsWorkspace(wsYSpaceSym, dataX, dataY, dataE, dataYFit, dataYSigma, Residuals)
    saveMinuitPlot(yFitIC, wsMinFit, m)

    # Calculate correlation matrix
    corrMatrix = m.covariance.correlation()
    corrMatrix *= 100

    # Create correlation tableWorkspace
    createCorrelationTableWorkspace(wsYSpaceSym, m.parameters, corrMatrix)

    # Run Minos
    fitCols = runMinos(m, yFitIC, constrFunc, wsYSpaceSym.name())

    # Create workspace with final fitting parameters and their errors
    createFitParametersTableWorkspace(wsYSpaceSym, *fitCols, chi2)
    return 


def extractFirstSpectra(ws):
    dataY = ws.extractY()[0]
    dataX = ws.extractX()[0]
    dataE = ws.extractE()[0]
    return dataX, dataY, dataE


def selectModelAndPars(modelFlag):
    """Selects the function to fit, the starting parameters of that function and the shared parameters in global fit."""

    if modelFlag == "SINGLE_GAUSSIAN":
        def model(x, A, x0, sigma):
            return  A / (2*np.pi)**0.5 / sigma * np.exp(-(x-x0)**2/2/sigma**2)

        defaultPars = {"y0":0, "A":1, "x0":0, "sigma":5}
        sharedPars = ["sigma"]    # Used only in Global fit

    elif (modelFlag=="GC_C4_C6"):
        def model(x, A, x0, sigma1, c4, c6):
            return  A * np.exp(-(x-x0)**2/2/sigma1**2) / (np.sqrt(2*np.pi*sigma1**2)) \
                    *(1 + c4/32*(16*((x-x0)/np.sqrt(2)/sigma1)**4 \
                    -48*((x-x0)/np.sqrt(2)/sigma1)**2+12) \
                    +c6/384*(64*((x-x0)/np.sqrt(2)/sigma1)**6 \
                    -480*((x-x0)/np.sqrt(2)/sigma1)**4 + 720*((x-x0)/np.sqrt(2)/sigma1)**2 - 120))
        
        defaultPars = {"A":1, "x0":0, "sigma1":6, "c4":0, "c6":0} 
        sharedPars = ["sigma1", "c4", "c6"]     # Used only in Global fit

    elif modelFlag=="GC_C4":
        def model(x, A, x0, sigma1, c4):
            return  A * np.exp(-(x-x0)**2/2/sigma1**2) / (np.sqrt(2*np.pi*sigma1**2)) \
                    *(1 + c4/32*(16*((x-x0)/np.sqrt(2)/sigma1)**4 \
                    -48*((x-x0)/np.sqrt(2)/sigma1)**2+12))
        
        defaultPars = {"A":1, "x0":0, "sigma1":6, "c4":0} 
        sharedPars = ["sigma1", "c4"]     # Used only in Global fit   
    
    elif modelFlag=="GC_C6":
        def model(x, A, x0, sigma1, c6):
            return  A * np.exp(-(x-x0)**2/2/sigma1**2) / (np.sqrt(2*np.pi*sigma1**2)) \
                    *(1 + +c6/384*(64*((x-x0)/np.sqrt(2)/sigma1)**6 \
                    -480*((x-x0)/np.sqrt(2)/sigma1)**4 + 720*((x-x0)/np.sqrt(2)/sigma1)**2 - 120))
        
        
        defaultPars = {"A":1, "x0":0, "sigma1":6, "c6":0} 
        sharedPars = ["sigma1", "c6"]     # Used only in Global fit   

    elif modelFlag=="DOUBLE_WELL":
        def model(x, A, d, R, sig1, sig2):
            h = 2.04
            theta = np.linspace(0, np.pi, 300)[:, np.newaxis]
            y = x[np.newaxis, :]

            sigTH = np.sqrt( sig1**2*np.cos(theta)**2 + sig2**2*np.sin(theta)**2 )
            alpha = 2*( d*sig2*sig1*np.sin(theta) / sigTH )**2
            beta = ( 2*sig1**2*d*np.cos(theta) / sigTH**2 ) * y
            denom = 2.506628 * sigTH * (1 + R**2 + 2*R*np.exp(-2*d**2*sig1**2))
            jp = np.exp( -y**2/(2*sigTH**2)) * (1 + R**2 + 2*R*np.exp(-alpha)*np.cos(beta)) / denom
            jp *= np.sin(theta)

            JBest = np.trapz(jp, x=theta, axis=0)
            JBest /= np.abs(np.trapz(JBest, x=y))
            JBest *= A
            return JBest

        defaultPars = {"A":1, "d":1, "R":1, "sig1":3, "sig2":5}  # TODO: Starting parameters and bounds?
        sharedPars = ["d", "R", "sig1", "sig2"]      # Only varying parameter is amplitude A     

    elif modelFlag=="DOUBLE_WELL_ANSIO":
        # Ansiotropic case
        def model(x, A, sig1, sig2):
            h = 2.04
            theta = np.linspace(0, np.pi, 300)[:, np.newaxis]
            y = x[np.newaxis, :]

            sigTH = np.sqrt( sig1**2*np.cos(theta)**2 + sig2**2*np.sin(theta)**2 )
            jp = np.exp( -y**2/(2*sigTH**2)) / (2.506628*sigTH)
            jp *= np.sin(theta)

            JBest = np.trapz(jp, x=theta, axis=0)
            JBest /= np.abs(np.trapz(JBest, x=y))
            JBest *= A
            return JBest

        defaultPars = {"A":1, "sig1":3, "sig2":5}
        sharedPars = ["sig1", "sig2"]           

    else:
        raise ValueError("Fitting Model not recognized, available options: 'SINGLE_GAUSSIAN', 'GC_C4_C6', 'GC_C4'")
    
    print("\nShared Parameters: ", [key for key in sharedPars])
    print("\nUnshared Parameters: ", [key for key in defaultPars if key not in sharedPars])
    
    assert all(isinstance(item, str) for item in sharedPars), "Parameters in list must be strings."
    assert describe(model)[-len(sharedPars):]==sharedPars, "Function signature needs to have shared parameters at the end: model(*unsharedPars, *sharedPars)"
    
    return model, defaultPars, sharedPars


def selectNonZeros(dataX, dataY, dataE):
    nonZeros = (dataE!=0) & (dataE!=np.nan) & (dataE!=np.inf) & (dataY!=np.nan)  # Invalid values should have errors=0, but cover other invalid cases as well
    dataXNZ = dataX[nonZeros]
    dataYNZ = dataY[nonZeros]
    dataENZ = dataE[nonZeros]   
    return dataXNZ, dataYNZ, dataENZ 


def createFitResultsWorkspace(wsYSpaceSym, dataX, dataY, dataE, dataYFit, dataYSigma, Residuals):
    """Creates workspace similar to the ones created by Mantid Fit."""

    wsMinFit = CreateWorkspace(DataX=np.concatenate((dataX, dataX, dataX)), 
                    DataY=np.concatenate((dataY, dataYFit, Residuals)), 
                    DataE=np.concatenate((dataE, dataYSigma, np.zeros(len(dataE)))),
                    NSpec=3,
                    OutputWorkspace=wsYSpaceSym.name()+"_Fitted_Minuit")
    return wsMinFit


def saveMinuitPlot(yFitIC, wsMinuitFit, mObj):

    leg = ""
    for p, v, e in zip(mObj.parameters, mObj.values, mObj.errors):
        leg += f"${p}={v:.2f} \pm {e:.2f}$\n"

    fig, ax = plt.subplots(subplot_kw={"projection":"mantid"})
    ax.errorbar(wsMinuitFit, "k.", wkspIndex=0, label="Weighted Avg")
    ax.errorbar(wsMinuitFit, "r-", wkspIndex=1, label=leg)
    ax.set_xlabel("YSpace")
    ax.set_ylabel("Counts")
    ax.set_title("Minuit Fit")
    ax.legend()

    fileName = wsMinuitFit.name()+".pdf"
    savePath = yFitIC.figSavePath / fileName
    plt.savefig(savePath, bbox_inches="tight")
    plt.close(fig)
    return


def createCorrelationTableWorkspace(wsYSpaceSym, parameters, corrMatrix):
    tableWS = CreateEmptyTableWorkspace(OutputWorkspace=wsYSpaceSym.name()+"_Fitted_Minuit_NormalizedCovarianceMatrix")
    tableWS.setTitle("Minuit Fit")
    tableWS.addColumn(type='str',name="Name")
    for p in parameters:
        tableWS.addColumn(type='float',name=p)
    for p, arr in zip(parameters, corrMatrix):
        tableWS.addRow([p] + list(arr))
 

def runMinos(mObj, yFitIC, constrFunc, wsName):
    """Outputs columns to be displayed in a table workspace"""

    # Extract info from fit before running any MINOS
    parameters = list(mObj.parameters)
    values = list(mObj.values)
    errors = list(mObj.errors)

    # If minos is set not to run, ouput columns with zeros on minos errors
    if not(yFitIC.runMinos):
        minosAutoErr = list(np.zeros((len(parameters), 2)))
        minosManErr = list(np.zeros((len(parameters), 2)))
        return parameters, values, errors, minosAutoErr, minosManErr
    
    bestFitVals = {}
    bestFitErrs = {}
    for p, v, e in zip(mObj.parameters, mObj.values, mObj.errors):
        bestFitVals[p] = v
        bestFitErrs[p] = e

    if (yFitIC.fitModel=="SINGLE_GAUSSIAN"):   # Case with no positivity constraint, can use automatic minos()
        mObj.minos()
        me = mObj.merrors

        # Build minos errors lists in suitable format
        minosAutoErr = []
        for p in parameters:
            minosAutoErr.append([me[p].lower, me[p].upper])
        minosManErr = list(np.zeros(np.array(minosAutoErr).shape))

        if yFitIC.showPlots:
            plotAutoMinos(mObj, wsName)

    else:   # Case with positivity constraint on function, use manual implementation
        merrors, fig = runAndPlotManualMinos(mObj, constrFunc, bestFitVals, bestFitErrs, yFitIC.showPlots)     # Changes values of minuit obj m, do not use m below this point
        
        # Same as above, but the other way around
        minosManErr = []
        for p in parameters:
            minosManErr.append(merrors[p])
        minosAutoErr = list(np.zeros(np.array(minosManErr).shape))

        if yFitIC.showPlots:
            fig.canvas.set_window_title(wsName+"_Manual_Implementation_MINOS")
            fig.show()

    return    parameters, values, errors, minosAutoErr, minosManErr


def runAndPlotManualMinos(minuitObj, constrFunc, bestFitVals, bestFitErrs, showPlots):
    """
    Runs brute implementation of minos algorithm and
    plots the profile for each parameter along the way.
    """
    # Reason for two distinct operations inside the same function is that its easier
    # to build the minos plots for each parameter as they are being calculated.
    print("\nRunning Minos ... \n")

    # Set format of subplots
    height = 2
    width = int(np.ceil(len(minuitObj.parameters)/2))
    figsize = (12, 7)
    # Output plot to Mantid
    fig, axs = plt.subplots(height, width, tight_layout=True, figsize=figsize, subplot_kw={'projection':'mantid'})  #subplot_kw={'projection':'mantid'}
    # fig.canvas.set_window_title("Plot of Manual Implementation MINOS")

    merrors = {}
    for p, ax in zip(minuitObj.parameters, axs.flat):
        lerr, uerr = runMinosForPar(minuitObj, constrFunc, p, 2, ax, bestFitVals, bestFitErrs, showPlots)
        merrors[p] = np.array([lerr, uerr])

    # if showPlots:
    # Hide plots not in use:
    for ax in axs.flat:
        if not ax.lines:   # If empty list
            ax.set_visible(False)

    # ALl axes share same legend, so set figure legend to first axis
    handle, label = axs[0, 0].get_legend_handles_labels()
    fig.legend(handle, label, loc='lower right')
        # fig.show()
    return merrors, fig


def runMinosForPar(minuitObj, constrFunc, var:str, bound:int, ax, bestFitVals, bestFitErrs, showPlots):

    resetMinuit(minuitObj, bestFitVals, bestFitErrs)
    # Run Fitting procedures again to be on the safe side and reset to minimum
    minuitObj.scipy(constraints=optimize.NonlinearConstraint(constrFunc, 0, np.inf))
    minuitObj.hesse()

    # Extract parameters from minimum
    varVal = minuitObj.values[var]
    varErr = minuitObj.errors[var]
    # Store fval of best fit
    fValsMin = minuitObj.fval      # Used to calculate error bands at the end

    varSpace = buildVarRange(bound, varVal, varErr) 
    
    # Split variable space into right and left side
    lhsVarSpace, rhsVarSpace = np.split(varSpace, 2)
    lhsVarSpace = np.flip(lhsVarSpace)   # Flip to start at minimum

    for minimizer in ("Scipy", "Migrad"):
        resetMinuit(minuitObj, bestFitVals, bestFitErrs)
        rhsMinos = runMinosOnRange(minuitObj, var, rhsVarSpace, minimizer, constrFunc)
        
        resetMinuit(minuitObj, bestFitVals, bestFitErrs)
        lhsMinos = runMinosOnRange(minuitObj, var, lhsVarSpace, minimizer, constrFunc)

        wholeMinos = np.concatenate((np.flip(lhsMinos), rhsMinos), axis=None)   # Flip left hand side again

        if minimizer == "Scipy":   # Calculate minos errors from constrained scipy
            lerr, uerr = errsFromMinosCurve(varSpace, varVal, wholeMinos, fValsMin, dChi2=1)
            ax.plot(varSpace, wholeMinos, label="fVals Constr Scipy")

        elif minimizer == "Migrad":   # Plot migrad as well to see the difference between constrained and unconstrained
            plotProfile(ax, var, varSpace, wholeMinos, lerr, uerr, fValsMin, varVal, varErr)
        else:
            raise ValueError("Minimizer not recognized.")

    resetMinuit(minuitObj, bestFitVals, bestFitErrs)
    return lerr, uerr


def resetMinuit(minuitObj, bestFitVals, bestFitErrs):
    # Set Minuit parameters to best fit values and errors
    for p in bestFitVals:
        minuitObj.values[p] = bestFitVals[p]
        minuitObj.errors[p] = bestFitErrs[p]
    return


def buildVarRange(bound, varVal, varErr):
    # Create variable space more dense near the minima using a quadratic density
    limit = (bound*varErr)**(1/2)     # Square root is corrected below
    varSpace = np.linspace(-limit, limit, 30)
    varSpace = varSpace**2 * np.sign(varSpace) + varVal
    assert len(varSpace)%2 == 0, "Number of points in Minos range needs to be even"
    return varSpace


def runMinosOnRange(minuitObj, var, varRange, minimizer, constrFunc):

    result = np.zeros(varRange.size)
    minuitObj.fixed[var] = True

    # Unconstrained fit over side range
    for i, value in enumerate(varRange):

        minuitObj.values[var] = value      # Fix variable

        if minimizer == "Migrad":
            minuitObj.migrad()                       # Fit 
        elif minimizer == "Scipy":
            minuitObj.scipy(constraints=optimize.NonlinearConstraint(constrFunc, 0, np.inf))

        result[i] = minuitObj.fval          # Store minimum

    minuitObj.fixed[var] = False 
    return result                    


def errsFromMinosCurve(varSpace, varVal, fValsScipy, fValsMin, dChi2=1):
    # Use intenpolation to create dense array of fmin values 
    varSpaceDense = np.linspace(np.min(varSpace), np.max(varSpace), 100000)
    fValsScipyDense = np.interp(varSpaceDense, varSpace, fValsScipy)
    # Calculate points of intersection with line delta fmin val = 1
    idxErr = np.argwhere(np.diff(np.sign(fValsScipyDense - fValsMin - 1)))
    
    if idxErr.size != 2:    # Intersections not found, do not plot error range
        lerr, uerr = 0., 0.   
    else:
        lerr, uerr = varSpaceDense[idxErr].flatten() - varVal
 
    return lerr, uerr


def plotAutoMinos(minuitObj, wsName):
    # Set format of subplots
    height = 2
    width = int(np.ceil(len(minuitObj.parameters)/2))
    figsize = (12, 7)
    # Output plot to Mantid
    fig, axs = plt.subplots(height, width, tight_layout=True, figsize=figsize, subplot_kw={'projection':'mantid'})
    fig.canvas.set_window_title(wsName+"_Plot_Automatic_MINOS")
 
    for p, ax in zip(minuitObj.parameters, axs.flat):
        loc, fvals, status = minuitObj.mnprofile(p, bound=2)

        minfval = minuitObj.fval
        minp = minuitObj.values[p]
        hessp = minuitObj.errors[p]
        lerr = minuitObj.merrors[p].lower
        uerr = minuitObj.merrors[p].upper
        plotProfile(ax, p, loc, fvals, lerr, uerr, minfval, minp, hessp)

    # Hide plots not in use:
    for ax in axs.flat:
        if not ax.lines:   # If empty list
            ax.set_visible(False)

    # ALl axes share same legend, so set figure legend to first axis
    handle, label = axs[0, 0].get_legend_handles_labels()
    fig.legend(handle, label, loc='lower right')
    fig.show()   


def plotProfile(ax, var, varSpace, fValsMigrad, lerr, uerr, fValsMin, varVal, varErr):
    """
    Plots likelihood profilef for the Migrad fvals.
    varSpace : x axis
    fValsMigrad : y axis
    """

    ax.set_title(var+f" = {varVal:.3f} {lerr:.3f} {uerr:+.3f}")

    ax.plot(varSpace, fValsMigrad, label="fVals Migrad")

    ax.axvspan(lerr+varVal, uerr+varVal, alpha=0.2, color="red", label="Minos error")
    ax.axvspan(varVal-varErr, varVal+varErr, alpha=0.2, color="green", label="Hessian Std error")
    
    ax.axvline(varVal, 0.03, 0.97, color="k", ls="--")
    ax.axhline(fValsMin+1, 0.03, 0.97, color="k")
    ax.axhline(fValsMin, 0.03, 0.97, color="k")


def createFitParametersTableWorkspace(wsYSpaceSym, parameters, values, errors, minosAutoErr, minosManualErr, chi2):
    # Create Parameters workspace
    tableWS = CreateEmptyTableWorkspace(OutputWorkspace=wsYSpaceSym.name()+"_Fitted_Minuit_Parameters")
    tableWS.setTitle("Minuit Fit")
    tableWS.addColumn(type='str', name="Name")
    tableWS.addColumn(type='float', name="Value")
    tableWS.addColumn(type='float', name="Error")
    tableWS.addColumn(type='float', name="Auto Minos Error-")
    tableWS.addColumn(type='float', name="Auto Minos Error+")
    tableWS.addColumn(type='float', name="Manual Minos Error-")
    tableWS.addColumn(type='float', name="Manual Minos Error+")

    for p, v, e, mae, mme in zip(parameters, values, errors, minosAutoErr, minosManualErr):
        tableWS.addRow([p, v, e, mae[0], mae[1], mme[0], mme[1]])

    tableWS.addRow(["Cost function", chi2, 0, 0, 0, 0, 0])
    return


def oddPointsRes(x, res):
    """
    Make a odd grid that ensures a resolution with a single peak at the center.
    """

    assert np.min(x) == -np.max(x), "Resolution needs to be in symetric range!"
    assert x.size == res.size, "x and res need to be the same size!"

    if res.size % 2 == 0:
        dens = res.size+1  # If even change to odd
    else:
        dens = res.size    # If odd, keep being odd

    xDense = np.linspace(np.min(x), np.max(x), dens)    # Make gridd with odd number of points - peak at center
    xDelta = xDense[1] - xDense[0]

    resDense = np.interp(xDense, x, res)

    return xDelta, resDense


def fitProfileMantidFit(yFitIC, wsYSpaceSym, wsRes):
    print('\nFitting on the sum of spectra in the West domain ...\n')     
    for minimizer in ['Levenberg-Marquardt','Simplex']:
        
        if yFitIC.fitModel=="SINGLE_GAUSSIAN":
            function=f"""composite=Convolution,FixResolution=true,NumDeriv=true;
            name=Resolution,Workspace={wsRes.name()},WorkspaceIndex=0;
            name=UserFunction,Formula=y0 + A*exp( -(x-x0)^2/2/sigma^2)/(2*3.1415*sigma^2)^0.5,
            y0=0,A=1,x0=0,sigma=5,   ties=()"""

        elif yFitIC.fitModel=="GC_C4_C6":
            function = f"""
            composite=Convolution,FixResolution=true,NumDeriv=true;
            name=Resolution,Workspace={wsRes.name()},WorkspaceIndex=0,X=(),Y=();
            name=UserFunction,Formula=y0 + A*exp( -(x-x0)^2/2./sigma1^2)/(sqrt(2.*3.1415*sigma1^2))
            *(1.+c4/32.*(16.*((x-x0)/sqrt(2)/sigma1)^4-48.*((x-x0)/sqrt(2)/sigma1)^2+12)+c6/384*(64*((x-x0)/sqrt(2)/sigma1)^6 - 480*((x-x0)/sqrt(2)/sigma1)^4 + 720*((x-x0)/sqrt(2)/sigma1)^2 - 120)),
            y0=0, A=1,x0=0,sigma1=4.0,c4=0.0,c6=0.0,ties=(),constraints=(0<c4,0<c6)
            """
        elif yFitIC.fitModel=="GC_C4":
            function = f"""
            composite=Convolution,FixResolution=true,NumDeriv=true;
            name=Resolution,Workspace={wsRes.name()},WorkspaceIndex=0,X=(),Y=();
            name=UserFunction,Formula=y0 + A*exp( -(x-x0)^2/2./sigma1^2)/(sqrt(2.*3.1415*sigma1^2))
            *(1.+c4/32.*(16.*((x-x0)/sqrt(2)/sigma1)^4-48.*((x-x0)/sqrt(2)/sigma1)^2+12)),
            y0=0, A=1,x0=0,sigma1=4.0,c4=0.0,ties=()
            """
        elif yFitIC.fitModel=="GC_C6":
            function = f"""
            composite=Convolution,FixResolution=true,NumDeriv=true;
            name=Resolution,Workspace={wsRes.name()},WorkspaceIndex=0,X=(),Y=();
            name=UserFunction,Formula=y0 + A*exp( -(x-x0)^2/2./sigma1^2)/(sqrt(2.*3.1415*sigma1^2))
            *(1.+c6/384*(64*((x-x0)/sqrt(2)/sigma1)^6 - 480*((x-x0)/sqrt(2)/sigma1)^4 + 720*((x-x0)/sqrt(2)/sigma1)^2 - 120)),
            y0=0, A=1,x0=0,sigma1=4.0,c6=0.0,ties=()
            """
        elif (yFitIC.fitModel=="DOUBLE_WELL") | (yFitIC.fitModel=="DOUBLE_WELL_ANSIO"):
            return
        else: raise ValueError("fitmodel not recognized.")

        outputName = wsYSpaceSym.name()+"_Fitted_"+minimizer
        CloneWorkspace(InputWorkspace = wsYSpaceSym, OutputWorkspace = outputName)

        Fit(
            Function=function, 
            InputWorkspace=outputName,
            Output=outputName,
            Minimizer=minimizer
            )
        # Fit produces output workspaces with results
    return 


def printYSpaceFitResults(wsJoYName):
    print("\nFit in Y Space results:")
    foundWS = []
    try:
        wsFitLM = mtd[wsJoYName + "_Fitted_Levenberg-Marquardt_Parameters"]
        foundWS.append(wsFitLM)
    except KeyError: pass
    try:
        wsFitSimplex = mtd[wsJoYName + "_Fitted_Simplex_Parameters"]
        foundWS.append(wsFitSimplex)
    except KeyError: pass
    try:
        wsFitMinuit = mtd[wsJoYName + "_Fitted_Minuit_Parameters"]
        foundWS.append(wsFitMinuit)
    except KeyError: pass

    for tableWS in foundWS:
        print("\n"+" ".join(tableWS.getName().split("_")[-3:])+":")
        # print("    ".join(tableWS.keys()))
        for key in tableWS.keys():
            if key=="Name":
                print(f"{key:>20s}:  "+"  ".join([f"{elem:7.8s}" for elem in tableWS.column(key)]))
            else:
                print(f"{key:>20s}: "+"  ".join([f"{elem:7.4f}" for elem in tableWS.column(key)]))
    print("\n")


class ResultsYFitObject:

    def __init__(self, ic, yFitIC, wsFinalName, wsYSpaceAvgName):
        # Extract most relevant information from ws
        wsFinal = mtd[wsFinalName]
        wsResSum = mtd[wsFinalName + "_Resolution_Sum"]

        # TODO: Issue here with extractng workspace
        wsJoYAvg = mtd[wsYSpaceAvgName]
        wsSubMassName = wsYSpaceAvgName.split("_JoY_")[0]
        wsMass0 = mtd[wsSubMassName]
        # if yFitIC.symmetrisationFlag:
        #     wsJoYAvg = mtd[wsSubMassName + "_JoY_WeightedAvg_Symmetrised"]
        # else:
        #     wsJoYAvg = mtd[wsSubMassName + "_JoY_WeightedAvg"]

        self.finalRawDataY = wsFinal.extractY()
        self.finalRawDataE = wsFinal.extractE()
        self.HdataY = wsMass0.extractY()
        self.YSpaceSymSumDataY = wsJoYAvg.extractY()
        self.YSpaceSymSumDataE = wsJoYAvg.extractE()
        self.resolution = wsResSum.extractY()

        # Extract best fit parameters from workspaces
        poptList = []
        perrList = []
        try:
            wsFitMinuit = mtd[wsJoYAvg.name() + "_Fitted_Minuit_Parameters"]
            poptList.append(wsFitMinuit.column("Value"))
            perrList.append(wsFitMinuit.column("Error"))
        except: pass
        try:
            wsFitLM = mtd[wsJoYAvg.name() + "_Fitted_Levenberg-Marquardt_Parameters"]
            poptList.append(wsFitLM.column("Value"))
            perrList.append(wsFitLM.column("Error"))
        except: pass
        try:
            wsFitSimplex = mtd[wsJoYAvg.name() + "_Fitted_Simplex_Parameters"]
            poptList.append(wsFitSimplex.column("Value"))
            perrList.append(wsFitSimplex.column("Error"))
        except: pass

        # Number of parameters might not be the same, need to add zeros to some lists to match length
        maxLen = max([len(l) for l in poptList])
        for pList in [poptList, perrList]:
            for l in pList:
                while len(l) < maxLen:
                    l.append(0)
        
        popt = np.array(poptList)
        perr = np.array(perrList)

        self.popt = popt
        self.perr = perr

        self.savePath = ic.ySpaceFitSavePath
        self.fitModel = yFitIC.fitModel


    def save(self):
        np.savez(self.savePath,
                 YSpaceSymSumDataY=self.YSpaceSymSumDataY,
                 YSpaceSymSumDataE=self.YSpaceSymSumDataE,
                 resolution=self.resolution, 
                 HdataY=self.HdataY,
                 finalRawDataY=self.finalRawDataY, 
                 finalRawDataE=self.finalRawDataE,
                 popt=self.popt, 
                 perr=self.perr)


def runGlobalFit(wsYSpace, wsRes, IC, yFitIC):

    print("\nRunning GLobal Fit ...\n")

    dataX, dataY, dataE, dataRes, instrPars = extractData(wsYSpace, wsRes, IC)   
    dataX, dataY, dataE, dataRes, instrPars = takeOutMaskedSpectra(dataX, dataY, dataE, dataRes, instrPars)

    idxList = groupDetectors(instrPars, yFitIC)
    dataX, dataY, dataE, dataRes = avgWeightDetGroups(dataX, dataY, dataE, dataRes, idxList, yFitIC)

    if yFitIC.symmetrisationFlag:  
        dataY, dataE = symmetrizeArr(dataY, dataE)

    model, defaultPars, sharedPars = selectModelAndPars(yFitIC.fitModel)   
    
    totCost = 0
    for i, (x, y, yerr, res) in enumerate(zip(dataX, dataY, dataE, dataRes)):
        totCost += calcCostFun(model, i, x, y, yerr, res, sharedPars)
    
    defaultPars["y0"] = 0    # Introduce default parameter for convolved model

    assert len(describe(totCost)) == len(sharedPars) + len(dataY)*(len(defaultPars)-len(sharedPars)), f"Wrong parameters for Global Fit:\n{describe(totCost)}"
   
    # Minuit Fit with global cost function and local+global parameters
    initPars = minuitInitialParameters(defaultPars, sharedPars, len(dataY))

    print("\nRunning Global Fit ...\n")
    m = Minuit(totCost, **initPars)

    for i in range(len(dataY)):     # Set limits for unshared parameters
        m.limits["A"+str(i)] = (0, np.inf)   

    if yFitIC.fitModel=="DOUBLE_WELL":  
        m.limits["d"] = (0, np.inf)     # Shared parameters
        m.limits["R"] = (0, np.inf) 

    t0 = time.time()
    if yFitIC.fitModel=="SINGLE_GAUSSIAN":
        m.simplex()
        m.migrad() 

    else:
        totSig = describe(totCost)   # This signature has 'x' already removed
        sharedIdxs = [totSig.index(shPar) for shPar in sharedPars]
        nCostFunctions = len(totCost)   # Number of individual cost functions
        x = dataX[0]

        def constr(*pars):
            """
            Constraint for positivity of Global Gram Carlier.
            Input: All parameters defined in global cost function.
            x is the range for each individual cost fun, defined outside function.
            Builds array with all constraints from individual functions.
            """

            sharedPars = [pars[i] for i in sharedIdxs]    # sigma1, c4, c6 in original GC
            unsharedPars = np.delete(pars, sharedIdxs, None)
            unsharedParsSplit = np.split(unsharedPars, nCostFunctions)   # Splits unshared parameters per individual cost fun

            joinedGC = np.zeros(nCostFunctions * x.size)  
            for i, unshParsModel in enumerate(unsharedParsSplit):    # Attention to format of unshared and shared parameters when calling model
                joinedGC[i*x.size : (i+1)*x.size] = model(x, *unshParsModel[1:], *sharedPars)   # Intercept is first of unshared parameters 
                 
            return joinedGC

        m.simplex()
        m.scipy(constraints=optimize.NonlinearConstraint(constr, 0, np.inf))
    
    t1 = time.time()
    print(f"\nTime of fitting: {t1-t0:.2f} seconds")
    
    # Explicitly calculate errors
    m.hesse()

    chi2 = m.fval / (np.sum(dataE!=0)-m.nfit)   # Number of non zero points (considered in the fit) minus no of parameters
    print(f"Value of Chi2/ndof: {chi2:.2f}")
    print(f"Migrad Minimum valid: {m.valid}")

    print("\nResults of Global Fit:\n")
    for p, v, e in zip(m.parameters, m.values, m.errors):
        print(f"{p:>7s} = {v:>8.4f} \u00B1 {e:<8.4f}")
    print("\n")

    if yFitIC.showPlots:
        plotGlobalFit(dataX, dataY, dataE, m, totCost, wsYSpace.name())
    
    return np.array(m.values), np.array(m.errors)     # Pass into array to store values in variable


def extractData(ws, wsRes, ic):
    dataY = ws.extractY()
    dataE = ws.extractE()
    dataX = ws.extractX()
    dataRes = wsRes.extractY()
    instrPars = loadInstrParsFileIntoArray(ic)
    assert len(instrPars) == len(dataY), "Load of IP file not working correctly, probable issue with indexing."
    return dataX, dataY, dataE, dataRes, instrPars    


def loadInstrParsFileIntoArray(ic):
    data = np.loadtxt(ic.InstrParsPath, dtype=str)[1:].astype(float)
    spectra = data[:, 0]
    select_rows = np.where((spectra >= ic.firstSpec) & (spectra <= ic.lastSpec))
    instrPars = data[select_rows]
    return instrPars


def takeOutMaskedSpectra(dataX, dataY, dataE, dataRes, instrPars):
    zerosRowMask = np.all(dataY==0, axis=1)
    dataY = dataY[~zerosRowMask]
    dataE = dataE[~zerosRowMask]
    dataX = dataX[~zerosRowMask]
    dataRes = dataRes[~zerosRowMask]
    instrPars = instrPars[~zerosRowMask]
    return dataX, dataY, dataE, dataRes, instrPars 


def minuitInitialParameters(defaultPars, sharedPars, nSpec):
    """Buids dictionary to initialize Minuit with starting global+local parameters"""
    
    initPars = {}
    # Populate with initial shared parameters
    for sp in sharedPars:
        initPars[sp] = defaultPars[sp]
    # Add initial unshared parameters
    unsharedPars = [key for key in defaultPars if key not in sharedPars]
    for up in unsharedPars:
        for i in range(nSpec):
            initPars[up+str(i)] = defaultPars[up]
    return initPars


def calcCostFun(model, i, x, y, yerr, res, sharedPars):
    "Returns cost function for one spectrum i to be summed to total cost function"
   
    xDelta, resDense = oddPointsRes(x, res)
    def convolvedModel(xrange, y0, *pars):
        """Performs convolution first on high density grid and interpolates to desired x range"""
        return y0 + signal.convolve(model(xrange, *pars), resDense, mode="same") * xDelta

    signature = describe(model)[:]
    signature[1:1] = ["y0"]

    costSig = [key if key in sharedPars else key+str(i) for key in signature]
    convolvedModel.func_code = make_func_code(costSig)

    # Select only valid data, i.e. when error is not 0 or nan or inf
    nonZeros= (yerr!=0) & (yerr!=np.nan) & (yerr!=np.inf) & (y!=np.nan)
    xNZ = x[nonZeros]
    yNZ = y[nonZeros]
    yerrNZ = yerr[nonZeros]

    costFun = cost.LeastSquares(xNZ, yNZ, yerrNZ, convolvedModel)
    return costFun


def plotGlobalFit(dataX, dataY, dataE, mObj, totCost, wsName):

    if len(dataY) > 10:    
        print("\nToo many axes to show in figure, skipping the plot ...\n")
        return

    rows = 2
    fig, axs = plt.subplots(
        rows, 
        int(np.ceil(len(dataY)/rows)),
        figsize=(15, 8), 
        tight_layout=True,
        subplot_kw={'projection':'mantid'}
    )
    fig.canvas.set_window_title(wsName+"_Plot_of_Global_Fit")

    # Data used in Global Fit
    for i, (x, y, yerr, ax) in enumerate(zip(dataX, dataY, dataE, axs.flat)):
        ax.errorbar(x, y, yerr, fmt="k.", label=f"Data Group {i}") 

    # Global Fit 
    for x, costFun, ax in zip(dataX, totCost, axs.flat):
        signature = describe(costFun)

        values = mObj.values[signature]
        errors = mObj.errors[signature]

        yfit = costFun.model(x, *values)

        # Build a decent legend
        leg = []
        for p, v, e in zip(signature, values, errors):
            leg.append(f"${p} = {v:.3f} \pm {e:.3f}$")

        ax.fill_between(x, yfit, label="\n".join(leg), alpha=0.4)
        ax.legend()
    fig.show()
    return

# ------- Groupings 

def groupDetectors(ipData, yFitIC):
    """
    Uses the method of k-means to find clusters in theta-L1 space.
    Input: instrument parameters to extract L1 and theta of detectors.
    Output: list of group lists containing the idx of spectra.
    """

    checkNGroupsValid(yFitIC, ipData)

    print(f"\nNumber of gropus: {yFitIC.nGlobalFitGroups}")

    L1 = ipData[:, -1].copy()
    theta = ipData[:, 2].copy()  

    # Normalize  ranges to similar values
    L1 /= np.sum(L1)       
    theta /= np.sum(theta)

    L1 *= 2           # Bigger weight to L1

    points = np.vstack((L1, theta)).T
    assert points.shape == (len(L1), 2), "Wrong shape."
    # Initial centers of groups
    startingIdxs = np.linspace(0, len(points)-1, yFitIC.nGlobalFitGroups).astype(int)
    centers = points[startingIdxs, :]    # Centers of cluster groups, NOT fitting parameter

    if False:    # Set to True to investigate problems with groupings
        plotDetsAndInitialCenters(L1, theta, centers)

    clusters, n = kMeansClustering(points, centers)
    idxList = formIdxList(clusters, n, len(L1))

    if yFitIC.showPlots:
        fig, ax = plt.subplots(tight_layout=True, subplot_kw={'projection':'mantid'})  
        fig.canvas.set_window_title("Grouping of detectors")
        plotFinalGroups(ax, ipData, idxList)
        fig.show()
    return idxList


def checkNGroupsValid(yFitIC, ipData):

    nSpectra = len(ipData)  # Number of spectra in the workspace

    if (yFitIC.nGlobalFitGroups=="ALL"):
        yFitIC.nGlobalFitGroups = nSpectra
    else:
        assert type(yFitIC.nGlobalFitGroups)==int, "Number of global groups needs to be an integer."
        assert yFitIC.nGlobalFitGroups<=nSpectra, "Number of global groups needs to be less or equal to the no of unmasked spectra."
        assert yFitIC.nGlobalFitGroups>0, "NUmber of global groups needs to be bigger than zero"
    return 


def plotDetsAndInitialCenters(L1, theta, centers):
    fig, ax = plt.subplots(tight_layout=True, subplot_kw={'projection':'mantid'})  
    fig.canvas.set_window_title("Starting centroids for groupings")
    ax.scatter(L1, theta, alpha=0.3, color="r", label="Detectors")
    ax.scatter(centers[:, 0], centers[:, 1], color="k", label="Starting centroids")
    ax.axes.xaxis.set_ticks([])  # Numbers plotted do not correspond to real numbers, so hide them
    ax.axes.yaxis.set_ticks([]) 
    ax.set_xlabel("L1")
    ax.set_ylabel("Theta")
    ax.legend()
    fig.show()


def plotFinalGroups(ax, ipData, idxList):
    for i, idxs in enumerate(idxList):
        L1 = ipData[idxs, -1]
        theta = ipData[idxs, 2]
        ax.scatter(L1, theta, label=f"Group {i}")

        dets = ipData[idxs, 0]
        for det, x, y in zip(dets, L1, theta):
            ax.text(x, y, str(int(det)), fontsize=8)

    ax.set_xlabel("L1")
    ax.set_ylabel("Theta")
    ax.legend()
    return


def kMeansClustering(points, centers):
    """
    Algorithm used to form groups of detectors.
    Works best for spherical groups with similar scaling on x and y axis.
    Fails in some rare cases, solution is to try a different number of groups.
    """

    prevCenters = centers
    while  True:
        clusters, nGroups = closestCenter(points, prevCenters)
        centers = calculateCenters(points, clusters, nGroups)

        if np.all(centers == prevCenters):
            break

        assert np.isfinite(centers).all(), f"Invalid centers found:\n{centers}\nTry a different number for the groupings."

        prevCenters = centers
    clusters, n = closestCenter(points, centers)
    return clusters, n


def closestCenter(points, centers):
    """Checks eahc point and assigns it to closest center."""

    clusters = np.zeros(len(points))
    for p in range(len(points)):

        minCenter = 0
        minDist = pairDistance(points[p], centers[0])
        for i in range(1, len(centers)): 

            dist = pairDistance(points[p], centers[i])

            if dist < minDist:
                minDist = dist
                minCenter = i
        clusters[p] = minCenter
    return clusters, len(centers)


def pairDistance(p1, p2):
    "Calculates the distance between two points."
    return np.sqrt(np.sum(np.square(p1-p2)))


def calculateCenters(points, clusters, nGroups):
    """Calculates centers for the given clusters"""

    centers = np.zeros((nGroups, 2))
    for i in range(nGroups):
        centers[i] = np.mean(points[clusters==i, :], axis=0)  # If cluster i is not present, returns nan
    return centers


def formIdxList(clusters, nGroups, lenPoints):
    """Converts information of clusters into a list of indexes."""

    idxList = []
    for i in range(nGroups):
        idxs = np.argwhere(clusters==i).flatten()
        idxList.append(list(idxs))

    print("\nGroups formed successfully:\n")
    groupLen = np.array([len(group) for group in idxList])
    unique, counts = np.unique(groupLen, return_counts=True)
    for length, no in zip(unique, counts):
        print(f"{no} groups with {length} detectors.")

    # Check that idexes are not repeated and not missing
    flatList = []
    for group in idxList:
        for elem in group:
            flatList.append(elem)
    assert np.all(np.sort(np.array(flatList))==np.arange(lenPoints)), "Groupings did not work!"
    
    return idxList

# ---------- Weighted Avgs of Groups

def avgWeightDetGroups(dataX, dataY, dataE, dataRes, idxList, yFitIC):
    """
    Performs weighted average on each detector group given by the index list.
    The imput arrays do not include masked spectra.
    """
    assert ~np.any(np.all(dataY==0, axis=1)), f"Input data should not include masked spectra at: {np.argwhere(np.all(dataY==0, axis=1))}"

    if (yFitIC.maskTOFRange!=None): 
        if (yFitIC.maskTypeProcedure=="NAN_&_BIN"):   # Exceptional case
            return avgGroupsWithBins(dataX, dataY, dataE, dataRes, idxList, yFitIC)
    
    return avgGroupsOverCols(dataX, dataY, dataE, dataRes, idxList)


def avgGroupsOverCols(dataX, dataY, dataE, dataRes, idxList):
    """Averaging used when JoY workspace is already Rebinned or Interpolated."""

    wDataX, wDataY, wDataE, wDataRes = initiateZeroArr((len(idxList), len(dataY[0])))

    for i, idxs in enumerate(idxList):
        groupX, groupY, groupE, groupRes = extractArrByIdx(dataX, dataY, dataE, dataRes, idxs)
        
        if len(groupY) == 1:   # Cannot use weight avg in single spec, wrong results
            meanY, meanE = groupY, groupE
            meanRes = groupRes

        else:
            meanY, meanE = weightedAvgArr(groupY, groupE)
            meanRes = np.nanmean(groupRes, axis=0)   # Nans are not present but safeguard

        assert np.all(groupX[0] == np.mean(groupX, axis=0)), "X values should not change with groups"
        
        wDataX[i] = groupX[0]
        wDataY[i] = meanY
        wDataE[i] = meanE
        wDataRes[i] = meanRes 
    
    assert ~np.any(np.all(wDataY==0, axis=1)), f"Some avg weights in groups are not being performed:\n{np.argwhere(np.all(wDataY==0, axis=1))}"
    return wDataX, wDataY, wDataE, wDataRes


def avgGroupsWithBins(dataX, dataY, dataE, dataRes, idxList, yFitIC):
    """Performed when mask with NaNs and Bins is turned on"""

    # Build range to average over
    meanX = buildXRangeFromRebinPars(yFitIC)  

    wDataX, wDataY, wDataE, wDataRes = initiateZeroArr((len(idxList), len(meanX)))
    for i, idxs in enumerate(idxList):
        groupX, groupY, groupE, groupRes = extractArrByIdx(dataX, dataY, dataE, dataRes, idxs)

        meanY, meanE = weightedAvgXBinsArr(groupX, groupY, groupE, meanX)
        
        meanRes = np.nanmean(groupRes, axis=0)   # Nans are not present but safeguard
        
        wDataX[i] = meanX
        wDataY[i] = meanY
        wDataE[i] = meanE
        wDataRes[i] = meanRes 
    
    return wDataX, wDataY, wDataE, wDataRes


def initiateZeroArr(shape):
    wDataX = np.zeros(shape)
    wDataY = np.zeros(shape)
    wDataE = np.zeros(shape)
    wDataRes = np.zeros(shape)  
    return  wDataX, wDataY, wDataE, wDataRes


def extractArrByIdx(dataX, dataY, dataE, dataRes, idxs):
    groupE = dataE[idxs, :]
    groupY = dataY[idxs, :]
    groupX = dataX[idxs, :]
    groupRes = dataRes[idxs, :]
    return groupX, groupY, groupE, groupRes


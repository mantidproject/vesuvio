import numpy as np
from mantid.simpleapi import *
from pathlib import Path

# Format print output of arrays
np.set_printoptions(suppress=True, precision=4, linewidth=150, threshold=sys.maxsize)
repoPath = Path(__file__).absolute().parent  # Path to the repository

def convertToYSpace(ws0, mass):
    ConvertToYSpace(
        InputWorkspace=ws0, Mass=mass, 
        OutputWorkspace=ws0.name()+"_JoY", QWorkspace=ws0.name()+"_Q"
        )
    rebinPars="-20, 0.5, 20" 
    Rebin(
        InputWorkspace=ws0.name()+"_JoY", Params=rebinPars, 
        FullBinsOnly=True, OutputWorkspace=ws0.name()+"_JoY"
        )
    normalise_workspace(ws0.name()+"_JoY")
    return mtd[ws0.name()+"_JoY"]

    
def normalise_workspace(ws_name):
    tmp_norm = Integration(ws_name)
    Divide(LHSWorkspace=ws_name,RHSWorkspace=tmp_norm,OutputWorkspace=ws_name)
    DeleteWorkspace("tmp_norm")


def replaceNonZeroNanValuesByOnesInWs(wsYSym):
    dataY = wsYSym.extractY()
    dataE = wsYSym.extractE()
    
    dataY[np.isnan(dataY)] = 0   # Safeguard agaist nans
    nonZerosMask = ~(dataY==0)
    dataYones = np.where(nonZerosMask, 1, 0)
    dataE = np.full(dataE.shape, 0.000001)  # Value from original script

    # Build Workspaces, couldn't find a method for this in Mantid
    wsOnes = CloneWorkspace(wsYSym)
    for i in range(wsYSym.getNumberHistograms()):
        wsOnes.dataY(i)[:] = dataYones[i, :]
        wsOnes.dataE(i)[:] = dataE[i, :]
    return wsOnes


def originalProcedure(wsYSpace):
    mantidSum = SumSpectra(wsYSpace)
    wsOnes = replaceNonZeroNanValuesByOnesInWs(wsYSpace)
    wsOnesSum = SumSpectra(wsOnes)
    averagedYSpace = Divide(
            LHSWorkspace=mantidSum, RHSWorkspace=wsOnesSum,
            OutputWorkspace="JOfY_averaged"
        )
    return averagedYSpace

def myProcedure(wsYSpace):
    dataY = wsYSpace.extractY()
    dataE = wsYSpace.extractE()
    npErr = np.sqrt(np.nansum(np.square(dataE), axis=0))
    ySum = np.sum(dataY, axis=0)
    npCount = np.sum(dataY!=0, axis=0)

    meanY = ySum / npCount
    meanE = npErr / npCount

    tempWs = SumSpectra(wsYSpace)
    newWs = CloneWorkspace(tempWs)
    newWs.dataY(0)[:] = meanY
    newWs.dataE(0)[:] = meanE
    return newWs

def avgSymmetrize(avgYSpace):
    dataX = avgYSpace.extractX()
    dataY = avgYSpace.extractY()
    dataE = avgYSpace.extractE()

    # Following symmetrization only works for symmetric binning 
    yFlip = np.flip(dataY)
    eFlip = np.flip(dataE)

    # Inverse variance weighting
    yMean = (dataY/dataE**2 + yFlip/eFlip**2) / (1/dataE**2 + 1/eFlip**2)
    eMean = 1 / np.sqrt(1/dataE**2 + 1/eFlip**2)

    Sym = CloneWorkspace(avgYSpace, OutputWorkspace="Symmetrised")
    Sym.dataY(0)[:] = yMean
    Sym.dataE(0)[:] = eMean
    return Sym


ws = Load(Filename=r"./input_ws/starch_80_RD_raw_forward.nxs",
             OutputWorkspace="ws_raw")
MaskDetectors(Workspace=ws, SpectraList=[173, 174, 179])   
wsYSpace = convertToYSpace(ws, 1)

ori = originalProcedure(wsYSpace)
opt = myProcedure(wsYSpace)
CompareWorkspaces(ori, opt)

Symmetric = avgSymmetrize(opt)
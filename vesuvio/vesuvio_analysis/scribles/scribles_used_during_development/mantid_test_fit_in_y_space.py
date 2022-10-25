
import numpy as np
import numpy.testing as nptest
import matplotlib.pyplot as plt
from mantid.simpleapi import *    
from pathlib import Path
from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=True)

currentPath = Path(__file__).absolute().parent  # Path to the repository


# Load example workspace 
exampleWorkspace = Load(Filename=r"../input_ws/starch_80_RD_raw.nxs", OutputWorkspace="starch_80_RD_raw")
name = exampleWorkspace.name()
# Same initial conditions
masses = [1.0079, 12, 16, 27]
# Load results that were obtained from the same workspace
dataFilePath = currentPath / "fixatures" / "data_to_test_func_sub_mass.npz"

def prepareFinalWsInYSpace(wsFinal, ncpForEachMass):
    wsSubMass = subtractAllMassesExceptFirst(wsFinal, ncpForEachMass)
    wsH = mtd[name + "_H"]
    massH = 1.0079
    wsYSpaceSym = convertToYSpaceAndSymetrise(wsSubMass, massH) 
    wsRes = calculate_mantid_resolutions(wsFinal, massH)
    return wsFinal, wsSubMass, wsYSpaceSym, wsRes


def subtractAllMassesExceptFirst(ws, ncpForEachMass):
    """Input: workspace from last iteration, ncpTotal for each mass
       Output: workspace with all the ncpTotal subtracted except for the first mass"""

    ncpForEachMass = switchFirstTwoAxis(ncpForEachMass)
    # Select all masses other than the first one
    ncpForEachMass = ncpForEachMass[1:, :, :]
    # Sum the ncpTotal for remaining masses
    ncpTotal = np.sum(ncpForEachMass, axis=0)
    dataY, dataX = ws.extractY(), ws.extractX() 
    
    dataY[:, :-1] -= ncpTotal * (dataX[:, 1:] - dataX[:, :-1])

    wsSubMass = CloneWorkspace(InputWorkspace=ws, OutputWorkspace=ws.name()+"_H")
    for i in range(wsSubMass.getNumberHistograms()):  # Keeps the faulty last column
        wsSubMass.dataY(i)[:] = dataY[i, :]

    # wsSubMass = CreateWorkspace(    # Discard the last colums of workspace
    #     DataX=dataX[:, :-1].flatten(), DataY=dataY[:, :-1].flatten(),
    #     DataE=dataE[:, :-1].flatten(), Nspec=len(dataX)
    #     )
    HSpectraToBeMasked = [173, 174, 179]
    Rebin(InputWorkspace=ws.name()+"_H",Params="110,1.,430", OutputWorkspace=ws.name()+"_H")
    MaskDetectors(Workspace=ws.name()+"_H",SpectraList=HSpectraToBeMasked)
    RemoveMaskedSpectra(InputWorkspace=ws.name()+"_H", OutputWorkspace=ws.name()+"_H")    # Probably not necessary
    return mtd[ws.name()+"_H"]


def switchFirstTwoAxis(A):
    """Exchanges the first two indices of an array A,
    rearranges matrices per spectrum for iteration of main fitting procedure
    """
    return np.stack(np.split(A, len(A), axis=0), axis=2)[0]


def convertToYSpaceAndSymetrise(ws0, mass):
    ConvertToYSpace(
        InputWorkspace=ws0, Mass=mass, 
        OutputWorkspace=ws0.name()+"_JoY", QWorkspace=ws0.name()+"_Q"
        )
#     max_Y = np.ceil(2.5*mass+27)  
#     # First bin boundary, width, last bin boundary
#     rebin_parameters = str(-max_Y)+","+str(2.*max_Y/120)+","+str(max_Y)
    rebin_parameters ='-20,0.5,20'
    Rebin(
        InputWorkspace=ws0.name()+"_JoY", Params=rebin_parameters, 
        FullBinsOnly=True, OutputWorkspace=ws0.name()+"_JoY"
        )
    normalise_workspace(ws0.name()+"_JoY")

    wsYSpace = mtd[ws0.name()+"_JoY"]
    dataY = wsYSpace.extractY() 
    dataE = wsYSpace.extractE()
    dataX = wsYSpace.extractX()

    # Symmetrize
    dataY = np.where(dataX<0, np.flip(dataY), dataY)
    dataE = np.where(dataX<0, np.flip(dataE), dataE)

    # Build normalization arrays
    dataY[np.isnan(dataY)] = 0   # Safeguard agaist nans
    nonZerosMask = ~(dataY==0)
    dataYnorm = np.where(nonZerosMask, 1, 0)
    dataEnorm = np.full(dataE.shape, 0.000001)

    wsYSym = CloneWorkspace(InputWorkspace=wsYSpace, OutputWorkspace=ws0.name()+"_JoY_Sym")
    wsYNorm = CloneWorkspace(InputWorkspace=wsYSpace, OutputWorkspace=ws0.name()+"_JoY_norm")
    for i in range(wsYSpace.getNumberHistograms()):
        wsYSym.dataY(i)[:] = dataY[i, :]
        wsYSym.dataE(i)[:] = dataE[i, :]
        wsYNorm.dataY(i)[:] = dataYnorm[i, :]
        wsYNorm.dataE(i)[:] = dataEnorm[i, :]

    # Sum of spectra
    SumSpectra(InputWorkspace=wsYSym, OutputWorkspace=ws0.name()+"_JoY_Sym")
    SumSpectra(InputWorkspace=wsYNorm, OutputWorkspace=ws0.name()+"_JoY_norm")

    # Normalize
    Divide(
        LHSWorkspace=ws0.name()+"_JoY_Sym", RHSWorkspace=ws0.name()+"_JoY_norm",
        OutputWorkspace=ws0.name()+'_JoY_sum_final'
    )
    return mtd[ws0.name()+"_JoY_sum_final"]


def normalise_workspace(ws_name):
    tmp_norm = Integration(ws_name)
    Divide(LHSWorkspace=ws_name,RHSWorkspace="tmp_norm",OutputWorkspace=ws_name)
    DeleteWorkspace("tmp_norm")


def calculate_mantid_resolutions(ws, mass):
    # Only for loop in this script because the fuction VesuvioResolution takes in one spectra at a time
    # Haven't really tested this one becuase it's not modified
    max_Y = np.ceil(2.5*mass+27)
    rebin_parameters = str(-max_Y)+","+str(2.*max_Y/240)+","+str(max_Y)
    for index in range(ws.getNumberHistograms()):
        VesuvioResolution(Workspace=ws, WorkspaceIndex=index,
                          Mass=mass, OutputWorkspaceYSpace="tmp")
        tmp = Rebin("tmp", rebin_parameters)
        if index == 0:
            RenameWorkspace("tmp", "resolution")
        else:
            AppendSpectra("resolution", "tmp", OutputWorkspace="resolution")
    SumSpectra(InputWorkspace="resolution", OutputWorkspace="resolution")
    normalise_workspace("resolution")
    DeleteWorkspace("tmp")
    return mtd["resolution"]


dataFile = np.load(dataFilePath)
ncpForEachMass = dataFile["all_ncp_for_each_mass"][0]
Hmass = 1.0079
wsRaw, wsSubMass, wsYSpaceSym, wsRes = prepareFinalWsInYSpace(exampleWorkspace, ncpForEachMass)
# rtol = 0.0001
# index = 0

# yRaw = wsRaw.dataY(index)
# eRaw = wsRaw.dataE(index)
# xRaw = wsRaw.dataX(index)
# 
# ySubM = wsSubMass.dataY(index)
# eSubM = wsSubMass.dataE(index)
# xSubM = wsSubMass.dataX(index)
# 
# plt.figure()
# plt.errorbar(xRaw, yRaw, yerr=eRaw, fmt="none", label=f"Raw Data, index={index}", alpha=0.7)
# plt.errorbar(xSubM, ySubM, yerr=eSubM, fmt="none", label=f"H Data, index={index}", alpha=0.7)
# plt.xlabel("TOF")
# plt.ylabel("Counts")
# plt.legend()
# plt.show()
# 
# datay = wsYSpaceSym.dataY(0)
# datae = wsYSpaceSym.dataE(0)
# datax = wsYSpaceSym.dataX(0)
# 
# yMean = wsYSpaceMean.dataY(0)
# eMean= wsYSpaceMean.dataE(0)
# xMean = wsYSpaceMean.dataX(0)
# 
# plt.figure()
# plt.errorbar(xMean, yMean, yerr=eMean, fmt="none", label="Mean Data")
# plt.errorbar(datax, datay, yerr=datae, fmt="none", label="Summed Symetrized Data")
# plt.xlabel("Y-Space")
# plt.ylabel("Counts")
# plt.legend()
# plt.show()
# 
    


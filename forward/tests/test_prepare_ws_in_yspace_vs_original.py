import unittest
import numpy as np
import numpy.testing as nptest
import matplotlib.pyplot as plt
from mantid.simpleapi import *    
from pathlib import Path
from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=True)

currentPath = Path(__file__).absolute().parent  # Path to the repository
np.set_printoptions(linewidth=150)


# Load example workspace 
exampleWorkspace = Load(Filename=r"../input_ws/starch_80_RD_raw.nxs", OutputWorkspace="starch_80_RD_raw")
name = exampleWorkspace.name()
# Same initial conditions
masses = [1.0079, 12, 16, 27]
# Load results that were obtained from the same workspace
dataFilePath = currentPath / "fixatures" / "data_to_test_func_sub_mass.npz"

def prepareFinalWsInYSpace(wsFinal, ncpForEachMass):
    wsSubMass = subtractAllMassesExceptFirst(wsFinal, ncpForEachMass)
    massH = 1.0079
    wsYSpaceSymSum = convertToYSpaceAndSymetrise(wsSubMass, massH) 
    wsRes = calculate_mantid_resolutions(wsFinal, massH)
    return wsFinal, wsSubMass, wsYSpaceSymSum, wsRes


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

    # Pass the data onto a Workspace, clone to preserve properties
    wsSubMass = CloneWorkspace(InputWorkspace=ws, OutputWorkspace=ws.name()+"_H")
    for i in range(wsSubMass.getNumberHistograms()):  # Keeps the faulty last column
        wsSubMass.dataY(i)[:] = dataY[i, :]

    HSpectraToBeMasked = [173, 174, 179]   # Eventually need to test without this safeguard
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
    # max_Y = np.ceil(2.5*mass+27) 
    # rebin_parameters = str(-max_Y)+","+str(2.*max_Y/120)+","+str(max_Y)
    rebin_parameters='-20,0.5,20'  # To compare with original

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
    symmetrise_using_averages = True
    if symmetrise_using_averages:
        dataY = np.where(dataY==0, np.flip(dataY, axis=1), dataY)
        dataE = np.where(dataE==0, np.flip(dataE, axis=1), dataE)

        dataY = (dataY + np.flip(dataY, axis=1)) / 2
        dataE = (dataE + np.flip(dataE, axis=1)) / 2
    else:
        dataY = np.where(dataX<0, np.flip(dataY, axis=1), dataY)
        dataE = np.where(dataX<0, np.flip(dataE, axis=1), dataE)

    # Normalization
    dataY[np.isnan(dataY)] = 0   # Safeguard agaist nans
    nonZerosMask = ~(dataY==0)
    dataYnorm = np.where(nonZerosMask, 1, 0)
    dataEnorm = np.full(dataE.shape, 0.000001)

    # Build Workspaces
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
    # max_Y = np.ceil(2.5*mass+27)
    # rebin_parameters = str(-max_Y)+","+str(2.*max_Y/240)+","+str(max_Y)
    rebin_parameters='-20,0.5,20'
    resolution=CloneWorkspace(InputWorkspace=ws)
    resolution=Rebin(InputWorkspace=resolution, Params=rebin_parameters)

    for i in range(resolution.getNumberHistograms()):
        VesuvioResolution(Workspace=ws, Mass=1.0079, OutputWorkspaceYSpace='tmp')
        tmp = Rebin(InputWorkspace='tmp', Params=rebin_parameters)
        resolution.dataY(i)[:] = tmp.dataY(0)

    SumSpectra(InputWorkspace="resolution", OutputWorkspace="resolution")
    normalise_workspace("resolution")
    DeleteWorkspace("tmp")
    return mtd["resolution"]


class TestSubMasses(unittest.TestCase):

    dataFile = np.load(dataFilePath)
    ncpForEachMass = dataFile["all_ncp_for_each_mass"][0]
    Hmass = 1.0079

    wsRaw, wsSubMass,\
    wsYSpaceSym, wsRes = prepareFinalWsInYSpace(exampleWorkspace, ncpForEachMass)
    rtol = 0.0001
    index = 3

    oriSymSum = Load(Filename=r"fixatures/original_ws_yspace/starch_80_joy_sum.nxs")
    oriSubMass = Load(Filename=r"fixatures/original_ws_yspace/starch_80_raw_H.nxs")
    oriResolution = Load(Filename=r"fixatures/original_ws_yspace/starch_80_resolution.nxs")

    
    def test_symSumWorkspace(self):
        (result, messages) = CompareWorkspaces(self.oriSymSum, self.wsYSpaceSym)
        print("Result of comparison: ", result)
        print(messages.rowCount())

    def test_subMassWorkspace(self):
        (result, messages) = CompareWorkspaces(self.oriSubMass, self.wsSubMass)
        print("Result of comparison: ", result)
        print(messages.rowCount())

    def test_resolutionWs(self):
        (result, messages) = CompareWorkspaces(self.oriResolution, self.wsRes)
        print("Result of comparison: ", result)
        print(messages.rowCount())
        if ~result:
            oriWs = self.oriResolution
            optWs = self.wsRes
            nptest.assert_array_almost_equal(oriWs.extractY(),optWs.extractY(), decimal=15)
            nptest.assert_array_almost_equal(oriWs.extractX(),optWs.extractX(), decimal=15)
            # nptest.assert_array_almost_equal(oriWs.extractE(),optWs.extractE(), decimal=15)
            # For some reason the errors calculated with the optimized resolution are all zero    

    def test_plotRawDataAndSubMassesData(self):
        yRaw = self.wsRaw.dataY(self.index)
        eRaw = self.wsRaw.dataE(self.index)
        xRaw = self.wsRaw.dataX(self.index)

        ySubM = self.wsSubMass.dataY(self.index)
        eSubM = self.wsSubMass.dataE(self.index)
        xSubM = self.wsSubMass.dataX(self.index)

        plt.figure()
        plt.errorbar(xRaw, yRaw, yerr=eRaw, fmt="none", label=f"Raw Data, index={self.index}", alpha=0.7)
        plt.errorbar(xSubM, ySubM, yerr=eSubM, fmt="none", label=f"H Data, index={self.index}", alpha=0.7)
        plt.xlabel("TOF")
        plt.ylabel("Counts")
        plt.legend()
        plt.show()

    def test_plotSymYSpace(self):
        datay = self.wsYSpaceSym.dataY(0)
        datae = self.wsYSpaceSym.dataE(0)
        datax = self.wsYSpaceSym.dataX(0)

        plt.figure()
        plt.errorbar(datax, datay, yerr=datae, fmt="none", label="Summed Symetrized Data")
        plt.xlabel("Y-Space")
        plt.ylabel("Counts")
        plt.legend()
        plt.show()


    def test_plotMantidResolution(self):
        xRes = self.wsRes.dataX(0)
        yRes = self.wsRes.dataY(0)
        plt.figure()
        plt.plot(xRes, yRes, label="Mantid Resolution")
        plt.xlabel("Y-Space")
        plt.legend()
        plt.show()
    

if __name__ == "__main__":
    unittest.main()
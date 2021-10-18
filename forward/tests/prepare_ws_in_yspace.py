import unittest
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
    massH = 1.0079
    wsYSpace, wsYSpaceSum, wsYSpaceMean, wsYSpaceSym = convertToYSpaceAndSymetrise(wsSubMass, massH) 
    wsRes = calculate_mantid_resolutions(wsFinal, massH)
    return wsFinal, wsSubMass, wsYSpace, wsYSpaceSum, wsYSpaceMean, wsYSpaceSym, wsRes


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

    # wsSubMass = CreateWorkspace(    # Discard the last colums of workspace
    #     DataX=dataX[:, :-1].flatten(), DataY=dataY[:, :-1].flatten(),
    #     DataE=dataE[:, :-1].flatten(), Nspec=len(dataX)
    #     )
    HSpectraToBeMasked = []
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
    max_Y = np.ceil(2.5*mass+27)  
    # First bin boundary, width, last bin boundary
    rebin_parameters = str(-max_Y)+","+str(2.*max_Y/120)+","+str(max_Y)
    Rebin(
        InputWorkspace=ws0.name()+"_JoY", Params=rebin_parameters, 
        FullBinsOnly=True, OutputWorkspace=ws0.name()+"_JoY"
        )

    wsYSpace = mtd[ws0.name()+"_JoY"]
    dataY = wsYSpace.extractY()  
    # safeguarding against nans as well
    nanOrZerosMask = (dataY==0) | np.isnan(dataY)
    noOfNonZerosRow = np.nansum(~nanOrZerosMask, axis=0)

    wsSumYSpace = SumSpectra(InputWorkspace=wsYSpace, OutputWorkspace=ws0.name()+"_JoY_sum")
    # SumSpectra can not handle nan values 
    # Nan values might be coming from the ncp
    wsSumYSpace.dataY(0)[:] = np.nansum(dataY, axis=0)
    

    tmp = CloneWorkspace(InputWorkspace=wsSumYSpace, OutputWorkspace="normalization")
    tmp.dataY(0)[:] = noOfNonZerosRow
    tmp.dataE(0)[:] = np.zeros(noOfNonZerosRow.shape)

    wsMean = Divide(                                  # Use of Divide and not nanmean, err are prop automatically
        LHSWorkspace=wsSumYSpace, RHSWorkspace="normalization", OutputWorkspace=ws0.name()+"_JoY_mean"
       )

    ws = CloneWorkspace(wsMean, OutputWorkspace=ws0.name()+"_JoY_Sym")
    datay = ws.readY(0)[:]
    # Next step ensures that nans do not count as a data point during the symetrization
    datay = np.where(np.isnan(datay), np.flip(datay), datay)      
    ws.dataY(0)[:] = (datay + np.flip(datay)) / 2

    datae = ws.dataE(0)[:]
    datae = np.where(np.isnan(datae), np.flip(datae), datae)
    ws.dataE(0)[:] = (datae + np.flip(datae)) / 2

    normalise_workspace(ws)
    # DeleteWorkspaces(
    #     [ws0.name()+"_JoY_sum", ws0.name()+"_JoY_mean", "normalization"]
    #     )
    return mtd[ws0.name()+"_JoY"], mtd[ws0.name()+"_JoY_sum"], \
        mtd[ws0.name()+"_JoY_mean"], mtd[ws0.name()+"_JoY_Sym"]


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


class TestSubMasses(unittest.TestCase):
    def setUp(self):
        dataFile = np.load(dataFilePath)
        ncpForEachMass = dataFile["all_ncp_for_each_mass"][0]
        Hmass = 1.0079

        self.wsRaw, self.wsSubMass,\
        self.wsYSpace, self.wsYSpaceSum, self.wsYSpaceMean, self.wsYSpaceSym, \
        self.wsRes = prepareFinalWsInYSpace(exampleWorkspace, ncpForEachMass)
        self.rtol = 0.0001
        self.index = 3
        # self.savePath = currentPath / "fixatures" / "data_for_fit_in_yspace"
    
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

        # yMean = self.wsYSpaceMean.dataY(0)
        # eMean= self.wsYSpaceMean.dataE(0)
        # xMean = self.wsYSpaceMean.dataX(0)

        plt.figure()
        # plt.errorbar(xMean, yMean, yerr=eMean, fmt="none", label="Mean Data")
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
    
    # def test_saveresults(self):
    #     dataYSym = self.wsYSpaceSym.dataY(0)
    #     dataXSym = self.wsYSpaceSym.dataX(0)
    #     dataESym = self.wsYSpaceSym.dataE(0)

    #     dataYRes = self.wsRes.dataY(0)
    #     dataXRes = self.wsRes.dataX(0)
    #     dataERes = self.wsRes.dataE(0)

    #     np.savez(self.savePath, dataX=dataXSym, dataY=dataYSym, dataE=dataESym,
    #                 dataYRes=dataYRes, dataXRes=dataXRes, dataERes=dataERes)

if __name__ == "__main__":
    unittest.main()
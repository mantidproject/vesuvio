import unittest
import numpy as np
import numpy.testing as nptest
import matplotlib.pyplot as plt
from mantid.simpleapi import *    
from pathlib import Path
currentPath = Path(__file__).absolute().parent  # Path to the repository

def convertWsToYSpaceAndSymetrise(wsName, mass):
    """input: TOF workspace
       output: workspace in y-space for given mass with dataY symetrised"""

    wsYSpace, wsQ = ConvertToYSpace(
        InputWorkspace=wsName, Mass=mass, OutputWorkspace=wsName+"_JoY", QWorkspace=wsName+"_Q"
        )
    max_Y = np.ceil(2.5*mass+27)  
    # First bin boundary, width, last bin boundary
    rebin_parameters = str(-max_Y)+","+str(2.*max_Y/120)+","+str(max_Y)
    wsYSpace = Rebin(
        InputWorkspace=wsYSpace, Params=rebin_parameters, FullBinsOnly=True, OutputWorkspace=wsName+"_JoY"
        )

    dataYSpace = wsYSpace.extractY()
    # safeguarding against nans as well
    nonZerosNansMask = (dataYSpace != 0) & (dataYSpace != np.nan)
    dataYSpace[nonZerosNansMask] = 1
    noOfNonZeroNanY = np.nansum(dataYSpace, axis=0)

    wsSumYSpace = SumSpectra(InputWorkspace=wsYSpace, OutputWorkspace=wsName+"_JoY")

    tmp = CloneWorkspace(InputWorkspace=wsSumYSpace)
    tmp.dataY(0)[:] = noOfNonZeroNanY
    tmp.dataE(0)[:] = np.zeros(tmp.blocksize())
    ws = Divide(                                  # Use of Divide and not nanmean, err are prop automatically
        LHSWorkspace=wsSumYSpace, RHSWorkspace=tmp, OutputWorkspace=wsName+"_JoY"
        )
    ws.dataY(0)[:] = (ws.readY(0)[:] + np.flip(ws.readY(0)[:])) / 2 
    ws.dataE(0)[:] = (ws.readE(0)[:] + np.flip(ws.readE(0)[:])) / 2 
    normalise_workspace(ws)
    return ws

def convert_to_y_space_and_symmetrise(ws_name,mass):
    max_Y = np.ceil(2.5*mass+27)
    rebin_parameters = str(-max_Y)+","+str(2.*max_Y/120)+","+str(max_Y)
    ConvertToYSpace(InputWorkspace=ws_name,Mass=mass,OutputWorkspace=ws_name+"_JoY",QWorkspace=ws_name+"_Q")
    Rebin(InputWorkspace=ws_name+"_JoY", Params = rebin_parameters,FullBinsOnly=True, OutputWorkspace= ws_name+"_JoY")
    tmp=CloneWorkspace(InputWorkspace=ws_name+"_JoY")
    for j in range(tmp.getNumberHistograms()):
        for k in range(tmp.blocksize()):
            tmp.dataE(j)[k] =0.
            if (tmp.dataY(j)[k]!=0):
                tmp.dataY(j)[k] =1.
    tmp=SumSpectra('tmp')
    SumSpectra(InputWorkspace=ws_name+"_JoY",OutputWorkspace=ws_name+"_JoY")
    Divide(LHSWorkspace=ws_name+"_JoY", RHSWorkspace="tmp", OutputWorkspace =ws_name+"_JoY")
    ws=mtd[ws_name+"_JoY"]
    tmp=CloneWorkspace(InputWorkspace=ws_name+"_JoY")
    for k in range(tmp.blocksize()):
        tmp.dataE(0)[k] =(ws.dataE(0)[k]+ws.dataE(0)[ws.blocksize()-1-k])/2.
        tmp.dataY(0)[k] =(ws.dataY(0)[k]+ws.dataY(0)[ws.blocksize()-1-k])/2
    RenameWorkspace(InputWorkspace="tmp",OutputWorkspace=ws_name+"_JoY")
    normalise_workspace(ws_name+"_JoY")
    return mtd[ws_name+"_JoY"]

def normalise_workspace(ws_name):
    tmp_norm = Integration(ws_name)
    Divide(LHSWorkspace=ws_name,RHSWorkspace="tmp_norm",OutputWorkspace=ws_name)
    DeleteWorkspace("tmp_norm")

# Load example workspace 
ws0 = Load(Filename=r"../input_ws/starch_80_RD_raw.nxs")
name = ws0.name()
ws1 = CloneWorkspace(ws0, OutputWorkspace=name+"1")
# Same initial conditions
masses = [1.0079, 12, 16, 27]
dataFilePath = currentPath / "fixatures" / "data_to_test_func_sub_mass.npz"

def createSlabGeometry(slabPars):
    name, vertical_width, horizontal_width, thickness = slabPars
    half_height, half_width, half_thick = 0.5*vertical_width, 0.5*horizontal_width, 0.5*thickness
    xml_str = \
        " <cuboid id=\"sample-shape\"> " \
        + "<left-front-bottom-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (half_width, -half_height, half_thick) \
        + "<left-front-top-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (half_width, half_height, half_thick) \
        + "<left-back-bottom-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (half_width, -half_height, -half_thick) \
        + "<right-front-bottom-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (-half_width, -half_height, half_thick) \
        + "</cuboid>"
    CreateSampleShape(name, xml_str)

vertical_width, horizontal_width, thickness = 0.1, 0.1, 0.001  # Expressed in meters
slabPars = [name, vertical_width, horizontal_width, thickness]
createSlabGeometry(slabPars)

class TestSubMasses(unittest.TestCase):
    def setUp(self):
        dataFile = np.load(dataFilePath)
        ncpForEachMass = dataFile["all_ncp_for_each_mass"][0]
        Hmass = 1.0079
    
        self.wsOri = convert_to_y_space_and_symmetrise(ws0.name(), Hmass)
        self.wsOpt = convertWsToYSpaceAndSymetrise(ws1.name(), Hmass)

        self.rtol = 0.0001

    def test_dataY(self):
        oriDataY = self.wsOri.extractY()#[:, :-2]   # Last two columns are the problem!
        optDataY = self.wsOpt.extractY()#[:, :-2]

        totalMask = np.isclose(
            optDataY, oriDataY, rtol=self.rtol
            )
        totalDiffMask = ~ totalMask
        noDiff = np.sum(totalDiffMask)
        maskSize = totalDiffMask.size
        print("\nNo of different dataY points:\n",
            noDiff, " out of ", maskSize,
            f"ie {100*noDiff/maskSize:.1f} %")
        
        print(oriDataY[0, :10])
        print(optDataY[0, :10])
        print(np.isnan(optDataY).sum())
        # plt.figure()
        # plt.imshow(totalMask, aspect="auto", cmap=plt.cm.RdYlGn, 
        #             interpolation="nearest", norm=None)
        # plt.title("Comparison between optDataY and oriDataY")
        # plt.xlabel("TOF")
        # plt.ylabel("Spectra")
        # plt.show()
        nptest.assert_almost_equal(
            oriDataY, optDataY, decimal=10
        )

if __name__ == "__main__":
    unittest.main()
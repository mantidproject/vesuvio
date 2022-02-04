# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
currentPath = Path(__file__).absolute().parent  # Path to the repository


# Load example workspace 
ws0 = Load(Filename=r"../input_ws/starch_80_RD_raw.nxs", OutputWorkspace="raw_data")
wsName = ws0.name()
ws1 = CloneWorkspace(wsName, OutputWorkspace=wsName+"_opt")

    
def convertToYSpaceAndSymetrise(ws0, mass):
    wsYSpace, wsQ = ConvertToYSpace(
        InputWorkspace=ws0, Mass=mass, OutputWorkspace=ws0.name()+"_JoY", QWorkspace=ws0.name()+"_Q"
        )
    max_Y = np.ceil(2.5*mass+27)  
    # First bin boundary, width, last bin boundary
    rebin_parameters = str(-max_Y)+","+str(2.*max_Y/120)+","+str(max_Y)
    wsYSpace = Rebin(
        InputWorkspace=wsYSpace, Params=rebin_parameters, FullBinsOnly=True, OutputWorkspace=ws0.name()+"_JoY"
        )

    dataY = wsYSpace.extractY()
    # safeguarding against nans as well
    nanOrZerosMask = (dataY==0) | np.isnan(dataY)
    noOfNonZerosRow = (~nanOrZerosMask).sum(axis=0)

    wsSumYSpace = SumSpectra(InputWorkspace=wsYSpace, OutputWorkspace=ws0.name()+"_JoY_sum")

    tmp = CloneWorkspace(InputWorkspace=wsSumYSpace)
    tmp.dataY(0)[:] = noOfNonZerosRow
    tmp.dataE(0)[:] = np.zeros(noOfNonZerosRow.shape)

    wsMean = Divide(                                  # Use of Divide and not nanmean, err are prop automatically
        LHSWorkspace=wsSumYSpace, RHSWorkspace=tmp, OutputWorkspace=ws0.name()+"_JoY_mean"
       )
    # Need to correct this up  
    ws = CloneWorkspace(wsMean, OutputWorkspace=ws0.name()+"_JoY_Sym")

    datay = ws.dataY(0)[:]
    datay = np.where(np.isnan(datay), np.flip(datay), datay)
    ws.dataY(0)[:] = (datay + np.flip(datay)) / 2

    datae = ws.dataE(0)[:]
    datae = np.where(np.isnan(datae), np.flip(datae), datae)
    ws.dataE(0)[:] = (datae + np.flip(datae)) / 2
    normalise_workspace(ws)
    DeleteWorkspaces(
        [ws0.name()+"_JoY_sum", ws0.name()+"_JoY_mean"]
        )


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

def test_function():
    # Same initial conditions
    Hmass = 1.0079
    wsOpt = convertToYSpaceAndSymetrise(ws1, Hmass)
    wsOri = convert_to_y_space_and_symmetrise(wsName, Hmass)
    
test_function()
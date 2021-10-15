# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
currentPath = Path(__file__).absolute().parent  # Path to the repository

# Load example workspace 
ws0 = Load(Filename=r"../input_ws/starch_80_RD_raw.nxs", OutputWorkspace="raw_data")
wsName = ws0.name()
# Same initial conditions
masses = [1.0079, 12, 16, 27]
dataFilePath = currentPath / "fixatures" / "data_to_test_func_sub_mass.npz"

# def createSlabGeometry(slabPars):
#     name, vertical_width, horizontal_width, thickness = slabPars
#     half_height, half_width, half_thick = 0.5*vertical_width, 0.5*horizontal_width, 0.5*thickness
#     xml_str = \
#         " <cuboid id=\"sample-shape\"> " \
#         + "<left-front-bottom-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (half_width, -half_height, half_thick) \
#         + "<left-front-top-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (half_width, half_height, half_thick) \
#         + "<left-back-bottom-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (half_width, -half_height, -half_thick) \
#         + "<right-front-bottom-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (-half_width, -half_height, half_thick) \
#         + "</cuboid>"
#     CreateSampleShape(name, xml_str)
# 
# vertical_width, horizontal_width, thickness = 0.1, 0.1, 0.001  # Expressed in meters
# slabPars = [name, vertical_width, horizontal_width, thickness]
# createSlabGeometry(slabPars)

# Testing section
dataFile = np.load(dataFilePath)
ncpForEachMass = dataFile["all_ncp_for_each_mass"][0]
Hmass = 1.0079

wsYSpace, wsQ = ConvertToYSpace(
    InputWorkspace=ws0, Mass=Hmass, OutputWorkspace=wsName+"_JoY", QWorkspace=wsName+"_Q"
    )
max_Y = np.ceil(2.5*Hmass+27)  
# First bin boundary, width, last bin boundary
rebin_parameters = str(-max_Y)+","+str(2.*max_Y/120)+","+str(max_Y)
wsYSpace = Rebin(
    InputWorkspace=wsYSpace, Params=rebin_parameters, FullBinsOnly=True, OutputWorkspace=wsName+"_JoY"
    )

dataY = wsYSpace.extractY()
# safeguarding against nans as well
nanOrZerosMask = (dataY==0) | np.isnan(dataY)
noOfNonZerosRow = (~nanOrZerosMask).sum(axis=0)

wsSumYSpace = SumSpectra(InputWorkspace=wsYSpace, OutputWorkspace=wsName+"_sum")

tmp = CloneWorkspace(InputWorkspace=wsSumYSpace)
tmp.dataY(0)[:] = noOfNonZerosRow
tmp.dataE(0)[:] = np.zeros(noOfNonZerosRow.shape)

wsMean = Divide(                                  # Use of Divide and not nanmean, err are prop automatically
    LHSWorkspace=wsSumYSpace, RHSWorkspace=tmp, OutputWorkspace=wsName+"_JoY_mean"
   )

# Need to correct this up  
ws = CloneWorkspace(wsMean, OutputWorkspace=wsName+"_Sym")

datay = ws.dataY(0)[:]
datay = np.where(np.isnan(datay), np.flip(datay), datay)
dataySym = datay + np.flip(dataY)
ws.dataY(0)[:] = dataySym
#ws.dataY(0)[:] = (ws.readY(0)[:] + np.flip(ws.readY(0)[:])) / 2 
# ws.dataE(0)[:] = (ws.readE(0)[:] + np.flip(ws.readE(0)[:])) / 2 

# def normalise_workspace(ws_name):
#     tmp_norm = Integration(ws_name)
#     Divide(LHSWorkspace=ws_name,RHSWorkspace="tmp_norm",OutputWorkspace=ws_name)
#     DeleteWorkspace("tmp_norm")
# normalise_workspace(ws)
#    
   
# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

def calculate_mantid_resolutions(ws_name, mass):
    max_Y = np.ceil(2.5*mass+27)
    rebin_parameters = str(-max_Y)+","+str(2.*max_Y/240)+","+str(max_Y)
    ws= mtd[ws_name]
    for index in range(ws.getNumberHistograms()):
        VesuvioResolution(Workspace=ws,WorkspaceIndex=index,Mass=mass,OutputWorkspaceYSpace="tmp")
        tmp=Rebin("tmp",rebin_parameters)
        if index == 0:
            RenameWorkspace("tmp","resolution")
        else:
            AppendSpectra("resolution", "tmp", OutputWorkspace= "resolution")
    print("before sum: ", mtd["resolution"].dataY(0)[:5])
    CloneWorkspace(InputWorkspace="resolution", OutputWorkspace="before_sum0")
    SumSpectra(InputWorkspace="resolution", OutputWorkspace="resolution")
    normalise_workspace("resolution")
    DeleteWorkspace("tmp")
    
def calculate_mantid_resolutions_improved(ws_name, mass):
    max_Y = np.ceil(2.5*mass+27)
    rebin_parameters = str(-max_Y)+","+str(2.*max_Y/240)+","+str(max_Y)
    ws= mtd[ws_name]
    tof, ysp = VesuvioResolution(Workspace=ws, Mass=mass, OutputWorkspaceYSpace="before_rebin1")
    ysp_rebin = Rebin(ysp, rebin_parameters, OutputWorkspace = "before_sum1")
    res = SumSpectra(InputWorkspace=ysp_rebin, OutputWorkspace="resolution1")
    normalise_workspace(res)
    DeleteWorkspace(tof)
    
def normalise_workspace(ws_name):
    tmp_norm = Integration(ws_name)
    Divide(LHSWorkspace=ws_name,RHSWorkspace="tmp_norm",OutputWorkspace=ws_name)
    DeleteWorkspace("tmp_norm")

ws = mtd["CePtGe12_100K_DD_"]
calculate_mantid_resolutions(ws.name(), 140)
calculate_mantid_resolutions_improved(ws.name(), 140)
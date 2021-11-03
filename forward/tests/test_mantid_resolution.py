import numpy as np
from mantid.simpleapi import *


rebinParametersForYSpaceFit = "-20, 0.5, 20"

def original_function(ws, mass):
    rebin_parameters=rebinParametersForYSpaceFit
    for index in range(ws.getNumberHistograms()):
        VesuvioResolution(Workspace=ws,WorkspaceIndex=index,Mass=mass,OutputWorkspaceYSpace="tmp")
        Rebin(InputWorkspace="tmp", Params=rebin_parameters, OutputWorkspace="tmp")
        if index == 0:
            RenameWorkspace("tmp","resolution1")
        else:
            AppendSpectra("resolution1", "tmp", OutputWorkspace= "resolution1")
    SumSpectra(InputWorkspace="resolution1",OutputWorkspace="resolution1")
    normalise_workspace("resolution1")
    DeleteWorkspace("tmp")
    return mtd["resolution1"]

def simpler_function(ws, mass):
    rebin_parameters=rebinParametersForYSpaceFit
    resolution = CloneWorkspace(InputWorkspace=ws)
    resolution = Rebin(InputWorkspace=resolution, Params=rebin_parameters)

    for i in range(resolution.getNumberHistograms()):
        VesuvioResolution(
            Workspace=ws, WorkspaceIndex=str(i), Mass=mass, OutputWorkspaceYSpace='tmp'
            )
        tmp = Rebin(InputWorkspace='tmp', Params=rebin_parameters)
        resolution.dataY(i)[:] = tmp.dataY(0)

    SumSpectra(InputWorkspace="resolution", OutputWorkspace="resolution")
    normalise_workspace("resolution")
    DeleteWorkspace("tmp")
    return mtd["resolution"]

def normalise_workspace(ws_name):
    tmp_norm = Integration(ws_name)
    Divide(LHSWorkspace=ws_name,RHSWorkspace=tmp_norm,OutputWorkspace=ws_name)
    DeleteWorkspace("tmp_norm")


def test_func():
    mass = 1.079
    wsTest = Load(Filename= r"./input_ws/starch_80_RD_raw.nxs")
    res1 = original_function(wsTest, mass)
    res2 = simpler_function(wsTest, mass)
    CompareWorkspaces(res1, res2)
    # Workspaces differ only in the units, which maybe is important for Fit algorithm

test_func()
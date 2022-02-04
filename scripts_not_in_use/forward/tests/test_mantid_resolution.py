import numpy as np
from mantid.simpleapi import *


rebinParametersForYSpaceFit = "-20, 0.5, 20"

def function_in_optimized(ws, mass):
    rebin_parameters=rebinParametersForYSpaceFit
    for index in range(ws.getNumberHistograms()):
        if np.all(ws.dataY(index)[:] == 0):
            pass
        else:
            VesuvioResolution(Workspace=ws,WorkspaceIndex=index,Mass=mass,OutputWorkspaceYSpace="tmp")
            Rebin(InputWorkspace="tmp", Params=rebin_parameters, OutputWorkspace="tmp")
            if index == 0:
                RenameWorkspace("tmp","resolution_opt")
            else:
                AppendSpectra("resolution_opt", "tmp", OutputWorkspace= "resolution_opt")
    SumSpectra(InputWorkspace="resolution_opt",OutputWorkspace="resolution_opt_sum")
    normalise_workspace("resolution_opt_sum")
    DeleteWorkspace("tmp")
    return mtd["resolution_opt_sum"]

def normalise_workspace(ws_name):
    tmp_norm = Integration(ws_name)
    Divide(LHSWorkspace=ws_name,RHSWorkspace=tmp_norm,OutputWorkspace=ws_name)
    DeleteWorkspace("tmp_norm")


def function_in_original(ws, mass):
    resolution=CloneWorkspace(InputWorkspace=ws)          # The clonning of joy workspace must be to preserve units
    resolution=Rebin(InputWorkspace='resolution',Params=rebinParametersForYSpaceFit)
    for i in range(resolution.getNumberHistograms()):
#         if np.all(ws.dataY(i)[:] == 0):
#             pass
#         else:
        VesuvioResolution(Workspace=ws, WorkspaceIndex=str(i), Mass=mass, OutputWorkspaceYSpace='tmp')
        tmp=Rebin(InputWorkspace='tmp',Params=rebinParametersForYSpaceFit)
        for p in range (tmp.blocksize()):
            resolution.dataY(i)[p]=tmp.dataY(0)[p]

    # Definition of the sum of resolution functions
    resolution_sum=SumSpectra('resolution')
    tmp=Integration('resolution_sum')
    resolution_sum=Divide('resolution_sum','tmp')
    DeleteWorkspace('tmp')  
    return mtd["resolution_sum"]


def test_func():
    mass = 1.0079
    wsTest = Load(Filename= r"./input_ws/starch_80_RD_raw.nxs")
    MaskDetectors(Workspace=wsTest, SpectraList=[173, 174, 179])
    res1 = function_in_original(wsTest, mass)
    res2 = function_in_optimized(wsTest, mass)
    CompareWorkspaces(res1, res2, CheckAxes=False)
    Integration(res1, OutputWorkspace="integration_ori")
    Integration(res2, OutputWorkspace="integration_opt")
    # Check if convert to y space alters the masked spectra
    ConvertToYSpace(InputWorkspace=wsTest,Mass=mass,OutputWorkspace='joy',QWorkspace='q')


test_func()
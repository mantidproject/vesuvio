import numpy as np
import numpy.testing as nptest
import matplotlib.pyplot as plt
from mantid.simpleapi import *    
from pathlib import Path
from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=True)


# -------------- Use updated version of sub mass, was already tesed -----
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
# --------------- end of optimized code -----------

currentPath = Path(__file__).absolute().parent  # Path to the repository
# Load example workspace 
exampleWorkspace = Load(Filename=r"../input_ws/starch_80_RD_raw.nxs", OutputWorkspace="starch_80_RD_raw_")

ws_name = exampleWorkspace.name()
# Same initial conditions
masses = [1.0079, 12, 16, 27]
# Load results that were obtained from the same workspace
dataFilePath = currentPath / "fixatures" / "data_to_test_func_sub_mass.npz"
dataFile = np.load(dataFilePath)
ncpForEachMass = dataFile["all_ncp_for_each_mass"][0]
Hmass = 1.0079
H_spectra_to_be_masked = [173, 174, 179]

first_ws = subtractAllMassesExceptFirst(exampleWorkspace, ncpForEachMass)

# Treat the nans in the workspace, otherwise SumSpectra does not work
# dataY = first_ws.extractY()
# dataY = np.where(np.isnan(dataY), 0.0, dataY)
# for i in range(first_ws.getNumberHistograms()):
#     first_ws.dataY(i)[:] = dataY[i, :]
# ----------------

#RenameWorkspace(InputWorkspace=first_ws, OutputWorkspace=ws_name+'_H')
Rebin(InputWorkspace=ws_name+'_H',Params="110,1.,430",OutputWorkspace=ws_name+'_H')
MaskDetectors(Workspace=ws_name+'_H',SpectraList=H_spectra_to_be_masked)
RemoveMaskedSpectra(InputWorkspace=ws_name+'_H', OutputWorkspace=ws_name+'_H')      

# Conversion to hydrogen West-scaling variable
rebin_params='-20,0.5,20'    
ConvertToYSpace(InputWorkspace=ws_name+'_H',Mass=1.0079,OutputWorkspace=ws_name+'joy',QWorkspace=ws_name+'q')
Rebin(InputWorkspace=ws_name+'joy',Params=rebin_params,OutputWorkspace=ws_name+'joy')
Rebin(InputWorkspace=ws_name+'q',Params=rebin_params,OutputWorkspace=ws_name+'q')
tmp=Integration(InputWorkspace=ws_name+'joy',RangeLower='-20',RangeUpper='20')
Divide(LHSWorkspace=ws_name+'joy',RHSWorkspace='tmp',OutputWorkspace=ws_name+'joy')

# Symmetrisation to remove the FSEs
ws=mtd[ws_name+'joy']
for j in range(ws.getNumberHistograms()):
    for k in range(ws.blocksize()):
        if (ws.dataX(j)[k]<0):
            ws.dataY(j)[k] =ws.dataY(j)[ws.blocksize()-1-k]
            ws.dataE(j)[k] =ws.dataE(j)[ws.blocksize()-1-k]


# Definition of a normalising workspace taking into consideration the kinematic constraints
ws=CloneWorkspace(InputWorkspace=ws_name+'joy')
for j in range(ws.getNumberHistograms()):
    for k in range(ws.blocksize()):
        ws.dataE(j)[k] =0.000001
        if (ws.dataY(j)[k]!=0):
            ws.dataY(j)[k] =1.
ws=SumSpectra('ws')
RenameWorkspace('ws',ws_name+'joy_sum_normalisation')


# Definition of the sum of all spectra
SumSpectra(ws_name+'joy',OutputWorkspace=ws_name+'joy_sum')
Divide(LHSWorkspace=ws_name+'joy_sum',RHSWorkspace=ws_name+'joy_sum_normalisation',OutputWorkspace=ws_name+'joy_sum')

# Definition of the resolution functions
resolution=CloneWorkspace(InputWorkspace=ws_name+'joy')
resolution=Rebin(InputWorkspace='resolution',Params='-20,0.5,20')
for i in range(resolution.getNumberHistograms()):
    VesuvioResolution(Workspace=ws_name, Mass=1.0079, OutputWorkspaceYSpace='tmp')
    tmp=Rebin(InputWorkspace='tmp',Params='-20,0.5,20')
    for p in range (tmp.blocksize()):
        resolution.dataY(i)[p]=tmp.dataY(0)[p]

# Definition of the sum of resolution functions
resolution_sum=SumSpectra('resolution')
tmp=Integration('resolution_sum')
resolution_sum=Divide('resolution_sum','tmp')
DeleteWorkspace('tmp') 
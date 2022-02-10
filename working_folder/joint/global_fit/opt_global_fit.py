# import sys 
# sys.path.append('..')

from mantid.simpleapi import *
from pathlib import Path
import numpy as np
currentPath = Path(__file__).absolute().parent 

isolatedPath = currentPath / "DHMT_300K_backward_deuteron_.nxs"
wsD = Load(str(isolatedPath), OutputWorkspace='DHMT_300K_backward_deuteron_')
rebin_params='-30,0.5,30'
name = 'DHMT_300K_backward_deuteron_'

wsD.dataY(3)[::3] = 0
wsD.dataY(5)[3::2] = 0 
wsD.dataY(8)[:] = 0



def normalise_workspace(ws_name):
    tmp_norm = Integration(ws_name)
    Divide(LHSWorkspace=ws_name,RHSWorkspace=tmp_norm,OutputWorkspace=ws_name)
    DeleteWorkspace("tmp_norm")

def convertToYSpace(rebinPars, ws0, mass):
    wsJoY, wsQ = ConvertToYSpace(
        InputWorkspace=ws0, Mass=mass, 
        OutputWorkspace=ws0.name()+"_JoY", QWorkspace=ws0.name()+"_Q"
        )
    wsJoY = Rebin(
        InputWorkspace=wsJoY, Params=rebinPars, 
        FullBinsOnly=True, OutputWorkspace=ws0.name()+"_JoY"
        )
    wsQ = Rebin(
        InputWorkspace=wsQ, Params=rebinPars, 
        FullBinsOnly=True, OutputWorkspace=ws0.name()+"_Q"
        )
    
    # If workspace has nans present, normalization will put zeros for the whole spectra
    assert np.any(np.isnan(wsJoY.extractY()))==False, "Nans present before normalization."
    
    normalise_workspace(wsJoY)
    return wsJoY, wsQ


def replaceNansWithZeros(ws):
    for j in range(ws.getNumberHistograms()):
        ws.dataY(j)[np.isnan(ws.dataY(j)[:])] = 0
        ws.dataE(j)[np.isnan(ws.dataE(j)[:])] = 0


def artificialErrorsInUnphysicalBins(wsJoY):
    wsGlobal = CloneWorkspace(InputWorkspace=wsJoY, OutputWorkspace=wsJoY.name()+'_Global')
    for j in range(wsGlobal.getNumberHistograms()):
        wsGlobal.dataE(j)[wsGlobal.dataE(j)[:]==0] = 0.1
    
    assert np.any(np.isnan(wsGlobal.extractE())) == False, "Nan present in input workspace need to be replaced by zeros."

    return wsGlobal


def createOneOverQWs(wsQ):
    wsInvQ = CloneWorkspace(InputWorkspace=wsQ, OutputWorkspace=wsQ.name()+"_Inverse")
    for j in range(wsInvQ.getNumberHistograms()):
        nonZeroFlag = wsInvQ.dataY(j)[:] != 0
        wsInvQ.dataY(j)[nonZeroFlag] = 1 / wsInvQ.dataY(j)[nonZeroFlag]

        ZeroIdxs = np.argwhere(wsInvQ.dataY(j)[:]==0)   # Indxs of zero elements
        if ZeroIdxs.size != 0:
            wsInvQ.dataY(j)[ZeroIdxs[0] - 1] = 0
    
    return wsInvQ


# Optimized procedure:
wsJoY, wsQ = convertToYSpace(rebin_params, wsD, 2.015)
replaceNansWithZeros(wsJoY)
wsGlobal = artificialErrorsInUnphysicalBins(wsJoY)
wsQInv = createOneOverQWs(wsQ)



# Original procedure:
ConvertToYSpace(InputWorkspace=name,Mass=2.015,OutputWorkspace=name+'joy',
                            QWorkspace=name+'q')
Rebin(InputWorkspace=name+'joy',Params=rebin_params,OutputWorkspace=name+'joy')
Rebin(InputWorkspace=name+'q',Params=rebin_params,OutputWorkspace=name+'q')  

# Normalisation 
tmp=Integration(InputWorkspace=name+"joy",RangeLower='-30',RangeUpper='30')
Divide(LHSWorkspace=name+"joy",RHSWorkspace='tmp',OutputWorkspace=name+"joy")
    

# Replacement of Nans with zeros
ws=CloneWorkspace(InputWorkspace=name+'joy')
for j in range(ws.getNumberHistograms()):
    for k in range(ws.blocksize()):
        if  np.isnan(ws.dataY(j)[k]):
            ws.dataY(j)[k] =0.
        if  np.isnan(ws.dataE(j)[k]):
            ws.dataE(j)[k] =0.
RenameWorkspace('ws',name+'joy')


CloneWorkspace(InputWorkspace=name+'joy', OutputWorkspace=name+'joy_global')
ws=mtd[name+'joy_global']
for j in range(ws.getNumberHistograms()):
    for k in range(ws.blocksize()):
        if (ws.dataE(j)[k]==0):
            ws.dataE(j)[k] =0.1
        if np.isnan(ws.dataE(j)[k]):
            ws.dataE(j)[k] =0.1


# Definition of the 1/Q workspace for correction of the FSE in the global fit
CloneWorkspace(InputWorkspace=name+'q',OutputWorkspace='one_over_q')
ws=mtd['one_over_q']
for j in range(ws.getNumberHistograms()):
    flag=True
    for k in range(ws.blocksize()):
        if (ws.dataY(j)[k]!=0):
            ws.dataY(j)[k] =1./ws.dataY(j)[k]
        if (ws.dataY(j)[k] == 0):
            if (flag):
                ws.dataY(j)[k-1] =0
                flag=False


# Compare Workspaces
CompareWorkspaces(name+"_JoY", name+"joy")
CompareWorkspaces(name+"_Q", name+"q")
CompareWorkspaces(name+"_JoY_Global", name+"joy_global")
CompareWorkspaces(name+"_Q_Inverse", "one_over_q")






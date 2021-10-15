import unittest
import numpy as np
import numpy.testing as nptest
import matplotlib.pyplot as plt
from mantid.simpleapi import *    
from pathlib import Path
currentPath = Path(__file__).absolute().parent  # Path to the repository

# Load example workspace 
wsExample = Load(Filename=r"../input_ws/starch_80_RD_raw.nxs")

# Same initial conditions
masses = [1.0079, 12, 16, 27]
dataFilePath = currentPath / "fixatures" / "data_to_test_func_sub_mass.npz"


class TestSubMasses(unittest.TestCase):
    def setUp(self):
        dataFile = np.load(dataFilePath)
        ncpForEachMass = dataFile["all_ncp_for_each_mass"][0]



def subtractAllMassesExceptFirst(ws, ncpForEachMass):
    """Input: workspace from last iteration, ncpTotal for each mass
       Output: workspace with all the ncpTotal subtracted except for the first mass"""

    ncpForEachMass = switchFirstTwoAxis(ncpForEachMass)
    # Select all masses other than the first one
    ncpForEachMass = ncpForEachMass[1:, :, :]
    # Sum the ncpTotal for remaining masses
    ncpTotal = np.sum(ncpForEachMass, axis=0)
    dataY, dataX, dataE = ws.extractY(), ws.extractX(), ws.extractE()

    # The original uses the mean points of the histograms, not dataX!
    dataY[:, :-1] -= ncpTotal * (dataX[:, 1:] - dataX[:, :-1])
    wsSubMass = CreateWorkspace(DataX=dataX.flatten(), DataY=dataY.flatten(), DataE=dataE.flatten(), Nspec=len(dataX))
    return wsSubMass

def switchFirstTwoAxis(A):
    """Exchanges the first two indices of an array A,
    rearranges matrices per spectrum for iteration of main fitting procedure
    """
    return np.stack(np.split(A, len(A), axis=0), axis=2)[0]


def originalFitProcedure(wsFinal, ncpForEachMass):
    ws_name = wsFinal.name()
    first_ws = subtractAllMassesExceptFirst(wsFinal, ncpForEachMass)

    RenameWorkspace("first_ws", ws_name+'_H')
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

    # # Definition of the resolution functions
    # resolution=CloneWorkspace(InputWorkspace=ws_name+'joy')
    # resolution=Rebin(InputWorkspace='resolution',Params='-20,0.5,20')
    # for i in range(resolution.getNumberHistograms()):
    #     VesuvioResolution(Workspace=ws_name+str(iteration),WorkspaceIndex=str(i), Mass=1.0079, OutputWorkspaceYSpace='tmp')
    #     tmp=Rebin(InputWorkspace='tmp',Params='-20,0.5,20')
    #     for p in range (tmp.blocksize()):
    #         resolution.dataY(i)[p]=tmp.dataY(0)[p]

    # # Definition of the sum of resolution functions
    # resolution_sum=SumSpectra('resolution')
    # tmp=Integration('resolution_sum')
    # resolution_sum=Divide('resolution_sum','tmp')
    # DeleteWorkspace('tmp')        


    ############################################################################
    ######
    ######              FIT OF THE SUM OF SPECTRA 
    ######
    ############################################################################
    # print('\n','Fit on the sum of spectra in the West domain','\n')
    # for minimizer_sum in ('Levenberg-Marquardt','Simplex'):
    #     CloneWorkspace(InputWorkspace = ws_name+'joy_sum', OutputWorkspace = ws_name+minimizer_sum+'_joy_sum_fitted')
        
    #     if (simple_gaussian_fit):
    #         function='''composite=Convolution,FixResolution=true,NumDeriv=true;
    #         ws_name=Resolution,Workspace=resolution_sum,WorkspaceIndex=0;
    #         ws_name=UserFunction,Formula=y0+A*exp( -(x-x0)^2/2/sigma^2)/(2*3.1415*sigma^2)^0.5,
    #         y0=0,A=1,x0=0,sigma=5,   ties=()'''
    #     else:
    #         function='''composite=Convolution,FixResolution=true,NumDeriv=true;
    #         ws_name=Resolution,Workspace=resolution_sum,WorkspaceIndex=0;
    #         ws_name=UserFunction,Formula=y0+A*exp( -(x-x0)^2/2/sigma1^2)/(2*3.1415*sigma1^2)^0.5
    #         +A*0.054*exp( -(x-x0)^2/2/sigma2^2)/(2*3.1415*sigma2^2)^0.5,
    #         y0=0,x0=0,A=0.7143,sigma1=4.76, sigma2=5,   ties=(sigma1=4.76)'''

    #     Fit(Function=function, InputWorkspace=ws_name+minimizer_sum+'_joy_sum_fitted', Output=ws_name+minimizer_sum+'_joy_sum_fitted',Minimizer=minimizer_sum)
        
    #     ws=mtd[ws_name+minimizer_sum+'_joy_sum_fitted_Parameters']
    #     print('Using the minimizer: ',minimizer_sum)
    #     print('Hydrogen standard deviation: ',ws.cell(3,1),' +/- ',ws.cell(3,2))

if __name__ == "__main__":
    unittest.main()
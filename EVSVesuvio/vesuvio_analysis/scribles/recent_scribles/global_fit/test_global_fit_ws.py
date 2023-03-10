
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

wsD.dataE(3)[::3] = 0
wsD.dataE(5)[3::2] = 0 
wsD.dataE(8)[:] = 0

# To convert this script into a unit tes, import the functions below

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
    
    # If workspace has nans present, normalization will put zeros on the full spectrum
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
        if ZeroIdxs.size != 0:     # When zeros are present
            wsInvQ.dataY(j)[ZeroIdxs[0] - 1] = 0       # Put a zero before the first zero
    
    return wsInvQ


# Optimized procedure:
wsJoY, wsQ = convertToYSpace(rebin_params, wsD, 2.015)


# Introduce zeros to test exceptional cases
wsQ.dataY(3)[::3] = np.nan
wsQ.dataY(5)[3::2] = 0 
wsQ.dataY(8)[:] = 0

wsJoY.dataY(3)[::3] = np.nan
wsJoY.dataY(5)[3::2] = 0 
wsJoY.dataY(8)[:] = 0


replaceNansWithZeros(wsJoY)
wsGlobal = artificialErrorsInUnphysicalBins(wsJoY)
wsQInv = createOneOverQWs(wsQ)



# Original procedure:
ConvertToYSpace(InputWorkspace=name,Mass=2.015,OutputWorkspace=name+'joy',
                            QWorkspace=name+'q')
wsOriJoY = Rebin(InputWorkspace=name+'joy',Params=rebin_params,OutputWorkspace=name+'joy')
wsOriQ = Rebin(InputWorkspace=name+'q',Params=rebin_params,OutputWorkspace=name+'q')  

# Normalisation 
tmp=Integration(InputWorkspace=name+"joy",RangeLower='-30',RangeUpper='30')
wsOriJoY = Divide(LHSWorkspace=name+"joy",RHSWorkspace='tmp',OutputWorkspace=name+"joy")
 

# Introduce same zeros in original
wsOriQ.dataY(3)[::3] = np.nan
wsOriQ.dataY(5)[3::2] = 0 
wsOriQ.dataY(8)[:] = 0

wsOriJoY.dataY(3)[::3] = np.nan
wsOriJoY.dataY(5)[3::2] = 0 
wsOriJoY.dataY(8)[:] = 0


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



def calculateMantidResolution(resPars, ws, mass):
    rebinPars=rebin_params
    for index in range(ws.getNumberHistograms()):
        if np.all(ws.dataY(index)[:] == 0):  # Ignore masked spectra
            pass
        else:
            VesuvioResolution(Workspace=ws,WorkspaceIndex=index,Mass=mass,OutputWorkspaceYSpace="tmp")
            Rebin(InputWorkspace="tmp", Params=resPars, OutputWorkspace="tmp")

            if index == 0:   # Ensures that workspace has desired units
                RenameWorkspace("tmp",  ws.name()+"Resolution")
            else:
                AppendSpectra(ws.name()+"Resolution", "tmp", OutputWorkspace= ws.name()+"Resolution")

    try:
        wsResSum = SumSpectra(InputWorkspace=ws.name()+"Resolution", OutputWorkspace=ws.name()+"Resolution_Sum")
    except ValueError:
        raise ValueError ("All the rows from the workspace to be fitted are Nan!")

    normalise_workspace(wsResSum)
    DeleteWorkspace("tmp")
    return wsResSum, mtd[ws.name()+"Resolution"]
    
    
def originalResolution(resPars, ws, mass):
    resolution=CloneWorkspace(InputWorkspace=name+'joy')
    resolution=Rebin(InputWorkspace='resolution',Params=resPars)
    for i in range(resolution.getNumberHistograms()):
        VesuvioResolution(Workspace=ws,WorkspaceIndex=str(i), Mass=mass, OutputWorkspaceYSpace='tmp')
        tmp=Rebin(InputWorkspace='tmp',Params=resPars)
        for p in range (tmp.blocksize()):
            resolution.dataY(i)[p]=tmp.dataY(0)[p]
    # Definition of the sum of resolution functions
    resolution_sum=SumSpectra('resolution')
    tmp=Integration('resolution_sum')
    resolution_sum=Divide('resolution_sum','tmp')
    DeleteWorkspace('tmp')     
    return 


resPars = "-30, 0.125, 30"
mass = 2.015
calculateMantidResolution(resPars, wsD, mass)
originalResolution(resPars, wsD, mass)

oriRes = mtd["resolution_sum"]
optRes = mtd[name+"Resolution_Sum"]
CompareWorkspaces(oriRes, optRes)
# Test that the original resolution function used is the same as the one I am using in my code
np.testing.assert_allclose(oriRes.extractY(), optRes.extractY())


if False:
    # original global fit
    # Global fit of the H J(y) with global values of Sigma, and  FSE in the harmonic approximation
    sample_ws = mtd[name+'joy_global']
    resolution_ws = mtd['resolution']

    #####   ----edits------
    simple_gaussian_fit = True
    verbose=True
    ##### ----------

    if (simple_gaussian_fit):
        convolution_template = """
        (composite=Convolution,$domains=({0});
        name=Resolution,Workspace={1},WorkspaceIndex={0};
        (
        name=UserFunction,Formula=
        A*exp( -(x-x0)^2/2/Sigma^2)/(2*3.1415*Sigma^2)^0.5,
        A=1.,x0=0.,Sigma=6.0,  ties=();
        (
                composite=ProductFunction,NumDeriv=false;name=TabulatedFunction,Workspace=one_over_q,WorkspaceIndex={0},ties=(Scaling=1,Shift=0,XScaling=1);
                name=UserFunction,Formula=
                Sigma*1.4142/12.*exp( -(x)^2/2/Sigma^2)/(2*3.1415*Sigma^2)^0.5
                *((8*((x)/sqrt(2.)/Sigma)^3-12*((x)/sqrt(2.)/Sigma))),
                Sigma=6.0);ties=()
        ))"""
    else:
        convolution_template = """
        (composite=Convolution,$domains=({0});
        name=Resolution,Workspace={1},WorkspaceIndex={0};
        (
        name=UserFunction,Formula=
        A*exp( -(x-x0)^2/2/Sigma^2)/(2*3.1415*Sigma^2)^0.5
        *(1+c4/32*(16*((x-x0)/sqrt(2.)/Sigma)^4-48*((x-x0)/sqrt(2.)/Sigma)^2+12)),
        A=1.,x0=0.,Sigma=6.0, c4=0, ties=();
        (
                composite=ProductFunction,NumDeriv=false;name=TabulatedFunction,Workspace=one_over_q,WorkspaceIndex={0},ties=(Scaling=1,Shift=0,XScaling=1);
                name=UserFunction,Formula=
                Sigma*1.4142/12.*exp( -(x)^2/2/Sigma^2)/(2*3.1415*Sigma^2)^0.5
                *((8*((x)/sqrt(2.)/Sigma)^3-12*((x)/sqrt(2.)/Sigma))),
                Sigma=6.0);ties=()
        ))"""    

    print('\n','Global fit in the West domain over 8 mixed banks','\n')
    ######################prova
    #   GLOBAL FIT - MIXED BANKS
    minimizer = "Simplex"
    w=[]
    for bank in range(8):
        dets=[ bank, bank+8, bank+16, bank+24]
        print(dets)
        convolved_funcs = []
        ties = []
        datasets = {}
        counter = 0
        for i in dets:
            spec_i = sample_ws.getSpectrum(i)
            det_i = sample_ws.getDetector(i)
            #print "Considering spectrum {0}".format(spec_i.getSpectrumNo())
            if (verbose):
                if det_i.isMasked():
                    #print "Skipping masked spectrum {0}".format(spec_i.getSpectrumNo())
                    continue
            f1 = convolution_template.format(counter, resolution_ws.getName())
            convolved_funcs.append(f1)
            # Tie widths together for spectra together
            if counter > 0:
                ties.append('f{0}.f1.f0.Sigma= f{0}.f1.f1.f1.Sigma=f0.f1.f0.Sigma'.format(counter))
                #ties.append('f{0}.f1.f0.c4=f0.f1.f0.c4'.format(counter))
                #ties.append('f{0}.f1.f1.f1.c3=f0.f1.f1.f1.c3'.format(counter))
            # Attach datasets
            attr = 'InputWorkspace_{0}'.format(counter) if counter > 0 else 'InputWorkspace'
            datasets[attr] = sample_ws.getName()
            attr = 'WorkspaceIndex_{0}'.format(counter) if counter > 0 else 'WorkspaceIndex'
            datasets[attr] = i
            counter += 1
    # end
        multifit_func = 'composite=MultiDomainFunction;' + ';'.join(convolved_funcs)  + ';ties=({0},f0.f1.f1.f1.Sigma=f0.f1.f0.Sigma)'.format(','.join(ties))
        minimizer_string=minimizer+',AbsError=0.00001,RealError=0.00001,MaxIterations=2000'
        Fit(multifit_func,Minimizer=minimizer_string, Output=name+'joy_mixed_banks_bank_'+str(bank)+'_fit', **datasets)
        ws=mtd[name+'joy_mixed_banks_bank_'+str(bank)+'_fit_Parameters']
        if (verbose):
            print('bank: ', str(bank), ' -- sigma= ', ws.cell(2,1), ' +/- ', ws.cell(2,2))
        w.append(ws.cell(2,1))
        if (not verbose):
            DeleteWorkspace(name+'joy_mixed_banks_bank_'+str(bank)+'_fit_NormalisedCovarianceMatrix')
            DeleteWorkspace(name+'joy_mixed_banks_bank_'+str(bank)+'_fit_Workspaces') 
    print('Average hydrogen standard deviation: ',np.mean(w),' +/- ', np.std(w))





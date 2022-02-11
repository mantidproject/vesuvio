from mantid.simpleapi import *
from pathlib import Path
import numpy as np
# from ..D_HMT import yfitIC
# from ..core_functions.fit_in_yspace import subtractAllMassesExceptFirst, averageJOfYOverAllSpectra, calculateMantidResolution
currentPath = Path(__file__).absolute().parent 

isolatedPath = currentPath / "DHMT_300K_backward_deuteron_.nxs"
wsD = Load(str(isolatedPath), OutputWorkspace='DHMT_300K_backward_deuteron_')

############################################################################
######
######              ISOLATION OF THE DEUTERON PEAK IN TOF
######
############################################################################
verbose = True
simple_gaussian_fit=True          
# masses = np.array([2.015, 12, 14, 27]) 
# CloneWorkspace(InputWorkspace='DHMT_300K_backward_1',OutputWorkspace='DHMT_300K_backward_1_deuteron_')
    
# for m in range(masses.__len__()-1):
#     ws = mtd['DHMT_300K_backward_1_deuteron_']
#     tmp=mtd['DHMT_300K_backward_1_tof_fitted_profile_'+str(m+2)]
#     for spec in range(tmp.getNumberHistograms()):
#         for bin in range(tmp.blocksize()):
#             if (ws.dataY(spec)[bin]!=0):
#                 ws.dataY(spec)[bin] -= tmp.dataY(spec)[bin]


############################################################################
######
######              CONVERSION TO HYDROGEN Y-SPACE
######
############################################################################


# Conversion to hydrogen West-scaling variable
rebin_params='-30,0.5,30'
name = 'DHMT_300K_backward_deuteron_'
ConvertToYSpace(InputWorkspace=name,Mass=2.015,OutputWorkspace=name+'joy',
                            QWorkspace=name+'q')
Rebin(InputWorkspace=name+'joy',Params=rebin_params,OutputWorkspace=name+'joy')
Rebin(InputWorkspace=name+'q',Params=rebin_params,OutputWorkspace=name+'q')  

# Symmetrisation 
# ws=mtd[name+"joy"]
# tmp=CloneWorkspace(InputWorkspace=name+"joy")
# for j in range(tmp.getNumberHistograms()):
#     for k in range(tmp.blocksize()):
#         tmp.dataE(j)[k] =(ws.dataE(j)[k]+ws.dataE(j)[ws.blocksize()-1-k])/2.
#         tmp.dataY(j)[k] =(ws.dataY(j)[k]+ws.dataY(j)[ws.blocksize()-1-k])/2.
# RenameWorkspace(InputWorkspace="tmp",OutputWorkspace=name+"joy_symmetrised")

# Normalisation 
# tmp=Integration(InputWorkspace=name+"joy_symmetrised",RangeLower='-30',RangeUpper='30')
# Divide(LHSWorkspace=name+"joy_symmetrised",RHSWorkspace='tmp',OutputWorkspace=name+"joy_symmetrised")
#  
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


# ws=CloneWorkspace(InputWorkspace=name+'joy_symmetrised')
# for j in range(ws.getNumberHistograms()):
#     for k in range(ws.blocksize()):
#         if  np.isnan(ws.dataY(j)[k]):
#             ws.dataY(j)[k] =0.
#         if  np.isnan(ws.dataE(j)[k]):
#             ws.dataE(j)[k] =0.
# RenameWorkspace('ws',name+'joy_symmetrised')
# 
# 

# SumSpectra(name+"joy_symmetrised",OutputWorkspace=name+"joy_symmetrised_sum")
# tmp_norm = Integration(name+"joy_symmetrised_sum")
# Divide(LHSWorkspace=name+"joy_symmetrised_sum",RHSWorkspace="tmp_norm",OutputWorkspace=name+"joy_symmetrised_sum")
# DeleteWorkspace("tmp_norm")
# 
# 

# Definition of the resolution functions
resolution=CloneWorkspace(InputWorkspace=name+'joy')
resolution=Rebin(InputWorkspace='resolution',Params='-30,0.125,30')
for i in range(resolution.getNumberHistograms()):
    VesuvioResolution(Workspace=name,WorkspaceIndex=str(i), Mass=2.015, OutputWorkspaceYSpace='tmp')
    tmp=Rebin(InputWorkspace='tmp',Params='-30,0.125,30')
    for p in range (tmp.blocksize()):
        resolution.dataY(i)[p]=tmp.dataY(0)[p]
# Definition of the sum of resolution functions
resolution_sum=SumSpectra('resolution')
tmp=Integration('resolution_sum')
resolution_sum=Divide('resolution_sum','tmp')
DeleteWorkspace('tmp')        


############################################################################
######
######              FIT OF THE SUM OF SPECTRA 
######
############################################################################

# print('\n','Fit on the sum of spectra in the West domain','\n')
# for minimizer_sum in ('Simplex','Levenberg-Marquardt'): 
#     if (simple_gaussian_fit):
#         function='''composite=Convolution,FixResolution=true,NumDeriv=true;
#             name=Resolution,Workspace=resolution_sum,WorkspaceIndex=0;
#             name=UserFunction,Formula=A*exp( -(x-x0)^2/2/sigma^2)/(2*3.1415*sigma^2)^0.5,
#             A=1,x0=0,sigma=6,   ties=()'''
#     else:
#         function='''composite=Convolution,FixResolution=true,NumDeriv=true;
#             name=Resolution,Workspace=resolution_sum,WorkspaceIndex=0;
#             name=UserFunction,Formula=A*exp( -(x-x0)^2/2/sigma^2)/(2*3.1415*sigma^2)^0.5
#             *(1+c4/32*(16*((x-x0)/sqrt(2.)/sigma)^4-48*((x-x0)/sqrt(2.)/sigma)^2+12)),
#             A=1,x0=0,sigma=6, c4=0,   ties=()'''
#     Fit(Function=function, InputWorkspace=name+"joy_symmetrised_sum", Output=name+minimizer_sum+'_joy_sum_fitted',
#                                 Minimizer=minimizer_sum)
#     ws=mtd[name+minimizer_sum+'_joy_sum_fitted_Parameters']
#     print('Using the minimizer: ',minimizer_sum)
#     print('Standard deviation: ',ws.cell(2,1),' +/- ',ws.cell(2,2))
#     sigma_to_energy = 1.5 * 2.0445**2 / masses[0] 
#     print("mean kinetic energy: ",sigma_to_energy*ws.cell(2,1)**2," +/- ", 2.*sigma_to_energy*ws.cell(2,1)*ws.cell(2,2), " meV ")

############################################################################
######
######              GLOBAL FIT ON BANKS OF SPECTRA (BOTH CONSECUTIVE AND MIXED DETECTORS)
######
############################################################################

# Definition of a workspace for the global fit with artificial error bars on the unphysical bins
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



# Global fit of the H J(y) with global values of Sigma, and  FSE in the harmonic approximation
sample_ws = mtd[name+'joy_global']
resolution_ws = mtd['resolution']

#####   ----edits------
simple_gaussian_fit = True
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


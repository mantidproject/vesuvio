from mantid.simpleapi import *
from pathlib import Path
import numpy as np
currentPath = Path(__file__).absolute().parent 

isolatedPath = currentPath / "DHMT_300K_backward_deuteron_.nxs"

wsGlobalPath = currentPath / "DHMT_global.nxs"
wsResolutionPath = currentPath / "DHMT_resolution.nxs"
wsQInvPath = currentPath / "DHMT_Q_inverse.nxs"


wsGlobal = Load(str(wsGlobalPath))
wsRes = Load(str(wsResolutionPath), OutputWorkspace="resolution")
wsQInv = Load(str(wsQInvPath), OutputWorkspace="wsQInv")

# Global fit of the H J(y) with global values of Sigma, and  FSE in the harmonic approximation

#####   ----edits------
simple_gaussian_fit = True
verbose = True
name = "someWs"
##### ----------
def globalFitProcedure(wsGlobal, wsQInv, wsRes):
    if (simple_gaussian_fit):
        convolution_template = """
        (composite=Convolution,$domains=({0});
        name=Resolution,Workspace={1},WorkspaceIndex={0};
        (
        name=UserFunction,Formula=
        A*exp( -(x-x0)^2/2/Sigma^2)/(2*3.1415*Sigma^2)^0.5,
        A=1.,x0=0.,Sigma=6.0,  ties=();
        (
                composite=ProductFunction,NumDeriv=false;name=TabulatedFunction,Workspace={2},WorkspaceIndex={0},ties=(Scaling=1,Shift=0,XScaling=1);
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
                composite=ProductFunction,NumDeriv=false;name=TabulatedFunction,Workspace={2},WorkspaceIndex={0},ties=(Scaling=1,Shift=0,XScaling=1);
                name=UserFunction,Formula=
                Sigma*1.4142/12.*exp( -(x)^2/2/Sigma^2)/(2*3.1415*Sigma^2)^0.5
                *((8*((x)/sqrt(2.)/Sigma)^3-12*((x)/sqrt(2.)/Sigma))),
                Sigma=6.0);ties=()
        ))"""    

    print('\n','Global fit in the West domain over 8 mixed banks','\n')
    ######################prova
    #   GLOBAL FIT - MIXED BANKS
    minimizer = "Simplex"
    widths = []  
    for bank in range(8):
        dets=[ bank, bank+8, bank+16, bank+24]
        convolvedFunctionsList = []
        ties = []
        datasets = {}
        counter = 0
        print("Detectors: ", dets)

        for i in dets:

            print(f"Considering spectrum {wsGlobal.getSpectrumNumbers()[i]}")
            if wsGlobal.spectrumInfo().isMasked(i):
                print(f"Skipping masked spectrum {wsGlobal.getSpectrumNumbers()[i]}")
                continue

            thisIterationFunction = convolution_template.format(counter, wsRes.name(), wsQInv.name())
            convolvedFunctionsList.append(thisIterationFunction)

            # Tie widths together for spectra together
            if counter > 0:
                ties.append('f{0}.f1.f0.Sigma= f{0}.f1.f1.f1.Sigma=f0.f1.f0.Sigma'.format(counter))
                # TODO: Ask if I should put a conditional here for the GC fit
                #ties.append('f{0}.f1.f0.c4=f0.f1.f0.c4'.format(counter))
                #ties.append('f{0}.f1.f1.f1.c3=f0.f1.f1.f1.c3'.format(counter))
            
            # Attach datasets
            attr = 'InputWorkspace_{0}'.format(counter) if counter > 0 else 'InputWorkspace'
            datasets[attr] = wsGlobal.name()
            attr = 'WorkspaceIndex_{0}'.format(counter) if counter > 0 else 'WorkspaceIndex'
            datasets[attr] = i

            counter += 1

        # TODO: Ties and dict could be initialized before loop
        multifit_func = 'composite=MultiDomainFunction;' + ';'.join(convolvedFunctionsList)  + ';ties=({0},f0.f1.f1.f1.Sigma=f0.f1.f0.Sigma)'.format(','.join(ties))
        minimizer_string=minimizer+',AbsError=0.00001,RealError=0.00001,MaxIterations=2000'

        # Unpack dictionary as arguments
        Fit(multifit_func, Minimizer=minimizer_string, Output=name+'joy_mixed_banks_bank_'+str(bank)+'_fit', **datasets)
        
        # Select ws with fit results
        ws=mtd[name+'joy_mixed_banks_bank_'+str(bank)+'_fit_Parameters']

        print('bank: ', str(bank), ' -- sigma= ', ws.cell(2,1), ' +/- ', ws.cell(2,2))
        widths.append(ws.cell(2,1))

        # DeleteWorkspace(name+'joy_mixed_banks_bank_'+str(bank)+'_fit_NormalisedCovarianceMatrix')
        # DeleteWorkspace(name+'joy_mixed_banks_bank_'+str(bank)+'_fit_Workspaces') 
    print('Average hydrogen standard deviation: ',np.mean(widths),' +/- ', np.std(widths))
    return widths

def originalProcedure(wsGlobal, wsQInv, wsRes):
    # original global fit
    # Global fit of the H J(y) with global values of Sigma, and  FSE in the harmonic approximation
    sample_ws = wsGlobal
    resolution_ws = wsRes
    RenameWorkspace(wsQInv, "one_over_q")

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
            print("Considering spectrum {0}".format(spec_i.getSpectrumNo()))
            if (verbose):
                if det_i.isMasked():
                    print("Skipping masked spectrum {0}".format(spec_i.getSpectrumNo()))
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
    return w

optWidths = globalFitProcedure(wsGlobal, wsQInv, wsRes)
oriWidths = originalProcedure(wsGlobal, wsQInv, wsRes)
   
np.testing.assert_allclose(optWidths, oriWidths)

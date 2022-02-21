from unicodedata import name
import numpy as np
from mantid.simpleapi import *
from scipy import optimize
from scipy import ndimage
from pathlib import Path
from iminuit import Minuit, cost, util
repoPath = Path(__file__).absolute().parent  # Path to the repository



def fitInYSpaceProcedure(ic, wsFinal, ncpForEachMass):
    firstMass = ic.masses[0]
    wsResSum, wsRes = calculateMantidResolution(ic.resolutionRebinPars, wsFinal, firstMass)
    
    wsSubMass = subtractAllMassesExceptFirst(ic, wsFinal, ncpForEachMass)
    wsYSpace, wsQ = convertToYSpace(ic.rebinParametersForYSpaceFit, wsSubMass, firstMass) 
    wsYSpaceAvg = weightedAvg(wsYSpace)
    
    if ic.symmetrisationFlag:
        wsYSpaceAvg = symmetrizeWs(ic.symmetriseHProfileUsingAveragesFlag, wsYSpaceAvg)

    fitProfileMinuit(ic, wsYSpaceAvg, wsResSum)
    fitProfileMantidFit(ic, wsYSpaceAvg, wsResSum)
    
    yfitResults = ResultsYFitObject(ic, wsFinal.name())
    yfitResults.printYSpaceFitResults()
    yfitResults.save()

    if ic.globalFitFlag:
        fitGlobalFit(wsYSpace, wsQ, wsRes, "Simplex", ic.singleGaussFitToHProfile, wsSubMass.name())


def calculateMantidResolution(rebinResPars, ws, mass):
    #TODO: Resolution function currently skips masked spectra and outputs ws with different size
    # Is this okay for the Global Fit?
    resName = ws.name()+"_Resolution"
    for index in range(ws.getNumberHistograms()):
        if np.all(ws.dataY(index)[:] == 0):  # Ignore masked spectra
            pass
        else:
            VesuvioResolution(Workspace=ws,WorkspaceIndex=index,Mass=mass,OutputWorkspaceYSpace="tmp")
            Rebin(InputWorkspace="tmp", Params=rebinResPars, OutputWorkspace="tmp")

            if index == 0:   # Ensures that workspace has desired units
                RenameWorkspace("tmp",  resName)
            else:
                AppendSpectra(resName, "tmp", OutputWorkspace=resName)
    try:
        wsResSum = SumSpectra(InputWorkspace=resName, OutputWorkspace=ws.name()+"_Resolution_Sum")
    except ValueError:
        raise ValueError ("All the rows from the workspace to be fitted are Nan!")

    normalise_workspace(wsResSum)
    DeleteWorkspace("tmp")
    return wsResSum, mtd[resName]

    
def normalise_workspace(ws_name):
    tmp_norm = Integration(ws_name)
    Divide(LHSWorkspace=ws_name,RHSWorkspace=tmp_norm,OutputWorkspace=ws_name)
    DeleteWorkspace("tmp_norm")


def subtractAllMassesExceptFirst(ic, ws, ncpForEachMass):
    """Input: workspace from last iteration, ncpTotal for each mass
       Output: workspace with all the ncpTotal subtracted except for the first mass"""

    ncpForEachMass = switchFirstTwoAxis(ncpForEachMass)
    # Select all masses other than the first one
    ncpForEachMassExceptFirst = ncpForEachMass[1:, :, :]
    # Sum the ncpTotal for remaining masses
    ncpTotalExceptFirst = np.sum(ncpForEachMassExceptFirst, axis=0)

    wsSubMass = CloneWorkspace(InputWorkspace=ws, OutputWorkspace=ws.name()+"_Mass0")
    for j in range(wsSubMass.getNumberHistograms()):
        if wsSubMass.spectrumInfo().isMasked(j):
            continue

        binWidths = wsSubMass.dataX(j)[1:] - wsSubMass.dataX(j)[:-1]
        wsSubMass.dataY(j)[:-1] -= ncpTotalExceptFirst[j] * binWidths

     # Mask spectra again, to be seen as masked from Mantid's perspective
    MaskDetectors(Workspace=wsSubMass, WorkspaceIndexList=ic.maskedDetectorIdx)  

    if np.any(np.isnan(wsSubMass.extractY())):
        raise ValueError("The workspace for the isolated first mass countains NaNs in non-masked spectra, might cause problems!")
    return wsSubMass


def switchFirstTwoAxis(A):
    """Exchanges the first two indices of an array A,
    rearranges matrices per spectrum for iteration of main fitting procedure
    """
    return np.stack(np.split(A, len(A), axis=0), axis=2)[0]


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


def weightedAvg(wsYSpace):
    dataY = wsYSpace.extractY()
    dataE = wsYSpace.extractE()

    dataY[dataY==0] = np.nan
    dataE[dataE==0] = np.nan

    meanY = np.nansum(dataY/np.square(dataE), axis=0) / np.nansum(1/np.square(dataE), axis=0)
    meanE = np.sqrt(1 / np.nansum(1/np.square(dataE), axis=0))

    tempWs = SumSpectra(wsYSpace)
    newWs = CloneWorkspace(tempWs, OutputWorkspace=wsYSpace.name()+"_Weighted_Avg")
    newWs.dataY(0)[:] = meanY
    newWs.dataE(0)[:] = meanE
    DeleteWorkspace(tempWs)
    return newWs


def symmetrizeWs(avgSymFlag, avgYSpace):
    """Symmetrizes workspace,
       Needs to have symmetric binning"""

    dataX = avgYSpace.extractX()
    dataY = avgYSpace.extractY()
    dataE = avgYSpace.extractE()

    yFlip = np.flip(dataY, axis=1)
    eFlip = np.flip(dataE, axis=1)

    if avgSymFlag:
        # Inverse variance weighting
        dataYSym = (dataY/dataE**2 + yFlip/eFlip**2) / (1/dataE**2 + 1/eFlip**2)
        dataESym = 1 / np.sqrt(1/dataE**2 + 1/eFlip**2)
    else:
        # Mirroring positive values from negative ones
        dataYSym = np.where(dataX>0, yFlip, dataY)
        dataESym = np.where(dataX>0, eFlip, dataE)

    Sym = CloneWorkspace(avgYSpace, OutputWorkspace=avgYSpace.name()+"_Symmetrised")
    for i in range(Sym.getNumberHistograms()):
        Sym.dataY(i)[:] = dataYSym[i]
        Sym.dataE(i)[:] = dataESym[i]
    return Sym


def fitProfileMinuit(ic, wsYSpaceSym, wsRes):
    resY = wsRes.extractY()[0]
    resX = wsRes. extractX()[0]

    # Interpolate Resolution to get single peak at zero
    # Otherwise if the resolution has two data points at the peak, the convolution will be skewed.
    start, interval, end = [float(i) for i in ic.rebinParametersForYSpaceFit.split(",")]
    resNewX = np.arange(start, end, interval)
    resolution = np.interp(resNewX, resX, resY)

    dataY = wsYSpaceSym.extractY()[0]
    dataX = wsYSpaceSym.extractX()[0]
    dataE = wsYSpaceSym.extractE()[0]

    histWidths = dataX[1:] - dataX[:-1]
    assert (np.max(histWidths)==np.min(histWidths)), "dataX spacings in ws need to be all equal for numerical convolution."


    if ic.singleGaussFitToHProfile:
        def convolvedModel(x, y0, A, x0, sigma):
            gauss = y0 + A / (2*np.pi)**0.5 / sigma * np.exp(-(x-x0)**2/2/sigma**2)
            histWidths = x[1] - x[0]     # Assumes all widhts are equal, take first
            return ndimage.convolve1d(gauss, resolution, mode="constant") * histWidths

        # Fit with Minuit
        costFun = cost.LeastSquares(dataX, dataY, dataE, convolvedModel)
        m = Minuit(costFun, y0=0, A=1, x0=0, sigma=5)
        m.limits["A"] = (0, None)
        m.simplex()
        m.migrad()

    else:
        def convolvedModel(x, A, x0, sigma1, c4, c6):
            gramCharlier =  A * np.exp(-(x-x0)**2/2/sigma1**2) / (np.sqrt(2*3.1415*sigma1**2)) \
                    *(1 + c4/32*(16*((x-x0)/np.sqrt(2)/sigma1)**4 \
                    -48*((x-x0)/np.sqrt(2)/sigma1)**2+12) \
                    +c6/384*(64*((x-x0)/np.sqrt(2)/sigma1)**6 \
                    -480*((x-x0)/np.sqrt(2)/sigma1)**4 + 720*((x-x0)/np.sqrt(2)/sigma1)**2 - 120))

            histWidths = x[1] - x[0]     # Assumes all widhts are equal, take first
            return ndimage.convolve1d(gramCharlier, resolution, mode="constant") * histWidths

        # Fit with Minuit
        costFun = cost.LeastSquares(dataX, dataY, dataE, convolvedModel)
        m = Minuit(costFun, A=1, x0=0, sigma1=4, c4=0, c6=0)
        m.limits["A"] = (0, None)
        m.simplex()
        constraints = optimize.NonlinearConstraint(lambda *pars: convolvedModel(dataX, *pars), 0, np.inf)
        m.scipy(constraints=constraints)

    # Explicit calculation of Hessian after the fit
    m.hesse()

    # Weighted Chi2
    chi2 = m.fval / (len(dataX)-m.nfit)

    # Propagate error to yfit
    # Takes in the best fit parameters and their covariance matrix
    # Outputs the best fit curve with std in the diagonal
    dataYFit, dataYCov = util.propagate(lambda pars: convolvedModel(dataX, *pars), m.values, m.covariance)
    dataYSigma = np.sqrt(np.diag(dataYCov))

    # Weight the confidence band
    # TODO: Shoud it be weighted?
    dataYSigma *= chi2

    Residuals = dataY - dataYFit

    # Create workspace to store best fit curve and errors on the fit
    CreateWorkspace(DataX=np.concatenate((dataX, dataX, dataX)), 
                    DataY=np.concatenate((dataY, dataYFit, Residuals)), 
                    DataE=np.concatenate((dataE, dataYSigma, np.zeros(len(dataE)))),
                    NSpec=3,
                    OutputWorkspace=wsYSpaceSym.name()+"_Fitted_Minuit")
    
    # Calculate correlation matrix
    corr = m.covariance.correlation()
    corr *= 100

    # Create correlation tableWorkspace
    tableWS = CreateEmptyTableWorkspace(OutputWorkspace=wsYSpaceSym.name()+"_Fitted_Minuit_NormalizedCovarianceMatrix")
    tableWS.setTitle("Minuit Fit")
    tableWS.addColumn(type='str',name="Name")
    for p in m.parameters:
        tableWS.addColumn(type='float',name=p)
    for p, arr in zip(m.parameters, corr):
        tableWS.addRow([p] + list(arr))
    

    # Create Parameters workspace
    tableWS = CreateEmptyTableWorkspace(OutputWorkspace=wsYSpaceSym.name()+"_Fitted_Minuit_Parameters")
    tableWS.setTitle("Minuit Fit")
    tableWS.addColumn(type='str', name="Name")
    tableWS.addColumn(type='float', name="Value")
    tableWS.addColumn(type='float', name="Error")
    tableWS.addColumn(type='float', name="Minos Error-")
    tableWS.addColumn(type='float', name="Minos Error+")

    try:  # Compute errors from MINOS, fails if constraint forces result away from minimum
        m.minos()
        me = m.merrors
        for p, v, e in zip(m.parameters, m.values, m.errors):
            tableWS.addRow([p, v, e, me[p].lower, me[p].upper])  

    except RuntimeError:
        for p, v, e in zip(m.parameters, m.values, m.errors):
            tableWS.addRow([p, v, e, 0, 0])

    tableWS.addRow(["Cost function", chi2, 0, 0, 0])
    return 


def fitProfileMantidFit(ic, wsYSpaceSym, wsRes):
    print('\n','Fitting on the sum of spectra in the West domain ...','\n')     
    for minimizer in ['Levenberg-Marquardt','Simplex']:
        outputName = wsYSpaceSym.name()+"_Fitted_"+minimizer
        CloneWorkspace(InputWorkspace = wsYSpaceSym, OutputWorkspace = outputName)
        
        if ic.singleGaussFitToHProfile:
            function=f"""composite=Convolution,FixResolution=true,NumDeriv=true;
            name=Resolution,Workspace={wsRes.name()},WorkspaceIndex=0;
            name=UserFunction,Formula=y0+A*exp( -(x-x0)^2/2/sigma^2)/(2*3.1415*sigma^2)^0.5,
            y0=0,A=1,x0=0,sigma=5,   ties=()"""
        else:
            function = f"""
            composite=Convolution,FixResolution=true,NumDeriv=true;
            name=Resolution,Workspace={wsRes.name()},WorkspaceIndex=0,X=(),Y=();
            name=UserFunction,Formula=A*exp( -(x-x0)^2/2./sigma1^2)/(sqrt(2.*3.1415*sigma1^2))
            *(1.+c4/32.*(16.*((x-x0)/sqrt(2)/sigma1)^4-48.*((x-x0)/sqrt(2)/sigma1)^2+12)+c6/384*(64*((x-x0)/sqrt(2)/sigma1)^6 - 480*((x-x0)/sqrt(2)/sigma1)^4 + 720*((x-x0)/sqrt(2)/sigma1)^2 - 120)),
            A=1,x0=0,sigma1=4.0,c4=0.0,c6=0.0,ties=(),constraints=(0<c4,0<c6)
            """

        Fit(
            Function=function, 
            InputWorkspace=outputName,
            Output=outputName,
            Minimizer=minimizer
            )
        # Fit produces output workspaces with results
    return 


class ResultsYFitObject:

    def __init__(self, ic, wsFinalName):
        # Extract most relevant information from ws
        wsFinal = mtd[wsFinalName]
        wsMass0 = mtd[wsFinalName + "_Mass0"]
        if ic.symmetrisationFlag:
            wsJoYAvg = mtd[wsFinalName + "_Mass0_JoY_Weighted_Avg_Symmetrised"]
        else:
            wsJoYAvg = mtd[wsFinalName + "_Mass0_JoY_Weighted_Avg"]
        wsResSum = mtd[wsFinalName + "_Resolution_Sum"]

        self.finalRawDataY = wsFinal.extractY()
        self.finalRawDataE = wsFinal.extractE()
        self.HdataY = wsMass0.extractY()
        self.YSpaceSymSumDataY = wsJoYAvg.extractY()
        self.YSpaceSymSumDataE = wsJoYAvg.extractE()
        self.resolution = wsResSum.extractY()

        # Extract best fit parameters from workspaces
        wsFitLM = mtd[wsJoYAvg.name() + "_Fitted_Levenberg-Marquardt_Parameters"]
        wsFitSimplex = mtd[wsJoYAvg.name() + "_Fitted_Simplex_Parameters"]
        wsFitMinuit = mtd[wsJoYAvg.name() + "_Fitted_Minuit_Parameters"]

        noPars = len(wsFitLM.column("Value"))
        popt = np.zeros((3, noPars))
        perr = np.zeros((3, noPars))
        for i, ws in enumerate([wsFitMinuit, wsFitLM, wsFitSimplex]):
            popt[i] = ws.column("Value")
            perr[i] = ws.column("Error")
        self.popt = popt
        self.perr = perr

        self.savePath = ic.ySpaceFitSavePath
        self.singleGaussFitToHProfile = ic.singleGaussFitToHProfile

    def printYSpaceFitResults(self):
        print("\nFit in Y Space results:")

        if self.singleGaussFitToHProfile:
            for i, fit in enumerate(["Minuit Fit", "Mantid Fit LM", "Mantid Fit Simplex"]):
                print(f"\n{fit:15s}")
                for par, popt, perr in zip(["y0:", "A:", "x0:", "sigma:"], self.popt[i], self.perr[i]):
                    print(f"{par:9s} {popt:8.4f} \u00B1 {perr:6.4f}")
                print(f"Cost function: {self.popt[i, -1]:5.3}")
        else:
            for i, fit in enumerate(["Minuit Fit", "Mantid Fit LM", "Mantid Fit Simplex"]):
                print(f"\n{fit:15s}")
                for par, popt, perr in zip(["A", "x0", "sigma:", "c4:", "c6:"], self.popt[i], self.perr[i]):
                    print(f"{par:9s} {popt:8.4f} \u00B1 {perr:6.4f}")
                print(f"Cost function: {self.popt[i, -1]:5.3}")

    def save(self):
        np.savez(self.savePath,
                 YSpaceSymSumDataY=self.YSpaceSymSumDataY,
                 YSpaceSymSumDataE=self.YSpaceSymSumDataE,
                 resolution=self.resolution, 
                 HdataY=self.HdataY,
                 finalRawDataY=self.finalRawDataY, 
                 finalRawDataE=self.finalRawDataE,
                 popt=self.popt, 
                 perr=self.perr)


# Functions for Global Fit

def fitGlobalFit(wsJoY, wsQ, wsRes, minimizer, gaussFitFlag, wsFirstMassName):
    replaceNansWithZeros(wsJoY)
    wsGlobal = artificialErrorsInUnphysicalBins(wsJoY)
    wsQInv = createOneOverQWs(wsQ)

    avgWidths = globalFitProcedure(wsGlobal, wsQInv, wsRes, minimizer, gaussFitFlag, wsFirstMassName)


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


def globalFitProcedure(wsGlobal, wsQInv, wsRes, minimizer, gaussFitFlag, wsFirstMassName):
    if gaussFitFlag:
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
                )
            )"""
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
                )
            )"""    

    print('\nGlobal fit in the West domain over 8 mixed banks\n')
    widths = []  
    for bank in range(8):
        dets=[bank, bank+8, bank+16, bank+24]

        convolvedFunctionsList = []
        ties = ["f0.f1.f1.f1.Sigma=f0.f1.f0.Sigma"]
        datasets = {'InputWorkspace' : wsGlobal.name(),
                    'WorkspaceIndex' : dets[0]}

        print("Detectors: ", dets)

        counter = 0
        for i in dets:

            print(f"Considering spectrum {wsGlobal.getSpectrumNumbers()[i]}")
            if wsGlobal.spectrumInfo().isMasked(i):
                print(f"Skipping masked spectrum {wsGlobal.getSpectrumNumbers()[i]}")
                continue

            thisIterationFunction = convolution_template.format(counter, wsRes.name(), wsQInv.name())
            convolvedFunctionsList.append(thisIterationFunction)

            if counter > 0:
                ties.append('f{0}.f1.f0.Sigma= f{0}.f1.f1.f1.Sigma=f0.f1.f0.Sigma'.format(counter))
                #TODO: Ask if conditional statement goes here
                #ties.append('f{0}.f1.f0.c4=f0.f1.f0.c4'.format(counter))
                #ties.append('f{0}.f1.f1.f1.c3=f0.f1.f1.f1.c3'.format(counter))

                # Attach datasets
                datasets[f"InputWorkspace_{counter}"] = wsGlobal.name()
                datasets[f"WorkspaceIndex_{counter}"] = i
            counter += 1

        multifit_func = f"composite=MultiDomainFunction; {';'.join(convolvedFunctionsList)}; ties=({','.join(ties)})"
        minimizer_string = f"{minimizer}, AbsError=0.00001, RealError=0.00001, MaxIterations=2000"

        # Unpack dictionary as arguments
        Fit(multifit_func, Minimizer=minimizer_string, Output=wsFirstMassName+f'Joy_Mixed_Banks_Bank_{str(bank)}_fit', **datasets)
        
        # Select ws with fit results
        ws=mtd[wsFirstMassName+f'Joy_Mixed_Banks_Bank_{str(bank)}_fit_Parameters']
        print(f"Bank: {str(bank)} -- sigma={ws.cell(2,1)} +/- {ws.cell(2,2)}")
        widths.append(ws.cell(2,1))

        # DeleteWorkspace(name+'joy_mixed_banks_bank_'+str(bank)+'_fit_NormalisedCovarianceMatrix')
        # DeleteWorkspace(name+'joy_mixed_banks_bank_'+str(bank)+'_fit_Workspaces') 
    print('\nAverage hydrogen standard deviation: ',np.mean(widths),' +/- ', np.std(widths))
    return widths

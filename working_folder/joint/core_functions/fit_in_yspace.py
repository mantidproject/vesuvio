import matplotlib.pyplot as plt
import numpy as np
from mantid.simpleapi import *
from scipy import optimize
from scipy import ndimage, signal
from pathlib import Path
from iminuit import Minuit, cost, util
from iminuit.util import make_func_code, describe
import time

repoPath = Path(__file__).absolute().parent  # Path to the repository



def fitInYSpaceProcedure(ic, wsFinal, ncpForEachMass):
    firstMass = ic.masses[0]
    wsResSum, wsRes = calculateMantidResolution(ic, wsFinal, firstMass)
    
    wsSubMass = subtractAllMassesExceptFirst(ic, wsFinal, ncpForEachMass)
    wsYSpace, wsQ = convertToYSpace(ic.rebinParametersForYSpaceFit, wsSubMass, firstMass) 
    wsYSpaceAvg = weightedAvg(wsYSpace)
    
    if ic.symmetrisationFlag:
        wsYSpaceAvg = symmetrizeWs(ic.symmetriseHProfileUsingAveragesFlag, wsYSpaceAvg)

    fitProfileMinuit(ic, wsYSpaceAvg, wsResSum)
    fitProfileMantidFit(ic, wsYSpaceAvg, wsResSum)
    
    printYSpaceFitResults(wsYSpaceAvg.name())

    yfitResults = ResultsYFitObject(ic, wsFinal.name())
    yfitResults.save()

    if ic.globalFitFlag:
        # fitGlobalFit(wsYSpace, wsQ, wsRes, "Simplex", ic.singleGaussFitToHProfile, wsSubMass.name())
        fitMinuitGlobalFit(wsYSpace, wsRes, ic)


def calculateMantidResolution(ic, ws, mass):
    resName = ws.name()+"_Resolution"
    for index in range(ws.getNumberHistograms()):
        VesuvioResolution(Workspace=ws,WorkspaceIndex=index,Mass=mass,OutputWorkspaceYSpace="tmp")
        Rebin(InputWorkspace="tmp", Params=ic.rebinParametersForYSpaceFit, OutputWorkspace="tmp")

        if index == 0:   # Ensures that workspace has desired units
            RenameWorkspace("tmp",  resName)
        else:
            AppendSpectra(resName, "tmp", OutputWorkspace=resName)
   
    MaskDetectors(resName, WorkspaceIndexList=ic.maskedDetectorIdx)
    wsResSum = SumSpectra(InputWorkspace=resName, OutputWorkspace=resName+"_Sum")
 
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
    """Returns ws with weighted avg of input ws"""
    
    dataY = wsYSpace.extractY()
    dataE = wsYSpace.extractE()

    # TODO: Revise this, some zeros might not be cut offs
    # TODO: If one column is all zeros ir puts dataY=nan and dataE=inf, will throw an error when fitting
    # Replace dataE at cut-offs by np.inf?

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

    # TODO: Maybe take out possibility of symmetrisation by mirror
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
    dataY = wsYSpaceSym.extractY()[0]
    dataX = wsYSpaceSym.extractX()[0]
    dataE = wsYSpaceSym.extractE()[0]

    resY = wsRes.extractY()[0]
    resX = wsRes. extractX()[0]

    if ic.singleGaussFitToHProfile:
        def model(x, y0, A, x0, sigma):
            return y0 + A / (2*np.pi)**0.5 / sigma * np.exp(-(x-x0)**2/2/sigma**2)

        funcSig = ["x", "y0", "A", "x0", "sigma"]
        initPars = {"y0":0, "A":1, "x0":0, "sigma":5}

    else:
        def model(x, A, x0, sigma1, c4, c6):
            return  A * np.exp(-(x-x0)**2/2/sigma1**2) / (np.sqrt(2*3.1415*sigma1**2)) \
                    *(1 + c4/32*(16*((x-x0)/np.sqrt(2)/sigma1)**4 \
                    -48*((x-x0)/np.sqrt(2)/sigma1)**2+12) \
                    +c6/384*(64*((x-x0)/np.sqrt(2)/sigma1)**6 \
                    -480*((x-x0)/np.sqrt(2)/sigma1)**4 + 720*((x-x0)/np.sqrt(2)/sigma1)**2 - 120))
        
        funcSig = ["x", "A", "x0", "sigma1", "c4", "c6"]
        initPars = {"A":1, "x0":0, "sigma1":4, "c4":0, "c6":0}

    xDense, xDelta, resDense = chooseXDense(resX, resY, True)

    def convolvedModel(x, *pars):
        convDense = signal.convolve(model(xDense, *pars), resDense, mode="same") * xDelta
        return np.interp(x, xDense, convDense)

    convolvedModel.func_code = make_func_code(funcSig)

    # Ignore values that are zero, eg. cut-offs
    nonZeros = dataY != 0
    dataXNZ = dataX[nonZeros]
    dataYNZ = dataY[nonZeros]
    dataENZ = dataE[nonZeros]

    # Fit with Minuit
    costFun = cost.LeastSquares(dataXNZ, dataYNZ, dataENZ, convolvedModel)
    m = Minuit(costFun, **initPars)
    m.limits["A"] = (0, None)

    m.simplex()
    if ic.singleGaussFitToHProfile:
        m.migrad()
    else:
        def constrFunc(*pars):
            return model(dataX, *pars)   # GC > 0 before convolution, i.e. physical system

        m.scipy(constraints=optimize.NonlinearConstraint(constrFunc, 0, np.inf))

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
    tableWS.addColumn(type='float', name="Auto Minos Error-")
    tableWS.addColumn(type='float', name="Auto Minos Error+")
    tableWS.addColumn(type='float', name="Manual Minos Error-")
    tableWS.addColumn(type='float', name="Manual Minos Error+")


    # Extract info from fit before running any MINOS
    parameters = list(m.parameters)
    values = list(m.values)
    errors = list(m.errors)
    
    bestFitVals = {}
    bestFitErrs = {}
    for p, v, e in zip(m.parameters, m.values, m.errors):
        bestFitVals[p] = v
        bestFitErrs[p] = e
    fValsMin = m.fval

    try:  # Compute errors from MINOS, fails if constraint forces result away from minimum
        if ic.forceManualMinos:
            try:
                constrFunc(*m.values)      # Check if constraint is present
                raise(RuntimeError)        # If so, jump to Manual MINOS

            except UnboundLocalError:      # Constraint not present, default to auto MINOS
                print("\nConstraint not present, using default Automatic MINOS ...\n")
                pass
        
        m.minos()
        me = m.merrors
        for p, v, e in zip(parameters, values, errors):
            tableWS.addRow([p, v, e, me[p].lower, me[p].upper, 0, 0])   
        plotAutoMinos(m)

    except RuntimeError:
        merrors = runAndPlotManualMinos(m, constrFunc, bestFitVals, bestFitErrs)     # Changes values of minuit obj m, do not use m below this point
        for p, v, e in zip(parameters, values, errors):
            tableWS.addRow([p, v, e, 0, 0, merrors[p][0], merrors[p][1]])

    tableWS.addRow(["Cost function", chi2, 0, 0, 0, 0, 0])
    return 


def chooseXDense(x, res, flag):
    # TODO: Need to sort out the best density for the convolution

    """Make high density symmetric grid for convolution"""

    assert np.min(x) == -np.max(x), "Resolution needs to be in symetric range!"
    assert x.size == res.size, "x and res need to be the same size!"

    if flag:
        if x.size % 2 == 0:
            dens = x.size+1  # If even change to odd
        else:
            dens = x.size    # If odd, keep being odd)
    else:
        dens = 1000

    xDense = np.linspace(np.min(x), np.max(x), dens)
    xDelta = xDense[1] - xDense[0]
    resDense = np.interp(xDense, x, res)
    return xDense, xDelta, resDense



def fitProfileMantidFit(ic, wsYSpaceSym, wsRes):
    print('\nFitting on the sum of spectra in the West domain ...\n')     
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


def runAndPlotManualMinos(minuitObj, constrFunc, bestFitVals, bestFitErrs):
    # Set format of subplots
    height = 2
    width = int(np.ceil(len(minuitObj.parameters)/2))
    figsize = (12, 7)
    # Output plot to Mantid
    fig, axs = plt.subplots(height, width, tight_layout=True, figsize=figsize, subplot_kw={'projection':'mantid'})  #subplot_kw={'projection':'mantid'}
    fig.canvas.set_window_title("Manual Implementation of Minos algorithm")

    merrors = {}
    for p, ax in zip(minuitObj.parameters, axs.flat):
        lerr, uerr = runMinosForPar(minuitObj, constrFunc, p, 2, ax, bestFitVals, bestFitErrs)
        merrors[p] = np.array([lerr, uerr])

    # Hide plots not in use:
    for ax in axs.flat:
        if not ax.lines:   # If empty list
            ax.set_visible(False)

    # ALl axes share same legend, so set figure legend to first axis
    handle, label = axs[0, 0].get_legend_handles_labels()
    fig.legend(handle, label, loc='lower right')
    fig.show()
    return merrors


def runMinosForPar(minuitObj, constrFunc, var:str, bound:int, ax, bestFitVals, bestFitErrs):

    # Set parameters to previously found minimum to restart procedure
    for p in bestFitVals:
        minuitObj.values[p] = bestFitVals[p]
        minuitObj.errors[p] = bestFitErrs[p]

    # Run Fitting procedures again to be on the safe side and reset to minimum
    minuitObj.scipy(constraints=optimize.NonlinearConstraint(constrFunc, 0, np.inf))
    minuitObj.hesse()

    # Extract parameters from minimum
    varVal = minuitObj.values[var]
    varErr = minuitObj.errors[var]
    # Store fval of best fit
    fValsMin = minuitObj.fval      # Used to calculate error bands at the end

    # Create variable space more dense near the minima
    limit = (bound*varErr)**(1/2)
    varSpace = np.linspace(-limit, limit, 30)
    varSpace = varSpace**2 * np.sign(varSpace) + varVal
    # varSpace = np.linspace(varVal-bound*varErr, varVal+bound*varErr, 30)

    fValsScipy = np.zeros(varSpace.shape)
    fValsMigrad = np.zeros(varSpace.shape)

    # Run Minos algorithm
    minuitObj.fixed[var] = True        # Variable is fixed at each iteration

    # Split variable space in two parts to start loop from minimum
    lhsRange, rhsRange = np.split(np.arange(varSpace.size), 2)
    betterRange = [rhsRange, np.flip(lhsRange)]  # First do rhs, then lhs, starting from minima
    for side in betterRange:
        # Reset values and errors to minima
        for p in bestFitVals:
            minuitObj.values[p] = bestFitVals[p]
            minuitObj.errors[p] = bestFitErrs[p]

        # Unconstrained fit
        for i in side.astype(int):
            minuitObj.values[var] = varSpace[i]      # Fix variable
            minuitObj.migrad()     
            fValsMigrad[i] = minuitObj.fval

        # Reset values and errors to minima
        for p in bestFitVals:
            minuitObj.values[p] = bestFitVals[p]
            minuitObj.errors[p] = bestFitErrs[p]

        # Constrained fit       
        for i in side.astype(int):
            minuitObj.values[var] = varSpace[i]      # Fix variable
            minuitObj.scipy(constraints=optimize.NonlinearConstraint(constrFunc, 0, np.inf))
            fValsScipy[i] = minuitObj.fval
        
    minuitObj.fixed[var] = False    # Release variable       

    # Use intenpolation to create dense array of fmin values 
    varSpaceDense = np.linspace(np.min(varSpace), np.max(varSpace), 100000)
    fValsScipyDense = np.interp(varSpaceDense, varSpace, fValsScipy)
    # Calculate points of intersection with line delta fmin val = 1
    idxErr = np.argwhere(np.diff(np.sign(fValsScipyDense - fValsMin - 1)))
    
    if idxErr.size != 2:    # Intersections not found, do not plot error range
        lerr, uerr = 0., 0.   
    else:
        lerr, uerr = varSpaceDense[idxErr].flatten() - varVal

    ax.plot(varSpaceDense, fValsScipyDense, label="fVals Constr Scipy")
    plotProfile(ax, var, varSpace, fValsMigrad, lerr, uerr, fValsMin, varVal, varErr)
  
    return lerr, uerr


def plotAutoMinos(minuitObj):
    # Set format of subplots
    height = 2
    width = int(np.ceil(len(minuitObj.parameters)/2))
    figsize = (12, 7)
    # Output plot to Mantid
    fig, axs = plt.subplots(height, width, tight_layout=True, figsize=figsize, subplot_kw={'projection':'mantid'})  #subplot_kw={'projection':'mantid'}
    fig.canvas.set_window_title("Plot of automatic Minos algorithm")

    for p, ax in zip(minuitObj.parameters, axs.flat):
        loc, fvals, status = minuitObj.mnprofile(p, bound=2)
        

        minfval = minuitObj.fval
        minp = minuitObj.values[p]
        hessp = minuitObj.errors[p]
        lerr = minuitObj.merrors[p].lower
        uerr = minuitObj.merrors[p].upper
        plotProfile(ax, p, loc, fvals, lerr, uerr, minfval, minp, hessp)

    # Hide plots not in use:
    for ax in axs.flat:
        if not ax.lines:   # If empty list
            ax.set_visible(False)

    # ALl axes share same legend, so set figure legend to first axis
    handle, label = axs[0, 0].get_legend_handles_labels()
    fig.legend(handle, label, loc='lower right')
    fig.show()   


def plotProfile(ax, var, varSpace, fValsMigrad, lerr, uerr, fValsMin, varVal, varErr):
    """
    Plots likelihood profilef for the Migrad fvals.
    varSpace : x axis
    fValsMigrad : y axis
    """

    ax.set_title(var+f" = {varVal:.3f} {lerr:.3f} {uerr:+.3f}")

    ax.plot(varSpace, fValsMigrad, label="fVals Migrad")

    ax.axvspan(lerr+varVal, uerr+varVal, alpha=0.2, color="red", label="Minos error")
    ax.axvspan(varVal-varErr, varVal+varErr, alpha=0.2, color="grey", label="Hessian Std error")
    
    ax.axvline(varVal, 0.03, 0.97, color="k", ls="--")
    ax.axhline(fValsMin+1, 0.03, 0.97, color="k")
    ax.axhline(fValsMin, 0.03, 0.97, color="k")


def printYSpaceFitResults(wsJoYName):
    print("\nFit in Y Space results:")

    wsFitLM = mtd[wsJoYName + "_Fitted_Levenberg-Marquardt_Parameters"]
    wsFitSimplex = mtd[wsJoYName + "_Fitted_Simplex_Parameters"]
    wsFitMinuit = mtd[wsJoYName + "_Fitted_Minuit_Parameters"]

    for tableWS in [wsFitLM, wsFitSimplex, wsFitMinuit]:
        print("\n"+" ".join(tableWS.getName().split("_")[-3:])+":")
        # print("    ".join(tableWS.keys()))
        for key in tableWS.keys():
            if key=="Name":
                print(f"{key:12s}:  "+"  ".join([f"{elem:7.8s}" for elem in tableWS.column(key)]))
            else:
                print(f"{key:12s}: "+"  ".join([f"{elem:7.4f}" for elem in tableWS.column(key)]))
    print("\n")


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


# Functions for Global Fit Mantid

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


# ------ Global Fit Minuit Procedure

def fitMinuitGlobalFit(ws, wsRes, ic):

    dataX, dataY, dataE, dataRes, instrPars = extractData(ws, wsRes, ic)   
    dataX, dataY, dataE, dataRes, instrPars = takeOutMaskedSpectra(dataX, dataY, dataE, dataRes, instrPars)

    idxList = groupDetectors(instrPars, ic.nGlobalFitGroups)
    dataX, dataY, dataE, dataRes = avgWeightDetGroups(dataX, dataY, dataE, dataRes, idxList)

    model, defaultPars, sharedPars = selectModelAndPars(ic.singleGaussFitToHProfile)   
 
    totCost = 0
    for i, (x, y, yerr, res) in enumerate(zip(dataX, dataY, dataE, dataRes)):
        totCost += calcCostFun(model, i, x, y, yerr, res, sharedPars)
    
    print("\nGlobal Fit Parameters:\n", describe(totCost))
    print("\nRunning Global Fit ...\n")

    initPars = minuitInitialParameters(defaultPars, sharedPars, len(dataY))

    m = Minuit(totCost, **initPars)

    for i in range(len(dataY)):     # Limit for both Gauss and Gram Charlier
        m.limits["A"+str(i)] = (0, np.inf)

    t0 = time.time()
    if ic.singleGaussFitToHProfile:

        #TODO Ask about the limits on y0
        for i in range(len(dataY)):
            m.limits["y0"+str(i)] = (0, np.inf)

        m.simplex()
        m.migrad() 

    else:

        x = dataX[0]
        def constr(*pars):
            """Constraint for positivity of Gram Carlier.
            Input: All parameters defined in original function.
            Format *pars to work with Minuit.
            x is defined outside function.
            Builds array with all constraints from individual functions."""

            sharedPars = pars[:3]    # sigma1, c4, c6
            joinedGC = np.zeros(int((len(pars)-3)/2) * x.size)
            for i, (A, x0) in enumerate(zip(pars[3::2], pars[4::2])):
                joinedGC[i*x.size : (i+1)*x.size] = model(x, *sharedPars, A, x0)
            
            assert np.all(joinedGC!=0), f"Args where zero: {np.argwhere(joinedGC==0)}"
     
            return joinedGC

        m.simplex()
        m.scipy(constraints=optimize.NonlinearConstraint(constr, 0, np.inf))
    
    # Explicitly calculate errors
    m.hesse()
    t1 = time.time()
    print(f"\nTime of fitting: {t1-t0:.2f} seconds")
    print(f"Value of minimum: {m.fval:.2f}")

    print("\nResults of Global Fit:\n")
    for p, v, e in zip(m.parameters, m.values, m.errors):
        print(f"{p:7s} = {v:7.3f} +/- {e:7.3f}")
    print("\n")

    plotGlobalFit(dataX, dataY, dataE, m, totCost)
    
    return np.array(m.values), np.array(m.errors)     # Pass into array to store values in variable


def extractData(ws, wsRes, ic):
    dataY = ws.extractY()
    dataE = ws.extractE()
    dataX = ws.extractX()
    dataRes = wsRes.extractY()
    instrPars = loadInstrParsFileIntoArray(ic)
    assert len(instrPars) == len(dataY), "Load of IP file not working correctly, probable issue with indexing."
    return dataX, dataY, dataE, dataRes, instrPars    


def loadInstrParsFileIntoArray(ic):
    data = np.loadtxt(ic.InstrParsPath, dtype=str)[1:].astype(float)
    spectra = data[:, 0]
    select_rows = np.where((spectra >= ic.firstSpec) & (spectra <= ic.lastSpec))
    instrPars = data[select_rows]
    return instrPars


def takeOutMaskedSpectra(dataX, dataY, dataE, dataRes, instrPars):
    zerosRowMask = np.all(dataY==0, axis=1)
    dataY = dataY[~zerosRowMask]
    dataE = dataE[~zerosRowMask]
    dataX = dataX[~zerosRowMask]
    dataRes = dataRes[~zerosRowMask]
    instrPars = instrPars[~zerosRowMask]
    return dataX, dataY, dataE, dataRes, instrPars 


def selectModelAndPars(gaussFlag):
    if gaussFlag:
        def model(x, sigma, y0, A, x0):
            gauss = y0 + A / (2*np.pi)**0.5 / sigma * np.exp(-(x-x0)**2/2/sigma**2)
            return gauss 

        defaultPars = {
            "sigma" : 5,
            "y0" : 0,
            "A" : 1,
            "x0" : 0,         
        }

        sharedPars = ["sigma"]

    else:
        def model(x, sigma1, c4, c6, A, x0):
            return A * np.exp(-(x-x0)**2/2/sigma1**2) / (np.sqrt(2*3.1415*sigma1**2)) \
                    *(1 + c4/32*(16*((x-x0)/np.sqrt(2)/sigma1)**4 \
                    -48*((x-x0)/np.sqrt(2)/sigma1)**2+12) \
                    +c6/384*(64*((x-x0)/np.sqrt(2)/sigma1)**6 \
                    -480*((x-x0)/np.sqrt(2)/sigma1)**4 + 720*((x-x0)/np.sqrt(2)/sigma1)**2 - 120)) 
        
        defaultPars = {
            "sigma1" : 6,
            "c4" : 0,
            "c6" : 0,
            "A" : 1,
            "x0" : 0          
        }

        sharedPars = ["sigma1", "c4", "c6"]  

    assert all(isinstance(item, str) for item in sharedPars), "Parameters in list must be strings."

    return model, defaultPars, sharedPars


def minuitInitialParameters(defaultPars, sharedPars, nSpec):
    initPars = {}
    # Populate with initial shared parameters
    for sp in sharedPars:
        initPars[sp] = defaultPars[sp]
    # Add initial unshared parameters
    unsharedPars = [key for key in defaultPars if key not in sharedPars]
    for up in unsharedPars:
        for i in range(nSpec):
            initPars[up+str(i)] = defaultPars[up]
    return initPars


def calcCostFun(model, i, x, y, yerr, res, sharedPars):
    "Returns cost function for one spectrum i to be summed to total cost function"
   
    xDense, xDelta, resDense = chooseXDense(x, res, True)
    def convolvedModel(xrange, *pars):
        """Performs convolution first on high density grid and interpolates to desired x range"""
        convDense = signal.convolve(model(xDense, *pars), resDense, mode="same") * xDelta
        return np.interp(xrange, xDense, convDense)

    costSig = [key if key in sharedPars else key+str(i) for key in describe(model)]
    convolvedModel.func_code = make_func_code(costSig)
    # print(describe(convolvedModel))

    # Data without cut-offs
    nonZeros = y != 0
    xNZ = x[nonZeros]
    yNZ = y[nonZeros]
    yerrNZ = yerr[nonZeros]

    costFun = cost.LeastSquares(xNZ, yNZ, yerrNZ, convolvedModel)
    return costFun


def chooseXDense(x, res, flag):
    """Make high density symmetric grid for convolution"""

    assert np.min(x) == -np.max(x), "Resolution needs to be in symetric range!"
    assert x.size == res.size, "x and res need to be the same size!"

    if flag:
        if x.size % 2 == 0:
            dens = x.size+1  # If even change to odd
        else:
            dens = x.size    # If odd, keep being odd)
    else:
        dens = 1000

    xDense = np.linspace(np.min(x), np.max(x), dens)
    xDelta = xDense[1] - xDense[0]
    resDense = np.interp(xDense, x, res)
    return xDense, xDelta, resDense


def plotGlobalFit(dataX, dataY, dataE, mObj, totCost):
    rows = 2

    fig, axs = plt.subplots(
        rows, 
        int(np.ceil(len(dataY)/rows)),
        figsize=(15, 8), 
        tight_layout=True,
        subplot_kw={'projection':'mantid'}
    )

    # Data used in Global Fit
    for x, y, yerr, ax in zip(dataX, dataY, dataE, axs.flat):
        ax.errorbar(x, y, yerr, fmt="k.", label="Data") 

    # Global Fit 
    for x, costFun, ax in zip(dataX, totCost, axs.flat):
        signature = describe(costFun)

        values = mObj.values[signature]
        errors = mObj.errors[signature]

        yfit = costFun.model(x, *values)

        # Build a decent legend
        leg = []
        for p, v, e in zip(signature, values, errors):
            leg.append(f"${p} = {v:.3f} \pm {e:.3f}$")

        ax.fill_between(x, yfit, label="\n".join(leg), alpha=0.4)
        ax.legend()


# ------- Groupings 

def groupDetectors(ipData, nGroups):
    """
    Uses the method of k-means to find clusters in theta-L1 space.
    Input: instrument parameters to extract L1 and theta of detectors.
    Output: list of group lists containing the idx of spectra.
    """
    assert nGroups > 0, "Number of groups must be bigger than zero."
    assert nGroups < len(ipData), "Number of groups cannot exceed no of detectors"
    print(f"\nNumber of gropus: {nGroups}")

    L1 = ipData[:, -1]    
    theta = ipData[:, 2]  

    # Normalize  ranges to similar values
    L1 /= np.sum(L1)       
    theta /= np.sum(theta)

    L1 *= 2           # Bigger weight to L1


    points = np.vstack((L1, theta)).T
    assert points.shape == (len(L1), 2), "Wrong shape."
    centers = points[np.linspace(0, len(points)-1, nGroups).astype(int), :]

    if True:
        plotDetsAndInitialCenters(L1, theta, centers)

    clusters, n = kMeansClustering(points, centers)
    idxList = formIdxList(clusters, n, len(L1))

    plotFinalGroups(points, clusters, n)

    return idxList


def plotDetsAndInitialCenters(L1, theta, centers):
    fig, ax = plt.subplots(tight_layout=True, subplot_kw={'projection':'mantid'})  
    ax.scatter(L1, theta, alpha=0.3, color="r", label="Detectors")
    ax.scatter(centers[:, 0], centers[:, 1], color="k", label="Starting centroids")
    ax.set_xlabel("L1")
    ax.set_ylabel("theta")
    ax.legend()

def plotFinalGroups(points, clusters, nGroups):
    fig, ax = plt.subplots(tight_layout=True, subplot_kw={'projection':'mantid'})  
    for i in range(nGroups):
        clus = points[clusters==i]
        ax.scatter(clus[:, 0], clus[:, 1], label=f"group {i}")
    ax.set_xlabel("L1")
    ax.set_ylabel("theta")
    ax.legend()


def kMeansClustering(points, centers):
    # Fails in some rare situations
    prevCenters = centers
    while  True:
        clusters, nGroups = closestCenter(points, prevCenters)
        centers = calculateCenters(points, clusters, nGroups)
        # print(centers)

        if np.all(centers == prevCenters):
            break

        assert np.isfinite(centers).all(), f"Invalid centers found:\n{centers}\nMaybe try a different number for the groupings."

        prevCenters = centers
    clusters, n = closestCenter(points, centers)
    return clusters, n


def closestCenter(points, centers):
    clusters = np.zeros(len(points))
    for p in range(len(points)):

        minCenter = 0
        minDist = pairDistance(points[p], centers[0])
        for i in range(1, len(centers)): 

            dist = pairDistance(points[p], centers[i])

            if dist < minDist:
                minDist = dist
                minCenter = i
        clusters[p] = minCenter
    return clusters, len(centers)


def pairDistance(p1, p2):
    "pairs have shape (1, 2)"
    return np.sqrt(np.sum(np.square(p1-p2)))


def calculateCenters(points, clusters, n):
    centers = np.zeros((n, 2))
    for i in range(n):
        centers[i] = np.mean(points[clusters==i, :], axis=0)  # If cluster i is not present, returns nan
    return centers


def formIdxList(clusters, n, lenPoints):
    # Form list with groups of idxs
    idxList = []
    for i in range(n):
        idxs = np.argwhere(clusters==i).flatten()
        idxList.append(list(idxs))
    print("\nList of idexes that will be used for idexing: \n", idxList)

    # Check that idexes are not repeated and not missing
    flatList = []
    for group in idxList:
        for elem in group:
            flatList.append(elem)
    assert np.all(np.sort(np.array(flatList))==np.arange(lenPoints)), "Groupings did not work!"
    
    return idxList

# ---------- Weighted Avgs of Groups

def avgWeightDetGroups(dataX, dataY, dataE, dataRes, idxList):
    wDataX, wDataY, wDataE, wDataRes = initiateZeroArr((len(idxList), len(dataY[0])))

    for i, idxs in enumerate(idxList):
        groupX, groupY, groupE, groupRes = extractArrByIdx(dataX, dataY, dataE, dataRes, idxs)
        
        if len(groupY) == 1:   # Cannot use weight avg in single spec, wrong results
            meanY, meanE = groupY, groupE
            meanRes = groupRes

        else:
            meanY, meanE = weightedAvgArr(groupY, groupE)
            meanRes = np.nanmean(groupRes, axis=0)

        assert np.all(groupX[0] == np.mean(groupX, axis=0)), "X values should not change with groups"
        
        wDataX[i] = groupX[0]
        wDataY[i] = meanY
        wDataE[i] = meanE
        wDataRes[i] = meanRes 
    
    assert ~np.any(np.all(wDataY==0, axis=1)), f"Some avg weights in groups are not being performed:\n{np.argwhere(np.all(wDataY==0, axis=1))}"

    return wDataX, wDataY, wDataE, wDataRes


def initiateZeroArr(shape):
    wDataX = np.zeros(shape)
    wDataY = np.zeros(shape)
    wDataE = np.zeros(shape)
    wDataRes = np.zeros(shape)  
    return  wDataX, wDataY, wDataE, wDataRes


def extractArrByIdx(dataX, dataY, dataE, dataRes, idxs):
    groupE = dataE[idxs, :]
    groupY = dataY[idxs, :]
    groupX = dataX[idxs, :]
    groupRes = dataRes[idxs, :]
    return groupX, groupY, groupE, groupRes


def weightedAvgArr(dataY, dataE):
    meanY = np.nansum(dataY/np.square(dataE), axis=0) / np.nansum(1/np.square(dataE), axis=0)
    meanE = np.sqrt(1 / np.nansum(1/np.square(dataE), axis=0))
    return meanY, meanE
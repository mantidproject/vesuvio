import numpy as np
from mantid.simpleapi import *
from scipy import optimize
from scipy import ndimage
from pathlib import Path
repoPath = Path(__file__).absolute().parent  # Path to the repository



class ResultsYFitObject:

    def __init__(self, ic, wsFinal, wsH, wsYSpaceSymSum, wsRes, popt, perr):
        self.finalRawDataY = wsFinal.extractY()
        self.finalRawDataE = wsFinal.extractE()
        self.HdataY = wsH.extractY()
        self.YSpaceSymSumDataY = wsYSpaceSymSum.extractY()
        self.YSpaceSymSumDataE = wsYSpaceSymSum.extractE()
        self.resolution = wsRes.extractY()
        self.popt = popt
        self.perr = perr

        self.savePath = ic.ySpaceFitSavePath
        self.singleGaussFitToHProfile = ic.singleGaussFitToHProfile

    def printYSpaceFitResults(self):
        print("\nFit in Y Space results:")

        if self.singleGaussFitToHProfile:
            for i, fit in enumerate(["Curve Fit", "Mantid Fit LM", "Mantid Fit Simplex"]):
                print(f"\n{fit:15s}")
                for par, popt, perr in zip(["y0:", "A:", "x0:", "sigma:"], self.popt[i], self.perr[i]):
                    print(f"{par:9s} {popt:8.4f} \u00B1 {perr:6.4f}")
                print(f"Cost function: {self.popt[i, -1]:5.3}")
        else:
            for i, fit in enumerate(["Curve Fit", "Mantid Fit LM", "Mantid Fit Simplex"]):
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



def fitInYSpaceProcedure(ic, wsFinal, ncpForEachMass):
    # ncpForEachMass = fittingResults.all_ncp_for_each_mass[-1]  # Select last iteration
    wsYSpaceSymSum, wsRes = isolateFirstMassProfileInYSpace(ic, wsFinal, ncpForEachMass)
    popt, perr = fitFirstMassProfileInYSpace(ic, wsYSpaceSymSum, wsRes)
    wsH = mtd[wsFinal.name()+"_H"]

    yfitResults = ResultsYFitObject(ic, wsFinal, wsH, wsYSpaceSymSum, wsRes, popt, perr)
    yfitResults.printYSpaceFitResults()
    yfitResults.save()

def isolateFirstMassProfileInYSpace(ic, wsFinal, ncpForEachMass):

    firstMass = ic.masses[0]
    wsRes = calculateMantidResolution(ic, wsFinal, firstMass)  

    wsSubMass = subtractAllMassesExceptFirst(ic, wsFinal, ncpForEachMass)
    averagedSpectraYSpace = averageJOfYOverAllSpectra(ic, wsSubMass, firstMass) 
    return averagedSpectraYSpace, wsRes


def calculateMantidResolution(ic, ws, mass):
    rebinPars=ic.rebinParametersForYSpaceFit
    for index in range(ws.getNumberHistograms()):
        if np.all(ws.dataY(index)[:] == 0):  # Ignore masked spectra
            pass
        else:
            VesuvioResolution(Workspace=ws,WorkspaceIndex=index,Mass=mass,OutputWorkspaceYSpace="tmp")
            Rebin(InputWorkspace="tmp", Params=rebinPars, OutputWorkspace="tmp")

            if index == 0:   # Ensures that workspace has desired units
                RenameWorkspace("tmp","resolution")
            else:
                AppendSpectra("resolution", "tmp", OutputWorkspace= "resolution")

    try:
        SumSpectra(InputWorkspace="resolution",OutputWorkspace="resolution")
    except ValueError:
        raise ValueError ("All the rows from the workspace to be fitted are Nan!")

    normalise_workspace("resolution")
    DeleteWorkspace("tmp")
    return mtd["resolution"]

    
def normalise_workspace(ws_name):
    tmp_norm = Integration(ws_name)
    Divide(LHSWorkspace=ws_name,RHSWorkspace=tmp_norm,OutputWorkspace=ws_name)
    DeleteWorkspace("tmp_norm")


def subtractAllMassesExceptFirst(ic, ws, ncpForEachMass):
    """Input: workspace from last iteration, ncpTotal for each mass
       Output: workspace with all the ncpTotal subtracted except for the first mass"""

    ncpForEachMass = switchFirstTwoAxis(ncpForEachMass)
    # Select all masses other than the first one
    ncpForEachMass = ncpForEachMass[1:, :, :]
    # Sum the ncpTotal for remaining masses
    ncpTotal = np.sum(ncpForEachMass, axis=0)

    dataY, dataX = ws.extractY(), ws.extractX() 
    
    # Subtract the ncp of all masses exept first to dataY
    dataY[:, :-1] -= ncpTotal * (dataX[:, 1:] - dataX[:, :-1])

    # Pass the data onto a Workspace, clone to preserve properties
    wsSubMass = CloneWorkspace(InputWorkspace=ws, OutputWorkspace=ws.name()+"_H")
    for i in range(wsSubMass.getNumberHistograms()):  # Keeps the faulty last column
        wsSubMass.dataY(i)[:] = dataY[i, :]

     # Mask spectra again, to be seen as masked from Mantid's perspective
    MaskDetectors(Workspace=wsSubMass, WorkspaceIndexList=ic.maskedDetectorIdx)  

    if np.any(np.isnan(mtd[ws.name()+"_H"].extractY())):
        raise ValueError("The workspace for the isolated H data countains NaNs, \
                            might cause problems!")
    return wsSubMass


def switchFirstTwoAxis(A):
    """Exchanges the first two indices of an array A,
    rearranges matrices per spectrum for iteration of main fitting procedure
    """
    return np.stack(np.split(A, len(A), axis=0), axis=2)[0]


def averageJOfYOverAllSpectra(ic, ws0, mass):
    wsYSpace = convertToYSpace(ic, ws0, mass)
    averagedSpectraYSpace = weightedAvg(wsYSpace)
    
    if ic.symmetrisationFlag == True:
        symAvgdSpecYSpace = symmetrizeWs(ic, averagedSpectraYSpace)
        return symAvgdSpecYSpace

    return averagedSpectraYSpace


def convertToYSpace(ic, ws0, mass):
    ConvertToYSpace(
        InputWorkspace=ws0, Mass=mass, 
        OutputWorkspace=ws0.name()+"_JoY", QWorkspace=ws0.name()+"_Q"
        )
    rebinPars=ic.rebinParametersForYSpaceFit
    Rebin(
        InputWorkspace=ws0.name()+"_JoY", Params=rebinPars, 
        FullBinsOnly=True, OutputWorkspace=ws0.name()+"_JoY"
        )
    normalise_workspace(ws0.name()+"_JoY")
    return mtd[ws0.name()+"_JoY"]


def weightedAvg(wsYSpace):
    dataY = wsYSpace.extractY()
    dataE = wsYSpace.extractE()

    dataY[dataY==0] = np.nan
    dataE[dataE==0] = np.nan

    meanY = np.nansum(dataY/np.square(dataE), axis=0) / np.nansum(1/np.square(dataE), axis=0)
    meanE = np.sqrt(1 / np.nansum(1/np.square(dataE), axis=0))

    tempWs = SumSpectra(wsYSpace)
    newWs = CloneWorkspace(tempWs, OutputWorkspace=wsYSpace.name()+"_weighted_avg")
    newWs.dataY(0)[:] = meanY
    newWs.dataE(0)[:] = meanE
    DeleteWorkspace(tempWs)
    return newWs


def symmetrizeWs(ic, avgYSpace):
    """Symmetrizes workspace with only one spectrum,
       Needs to have symmetric binning"""

    dataX = avgYSpace.extractX()
    dataY = avgYSpace.extractY()
    dataE = avgYSpace.extractE()

    yFlip = np.flip(dataY)
    eFlip = np.flip(dataE)

    if ic.symmetriseHProfileUsingAveragesFlag:
        # Inverse variance weighting
        dataYSym = (dataY/dataE**2 + yFlip/eFlip**2) / (1/dataE**2 + 1/eFlip**2)
        dataESym = 1 / np.sqrt(1/dataE**2 + 1/eFlip**2)
    else:
        # Mirroring positive values from negative ones
        dataYSym = np.where(dataX>0, yFlip, dataY)
        dataESym = np.where(dataX>0, eFlip, dataE)

    Sym = CloneWorkspace(avgYSpace, OutputWorkspace=avgYSpace.name()+"_symmetrised")
    Sym.dataY(0)[:] = dataYSym
    Sym.dataE(0)[:] = dataESym
    return Sym


def fitFirstMassProfileInYSpace(ic, wsYSpaceSym, wsRes):
    # if ic.useScipyCurveFitToHProfileFlag:
    poptCurveFit, pcovCurveFit = fitProfileCurveFit(ic, wsYSpaceSym, wsRes)
    perrCurveFit = np.sqrt(np.diag(pcovCurveFit))
    # else:
    poptMantidFit, perrMantidFit = fitProfileMantidFit(ic, wsYSpaceSym, wsRes)
    
    #TODO: Add the Cost function as the last parameter
    poptCurveFit = np.append(poptCurveFit, np.nan)
    perrCurveFit = np.append(perrCurveFit, np.nan)

    popt = np.vstack((poptCurveFit, poptMantidFit))
    perr = np.vstack((perrCurveFit, perrMantidFit))

    return popt, perr


def fitProfileCurveFit(ic, wsYSpaceSym, wsRes):
    res = wsRes.extractY()[0]
    resX = wsRes. extractX()[0]

    # Interpolate Resolution to get single peak at zero
    # Otherwise if the resolution has two data points at the peak,
    # the convolution will be skewed.
    start, interval, end = [float(i) for i in ic.rebinParametersForYSpaceFit.split(",")]
    resNewX = np.arange(start, end, interval)
    res = np.interp(resNewX, resX, res)

    dataY = wsYSpaceSym.extractY()[0]
    dataX = wsYSpaceSym.extractX()[0]
    dataE = wsYSpaceSym.extractE()[0]

    if ic.singleGaussFitToHProfile:
        def convolvedFunction(x, y0, A, x0, sigma):
            histWidths = x[1:] - x[:-1]
            if ~ (np.max(histWidths)==np.min(histWidths)):
                raise AssertionError("The histograms widhts need to be the same for the discrete convolution to work!")

            gaussFunc = gaussianFit(x, y0, x0, A, sigma)
            convGauss = ndimage.convolve1d(gaussFunc, res, mode="constant") * histWidths[0]  
            return convGauss
        p0 = [0, 1, 0, 5]
        bounds = [-np.inf, np.inf]  # Applied to all parameters

    else:
        # # Double Gaussian
        # def convolvedFunction(x, y0, x0, A, sigma):
        #     histWidths = x[1:] - x[:-1]
        #     if ~ (np.max(histWidths)==np.min(histWidths)):
        #         raise AssertionError("The histograms widhts need to be the same for the discrete convolution to work!")

        #     gaussFunc = gaussianFit(x, y0, x0, A, 4.76) + gaussianFit(x, 0, x0, 0.054*A, sigma)
        #     convGauss = ndimage.convolve1d(gaussFunc, res, mode="constant") * histWidths[0]
        #     return convGauss
        # p0 = [0, 0, 0.7143, 5]

        def HermitePolynomial(x, A, x0, sigma1, c4, c6):
            return A * np.exp(-(x-x0)**2/2/sigma1**2) / (np.sqrt(2*3.1415*sigma1**2)) \
                    *(1 + c4/32*(16*((x-x0)/np.sqrt(2)/sigma1)**4 \
                    -48*((x-x0)/np.sqrt(2)/sigma1)**2+12) \
                    +c6/384*(64*((x-x0)/np.sqrt(2)/sigma1)**6 \
                    -480*((x-x0)/np.sqrt(2)/sigma1)**4 + 720*((x-x0)/np.sqrt(2)/sigma1)**2 - 120))
        
        def convolvedFunction(x, A, x0, sigma1, c4, c6):
            histWidths = x[1:] - x[:-1]
            if ~ (np.max(histWidths)==np.min(histWidths)):
                raise AssertionError("The histograms widhts need to be the same for the discrete convolution to work!")

            hermiteFunc = HermitePolynomial(x, A, x0, sigma1, c4, c6)
            convFunc = ndimage.convolve1d(hermiteFunc, res, mode="constant") * histWidths[0]
            return convFunc
        p0 = [1, 0, 4, 0, 0]     
        # The bounds on curve_fit() are set up diferently than on minimize()
        bounds = [[0, -np.inf, -np.inf, 0, 0], [np.inf, np.inf, np.inf, np.inf, np.inf]] 


    popt, pcov = optimize.curve_fit(
        convolvedFunction, 
        dataX, 
        dataY, 
        p0=p0,
        sigma=dataE,
        bounds=bounds
    )
    yfit = convolvedFunction(dataX, *popt)
    Residuals = dataY - yfit
    
    # Create Workspace with the fit results
    # TODO add DataE 
    CreateWorkspace(DataX=np.concatenate((dataX, dataX, dataX)), 
                    DataY=np.concatenate((dataY, yfit, Residuals)), 
                    NSpec=3,
                    OutputWorkspace=wsYSpaceSym.name()+"_fitted_CurveFit")
    return popt, pcov


def gaussianFit(x, y0, x0, A, sigma):
    """Gaussian centered at zero"""
    return y0 + A / (2*np.pi)**0.5 / sigma * np.exp(-(x-x0)**2/2/sigma**2)


def fitProfileMantidFit(ic, wsYSpaceSym, wsRes):

    if ic.singleGaussFitToHProfile:
        popt, perr = np.zeros((2, 5)), np.zeros((2, 5))
    else:
        # popt, perr = np.zeros((2, 6)), np.zeros((2, 6))
        popt, perr = np.zeros((2, 6)), np.zeros((2, 6))


    print('\n','Fitting on the sum of spectra in the West domain ...','\n')     
    for i, minimizer in enumerate(['Levenberg-Marquardt','Simplex']):
        outputName = wsYSpaceSym.name()+"_fitted_"+minimizer
        CloneWorkspace(InputWorkspace = wsYSpaceSym, OutputWorkspace = outputName)
        
        if ic.singleGaussFitToHProfile:
            function='''composite=Convolution,FixResolution=true,NumDeriv=true;
            name=Resolution,Workspace=resolution,WorkspaceIndex=0;
            name=UserFunction,Formula=y0+A*exp( -(x-x0)^2/2/sigma^2)/(2*3.1415*sigma^2)^0.5,
            y0=0,A=1,x0=0,sigma=5,   ties=()'''
        else:
            # # Function for Double Gaussian
            # function='''composite=Convolution,FixResolution=true,NumDeriv=true;
            # name=Resolution,Workspace=resolution,WorkspaceIndex=0;
            # name=UserFunction,Formula=y0+A*exp( -(x-x0)^2/2/sigma1^2)/(2*3.1415*sigma1^2)^0.5
            # +A*0.054*exp( -(x-x0)^2/2/sigma2^2)/(2*3.1415*sigma2^2)^0.5,
            # y0=0,x0=0,A=0.7143,sigma1=4.76, sigma2=5,   ties=(sigma1=4.76)'''
            
            # TODO: Check that this function is correct
            function = """
            composite=Convolution,FixResolution=true,NumDeriv=true;
            name=Resolution,Workspace=resolution,WorkspaceIndex=0,X=(),Y=();
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
        
        ws=mtd[outputName+"_Parameters"]
        popt[i] = ws.column("Value")
        perr[i] = ws.column("Error")
    return popt, perr
# import enum
# from sys import stdin
import numpy as np
# import mantid
# from mantid.api import AnalysisDataService
from mantid.simpleapi import *
# from numpy.core.numeric import True_
from scipy import optimize
# from scipy import ndimage
# import time
# from pathlib import Path
from functools import partial

# Format print output of arrays
np.set_printoptions(suppress=True, precision=4, linewidth=150, threshold=sys.maxsize)
# repoPath = Path(__file__).absolute().parent  # Path to the repository


# '''
# The InitialConditions class allows the user to change the starting parameters.
# The script includes everything required for forward and backscattering.
# The user can choose to perform back or forward scattering individually or to connect the mean widths from 
# backscattering onto the initial parameters for forward scattering.

# The fit procedure in the time-of-flight domain is  based on the scipy.minimize.optimize() tool,
# used with the SLSQP minimizer, that can handle both boundaries and constraints for fitting parameters.
# '''


# class InitialConditions:

#     # Initialize object to use self methods
#     def __init__(self):
#         return None

#     # Multiscaterring Correction Parameters
#     HToMass0Ratio = 19.0620008206

#     transmission_guess =  0.8537        # Experimental value from VesuvioTransmission
#     multiple_scattering_order, number_of_events = 2, 1.e5   
#     hydrogen_peak = True                 # Hydrogen multiple scattering
    
#     # Sample slab parameters
#     vertical_width, horizontal_width, thickness = 0.1, 0.1, 0.001  # Expressed in meters
  
#     modeRunning = "None"     # Stores wether is running forward or backward

#     # Paths to save results for back and forward scattering
#     testingCleaning = True
#     if testingCleaning:     
#         pathForTesting = repoPath / "tests" / "cleaning"  
#         forwardScatteringSavePath = pathForTesting / "current_forward.npz" 
#         backScatteringSavePath = pathForTesting / "current_backward.npz" 
#     else:
#         forwardScatteringSavePath = repoPath / "tests" / "fixatures" / "4iter_forward_GB_MS_opt.npz" 
#         backScatteringSavePath = repoPath / "tests" / "fixatures" / "4iter_backward_MS_opt.npz"
    

#     def setBackscatteringInitialConditions(self):
#         self.modeRunning = "BACKWARD"

#         # Parameters to load Raw and Empty Workspaces
#         self.userWsRawPath = r"./input_ws/starch_80_RD_raw_backward.nxs"
#         self.userWsEmptyPath = r"./input_ws/starch_80_RD_empty_backward.nxs"

#         self.name = "starch_80_RD_backward_"
#         self.runs='43066-43076'  # 77K             # The numbers of the runs to be analysed
#         self.empty_runs='41876-41923'   # 77K             # The numbers of the empty runs to be subtracted
#         self.spectra='3-134'                            # Spectra to be analysed
#         self.tof_binning='275.,1.,420'                    # Binning of ToF spectra
#         self.mode='DoubleDifference'
#         self.ipfile=r'./ip2019.par' 

#         # Masses, instrument parameters and initial fitting parameters
#         self.masses = np.array([12, 16, 27])
#         self.noOfMasses = len(self.masses)
#         self.InstrParsPath = repoPath / 'ip2018_3.par'

#         self.initPars = np.array([ 
#         # Intensities, NCP widths, NCP centers   
#             1, 12, 0.,    
#             1, 12, 0.,   
#             1, 12.5, 0.    
#         ])
#         self.bounds = np.array([
#             [0, np.nan], [8, 16], [-3, 1],
#             [0, np.nan], [8, 16], [-3, 1],
#             [0, np.nan], [11, 14], [-3, 1]
#         ])
#         self.constraints = ()

#         self.noOfMSIterations = 4     #4
#         self.firstSpec = 3    #3
#         self.lastSpec = 134    #134

#         # Boolean Flags to control script
#         self.loadWsFromUserPathFlag = True
#         self.scaleParsFlag = False
#         self.MSCorrectionFlag = True
#         self.GammaCorrectionFlag = False
#         maskedSpecAllNo = np.array([18, 34, 42, 43, 59, 60, 62, 118, 119, 133])

#         # Parameters below are not to be changed
#         self.firstSpecIdx = 0
#         self.lastSpecIdx = self.lastSpec - self.firstSpec

#         # Consider only the masked spectra between first and last spectrum
#         self.maskedSpecNo = maskedSpecAllNo[
#             (maskedSpecAllNo >= self.firstSpec) & (maskedSpecAllNo <= self.lastSpec)
#         ]
#         self.maskedDetectorIdx = self.maskedSpecNo - self.firstSpec

#         # Set scaling factors for the fitting parameters, default is ones
#         self.scalingFactors = np.ones(self.initPars.shape)
#         if self.scaleParsFlag:        # Scale fitting parameters using initial values
#                 self.initPars[2::3] = np.ones((1, self.noOfMasses))  # Main problem is that zeros have to be replaced by non zeros
#                 self.scalingFactors = 1 / self.initPars



#     def setForwardScatteringInitialConditions(self):
#         self.modeRunning = "FORWARD"  # Used to control MS correction

#         self.userWsRawPath = r"./input_ws/starch_80_RD_raw_forward.nxs"
#         self.userWsEmptyPath = r"./input_ws/starch_80_RD_raw_forward.nxs"

#         self.name = "starch_80_RD_forward_"
#         self.runs='43066-43076'         # 100K        # The numbers of the runs to be analysed
#         self.empty_runs='43868-43911'   # 100K        # The numbers of the empty runs to be subtracted
#         self.spectra='144-182'                        # Spectra to be analysed
#         self.tof_binning="110,1.,430"                 # Binning of ToF spectra
#         self.mode='SingleDifference'
#         self.ipfile=r'./ip2018_3.par'

#         self.masses = np.array([1.0079, 12, 16, 27]) 
#         self.noOfMasses = len(self.masses)
#         self.InstrParsPath = repoPath / 'ip2018_3.par'

#         self.initPars = np.array([ 
#         # Intensities, NCP widths, NCP centers  
#             1, 4.7, 0, 
#             1, 12.71, 0.,    
#             1, 8.76, 0.,   
#             1, 13.897, 0.    
#         ])
#         self.bounds = np.array([
#             [0, np.nan], [3, 6], [-3, 1],
#             [0, np.nan], [12.71, 12.71], [-3, 1],
#             [0, np.nan], [8.76, 8.76], [-3, 1],
#             [0, np.nan], [13.897, 13.897], [-3, 1]
#         ])
#         self.constraints = ()

#         self.noOfMSIterations = 2     #4
#         self.firstSpec = 164   #144
#         self.lastSpec = 175    #182

#         # Boolean Flags to control script
#         self.loadWsFromUserPathFlag = True
#         self.scaleParsFlag = False
#         self.MSCorrectionFlag = True
#         self.GammaCorrectionFlag = True

#         # Parameters to control fit in Y-Space
#         self.symmetrisationFlag = True
#         self.symmetriseHProfileUsingAveragesFlag = True      # When False, use mirror sym
#         self.rebinParametersForYSpaceFit = "-20, 0.5, 20"    # Needs to be symetric
#         self.singleGaussFitToHProfile = True      # When False, use Hermite expansion
#         maskedSpecAllNo = np.array([173, 174, 179])

#         # Parameters below are not to be changed
#         self.firstSpecIdx = 0
#         self.lastSpecIdx = self.lastSpec - self.firstSpec

#         # Consider only the masked spectra between first and last spectrum
#         self.maskedSpecNo = maskedSpecAllNo[
#             (maskedSpecAllNo >= self.firstSpec) & (maskedSpecAllNo <= self.lastSpec)
#         ]
#         self.maskedDetectorIdx = self.maskedSpecNo - self.firstSpec

#         # Set scaling factors for the fitting parameters, default is ones
#         self.scalingFactors = np.ones(self.initPars.shape)
#         if self.scaleParsFlag:        # Scale fitting parameters using initial values
#                 self.initPars[2::3] = np.ones((1, self.noOfMasses))  # Main problem is that zeros have to be replaced by non zeros
#                 self.scalingFactors = 1 / self.initPars


def printInitialParameters(ic):
    print("\nRUNNING ", ic.modeRunning, " SCATTERING.")
    print("\n\nH to first mass ratio: ", ic.HToMass0Ratio,
            "\n\nForward scattering initial fitting parameters:\n", 
            ic.initPars.reshape((ic.masses.size, 3)),
            "\n\nForward scattering initial fitting bounds:\n", 
            ic.bounds, "\n")


# This is the only variable with global behaviour, all functions are defined to use attributes of ic
# ic = InitialConditions() 


# def runOnlyBackScattering(bckwdIC):
#     AnalysisDataService.clear()
#     # ic.setBackscatteringInitialConditions()
#     printInitialParameters(bckwdIC)
#     wsFinal, backScatteringResults = iterativeFitForDataReduction(bckwdIC)
#     backScatteringResults.save(bckwdIC) 


# def runOnlyForwardScattering(fwdIC):
#     AnalysisDataService.clear()
#     # ic.setForwardScatteringInitialConditions()
#     printInitialParameters(fwdIC)
#     wsFinal, forwardScatteringResults = iterativeFitForDataReduction(fwdIC)
#     fitInYSpaceProcedure(fwdIC, wsFinal, forwardScatteringResults)
#     forwardScatteringResults.save(fwdIC)


# def runSequenceForKnownRatio(bckwdIC, fwdIC):
#     AnalysisDataService.clear()
#     # If H to first mass ratio is known, can run MS correction for backscattering
#     # Back scattering produces precise results for widhts and intensity ratios for non-H masses
#     printInitialParameters(bckwdIC)
#     wsFinal, backScatteringResults = iterativeFitForDataReduction(bckwdIC)
#     backScatteringResults.save(bckwdIC) 

#     setInitFwdParsFromBackResults(backScatteringResults, bckwdIC.HToMass0Ratio, fwdIC)
#     printInitialParameters(fwdIC)
#     wsFinal, forwardScatteringResults = iterativeFitForDataReduction(fwdIC)
#     fitInYSpaceProcedure(fwdIC, wsFinal, forwardScatteringResults)
#     forwardScatteringResults.save(fwdIC)


# def runSequenceRatioNotKnown(bckwdIC, fwdIC):
#     # Run preliminary forward with a good guess for the widths of non-H masses
#     # ic.setForwardScatteringInitialConditions()
#     printInitialParameters(fwdIC)
#     wsFinal, forwardScatteringResults = iterativeFitForDataReduction(fwdIC)
#     for i in range(2):    # Loop until convergence is achieved
#         AnalysisDataService.clear()    # Clears all Workspaces

#         # Get first estimate of H to mass0 ratio
#         fwdMeanIntensityRatios = forwardScatteringResults.all_mean_intensities[-1] 
#         bckwdIC.HToMass0Ratio = fwdMeanIntensityRatios[0] / fwdMeanIntensityRatios[1]
       
#         # Run backward procedure with this estimate
#         # ic.setBackscatteringInitialConditions()
#         printInitialParameters(bckwdIC)
#         wsFinal, backScatteringResults = iterativeFitForDataReduction(bckwdIC)

#         # ic.setForwardScatteringInitialConditions()
#         setInitFwdParsFromBackResults(backScatteringResults, bckwdIC.HToMass0Ratio, fwdIC)
#         printInitialParameters(fwdIC)
#         wsFinal, forwardScatteringResults = iterativeFitForDataReduction(fwdIC)

#     fitInYSpaceProcedure(fwdIC, wsFinal, forwardScatteringResults)
#     backScatteringResults.save(bckwdIC)
#     forwardScatteringResults.save(fwdIC)


# def setInitFwdParsFromBackResults(backScatteringResults, HToMass0Ratio, fwdIC):
#     """Takes widths and intensity ratios obtained in backscattering
#     and uses them as the initial conditions for forward scattering """

#     # Get widts and intensity ratios from backscattering results
#     backMeanWidths = backScatteringResults.all_mean_widths[-1]
#     backMeanIntensityRatios = backScatteringResults.all_mean_intensities[-1] 

#     HIntensity = HToMass0Ratio * backMeanIntensityRatios[0]
#     initialFwdIntensityRatios = np.append([HIntensity], backMeanIntensityRatios)
#     initialFwdIntensityRatios /= np.sum(initialFwdIntensityRatios)

#     # Set starting conditions for forward scattering
#     # Fix known widths and intensity ratios from back scattering results
#     fwdIC.initPars[4::3] = backMeanWidths
#     fwdIC.initPars[0::3] = initialFwdIntensityRatios
#     fwdIC.bounds[4::3] = backMeanWidths[:, np.newaxis] * np.ones((1,2))
#     # Fix the intensity ratios 
#     # ic.bounds[0::3] = initialFwdIntensityRatios[:, np.newaxis] * np.ones((1,2)) 

#     print("\nChanged initial conditions of forward scattering \
#         according to mean widhts and intensity ratios from backscattering.\n")
#     return



# """
# All the functions required for the procedures above are listed below, in order of appearance
# """


def iterativeFitForDataReduction(ic):
    printInitialParameters(ic)

    initialWs = loadVesuvioDataWorkspaces(ic)   
    cropedWs = cropAndMaskWorkspace(ic, initialWs)
    wsToBeFitted = CloneWorkspace(InputWorkspace=cropedWs, OutputWorkspace=cropedWs.name()+"0")
   
    createSlabGeometry(ic)

    # Initialize arrays to store script results
    fittingResults = resultsObject(ic)

    for iteration in range(ic.noOfMSIterations):
        # Workspace from previous iteration
        wsToBeFitted = mtd[ic.name+str(iteration)]
        SumSpectra(InputWorkspace=wsToBeFitted, OutputWorkspace=wsToBeFitted.name()+"_sum")

        print("\nFitting spectra ", ic.firstSpec, " - ", ic.lastSpec, "\n.............")
        
        # The fitting procedure stores results in the fittingResults object
        fitNcpToWorkspace(ic, wsToBeFitted, fittingResults)
        
        fittingResults.calculateMeansAndStd()
        fittingResults.printCurrentIrerationResults()

        # When last iteration, skip MS and GC
        if iteration == ic.noOfMSIterations - 1:
          break 

        meanWidths = fittingResults.all_mean_widths[iteration]
        meanIntensityRatios = fittingResults.all_mean_intensities[iteration]
        
        CloneWorkspace(InputWorkspace=ic.name, OutputWorkspace="tmpNameWs")

        if ic.MSCorrectionFlag:
            createWorkspacesForMSCorrection(ic, meanWidths, meanIntensityRatios)
            Minus(LHSWorkspace="tmpNameWs", RHSWorkspace=ic.name+"_MulScattering",
                    OutputWorkspace="tmpNameWs")

        if ic.GammaCorrectionFlag:  
            createWorkspacesForGammaCorrection(ic, meanWidths, meanIntensityRatios)
            Minus(LHSWorkspace="tmpNameWs", RHSWorkspace=ic.name+"_gamma_background", 
                    OutputWorkspace="tmpNameWs")

        RenameWorkspace(InputWorkspace="tmpNameWs", OutputWorkspace=ic.name+str(iteration+1))

    wsFinal = mtd[ic.name+str(ic.noOfMSIterations - 1)]
    fittingResults.save()
    return wsFinal, fittingResults


def loadVesuvioDataWorkspaces(ic):
    """Loads raw and empty workspaces from either LoadVesuvio or user specified path"""
    if ic.loadWsFromUserPathFlag:
        wsToBeFitted =  loadRawAndEmptyWsFromUserPath(ic)
    else:
        wsToBeFitted = loadRawAndEmptyWsFromLoadVesuvio(ic)
    return wsToBeFitted


def loadRawAndEmptyWsFromUserPath(ic):

    print('\n', 'Loading the sample runs: ', ic.runs, '\n')
    Load(Filename=ic.userWsRawPath, OutputWorkspace=ic.name+"raw")
    Rebin(InputWorkspace=ic.name+'raw', Params=ic.tof_binning,
          OutputWorkspace=ic.name+'raw')
    SumSpectra(InputWorkspace=ic.name+'raw', OutputWorkspace=ic.name+'raw'+'_sum')
    wsToBeFitted = CloneWorkspace(InputWorkspace=ic.name+'raw', OutputWorkspace=ic.name+"uncroped_unmasked")

    if ic.mode=="DoubleDifference":
        print('\n', 'Loading the empty runs: ', ic.empty_runs, '\n')
        Load(Filename=ic.userWsEmptyPath, OutputWorkspace=ic.name+"empty")
        Rebin(InputWorkspace=ic.name+'empty', Params=ic.tof_binning,
            OutputWorkspace=ic.name+'empty')
        wsToBeFitted = Minus(LHSWorkspace=ic.name+'raw', RHSWorkspace=ic.name+'empty',
                            OutputWorkspace=ic.name+"uncroped_unmasked")
    return wsToBeFitted


def loadRawAndEmptyWsFromLoadVesuvio(ic):
    
    print('\n', 'Loading the sample runs: ', ic.runs, '\n')
    LoadVesuvio(Filename=ic.runs, SpectrumList=ic.spectra, Mode=ic.mode,
                InstrumentParFile=ic.ipfile, OutputWorkspace=ic.name+'raw')
    Rebin(InputWorkspace=ic.name+'raw', Params=ic.tof_binning,
          OutputWorkspace=ic.name+'raw')
    SumSpectra(InputWorkspace=ic.name+'raw', OutputWorkspace=ic.name+'raw'+'_sum')
    wsToBeFitted = CloneWorkspace(InputWorkspace=ic.name+'raw', OutputWorkspace=ic.name+"uncroped_unmasked")

    if ic.mode=="DoubleDifference":
        print('\n', 'Loading the empty runs: ', ic.empty_runs, '\n')
        LoadVesuvio(Filename=ic.empty_runs, SpectrumList=ic.spectra, Mode=ic.mode,
                    InstrumentParFile=ic.ipfile, OutputWorkspace=ic.name+'empty')
        Rebin(InputWorkspace=ic.name+'empty', Params=ic.tof_binning,
            OutputWorkspace=ic.name+'empty')
        wsToBeFitted = Minus(LHSWorkspace=ic.name+'raw', RHSWorkspace=ic.name+'empty', 
                            OutputWorkspace=ic.name+"uncroped_unmasked")
    return wsToBeFitted


def cropAndMaskWorkspace(ic, ws):
    """Returns cloned and cropped workspace with modified name"""
    # Read initial Spectrum number
    wsFirstSpec = ws.getSpectrumNumbers()[0]
    assert ic.firstSpec >= wsFirstSpec, "Can't crop workspace, firstSpec < first spectrum in workspace."
    
    initialIdx = ic.firstSpec - wsFirstSpec
    lastIdx = ic.lastSpec - wsFirstSpec
    
    newWsName = ws.name().split("uncroped")[0]  # Retrieve original name
    cropedWs = CropWorkspace(
        InputWorkspace=ws, 
        StartWorkspaceIndex=initialIdx, EndWorkspaceIndex=lastIdx, 
        OutputWorkspace=newWsName
        )
    MaskDetectors(Workspace=cropedWs, WorkspaceIndexList=ic.maskedDetectorIdx)
    return cropedWs


def createSlabGeometry(ic):
    half_height, half_width, half_thick = 0.5*ic.vertical_width, 0.5*ic.horizontal_width, 0.5*ic.thickness
    xml_str = \
        " <cuboid id=\"sample-shape\"> " \
        + "<left-front-bottom-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (half_width, -half_height, half_thick) \
        + "<left-front-top-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (half_width, half_height, half_thick) \
        + "<left-back-bottom-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (half_width, -half_height, -half_thick) \
        + "<right-front-bottom-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (-half_width, -half_height, half_thick) \
        + "</cuboid>"
    CreateSampleShape(ic.name, xml_str)


def fitNcpToWorkspace(ic, ws, fittingResults):
    """Firstly calculates matrices for all spectrums,
    then iterates over each spectrum
    """
    wsDataY = ws.extractY()       #DataY unaltered
    fittingResults.addDataY(wsDataY)

    dataY, dataX, dataE = loadWorkspaceIntoArrays(ws)                     
    resolutionPars, instrPars, kinematicArrays, ySpacesForEachMass = prepareFitArgs(ic, dataX)
    
    # Fit all spectrums
    fitPars = np.array(list(map(
        partial(fitNcpToSingleSpec, ic=ic), 
        dataY, dataE, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays
    )))
    fittingResults.addFitPars(fitPars)

    # Create ncpTotal workspaces
    mainPars = fittingResults.getMainPars()
    ncpForEachMass = np.array(list(map(
        partial(buildNcpFromSpec, ic=ic), 
        mainPars , ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays
    )))
    ncpTotal = np.sum(ncpForEachMass, axis=1)  
    createNcpWorkspaces(ncpForEachMass, ncpTotal, ws)  
    # Adds individual and total ncp
    fittingResults.addNCP(ncpForEachMass, ncpTotal)

 
class resultsObject:
    """Used to store fitting results of the script at each iteration"""

    def __init__(self, ic):
        """Initializes arrays full of zeros"""

        self.all_fit_workspaces = None 
        self.all_spec_best_par_chi_nit = None 
        self.all_tot_ncp = None 
        self.all_ncp_for_each_mass = None
        
        self.all_mean_widths = None
        self.all_mean_intensities = None 
        self.all_std_widths = None 
        self.all_std_intensities = None 

        # Pass all attributes of ic into attributes to be used whithin this object
        self.maskedDetectorIdx = ic.maskedDetectorIdx
        self.masses = ic.masses
        self.noOfMasses = ic.noOfMasses
        self.resultsSavePath = ic.resultsSavePath
        self.singleGaussFitToHProfile = ic.singleGaussFitToHProfile

    def addDataY(self, dataY):
        if self.all_fit_workspaces is None:
            self.all_fit_workspaces = dataY[np.newaxis, :, :]
        else:
            self.all_fit_workspaces = np.append(self.all_fit_workspaces, dataY[np.newaxis, :, :], axis=0)
        # self.all_fit_workspaces[MSiter] = dataY
    

    def addFitPars(self, fitPars):
        if self.all_spec_best_par_chi_nit is None:
            self.all_spec_best_par_chi_nit = fitPars[np.newaxis, :, :]
        else:
            self.all_spec_best_par_chi_nit = np.append(self.all_spec_best_par_chi_nit, fitPars[np.newaxis, :, :], axis=0)
       
 
    def getMainPars(self):
        # Since only want a reading, use copy() not to risk changing the array
        return self.all_spec_best_par_chi_nit[-1][:, 1:-2].copy()
    

    def addNCP(self, ncpForEachMass, ncpTotal):
        if self.all_tot_ncp is None:
            self.all_ncp_for_each_mass = ncpForEachMass[np.newaxis, :, :, :]
            self.all_tot_ncp = ncpTotal[np.newaxis, :, :]
        else:
            self.all_ncp_for_each_mass = np.append(self.all_ncp_for_each_mass, ncpForEachMass[np.newaxis, :, :, :], axis=0)
            self.all_tot_ncp = np.append(self.all_tot_ncp, ncpTotal[np.newaxis, :, :], axis=0)


    def calculateMeansAndStd(self):
        # Copy arrays to avoid changing stored values
        # Transpose to horizontal shape
        intensities = self.all_spec_best_par_chi_nit[-1][:, 1:-2:3].copy().T
        widths = self.all_spec_best_par_chi_nit[-1][:, 2:-2:3].copy().T
        noOfMasses = self.noOfMasses

        # Replace zeros from masked spectra with nans
        widths[:, self.maskedDetectorIdx] = np.nan
        intensities[:, self.maskedDetectorIdx] = np.nan

        meanWidths = np.nanmean(widths, axis=1).reshape(noOfMasses, 1)  
        stdWidths = np.nanstd(widths, axis=1).reshape(noOfMasses, 1)

        # Subtraction row by row
        widthDeviation = np.abs(widths - meanWidths)
        # Where True, replace by nan
        betterWidths = np.where(widthDeviation > stdWidths, np.nan, widths)
        betterIntensities = np.where(widthDeviation > stdWidths, np.nan, intensities)

        meanWidths = np.nanmean(betterWidths, axis=1)  
        stdWidths = np.nanstd(betterWidths, axis=1)

        # Not nansum(), to propagate nan
        normalization = np.sum(betterIntensities, axis=0)
        intensityRatios = betterIntensities / normalization

        meanIntensityRatios = np.nanmean(intensityRatios, axis=1)
        stdIntensityRatios = np.nanstd(intensityRatios, axis=1)

        # Append in the same fashion as above
        if self.all_mean_intensities is None:
            self.all_mean_widths = meanWidths[np.newaxis, :]
            self.all_std_widths = stdWidths[np.newaxis, :]
            self.all_mean_intensities = meanIntensityRatios[np.newaxis, :]
            self.all_std_intensities = stdIntensityRatios[np.newaxis, :]
        else:
            self.all_mean_widths = np.append(self.all_mean_widths, meanWidths[np.newaxis, :], axis=0) 
            self.all_std_widths = np.append(self.all_std_widths, stdWidths[np.newaxis, :], axis=0)
            self.all_mean_intensities = np.append(self.all_mean_intensities, meanIntensityRatios[np.newaxis, :], axis=0)
            self.all_std_intensities = np.append(self.all_std_intensities, stdIntensityRatios[np.newaxis, :], axis=0)


    def printCurrentIrerationResults(self):

        print("\n\nSpec ------- Main Pars ----------- Chi Nit:\n")
        print(self.all_spec_best_par_chi_nit[-1])

        for i, mass in enumerate(self.masses):
            print(f"\nMass = {mass:.2f} amu:")
            print(f"Width:     {self.all_mean_widths[-1, i]:.3f} \u00B1 {self.all_std_widths[-1, i]:.3f} ")
            print(f"Intensity: {self.all_mean_intensities[-1, i]:.3f} \u00B1 {self.all_std_intensities[-1, i]:.3f} ")

        print("\nCheck masses are correct:\n")
        print(self.all_mean_widths[-1])
        print(self.all_std_widths[-1])
        print(self.all_mean_intensities[-1])
        print(self.all_std_intensities[-1])

    # Set default of yspace fit parameters to zero, in case they don't get used
    # YSpaceSymSumDataY = 0
    # YSpaceSymSumDataE = 0
    # resolution = 0
    # finalRawDataY = 0
    # finalRawDataE = 0
    # HdataY = 0
    # popt = 0
    # perr = 0


    # def storeResultsOfYSpaceFit(self, wsFinal, wsH, wsYSpaceSymSum, wsRes, popt, perr):
    #     self.finalRawDataY = wsFinal.extractY()
    #     self.finalRawDataE = wsFinal.extractE()
    #     self.HdataY = wsH.extractY()
    #     self.YSpaceSymSumDataY = wsYSpaceSymSum.extractY()
    #     self.YSpaceSymSumDataE = wsYSpaceSymSum.extractE()
    #     self.resolution = wsRes.extractY()
    #     self.popt = popt
    #     self.perr = perr


    # def printYSpaceFitResults(self):
    #     print("\nFit in Y Space results:")
    #     # print("Fit algorithm rows: \nCurve Fit \nMantid Fit LM \nMantid Fit Simplex")
    #     # print("\nOrder: [y0, A, x0, sigma]")
    #     # print("\npopt:\n", self.popt)
    #     # print("\nperr:\n", self.perr, "\n")

    #     if self.singleGaussFitToHProfile:
    #         for i, fit in enumerate(["Curve Fit", "Mantid Fit LM", "Mantid Fit Simplex"]):
    #             print(f"\n{fit:15s}")
    #             for par, popt, perr in zip(["y0:", "A:", "x0:", "sigma:", "Cost Fun:"], self.popt[i], self.perr[i]):
    #                 print(f"{par:9s} {popt:8.4f} \u00B1 {perr:6.4f}")
    #     else:
    #         for i, fit in enumerate(["Curve Fit", "Mantid Fit LM", "Mantid Fit Simplex"]):
    #             print(f"\n{fit:15s}")
    #             for par, popt, perr in zip(["sigma:", "c4:", "c6:"], self.popt[i], self.perr[i]):
    #                 print(f"{par:9s} {popt:8.4f} \u00B1 {perr:6.4f}")


    def save(self):
        """Saves all of the arrays stored in this object"""

        # TODO: Take out nans next time when running original results
        # Because original results were recently saved with nans, mask spectra with nans
        self.all_spec_best_par_chi_nit[:, self.maskedDetectorIdx, :] = np.nan
        self.all_ncp_for_each_mass[:, self.maskedDetectorIdx, :, :] = np.nan
        self.all_tot_ncp[:, self.maskedDetectorIdx, :] = np.nan

        savePath = self.resultsSavePath
        np.savez(savePath,
                 all_fit_workspaces=self.all_fit_workspaces,
                 all_spec_best_par_chi_nit=self.all_spec_best_par_chi_nit,
                 all_mean_widths=self.all_mean_widths,
                 all_mean_intensities=self.all_mean_intensities,
                 all_std_widths=self.all_std_widths,
                 all_std_intensities=self.all_std_intensities,
                 all_tot_ncp=self.all_tot_ncp,
                 all_ncp_for_each_mass=self.all_ncp_for_each_mass)
                #  YSpaceSymSumDataY=self.YSpaceSymSumDataY,
                #  YSpaceSymSumDataE=self.YSpaceSymSumDataE,
                #  resolution=self.resolution, 
                #  HdataY=self.HdataY,
                #  finalRawDataY=self.finalRawDataY, 
                #  finalRawDataE=self.finalRawDataE,
                #  popt=self.popt, 
                #  perr=self.perr)

    

def loadWorkspaceIntoArrays(ws):
    """Output: dataY, dataX and dataE as arrays and converted to point data"""
    dataY = ws.extractY()
    dataE = ws.extractE()
    dataX = ws.extractX()

    histWidths = dataX[:, 1:] - dataX[:, :-1]
    dataY = dataY[:, :-1] / histWidths
    dataE = dataE[:, :-1] / histWidths
    dataX = (dataX[:, 1:] + dataX[:, :-1]) / 2
    return dataY, dataX, dataE


def prepareFitArgs(ic, dataX):
    instrPars = loadInstrParsFileIntoArray(ic.InstrParsPath, ic.firstSpec, ic.lastSpec)       
    resolutionPars = loadResolutionPars(instrPars)                                   

    v0, E0, delta_E, delta_Q = calculateKinematicsArrays(dataX, instrPars)   
    kinematicArrays = np.array([v0, E0, delta_E, delta_Q])
    ySpacesForEachMass = convertDataXToYSpacesForEachMass(dataX, ic.masses, delta_Q, delta_E)        
    
    kinematicArrays = reshapeArrayPerSpectrum(kinematicArrays)
    ySpacesForEachMass = reshapeArrayPerSpectrum(ySpacesForEachMass)
    return resolutionPars, instrPars, kinematicArrays, ySpacesForEachMass


def loadInstrParsFileIntoArray(InstrParsPath, firstSpec, lastSpec):
    """Loads instrument parameters into array, from the file in the specified path"""

    # For some odd reason, np.loadtxt() is only working with Path object
    # So transform string path into Path object
    # IPFileName = IPFileString.split("/")[-1]
    # InstrParsPath = repoPath / IPFileName

    data = np.loadtxt(InstrParsPath, dtype=str)[1:].astype(float)

    spectra = data[:, 0]
    select_rows = np.where((spectra >= firstSpec) & (spectra <= lastSpec))
    instrPars = data[select_rows]
    return instrPars


def loadResolutionPars(instrPars):
    """Resolution of parameters to propagate into TOF resolution
       Output: matrix with each parameter in each column"""
    spectrums = instrPars[:, 0] 
    L = len(spectrums)
    #for spec no below 135, back scattering detectors, mode is double difference
    #for spec no 135 or above, front scattering detectors, mode is single difference
    dE1 = np.where(spectrums < 135, 88.7, 73)       #meV, STD
    dE1_lorz = np.where(spectrums < 135, 40.3, 24)  #meV, HFHM
    dTOF = np.repeat(0.37, L)      #us
    dTheta = np.repeat(0.016, L)   #rad
    dL0 = np.repeat(0.021, L)      #meters
    dL1 = np.repeat(0.023, L)      #meters
    
    resolutionPars = np.vstack((dE1, dTOF, dTheta, dL0, dL1, dE1_lorz)).transpose() 
    return resolutionPars 


def calculateKinematicsArrays(dataX, instrPars):          
    """Kinematics quantities calculated from TOF data"""   

    mN, Ef, en_to_vel, vf, hbar = loadConstants()    
    det, plick, angle, T0, L0, L1 = np.hsplit(instrPars, 6)     #each is of len(dataX)
    t_us = dataX - T0                                           #T0 is electronic delay due to instruments
    v0 = vf * L0 / ( vf * t_us - L1 )
    E0 =  np.square( v0 / en_to_vel )            #en_to_vel is a factor used to easily change velocity to energy and vice-versa
    
    delta_E = E0 - Ef  
    delta_Q2 = 2. * mN / hbar**2 * ( E0 + Ef - 2. * np.sqrt(E0*Ef) * np.cos(angle/180.*np.pi) )
    delta_Q = np.sqrt( delta_Q2 )
    return v0, E0, delta_E, delta_Q              #shape(no of spectrums, no of bins)


def reshapeArrayPerSpectrum(A):
    """Exchanges the first two indices of an array A,
    ao rearranges array to match iteration per spectrum of main fitting map()
    """
    return np.stack(np.split(A, len(A), axis=0), axis=2)[0]


def convertDataXToYSpacesForEachMass(dataX, masses, delta_Q, delta_E):
    "Calculates y spaces from TOF data, each row corresponds to one mass" 
    
    #prepare arrays to broadcast
    dataX = dataX[np.newaxis, :, :]
    delta_Q = delta_Q[np.newaxis, :, :]
    delta_E = delta_E[np.newaxis, :, :]  

    mN, Ef, en_to_vel, vf, hbar = loadConstants()
    masses = masses.reshape(masses.size, 1, 1)

    energyRecoil = np.square( hbar * delta_Q ) / 2. / masses              
    ySpacesForEachMass = masses / hbar**2 /delta_Q * (delta_E - energyRecoil)    #y-scaling  
    return ySpacesForEachMass


def fitNcpToSingleSpec(dataY, dataE, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays, ic):
    """Fits the NCP and returns the best fit parameters for one spectrum"""

    if np.all(dataY == 0) : 
        return np.zeros(len(ic.initPars)+3)  
    
    scaledPars = ic.initPars * ic.scalingFactors
    scaledBounds = ic.bounds * ic.scalingFactors[:, np.newaxis]

    result = optimize.minimize(
        errorFunction, 
        scaledPars, 
        args=(dataY, dataE, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays, ic),
        method='SLSQP', 
        bounds = scaledBounds, 
        constraints=ic.constraints
        )

    fitScaledPars = result["x"]
    fitPars = fitScaledPars / ic.scalingFactors

    noDegreesOfFreedom = len(dataY) - len(fitPars)
    specFitPars = np.append(instrPars[0], fitPars)
    return np.append(specFitPars, [result["fun"] / noDegreesOfFreedom, result["nit"]])


def errorFunction(scaledPars, dataY, dataE, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays, ic):
    """Error function to be minimized, operates in TOF space"""

    unscaledPars = scaledPars / ic.scalingFactors
    ncpForEachMass, ncpTotal = calculateNcpSpec(ic, unscaledPars, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays)

    if np.all(dataE == 0) | np.all(np.isnan(dataE)):
        # This condition is not usually satisfied but in the exceptional case that it is,
        # we can use a statistical weight to make sure the chi2 used is not too small for the 
        # optimization algorithm. This is not used in the original script
        chi2 = (ncpTotal - dataY)**2 / dataY**2
    else:
        chi2 =  (ncpTotal - dataY)**2 / dataE**2    
    return np.sum(chi2)


def calculateNcpSpec(ic, unscaledPars, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays):    
    """Creates a synthetic C(t) to be fitted to TOF values of a single spectrum, from J(y) and resolution functions
       Shapes: datax (1, n), ySpacesForEachMass (4, n), res (4, 2), deltaQ (1, n), E0 (1,n),
       where n is no of bins"""
    
    masses, intensities, widths, centers = prepareArraysFromPars(ic, unscaledPars) 
    v0, E0, deltaE, deltaQ = kinematicArrays
    
    gaussRes, lorzRes = caculateResolutionForEachMass(
        masses, ySpacesForEachMass, centers, resolutionPars, instrPars, kinematicArrays
        )
    totalGaussWidth = np.sqrt(widths**2 + gaussRes**2)                 
    
    JOfY = pseudoVoigt(ySpacesForEachMass - centers, totalGaussWidth, lorzRes)  
    
    FSE =  - numericalThirdDerivative(ySpacesForEachMass, JOfY) * widths**4 / deltaQ * 0.72 
    
    ncpForEachMass = intensities * (JOfY + FSE) * E0 * E0**(-0.92) * masses / deltaQ   
    ncpTotal = np.sum(ncpForEachMass, axis=0)
    return ncpForEachMass, ncpTotal


def prepareArraysFromPars(ic, initPars):
    """Extracts the intensities, widths and centers from the fitting parameters
        Reshapes all of the arrays to collumns, for the calculation of the ncp,"""

    masses = ic.masses[:, np.newaxis]    
    intensities = initPars[::3].reshape(masses.shape)
    widths = initPars[1::3].reshape(masses.shape)
    centers = initPars[2::3].reshape(masses.shape)  
    return masses, intensities, widths, centers 


def caculateResolutionForEachMass(masses, ySpacesForEachMass, centers, resolutionPars, instrPars, kinematicArrays):    
    """Calculates the gaussian and lorentzian resolution
    output: two column vectors, each row corresponds to each mass"""
    
    v0, E0, delta_E, delta_Q = kinematicsAtYCenters(ySpacesForEachMass, centers, kinematicArrays)
    
    gaussianResWidth = calcGaussianResolution(masses, v0, E0, delta_E, delta_Q, resolutionPars, instrPars)
    lorentzianResWidth = calcLorentzianResolution(masses, v0, E0, delta_E, delta_Q, resolutionPars, instrPars)
    return gaussianResWidth, lorentzianResWidth


def kinematicsAtYCenters(ySpacesForEachMass, centers, kinematicArrays):
    """v0, E0, deltaE, deltaQ at the peak of the ncpTotal for each mass"""

    shapeOfArrays = centers.shape
    proximityToYCenters = np.abs(ySpacesForEachMass - centers)
    yClosestToCenters = proximityToYCenters.min(axis=1).reshape(shapeOfArrays)
    yCentersMask = proximityToYCenters == yClosestToCenters

    v0, E0, deltaE, deltaQ = kinematicArrays

    # Expand arrays to match shape of yCentersMask
    v0 = v0 * np.ones(shapeOfArrays)
    E0 = E0 * np.ones(shapeOfArrays)
    deltaE = deltaE * np.ones(shapeOfArrays)
    deltaQ = deltaQ * np.ones(shapeOfArrays)

    v0 = v0[yCentersMask].reshape(shapeOfArrays)
    E0 = E0[yCentersMask].reshape(shapeOfArrays)
    deltaE = deltaE[yCentersMask].reshape(shapeOfArrays)
    deltaQ = deltaQ[yCentersMask].reshape(shapeOfArrays)
    return v0, E0, deltaE, deltaQ


def calcGaussianResolution(masses, v0, E0, delta_E, delta_Q, resolutionPars, instrPars):
    # Currently the function that takes the most time in the fitting
    assert masses.shape == (masses.size, 1), f"masses.shape: {masses.shape}. The shape of the masses array needs to be a collumn!"

    det, plick, angle, T0, L0, L1 = instrPars
    dE1, dTOF, dTheta, dL0, dL1, dE1_lorz = resolutionPars
    mN, Ef, en_to_vel, vf, hbar = loadConstants()

    angle = angle * np.pi/180

    dWdE1 = 1. + (E0 / Ef)**1.5 * (L1 / L0)
    dWdTOF = 2. * E0 * v0 / L0
    dWdL1 = 2. * E0**1.5 / Ef**0.5 / L0
    dWdL0 = 2. * E0 / L0

    dW2 = dWdE1**2*dE1**2 + dWdTOF**2*dTOF**2 + dWdL1**2*dL1**2 + dWdL0**2*dL0**2
    # conversion from meV^2 to A^-2, dydW = (M/q)^2
    dW2 *= (masses / hbar**2 / delta_Q)**2

    dQdE1 = 1. - (E0 / Ef)**1.5 * L1/L0 - np.cos(angle) * ((E0 / Ef)**0.5 - L1/L0 * E0/Ef)
    dQdTOF = 2.*E0 * v0/L0
    dQdL1 = 2.*E0**1.5 / L0 / Ef**0.5
    dQdL0 = 2.*E0 / L0
    dQdTheta = 2. * np.sqrt(E0 * Ef) * np.sin(angle)

    dQ2 = dQdE1**2*dE1**2 + (dQdTOF**2*dTOF**2 + dQdL1**2*dL1**2 + dQdL0 **
                             2*dL0**2)*np.abs(Ef/E0*np.cos(angle)-1) + dQdTheta**2*dTheta**2
    dQ2 *= (mN / hbar**2 / delta_Q)**2

    # in A-1    #same as dy^2 = (dy/dw)^2*dw^2 + (dy/dq)^2*dq^2
    gaussianResWidth = np.sqrt(dW2 + dQ2)
    return gaussianResWidth


def calcLorentzianResolution(masses, v0, E0, delta_E, delta_Q, resolutionPars, instrPars):
    assert masses.shape == (masses.size, 1), "The shape of the masses array needs to be a collumn!"
        
    det, plick, angle, T0, L0, L1 = instrPars
    dE1, dTOF, dTheta, dL0, dL1, dE1_lorz = resolutionPars
    mN, Ef, en_to_vel, vf, hbar = loadConstants()

    angle = angle * np.pi / 180

    dWdE1_lor = (1. + (E0/Ef)**1.5 * (L1/L0))**2
    # conversion from meV^2 to A^-2
    dWdE1_lor *= (masses / hbar**2 / delta_Q)**2

    dQdE1_lor = (1. - (E0/Ef)**1.5 * L1/L0 - np.cos(angle)
                 * ((E0/Ef)**0.5 + L1/L0 * E0/Ef))**2
    dQdE1_lor *= (mN / hbar**2 / delta_Q)**2

    lorentzianResWidth = np.sqrt(dWdE1_lor + dQdE1_lor) * dE1_lorz   # in A-1
    return lorentzianResWidth


def loadConstants():
    """Output: the mass of the neutron, final energy of neutrons (selected by gold foil),
    factor to change energies into velocities, final velocity of neutron and hbar"""
    mN=1.008    #a.m.u.
    Ef=4906.         # meV
    en_to_vel = 4.3737 * 1.e-4
    vf = np.sqrt(Ef) * en_to_vel  # m/us
    hbar = 2.0445
    return mN, Ef, en_to_vel, vf, hbar


def pseudoVoigt(x, sigma, gamma):
    """Convolution between Gaussian with std sigma and Lorentzian with HWHM gamma"""
    fg, fl = 2.*sigma*np.sqrt(2.*np.log(2.)), 2.*gamma
    f = 0.5346 * fl + np.sqrt(0.2166*fl**2 + fg**2)
    eta = 1.36603 * fl/f - 0.47719 * (fl/f)**2 + 0.11116 * (fl/f)**3
    sigma_v, gamma_v = f/(2.*np.sqrt(2.*np.log(2.))), f / 2.
    pseudo_voigt = eta * \
        lorentizian(x, gamma_v) + (1.-eta) * gaussian(x, sigma_v)
    norm = np.sum(pseudo_voigt)*(x[1]-x[0])
    return pseudo_voigt  # /np.abs(norm)


def gaussian(x, sigma):
    """Gaussian function centered at zero"""
    gaussian = np.exp(-x**2/2/sigma**2)
    gaussian /= np.sqrt(2.*np.pi)*sigma
    return gaussian


def lorentizian(x, gamma):
    """Lorentzian centered at zero"""
    lorentzian = gamma/np.pi / (x**2 + gamma**2)
    return lorentzian


def numericalThirdDerivative(x, fun):
    k6 = (- fun[:, 12:] + fun[:, :-12]) * 1
    k5 = (+ fun[:, 11:-1] - fun[:, 1:-11]) * 24
    k4 = (- fun[:, 10:-2] + fun[:, 2:-10]) * 192
    k3 = (+ fun[:,  9:-3] - fun[:, 3:-9]) * 488
    k2 = (+ fun[:,  8:-4] - fun[:, 4:-8]) * 387
    k1 = (- fun[:,  7:-5] + fun[:, 5:-7]) * 1584

    dev = k1 + k2 + k3 + k4 + k5 + k6
    dev /= np.power(x[:, 7:-5] - x[:, 6:-6], 3)
    dev /= 12**3

    derivative = np.zeros(fun.shape)
    derivative[:, 6:-6] = dev
    # Padded with zeros left and right to return array with same shape
    return derivative


def buildNcpFromSpec(initPars, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays, ic):
    """input: all row shape
       output: row shape with the ncpTotal for each mass"""

    if np.all(initPars==0):  
        return np.zeros(ySpacesForEachMass.shape) 
    
    ncpForEachMass, ncpTotal = calculateNcpSpec(ic, initPars, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays)        
    return ncpForEachMass


def createNcpWorkspaces(ncpForEachMass, ncpTotal, ws):
    """Creates workspaces from ncp array data"""

    # Need to rearrage array of yspaces into seperate arrays for each mass
    ncpForEachMass = switchFirstTwoAxis(ncpForEachMass)
    dataX = ws.extractX()
    dataE = np.zeros(dataX.shape)

    # Original script does not have this step to multiply by histogram widths
    histWidths = dataX[:, 1:] - dataX[:, :-1]
    dataX = dataX[:, :-1]       # Cut last column to match ncpTotal length
    dataY = ncpTotal * histWidths
    dataE = dataE[:, :-1] * histWidths

    ncpTotWs = CreateWorkspace(DataX=dataX.flatten(), DataY=dataY.flatten(), DataE=dataE.flatten(),
                     Nspec=len(dataX), OutputWorkspace=ws.name()+"_tof_fitted_profiles")
    SumSpectra(InputWorkspace=ncpTotWs, OutputWorkspace=ncpTotWs.name()+"_sum" )

    for i, ncp_m in enumerate(ncpForEachMass):
        ncpMWs = CreateWorkspace(DataX=dataX.flatten(), DataY=ncp_m.flatten(), Nspec=len(dataX),
                        OutputWorkspace=ws.name()+"_tof_fitted_profile_"+str(i+1))
        SumSpectra(InputWorkspace=ncpMWs, OutputWorkspace=ncpMWs.name()+"_sum" )



def switchFirstTwoAxis(A):
    """Exchanges the first two indices of an array A,
    rearranges matrices per spectrum for iteration of main fitting procedure
    """
    return np.stack(np.split(A, len(A), axis=0), axis=2)[0]


def createWorkspacesForMSCorrection(ic, meanWidths, meanIntensityRatios):
    """Creates _MulScattering and _TotScattering workspaces used for the MS correction"""

    sampleProperties = calcMSCorrectionSampleProperties(ic, meanWidths, meanIntensityRatios)
    print("\nThe sample properties for Multiple Scattering correction are:\n\n", 
            sampleProperties, "\n")
    createMulScatWorkspaces(ic, ic.name, sampleProperties)


def calcMSCorrectionSampleProperties(ic, meanWidths, meanIntensityRatios):
    masses = ic.masses.flatten()

    # If H not present ie backward scattering, add it to sample properties
    if (ic.modeRunning == "BACKWARD") and ic.hydrogen_peak:   
        masses = np.append(masses, 1.0079)
        meanWidths = np.append(meanWidths, 5.0)
        HIntensity = ic.HToMass0Ratio * meanIntensityRatios[0]
        meanIntensityRatios = np.append(meanIntensityRatios, HIntensity)
        meanIntensityRatios /= np.sum(meanIntensityRatios)

    MSProperties = np.zeros(3*len(masses))
    MSProperties[::3] = masses
    MSProperties[1::3] = meanIntensityRatios
    MSProperties[2::3] = meanWidths
    sampleProperties = list(MSProperties)   

    return sampleProperties


def createMulScatWorkspaces(ic, wsName, sampleProperties):
    """Uses the Mantid algorithm for the MS correction to create two Workspaces _TotScattering and _MulScattering"""

    print("\nEvaluating the Multiple Scattering Correction...\n")
    # selects only the masses, every 3 numbers
    MS_masses = sampleProperties[::3]
    # same as above, but starts at first intensities
    MS_amplitudes = sampleProperties[1::3]

    dens, trans = VesuvioThickness(
        Masses=MS_masses, Amplitudes=MS_amplitudes, TransmissionGuess=ic.transmission_guess, Thickness=0.1
        )

    _TotScattering, _MulScattering = VesuvioCalculateMS(
        wsName, 
        NoOfMasses=len(MS_masses), 
        SampleDensity=dens.cell(9, 1),
        AtomicProperties=sampleProperties, 
        BeamRadius=2.5,
        NumScatters=ic.multiple_scattering_order,
        NumEventsPerRun=int(ic.number_of_events)
        )

    data_normalisation = Integration(wsName)
    simulation_normalisation = Integration("_TotScattering")
    for workspace in ("_MulScattering", "_TotScattering"):
        Divide(LHSWorkspace=workspace, RHSWorkspace=simulation_normalisation, 
               OutputWorkspace=workspace)
        Multiply(LHSWorkspace=workspace, RHSWorkspace=data_normalisation, 
                 OutputWorkspace=workspace)
        RenameWorkspace(InputWorkspace=workspace,
                        OutputWorkspace=str(wsName)+workspace)
    DeleteWorkspaces(
        [data_normalisation, simulation_normalisation, trans, dens]
        )
  # The only remaining workspaces are the _MulScattering and _TotScattering


def createWorkspacesForGammaCorrection(ic, meanWidths, meanIntensityRatios):
    """Creates _gamma_background correction workspace to be subtracted from the main workspace"""

    # I do not know why, but setting these instrument parameters is required
    SetInstrumentParameter(ic.name, ParameterName='hwhm_lorentz', 
                            ParameterType='Number', Value='24.0')
    SetInstrumentParameter(ic.name, ParameterName='sigma_gauss', 
                            ParameterType='Number', Value='73.0')

    profiles = calcGammaCorrectionProfiles(ic.masses, meanWidths, meanIntensityRatios)
    background, corrected = VesuvioCalculateGammaBackground(
        InputWorkspace=ic.name, ComptonFunction=profiles
        )
    RenameWorkspace(InputWorkspace= background, OutputWorkspace = ic.name+"_gamma_background")
    Scale(InputWorkspace = ic.name+"_gamma_background", OutputWorkspace = ic.name+"_gamma_background", 
        Factor=0.9, Operation="Multiply")

    DeleteWorkspace(corrected)


def calcGammaCorrectionProfiles(masses, meanWidths, meanIntensityRatios):
    masses = masses.flatten()
    profiles = ""
    for mass, width, intensity in zip(masses, meanWidths, meanIntensityRatios):
        profiles += "name=GaussianComptonProfile,Mass="   \
                    + str(mass) + ",Width=" + str(width)  \
                    + ",Intensity=" + str(intensity) + ';'
    print("\n The sample properties for Gamma Correction are:\n",
            profiles)
    return profiles


# def fitInYSpaceProcedure(ic, wsFinal, fittingResults):
#     ncpForEachMass = fittingResults.all_ncp_for_each_mass[-1]  # Select last iteration
#     wsYSpaceSymSum, wsRes = isolateHProfileInYSpace(ic, wsFinal, ncpForEachMass)
#     popt, perr = fitTheHProfileInYSpace(ic, wsYSpaceSymSum, wsRes)
#     wsH = mtd[wsFinal.name()+"_H"]

#     fittingResults.storeResultsOfYSpaceFit(wsFinal, wsH, wsYSpaceSymSum, wsRes, popt, perr)
#     fittingResults.printYSpaceFitResults(ic)


# def isolateHProfileInYSpace(ic, wsFinal, ncpForEachMass):
#     massH = 1.0079
#     wsRes = calculateMantidResolution(ic, wsFinal, massH)  

#     wsSubMass = subtractAllMassesExceptFirst(ic, wsFinal, ncpForEachMass)
#     averagedSpectraYSpace = averageJOfYOverAllSpectra(ic, wsSubMass, massH) 
#     return averagedSpectraYSpace, wsRes


# def calculateMantidResolution(ic, ws, mass):
#     rebinPars=ic.rebinParametersForYSpaceFit
#     for index in range(ws.getNumberHistograms()):
#         if np.all(ws.dataY(index)[:] == 0):  # Ignore masked spectra
#             pass
#         else:
#             VesuvioResolution(Workspace=ws,WorkspaceIndex=index,Mass=mass,OutputWorkspaceYSpace="tmp")
#             Rebin(InputWorkspace="tmp", Params=rebinPars, OutputWorkspace="tmp")

#             if index == 0:   # Ensures that workspace has desired units
#                 RenameWorkspace("tmp","resolution")
#             else:
#                 AppendSpectra("resolution", "tmp", OutputWorkspace= "resolution")

#     try:
#         SumSpectra(InputWorkspace="resolution",OutputWorkspace="resolution")
#     except ValueError:
#         raise ValueError ("All the rows from the workspace to be fitted are Nan!")

#     normalise_workspace("resolution")
#     DeleteWorkspace("tmp")
#     return mtd["resolution"]

    
# def normalise_workspace(ws_name):
#     tmp_norm = Integration(ws_name)
#     Divide(LHSWorkspace=ws_name,RHSWorkspace=tmp_norm,OutputWorkspace=ws_name)
#     DeleteWorkspace("tmp_norm")


# def subtractAllMassesExceptFirst(ic, ws, ncpForEachMass):
#     """Input: workspace from last iteration, ncpTotal for each mass
#        Output: workspace with all the ncpTotal subtracted except for the first mass"""

#     ncpForEachMass = switchFirstTwoAxis(ncpForEachMass)
#     # Select all masses other than the first one
#     ncpForEachMass = ncpForEachMass[1:, :, :]
#     # Sum the ncpTotal for remaining masses
#     ncpTotal = np.sum(ncpForEachMass, axis=0)

#     dataY, dataX = ws.extractY(), ws.extractX() 
    
#     # Subtract the ncp of all masses exept first to dataY
#     dataY[:, :-1] -= ncpTotal * (dataX[:, 1:] - dataX[:, :-1])

#     # Pass the data onto a Workspace, clone to preserve properties
#     wsSubMass = CloneWorkspace(InputWorkspace=ws, OutputWorkspace=ws.name()+"_H")
#     for i in range(wsSubMass.getNumberHistograms()):  # Keeps the faulty last column
#         wsSubMass.dataY(i)[:] = dataY[i, :]

#      # Mask spectra again, to be seen as masked from Mantid's perspective
#     MaskDetectors(Workspace=wsSubMass, WorkspaceIndexList=ic.maskedDetectorIdx)  

#     if np.any(np.isnan(mtd[ws.name()+"_H"].extractY())):
#         raise ValueError("The workspace for the isolated H data countains NaNs, \
#                             might cause problems!")
#     return wsSubMass


# def averageJOfYOverAllSpectra(ic, ws0, mass):
#     wsYSpace = convertToYSpace(ic, ws0, mass)
#     averagedSpectraYSpace = weightedAvg(wsYSpace)
    
#     if ic.symmetrisationFlag == True:
#         symAvgdSpecYSpace = symmetrizeWs(ic, averagedSpectraYSpace)
#         return symAvgdSpecYSpace

#     return averagedSpectraYSpace


# def convertToYSpace(ic, ws0, mass):
#     ConvertToYSpace(
#         InputWorkspace=ws0, Mass=mass, 
#         OutputWorkspace=ws0.name()+"_JoY", QWorkspace=ws0.name()+"_Q"
#         )
#     rebinPars=ic.rebinParametersForYSpaceFit
#     Rebin(
#         InputWorkspace=ws0.name()+"_JoY", Params=rebinPars, 
#         FullBinsOnly=True, OutputWorkspace=ws0.name()+"_JoY"
#         )
#     normalise_workspace(ws0.name()+"_JoY")
#     return mtd[ws0.name()+"_JoY"]


# def weightedAvg(wsYSpace):
#     dataY = wsYSpace.extractY()
#     dataE = wsYSpace.extractE()

#     dataY[dataY==0] = np.nan
#     dataE[dataE==0] = np.nan

#     meanY = np.nansum(dataY/np.square(dataE), axis=0) / np.nansum(1/np.square(dataE), axis=0)
#     meanE = np.sqrt(1 / np.nansum(1/np.square(dataE), axis=0))

#     tempWs = SumSpectra(wsYSpace)
#     newWs = CloneWorkspace(tempWs, OutputWorkspace=wsYSpace.name()+"_weighted_avg")
#     newWs.dataY(0)[:] = meanY
#     newWs.dataE(0)[:] = meanE
#     DeleteWorkspace(tempWs)
#     return newWs


# def symmetrizeWs(ic, avgYSpace):
#     """Symmetrizes workspace with only one spectrum,
#        Needs to have symmetric binning"""

#     dataX = avgYSpace.extractX()
#     dataY = avgYSpace.extractY()
#     dataE = avgYSpace.extractE()

#     yFlip = np.flip(dataY)
#     eFlip = np.flip(dataE)

#     if ic.symmetriseHProfileUsingAveragesFlag:
#         # Inverse variance weighting
#         dataYSym = (dataY/dataE**2 + yFlip/eFlip**2) / (1/dataE**2 + 1/eFlip**2)
#         dataESym = 1 / np.sqrt(1/dataE**2 + 1/eFlip**2)
#     else:
#         # Mirroring positive values from negative ones
#         dataYSym = np.where(dataX>0, yFlip, dataY)
#         dataESym = np.where(dataX>0, eFlip, dataE)

#     Sym = CloneWorkspace(avgYSpace, OutputWorkspace=avgYSpace.name()+"_symmetrised")
#     Sym.dataY(0)[:] = dataYSym
#     Sym.dataE(0)[:] = dataESym
#     return Sym


# def fitTheHProfileInYSpace(ic, wsYSpaceSym, wsRes):
#     # if ic.useScipyCurveFitToHProfileFlag:
#     poptCurveFit, pcovCurveFit = fitProfileCurveFit(ic, wsYSpaceSym, wsRes)
#     perrCurveFit = np.sqrt(np.diag(pcovCurveFit))
#     # else:
#     poptMantidFit, perrMantidFit = fitProfileMantidFit(ic, wsYSpaceSym, wsRes)
    
#     #TODO: Add the Cost function as the last parameter
#     poptCurveFit = np.append(poptCurveFit, np.nan)
#     perrCurveFit = np.append(perrCurveFit, np.nan)

#     popt = np.vstack((poptCurveFit, poptMantidFit))
#     perr = np.vstack((perrCurveFit, perrMantidFit))

#     return popt, perr


# def fitProfileCurveFit(ic, wsYSpaceSym, wsRes):
#     res = wsRes.extractY()[0]
#     resX = wsRes. extractX()[0]

#     # Interpolate Resolution to get single peak at zero
#     # Otherwise if the resolution has two data points at the peak,
#     # the convolution will be skewed.
#     start, interval, end = [float(i) for i in ic.rebinParametersForYSpaceFit.split(",")]
#     resNewX = np.arange(start, end, interval)
#     res = np.interp(resNewX, resX, res)

#     dataY = wsYSpaceSym.extractY()[0]
#     dataX = wsYSpaceSym.extractX()[0]
#     dataE = wsYSpaceSym.extractE()[0]

#     if ic.singleGaussFitToHProfile:
#         def convolvedFunction(x, y0, A, x0, sigma):
#             histWidths = x[1:] - x[:-1]
#             if ~ (np.max(histWidths)==np.min(histWidths)):
#                 raise AssertionError("The histograms widhts need to be the same for the discrete convolution to work!")

#             gaussFunc = gaussianFit(x, y0, x0, A, sigma)
#             convGauss = ndimage.convolve1d(gaussFunc, res, mode="constant") * histWidths[0]  
#             return convGauss
#         p0 = [0, 1, 0, 5]
#         bounds = [-np.inf, np.inf]  # Applied to all parameters

#     else:
#         # # Double Gaussian
#         # def convolvedFunction(x, y0, x0, A, sigma):
#         #     histWidths = x[1:] - x[:-1]
#         #     if ~ (np.max(histWidths)==np.min(histWidths)):
#         #         raise AssertionError("The histograms widhts need to be the same for the discrete convolution to work!")

#         #     gaussFunc = gaussianFit(x, y0, x0, A, 4.76) + gaussianFit(x, 0, x0, 0.054*A, sigma)
#         #     convGauss = ndimage.convolve1d(gaussFunc, res, mode="constant") * histWidths[0]
#         #     return convGauss
#         # p0 = [0, 0, 0.7143, 5]

#         def HermitePolynomial(x, sigma1, c4, c6):
#             return np.exp(-x**2/2/sigma1**2) / (np.sqrt(2*3.1415*sigma1**2)) \
#                     *(1 + c4/32*(16*(x/np.sqrt(2)/sigma1)**4 \
#                     -48*(x/np.sqrt(2)/sigma1)**2+12) \
#                     +c6/384*(64*(x/np.sqrt(2)/sigma1)**6 \
#                     -480*(x/np.sqrt(2)/sigma1)**4 + 720*(x/np.sqrt(2)/sigma1)**2 - 120))
        
#         def convolvedFunction(x, sigma1, c4, c6):
#             histWidths = x[1:] - x[:-1]
#             if ~ (np.max(histWidths)==np.min(histWidths)):
#                 raise AssertionError("The histograms widhts need to be the same for the discrete convolution to work!")

#             hermiteFunc = HermitePolynomial(x, sigma1, c4, c6)
#             convFunc = ndimage.convolve1d(hermiteFunc, res, mode="constant") * histWidths[0]
#             return convFunc
#         p0 = [4, 0, 0]     
#         # The bounds on curve_fit() are set up diferently than on minimize()
#         bounds = [[-np.inf, 0, 0], [np.inf, np.inf, np.inf]] 


#     popt, pcov = optimize.curve_fit(
#         convolvedFunction, 
#         dataX, 
#         dataY, 
#         p0=p0,
#         sigma=dataE,
#         bounds=bounds
#     )
#     yfit = convolvedFunction(dataX, *popt)
#     Residuals = dataY - yfit
    
#     # Create Workspace with the fit results
#     # TODO add DataE 
#     CreateWorkspace(DataX=np.concatenate((dataX, dataX, dataX)), 
#                     DataY=np.concatenate((dataY, yfit, Residuals)), 
#                     NSpec=3,
#                     OutputWorkspace=wsYSpaceSym.name()+"_fitted_CurveFit")
#     return popt, pcov


# def gaussianFit(x, y0, x0, A, sigma):
#     """Gaussian centered at zero"""
#     return y0 + A / (2*np.pi)**0.5 / sigma * np.exp(-(x-x0)**2/2/sigma**2)


# def fitProfileMantidFit(ic, wsYSpaceSym, wsRes):

#     if ic.singleGaussFitToHProfile:
#         popt, perr = np.zeros((2, 5)), np.zeros((2, 5))
#     else:
#         # popt, perr = np.zeros((2, 6)), np.zeros((2, 6))
#         popt, perr = np.zeros((2, 4)), np.zeros((2, 4))


#     print('\n','Fitting on the sum of spectra in the West domain ...','\n')     
#     for i, minimizer in enumerate(['Levenberg-Marquardt','Simplex']):
#         outputName = wsYSpaceSym.name()+"_fitted_"+minimizer
#         CloneWorkspace(InputWorkspace = wsYSpaceSym, OutputWorkspace = outputName)
        
#         if ic.singleGaussFitToHProfile:
#             function='''composite=Convolution,FixResolution=true,NumDeriv=true;
#             name=Resolution,Workspace=resolution,WorkspaceIndex=0;
#             name=UserFunction,Formula=y0+A*exp( -(x-x0)^2/2/sigma^2)/(2*3.1415*sigma^2)^0.5,
#             y0=0,A=1,x0=0,sigma=5,   ties=()'''
#         else:
#             # # Function for Double Gaussian
#             # function='''composite=Convolution,FixResolution=true,NumDeriv=true;
#             # name=Resolution,Workspace=resolution,WorkspaceIndex=0;
#             # name=UserFunction,Formula=y0+A*exp( -(x-x0)^2/2/sigma1^2)/(2*3.1415*sigma1^2)^0.5
#             # +A*0.054*exp( -(x-x0)^2/2/sigma2^2)/(2*3.1415*sigma2^2)^0.5,
#             # y0=0,x0=0,A=0.7143,sigma1=4.76, sigma2=5,   ties=(sigma1=4.76)'''
            
#             # TODO: Check that this function is correct
#             function = """
#             composite=Convolution,FixResolution=true,NumDeriv=true;
#             name=Resolution,Workspace=resolution,WorkspaceIndex=0,X=(),Y=();
#             name=UserFunction,Formula=exp( -x^2/2./sigma1^2)/(sqrt(2.*3.1415*sigma1^2))
#             *(1.+c4/32.*(16.*(x/sqrt(2)/sigma1)^4-48.*(x/sqrt(2)/sigma1)^2+12)+c6/384*(64*(x/sqrt(2)/sigma1)^6 - 480*(x/sqrt(2)/sigma1)^4 + 720*(x/sqrt(2)/sigma1)^2 - 120)),
#             sigma1=4.0,c4=0.0,c6=0.0,ties=(),constraints=(0<c4,0<c6)
#             """

#         Fit(
#             Function=function, 
#             InputWorkspace=outputName,
#             Output=outputName,
#             Minimizer=minimizer
#             )
        
#         ws=mtd[outputName+"_Parameters"]
#         popt[i] = ws.column("Value")
#         perr[i] = ws.column("Error")
#     return popt, perr


# start_time = time.time()
# runSequenceRatioNotKnown()
# runSequenceForKnownRatio()
# runOnlyBackScattering()
# runOnlyForwardScattering()

# end_time = time.time()
# print("\nRunning time: ", end_time-start_time, " seconds")

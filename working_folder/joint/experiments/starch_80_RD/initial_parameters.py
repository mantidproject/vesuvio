from ...core_functions.analysis_functions import iterativeFitForDataReduction
from ...core_functions.fit_in_yspace import fitInYSpaceProcedure
from mantid.api import AnalysisDataService, mtd
import time
import numpy as np
from pathlib import Path
experimentPath = Path(__file__).absolute().parent  # Path to the repository



testingCleaning = True
if testingCleaning:     
    cleaningPath = experimentPath / "output" / "testing" / "cleaning"


    forwardScatteringSavePath = cleaningPath / "current_forward.npz" 
    backScatteringSavePath = cleaningPath / "current_backward.npz" 
    ySpaceFitSavePath = cleaningPath / "current_yspacefit.npz"

else:
    outputPath = experimentPath / "output" / "testing" / "original" / "current_data"

    forwardScatteringSavePath = outputPath / "4iter_forward_GM_MS.npz"
    backScatteringSavePath = outputPath / "4iter_backward_MS.npz"
    ySpaceFitSavePath = outputPath / "4iter_yspacefit.npz"


ipFilePath =  experimentPath / "ip2018_3.par"  

inputWsPath = experimentPath / "input_ws"

# Default in case of no DoubleDifference
# TODO: Sort out Double difference in a more elegant manner
frontWsEmptyPath = None
backWsEmptyPath = None
for wsPath in inputWsPath.iterdir():
    keywords = wsPath.name.split(".")[0].split("_")
    if "raw" in keywords and "backward" in keywords:
        backWsRawPath = wsPath 
    if "empty" in keywords and "backward" in keywords:
        backWsEmptyPath = wsPath 
    if "raw" in keywords and "forward" in keywords:
        frontWsRawPath = wsPath
    if "empty" in keywords and "forward" in keywords:
        frontWsEmptyPath = wsPath 


class BackwardInitialConditions:
    # Multiscaterring Correction Parameters
    HToMass0Ratio = 19.0620008206

    resultsSavePath = backScatteringSavePath

    transmission_guess =  0.8537        # Experimental value from VesuvioTransmission
    multiple_scattering_order, number_of_events = 2, 1.e5   
    hydrogen_peak = True                 # Hydrogen multiple scattering
    
    # Sample slab parameters
    vertical_width, horizontal_width, thickness = 0.1, 0.1, 0.001  # Expressed in meters

    modeRunning = "BACKWARD"

    # Parameters to Load Raw and Empty Workspaces
    userWsRawPath = str(backWsRawPath)
    userWsEmptyPath = str(backWsEmptyPath)
    name = "starch_80_RD_backward_"
    runs='43066-43076'  # 77K             # The numbers of the runs to be analysed
    empty_runs='41876-41923'   # 77K             # The numbers of the empty runs to be subtracted
    spectra='3-134'                            # Spectra to be analysed
    tof_binning='275.,1.,420'                    # Binning of ToF spectra
    mode='DoubleDifference'
    # ipfile='./ip2019.par'

    # Masses, instrument parameters and initial fitting parameters
    masses = np.array([12, 16, 27])
    noOfMasses = len(masses)
    InstrParsPath = ipFilePath

    initPars = np.array([ 
    # Intensities, NCP widths, NCP centers   
            1, 12, 0.,    
            1, 12, 0.,   
            1, 12.5, 0.    
        ])
    bounds = np.array([
            [0, np.nan], [8, 16], [-3, 1],
            [0, np.nan], [8, 16], [-3, 1],
            [0, np.nan], [11, 14], [-3, 1]
        ])
    constraints = ()

    noOfMSIterations = 2     #4
    firstSpec = 3    #3
    lastSpec = 134    #134

    # Boolean Flags to control script
    # loadWsFromUserPathFlag = True
    scaleParsFlag = False
    MSCorrectionFlag = True
    GammaCorrectionFlag = False
    maskedSpecAllNo = np.array([18, 34, 42, 43, 59, 60, 62, 118, 119, 133])

    # Parameters below are not to be changed
    firstSpecIdx = 0
    lastSpecIdx = lastSpec - firstSpec

    # Consider only the masked spectra between first and last spectrum
    maskedSpecNo = maskedSpecAllNo[
        (maskedSpecAllNo >= firstSpec) & (maskedSpecAllNo <= lastSpec)
    ]
    maskedDetectorIdx = maskedSpecNo - firstSpec

    # Set scaling factors for the fitting parameters, default is ones
    scalingFactors = np.ones(initPars.shape)
    if scaleParsFlag:        # Scale fitting parameters using initial values
            initPars[2::3] = np.ones((1, noOfMasses))  # Main problem is that zeros have to be replaced by non zeros
            scalingFactors = 1 / initPars


class ForwardInitialConditions:

    resultsSavePath = forwardScatteringSavePath
    ySpaceFitSavePath = ySpaceFitSavePath

    HToMass0Ratio = None

    transmission_guess =  0.8537        # Experimental value from VesuvioTransmission
    multiple_scattering_order, number_of_events = 2, 1.e5   
    hydrogen_peak = True                 # Hydrogen multiple scattering
    
    # Sample slab parameters
    vertical_width, horizontal_width, thickness = 0.1, 0.1, 0.001  # Expressed in meters

    modeRunning = "FORWARD"  # Used to control MS correction

    userWsRawPath = str(frontWsRawPath)
    userWsEmptyPath = str(frontWsEmptyPath)

    name = "starch_80_RD_forward_"
    runs='43066-43076'         # 100K        # The numbers of the runs to be analysed
    empty_runs='43868-43911'   # 100K        # The numbers of the empty runs to be subtracted
    spectra='144-182'                        # Spectra to be analysed
    tof_binning="110,1.,430"                 # Binning of ToF spectra
    mode='SingleDifference'
    # ipfile='./ip2018_3.par'

    masses = np.array([1.0079, 12, 16, 27]) 
    noOfMasses = len(masses)
    InstrParsPath = ipFilePath

    initPars = np.array([ 
    # Intensities, NCP widths, NCP centers  
        1, 4.7, 0, 
        1, 12.71, 0.,    
        1, 8.76, 0.,   
        1, 13.897, 0.    
    ])
    bounds = np.array([
        [0, np.nan], [3, 6], [-3, 1],
        [0, np.nan], [12.71, 12.71], [-3, 1],
        [0, np.nan], [8.76, 8.76], [-3, 1],
        [0, np.nan], [13.897, 13.897], [-3, 1]
    ])
    constraints = ()

    noOfMSIterations = 2   #4
    firstSpec = 164   #144
    lastSpec = 175    #182

    # Boolean Flags to control script
    # loadWsFromUserPathFlag = True
    scaleParsFlag = False
    MSCorrectionFlag = True
    GammaCorrectionFlag = True

    # Parameters to control fit in Y-Space
    # symmetrisationFlag = True
    # symmetriseHProfileUsingAveragesFlag = True      # When False, use mirror sym
    # rebinParametersForYSpaceFit = "-20, 0.5, 20"    # Needs to be symetric
    # singleGaussFitToHProfile = True      # When False, use Hermite expansion
    maskedSpecAllNo = np.array([173, 174, 179])

    # Parameters below are not to be changed
    firstSpecIdx = 0
    lastSpecIdx = lastSpec - firstSpec

    # Consider only the masked spectra between first and last spectrum
    maskedSpecNo = maskedSpecAllNo[
        (maskedSpecAllNo >= firstSpec) & (maskedSpecAllNo <= lastSpec)
    ]
    maskedDetectorIdx = maskedSpecNo - firstSpec

    # Set scaling factors for the fitting parameters, default is ones
    scalingFactors = np.ones(initPars.shape)
    if scaleParsFlag:        # Scale fitting parameters using initial values
            initPars[2::3] = np.ones((1, noOfMasses))  # Main problem is that zeros have to be replaced by non zeros
            scalingFactors = 1 / initPars


# Make a child class for the parameters of yspace fitting
# This class inherits all of the atributes in ForwardInitialConditions
class YSpaceFitInitialConditions(ForwardInitialConditions):
    ySpaceFitSavePath = ySpaceFitSavePath

    symmetrisationFlag = True
    symmetriseHProfileUsingAveragesFlag = True      # When False, use mirror sym
    rebinParametersForYSpaceFit = "-20, 0.5, 20"    # Needs to be symetric
    singleGaussFitToHProfile = True      # When False, use Hermite expansion
    

bckwdIC = BackwardInitialConditions
fwdIC = ForwardInitialConditions
yfitIC = YSpaceFitInitialConditions


def runIndependentIterativeProcedure(IC):
    """Runs the iterative fitting of NCP.
    input: Backward or Forward scattering initial conditions object
    output: Final workspace that was fitted, object with results arrays"""

    AnalysisDataService.clear()
    wsFinal, ncpFitResultsObject = iterativeFitForDataReduction(IC)
    return wsFinal, ncpFitResultsObject


def runSequenceForKnownRatio(bckwdIC, fwdIC):
    AnalysisDataService.clear()
    # If H to first mass ratio is known, can run MS correction for backscattering
    # Back scattering produces precise results for widhts and intensity ratios for non-H masses
    wsFinal, backScatteringResults = iterativeFitForDataReduction(bckwdIC)
    setInitFwdParsFromBackResults(backScatteringResults, bckwdIC.HToMass0Ratio, fwdIC)
    wsFinal, forwardScatteringResults = iterativeFitForDataReduction(fwdIC)
    return wsFinal, forwardScatteringResults
    # fitInYSpaceProcedure(fwdIC, wsFinal, forwardScatteringResults.all_ncp_for_each_mass[-1])


def runSequenceRatioNotKnown(bckwdIC, fwdIC):
    # Run preliminary forward with a good guess for the widths of non-H masses
    wsFinal, forwardScatteringResults = iterativeFitForDataReduction(fwdIC)
    for i in range(2):    # Loop until convergence is achieved
        AnalysisDataService.clear()    # Clears all Workspaces
        # Get first estimate of H to mass0 ratio
        fwdMeanIntensityRatios = forwardScatteringResults.all_mean_intensities[-1] 
        bckwdIC.HToMass0Ratio = fwdMeanIntensityRatios[0] / fwdMeanIntensityRatios[1]
        # Run backward procedure with this estimate
        wsFinal, backScatteringResults = iterativeFitForDataReduction(bckwdIC)
        setInitFwdParsFromBackResults(backScatteringResults, bckwdIC.HToMass0Ratio, fwdIC)
        wsFinal, forwardScatteringResults = iterativeFitForDataReduction(fwdIC)
    # fitInYSpaceProcedure(fwdIC, wsFinal, forwardScatteringResults.all_ncp_for_each_mass[-1])
    return wsFinal, forwardScatteringResults
 


def setInitFwdParsFromBackResults(backScatteringResults, HToMass0Ratio, fwdIC):
    """Takes widths and intensity ratios obtained in backscattering
    and uses them as the initial conditions for forward scattering """

    # Get widts and intensity ratios from backscattering results
    backMeanWidths = backScatteringResults.all_mean_widths[-1]
    backMeanIntensityRatios = backScatteringResults.all_mean_intensities[-1] 

    if fwdIC.masses[0] == 1.0079:
        HIntensity = HToMass0Ratio * backMeanIntensityRatios[0]
        initialFwdIntensityRatios = np.append([HIntensity], backMeanIntensityRatios)
        initialFwdIntensityRatios /= np.sum(initialFwdIntensityRatios)
        # Set starting conditions for forward scattering
        # Fix known widths and intensity ratios from back scattering results
        fwdIC.initPars[4::3] = backMeanWidths
        fwdIC.bounds[4::3] = backMeanWidths[:, np.newaxis] * np.ones((1,2))
        fwdIC.initPars[0::3] = initialFwdIntensityRatios

    else:
        fwdIC.initPars[1::3] = backMeanWidths
        # First width is set to vary
        fwdIC.bounds[4::3] = backMeanWidths[1:][:, np.newaxis] * np.ones((1,2))
        fwdIC.initPars[0::3] = backMeanIntensityRatios

    print("\nChanged initial conditions of forward scattering according to mean widhts and intensity ratios from backscattering.\n")

start_time = time.time()
# Interactive section 

wsFinal, fwdResults = runIndependentIterativeProcedure(fwdIC)
# wsFinal = mtd["starch_80_RD_forward_1"]

# Choose whether to get ncp from the workspace or from results object
# Useful when running in Mantid, can select workspace to fit 
# by assigning it to wsFinal
ncpFromWs = True
if ncpFromWs:  
    allNCP = mtd[wsFinal.name()+"_tof_fitted_profile_1"].extractY()[np.newaxis, :, :]
    for i in range(2, fwdIC.noOfMasses+1):
        ncpToAppend = mtd[wsFinal.name()+"_tof_fitted_profile_" + str(i)].extractY()[np.newaxis, :, :]
        allNCP = np.append(allNCP, ncpToAppend, axis=0)
else:
    lastIterationNCP = fwdResults.all_ncp_for_each_mass[-1]
    allNCP = lastIterationNCP

def switchFirstTwoAxis(A):
    return np.stack(np.split(A, len(A), axis=0), axis=2)[0]

allNCP = switchFirstTwoAxis(allNCP)
print("\nShape of all NCP: ", allNCP.shape)

print("\nFitting workspace in Y Space: ", wsFinal.name())

fitInYSpaceProcedure(yfitIC, wsFinal, allNCP)


# End of iteractive section
end_time = time.time()
print("\nRunning time: ", end_time-start_time, " seconds")

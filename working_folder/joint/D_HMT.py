# from core_functions.analysis_functions import iterativeFitForDataReduction, switchFirstTwoAxis
from core_functions.fit_in_yspace import fitInYSpaceProcedure
from core_functions.procedures import runJointBackAndForward, extractNCPFromWorkspaces
from mantid.api import AnalysisDataService, mtd
import time
import numpy as np
from pathlib import Path

scriptName =  Path(__file__).name.split(".")[0]
experimentPath = Path(__file__).absolute().parent / "experiments" / scriptName  # Path to the repository

# Set output path
testingCleaning = False
if testingCleaning:     
    cleaningPath = experimentPath / "output" / "testing" / "cleaning"

    forwardScatteringSavePath = cleaningPath / "current_forward.npz" 
    backScatteringSavePath = cleaningPath / "current_backward.npz" 
    ySpaceFitSavePath = cleaningPath / "current_yspacefit.npz"
else:
    outputPath = experimentPath / "output" 

    forwardScatteringSavePath = outputPath / "2iter_forward_GM_MS.npz"
    backScatteringSavePath = outputPath / "2iter_backward_MS.npz"
    ySpaceFitSavePath = outputPath / "2iter_yspacefit.npz"


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


class GeneralInitialConditions:
    """Used to define initial conditions shared by both Back and Forward scattering"""
    
    transmission_guess =  0.92        # Experimental value from VesuvioTransmission
    multiple_scattering_order, number_of_events = 2, 1.e5
    # Sample slab parameters
    vertical_width, horizontal_width, thickness = 0.1, 0.1, 0.001  # Expressed in meters


class BackwardInitialConditions(GeneralInitialConditions):

    modeRunning = "BACKWARD"

    resultsSavePath = backScatteringSavePath
    userWsRawPath = str(backWsRawPath)
    userWsEmptyPath = str(backWsEmptyPath)
    InstrParsPath = ipFilePath

    addHToMS = False
    HToMass0Ratio = None

    # Masses, instrument parameters and initial fitting parameters
    masses = np.array([2.015, 12, 14, 27])
    noOfMasses = len(masses)

    initPars = np.array([ 
    # Intensities, NCP widths, NCP centers   
            1, 6, 0.,     
            1, 12, 0.,    
            1, 12, 0.,   
            1, 12.5, 0.    
        ])
    bounds = np.array([
            [0, np.nan], [3.53, 20], [-3, 1],
            [0, np.nan], [8.62, 20], [-3, 1],
            [0, np.nan], [9.31, 20], [-3, 1],
            [0, np.nan], [12.93, 25], [-3, 1]
        ])
    constraints = ({'type': 'eq', 'fun': lambda par:  par[0] - 2.7527*par[3] },{'type': 'eq', 'fun': lambda par:  par[3] - 0.7234*par[6] })

    noOfMSIterations = 2     #4
    firstSpec = 3    #3
    lastSpec = 134    #134

    maskedSpecAllNo = np.array([18, 34, 42, 43, 59, 60, 62, 118, 119, 133])

    # Boolean Flags to control script
    scaleParsFlag = False
    MSCorrectionFlag = True
    GammaCorrectionFlag = False

    # Parameters of workspaces in input_ws
    name = "DHMT_300K_backward_"
    runs='36517-36556'              # The numbers of the runs to be analysed
    empty_runs='34038-34045'                # The numbers of the empty runs to be subtracted
    spectra='3-134'                            # Spectra to be analysed
    tof_binning='100.,1.,420'                    # Binning of ToF spectra
    mode = 'DoubleDifference'

    # Parameters below are not to be changed

    # Masked spectra between first and last spectrum
    maskedSpecNo = maskedSpecAllNo[
        (maskedSpecAllNo >= firstSpec) & (maskedSpecAllNo <= lastSpec)
    ]
    maskedDetectorIdx = maskedSpecNo - firstSpec

    # Set scaling factors for the fitting parameters, default is ones
    scalingFactors = np.ones(initPars.shape)
    if scaleParsFlag:        # Scale fitting parameters using initial values
            initPars[2::3] = np.ones((1, noOfMasses))  # Main problem is that zeros have to be replaced by non zeros
            scalingFactors = 1 / initPars


class ForwardInitialConditions(GeneralInitialConditions):

    modeRunning = "FORWARD"  # Used to control MS correction

    resultsSavePath = forwardScatteringSavePath
    userWsRawPath = str(frontWsRawPath)
    userWsEmptyPath = str(frontWsEmptyPath)
    InstrParsPath = ipFilePath

    # HToMass0Ratio = None

    masses = np.array([2.015, 12, 14, 27]) 
    noOfMasses = len(masses)

    initPars = np.array([ 
    # Intensities, NCP widths, NCP centers  
        0.4569, 6.5532, 0.,     
        0.166, 12.1585, 0.,    
        0.2295, 13.4784, 0.,   
        0.1476, 17.0095, 0. 
    ])
    bounds = np.array([
        [0, np.nan], [5, 8], [-3, 1],
        [0, np.nan], [12.1585, 12.1585], [-3, 1],
        [0, np.nan], [13.4784, 13.4784], [-3, 1],
        [0, np.nan], [17.0095, 17.0095], [-3, 1]
    ])
    constraints = ({'type': 'eq', 'fun': lambda par:  par[0] - 2.7527*par[3] },{'type': 'eq', 'fun': lambda par:  par[3] - 0.7234*par[6] })
    
    noOfMSIterations = 2   #4
    firstSpec = 135   #135
    lastSpec = 182   #182

    # Boolean Flags to control script
    scaleParsFlag = False
    MSCorrectionFlag = True
    GammaCorrectionFlag = True

    maskedSpecAllNo = np.array([180])

    name = "DHMT_300K_RD_forward_"
    runs='36517-36556'                       # The numbers of the runs to be analysed
    empty_runs='34038-34045'                 # The numbers of the empty runs to be subtracted
    spectra='135-182'                        # Spectra to be analysed
    tof_binning="110,1.,430"                 # Binning of ToF spectra
    mode='SingleDifference'

    # Parameters below are not to be changed

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


# This class inherits all of the atributes in ForwardInitialConditions
class YSpaceFitInitialConditions(ForwardInitialConditions):
    ySpaceFitSavePath = ySpaceFitSavePath

    symmetrisationFlag = True
    symmetriseHProfileUsingAveragesFlag = True      # When False, use mirror sym
    rebinParametersForYSpaceFit = "-30, 0.5, 30"    # Needs to be symetric
    resolutionRebinPars = "-30, 0.125, 30" 
    singleGaussFitToHProfile = True      # When False, use Hermite expansion
    

bckwdIC = BackwardInitialConditions
fwdIC = ForwardInitialConditions
yfitIC = YSpaceFitInitialConditions


start_time = time.time()
# Start of interactive section 

runOnlyYSpaceFit = True
if runOnlyYSpaceFit:
    wsFinal = mtd["DHMT_300K_backward_1"]
    allNCP = extractNCPFromWorkspaces(wsFinal)
else:
    wsFinal, forwardScatteringResults = runJointBackAndForward(bckwdIC, fwdIC)
    lastIterationNCP = forwardScatteringResults.all_ncp_for_each_mass[-1]
    allNCP = lastIterationNCP


print("\nFitting workspace ", wsFinal.name(), " in Y Space.")
fitInYSpaceProcedure(yfitIC, wsFinal, allNCP)


# End of iteractive section
end_time = time.time()
print("\nRunning time: ", end_time-start_time, " seconds")

from vesuvio_analysis.core_functions.fit_in_yspace import fitInYSpaceProcedure
from vesuvio_analysis.core_functions.procedures import runIndependentIterativeProcedure, extractNCPFromWorkspaces
from experiments.directories_helpers import IODirectoriesForSample, loadWsFromLoadVesuvio
from mantid.api import AnalysisDataService, mtd
import time
import numpy as np
from pathlib import Path

scriptName =  Path(__file__).name.split(".")[0]  # Take out .py
experimentPath = Path(__file__).absolute().parent / "experiments" / scriptName  # Path to the repository
ipFilesPath = Path(__file__).absolute().parent / "vesuvio_analysis" / "ip_files"


class LoadVesuvioBackParameters:
    runs="43066-43076"         # 77K         # The numbers of the runs to be analysed
    empty_runs="41876-41923"   # 77K         # The numbers of the empty runs to be subtracted
    spectra='3-134'                          # Spectra to be analysed
    mode='DoubleDifference'
    ipfile=str(ipFilesPath / "ip2019.par")   


class LoadVesuvioFrontParameters:
    runs='43066-43076'         # 100K        # The numbers of the runs to be analysed
    empty_runs='43868-43911'   # 100K        # The numbers of the empty runs to be subtracted
    spectra='144-182'                        # Spectra to be analysed
    mode='SingleDifference'
    ipfile=str(ipFilesPath / "ip2018_3.par") 


wspBack = LoadVesuvioBackParameters
wspFront = LoadVesuvioFrontParameters


# Check if directories of input ws exist
inputWSPath, inputPaths, outputPaths = IODirectoriesForSample(scriptName)

# If input ws are not detected, load locally with Mantid
if all(path==None for path in inputPaths):
    loadWsFromLoadVesuvio(wspBack, inputWSPath, scriptName)
    loadWsFromLoadVesuvio(wspFront, inputWSPath, scriptName)
    inputWSPath, inputPaths, outputPaths = IODirectoriesForSample(scriptName)
    assert any(path!=None for path in inputPaths), "Automatic loading of workspaces failed, usage: scriptName_raw_backward.nxs"

# Extract all required input and output paths
backWsRawPath, frontWsRawPath, backWsEmptyPath, frontWsEmptyPath = inputPaths
forwardSavePath, backSavePath, ySpaceFitSavePath = outputPaths
ipFileBackPath = ipFilesPath / "ip2018_3.par"  
ipFileFrontPath = ipFilesPath / "ip2018_3.par"  


class GeneralInitialConditions:
    """Used to define initial conditions shared by both Back and Forward scattering"""
    
    transmission_guess =  0.8537        # Experimental value from VesuvioTransmission
    multiple_scattering_order, number_of_events = 2, 1.e5
    # Sample slab parameters
    vertical_width, horizontal_width, thickness = 0.1, 0.1, 0.001  # Expressed in meters


class BackwardInitialConditions(GeneralInitialConditions):

    modeRunning = "BACKWARD"

    resultsSavePath = backSavePath
    userWsRawPath = str(backWsRawPath)
    userWsEmptyPath = str(backWsEmptyPath)
    InstrParsPath = ipFileBackPath

    HToMass0Ratio = 19.0620008206  # Set to zero or None when H is not present

    # Masses, instrument parameters and initial fitting parameters
    masses = np.array([12, 16, 27])
    noOfMasses = len(masses)

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

    noOfMSIterations = 4     #4
    firstSpec = 3    #3
    lastSpec = 134    #134

    maskedSpecAllNo = np.array([18, 34, 42, 43, 59, 60, 62, 118, 119, 133])

    # Boolean Flags to control script
    MSCorrectionFlag = True
    GammaCorrectionFlag = False

    # # Parameters of workspaces in input_ws
    tof_binning='275.,1.,420'                    # Binning of ToF spectra

    # Parameters below are not to be changed
    name = scriptName+"_"+modeRunning+"_"
    mode = wspBack.mode

    # Masked spectra between first and last spectrum
    maskedSpecNo = maskedSpecAllNo[
        (maskedSpecAllNo >= firstSpec) & (maskedSpecAllNo <= lastSpec)
    ]
    maskedDetectorIdx = maskedSpecNo - firstSpec


class ForwardInitialConditions(GeneralInitialConditions):

    modeRunning = "FORWARD"  # Used to control MS correction

    resultsSavePath = forwardSavePath
    userWsRawPath = str(frontWsRawPath)
    userWsEmptyPath = str(frontWsEmptyPath)
    InstrParsPath = ipFileFrontPath

    # HToMass0Ratio = None

    masses = np.array([1.0079, 12, 16, 27]) 
    noOfMasses = len(masses)

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
    MSCorrectionFlag = True
    GammaCorrectionFlag = True

    maskedSpecAllNo = np.array([173, 174, 179])

    tof_binning="110,1.,430"                 # Binning of ToF spectra
 
    # Parameters below are not to be changed
    name = scriptName+"_"+modeRunning+"_"
    mode = wspFront.mode

    # Consider only the masked spectra between first and last spectrum
    maskedSpecNo = maskedSpecAllNo[
        (maskedSpecAllNo >= firstSpec) & (maskedSpecAllNo <= lastSpec)
    ]
    maskedDetectorIdx = maskedSpecNo - firstSpec


# This class inherits all of the atributes in ForwardInitialConditions
class YSpaceFitInitialConditions(ForwardInitialConditions):
    ySpaceFitSavePath = ySpaceFitSavePath

    symmetrisationFlag = True
    rebinParametersForYSpaceFit = "-30, 0.5, 30"    # Needs to be symetric
    singleGaussFitToHProfile = True      # When False, use Hermite expansion
    globalFitFlag = True
    forceManualMinos = False
    nGlobalFitGroups = 8
   

bckwdIC = BackwardInitialConditions
fwdIC = ForwardInitialConditions
yfitIC = YSpaceFitInitialConditions

# if __name__ == "main":
start_time = time.time()
# Interactive section 

wsName = "starch_80_RD_forward_0"
if wsName in mtd:
    wsFinal = mtd["starch_80_RD_forward_0"]
    allNCP = extractNCPFromWorkspaces(wsFinal)
else:
    wsFinal, forwardScatteringResults = runIndependentIterativeProcedure(fwdIC)
    lastIterationNCP = forwardScatteringResults.all_ncp_for_each_mass[-1]
    allNCP = lastIterationNCP


print("\nFitting workspace ", wsFinal.name(), " in Y Space.")
fitInYSpaceProcedure(yfitIC, wsFinal, allNCP)


# End of iteractive section
end_time = time.time()
print("\nRunning time: ", end_time-start_time, " seconds")

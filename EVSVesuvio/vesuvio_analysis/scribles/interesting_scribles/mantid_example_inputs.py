
"""
This script includes extra comments to aid the implementation of Mantid GUI.
The script is identical to the scripts used to run the optimized VesuvioAnalysis algorithm.
It includes all of the inputs required to run the algorithm in seperate classes,
depending on the version of the algorithm to run, ie. either backscattering or forward scattering.

I have included comments explaining the inputs, and what each import is doing.
This script includes all of the sample dependent variables, so the connection
to Mantid GUI should be a matter of replicating the inputs in this script.
The only thing that I see differing is the Loading of workspaces, which I have
explained in more detail below.
"""

# Import data reduction algorithms
from vesuvio_analysis.core_functions.fit_in_yspace import fitInYSpaceProcedure
from vesuvio_analysis.core_functions.procedures import runIndependentIterativeProcedure, extractNCPFromWorkspaces

# These imports deal with input-output data directories and with the loading of workspaces
from vesuvio_analysis.directories_helpers import IODirectoriesForSample, loadWsFromLoadVesuvio
from pathlib import Path

# Other useful imports
from mantid.api import AnalysisDataService, mtd
import time
import numpy as np

# Set some paths for local inputs and outputs, ignore this for Mantid implementation
scriptName =  Path(__file__).name.split(".")[0]  # Take out .py
experimentPath = Path(__file__).absolute().parent / "experiments" / scriptName  # Path to the repository

# Path to IP file containing all IP files
ipFilesPath = Path(__file__).absolute().parent / "vesuvio_analysis" / "ip_files"


# The following classes are used as inputs to load workspaces in a local directory of the repository
# In Mantid, these are the inputs to LoadVesuvio
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


# This function sorts out input and output directories.
# The input directories are used to load locally stored workspaces.
# The output directories are used to store .npz files with some results arrays.
# Since in mantid, the input workspaces will be loaded with LoadVesuvio or already 
# present in mtd, these input paths will not be used.
# The ouput .npz files can also be turned off, so they are also not used in the Mantid implementation
inputWSPath, inputPaths, outputPaths = IODirectoriesForSample(scriptName)

# Below is a check that will perform the loading of workspaces locally
# if they are not so already.
# Not to be used in Mantid implementation
if all(path==None for path in inputPaths):
    loadWsFromLoadVesuvio(wspBack, inputWSPath, scriptName)
    loadWsFromLoadVesuvio(wspFront, inputWSPath, scriptName)
    inputWSPath, inputPaths, outputPaths = IODirectoriesForSample(scriptName)
    assert any(path!=None for path in inputPaths), "Automatic loading of workspaces failed, usage: scriptName_raw_backward.nxs"

# Ignore input and output paths below
backWsRawPath, frontWsRawPath, backWsEmptyPath, frontWsEmptyPath = inputPaths
forwardSavePath, backSavePath, ySpaceFitSavePath = outputPaths

# Paths to ip files
ipFileBackPath = ipFilesPath / "ip2018_3.par"  
ipFileFrontPath = ipFilesPath / "ip2018_3.par"  


# Inputs class with arguments that are shared between backward and forward scattering
class GeneralInitialConditions:
    """Used to define initial conditions shared by both Back and Forward scattering"""
    
    transmission_guess =  0.8537        # Experimental value from VesuvioTransmission
    multiple_scattering_order, number_of_events = 2, 1.e5
    # Sample slab parameters
    vertical_width, horizontal_width, thickness = 0.1, 0.1, 0.001  # Expressed in meters


# Inputs for backward scattering
class BackwardInitialConditions(GeneralInitialConditions):

    modeRunning = "BACKWARD"

    resultsSavePath = backSavePath         # This could be set to None to disable ouput files
    userWsRawPath = str(backWsRawPath)      # Instead of ws path, could modify script to allow name of ws already in mtd
    userWsEmptyPath = str(backWsEmptyPath)  # Same idea, use name of ws intead of Load() from local path
    InstrParsPath = ipFileBackPath       # local path to IP file to be used, Path() object.

    # The following variable is only present in backward scattering
    HToMass0Ratio = 19.0620008206  # Set to zero or None when H is not present

    # Masses, float
    masses = np.array([12, 16, 27])
    noOfMasses = len(masses)

    # Initial guesses for the fitting parameters
    # initPars is an array containing the initial guesses for intensities, widths and centers
    # Format: [intensity0, width0, center0, intensity1, with1, center1, ... ]
    initPars = np.array([ 
    # Intensities, NCP widths, NCP centers   
            1, 12, 0.,    
            1, 12, 0.,   
            1, 12.5, 0.    
        ])
    # Set boundaries for each initial guess
    bounds = np.array([
            [0, np.nan], [8, 16], [-3, 1],
            [0, np.nan], [8, 16], [-3, 1],
            [0, np.nan], [11, 14], [-3, 1]
        ])
    constraints = ()  # Used in optimize.minimize()

    # Number of iterations and select spectra for analysis
    noOfMSIterations = 4     # Not more than 4
    firstSpec = 3    # Not less than minimum spec
    lastSpec = 134    # Not more than max spec

    maskedSpecAllNo = np.array([18, 34, 42, 43, 59, 60, 62, 118, 119, 133])

    # Boolean Flags to control corrections after each iteration 
    MSCorrectionFlag = True
    GammaCorrectionFlag = False

    # Binning of input ws in TOF
    tof_binning='275.,1.,420'         

    # ---- Parameters below are calculated from inputs above

    name = scriptName+"_"+modeRunning+"_"
    mode = wspBack.mode

    # Masked spectra between first and last spectrum
    maskedSpecNo = maskedSpecAllNo[
        (maskedSpecAllNo >= firstSpec) & (maskedSpecAllNo <= lastSpec)
    ]
    maskedDetectorIdx = maskedSpecNo - firstSpec


# The following class has the same structure as above, except for HToMass0Ratio
class ForwardInitialConditions(GeneralInitialConditions):

    modeRunning = "FORWARD"  # Used to control MS correction

    resultsSavePath = forwardSavePath
    userWsRawPath = str(frontWsRawPath)
    userWsEmptyPath = str(frontWsEmptyPath)
    InstrParsPath = ipFileFrontPath

    # HToMass0Ratio is not present in forward scattering

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
 
    # ---- Parameters below are not to be changed

    name = scriptName+"_"+modeRunning+"_"
    mode = wspFront.mode

    # Consider only the masked spectra between first and last spectrum
    maskedSpecNo = maskedSpecAllNo[
        (maskedSpecAllNo >= firstSpec) & (maskedSpecAllNo <= lastSpec)
    ]
    maskedDetectorIdx = maskedSpecNo - firstSpec


# The following class is an input of a different algorithm.
# This algorithm is used for the analysis in yspace of the final output ws from forward scattering analysis.
# This is meant to be implemented as a seperate algorithm from VesuvioAnalysis.
class YSpaceFitInitialConditions(ForwardInitialConditions):  # Inherits from forward scattering
    ySpaceFitSavePath = ySpaceFitSavePath    # Output path could be disabled by setting it to None

    symmetrisationFlag = True     
    rebinParametersForYSpaceFit = "-30, 0.5, 30"    # Needs to be symetric, otherwise forbid from running.
    singleGaussFitToHProfile = True      # When False, use Hermite expansion
    globalFitFlag = True
    forceManualMinos = False
    nGlobalFitGroups = 8        # Not more than (noOfWSSpectra - noOfMaskedSpectra)
   

# Initialize conditions
bckwdIC = BackwardInitialConditions
fwdIC = ForwardInitialConditions
yfitIC = YSpaceFitInitialConditions


start_time = time.time()
# Interactive section 

# Run either forward or backward using the corresponding class and procedure
wsFinal, forwardScatteringResults = runIndependentIterativeProcedure(fwdIC)
allNCP = forwardScatteringResults.all_ncp_for_each_mass[-1]

# From the results of forward scattering, run analysis (fit) in yspace
fitInYSpaceProcedure(yfitIC, wsFinal, allNCP)

# End of iteractive section
end_time = time.time()
print("\nRunning time: ", end_time-start_time, " seconds")

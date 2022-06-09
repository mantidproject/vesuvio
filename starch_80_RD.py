
import time
import numpy as np
from pathlib import Path
from vesuvio_analysis.core_functions.bootstrap_analysis import runAnalysisOfStoredBootstrap
from vesuvio_analysis.core_functions.run_script import runScript

scriptName =  Path(__file__).name.split(".")[0]  # Take out .py
experimentPath = Path(__file__).absolute().parent / "experiments" / scriptName  # Path to the repository
ipFilesPath = Path(__file__).absolute().parent / "vesuvio_analysis" / "ip_files"


class LoadVesuvioBackParameters:
    runs = "43066-43076"         # 77K         # The numbers of the runs to be analysed
    empty_runs = "41876-41923"   # 77K         # The numbers of the empty runs to be subtracted
    spectra = '3-134'                          # Spectra to be analysed
    mode = 'DoubleDifference'
    ipfile = ipFilesPath / "ip2019.par"  

    subEmptyFromRaw = True         # Flag to control wether empty ws gets subtracted from raw
    scaleEmpty = 1       # None or scaling factor 
    scaleRaw = 1

class LoadVesuvioFrontParameters:
    runs = '43066-43076'         # 100K        # The numbers of the runs to be analysed
    empty_runs = '43868-43911'   # 100K        # The numbers of the empty runs to be subtracted
    spectra = '144-182'                        # Spectra to be analysed
    mode = 'SingleDifference'  
    ipfile = ipFilesPath / "ip2018_3.par"

    subEmptyFromRaw = False         # Flag to control wether empty ws gets subtracted from raw
    scaleEmpty = 1       # None or scaling factor 
    scaleRaw = 1

class GeneralInitialConditions:
    """Used to define initial conditions shared by both Back and Forward scattering"""
    
    transmission_guess =  0.8537        # Experimental value from VesuvioTransmission
    multiple_scattering_order, number_of_events = 2, 1.e5
    # Sample slab parameters
    vertical_width, horizontal_width, thickness = 0.1, 0.1, 0.001  # Expressed in meters


class BackwardInitialConditions(GeneralInitialConditions):
    InstrParsPath = ipFilesPath / "ip2018_3.par" 

    HToMass0Ratio = 19.0620008206  # Set to None when either unknown or H not present
    HToMassIdx = 0   # Idx of mass to take the ratio with

    # Masses, instrument parameters and initial fitting parameters
    masses = np.array([12, 16, 27])

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
    lastSpec = 134   #134

    maskedSpecAllNo = np.array([18, 34, 42, 43, 59, 60, 62, 118, 119, 133])

    # Boolean Flags to control script
    MSCorrectionFlag = True
    GammaCorrectionFlag = False

    # # Parameters of workspaces in input_ws
    tof_binning='275.,1.,420'                    # Binning of ToF spectra


class ForwardInitialConditions(GeneralInitialConditions):
    # InstrParsPath = ipFilesPath / "ip2018_3.par" 

    masses = np.array([1.0079, 12, 16, 27]) 

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

    noOfMSIterations = 4   #4
    firstSpec = 144   #144
    lastSpec = 182   #182

    # Boolean Flags to control script
    MSCorrectionFlag = True
    GammaCorrectionFlag = True

    maskedSpecAllNo = np.array([173, 174, 179])

    tof_binning="110,1,430"                 # Binning of ToF spectra
 

# This class inherits all of the atributes in ForwardInitialConditions
class YSpaceFitInitialConditions:
    showPlots = True
    symmetrisationFlag = True
    rebinParametersForYSpaceFit = "-25, 0.5, 25"    # Needs to be symetric
    fitModel = "SINGLE_GAUSSIAN"     # Options: 'SINGLE_GAUSSIAN', 'GC_C4', 'GC_C6', 'GC_C4_C6'
    globalFit = None                 # Options: None, 'Mantid', 'MINUIT' 
    forceManualMinos = False
    nGlobalFitGroups = 4               # Number or string "ALL"


class BootstrapInitialConditions:
    runningJackknife = False
    nSamples = 650
    skipMSIterations = False
    userConfirmation = True


class UserScriptControls:
    # Choose main procedure to run
    procedure = None  # Options: None, "BACKWARD", "FORWARD", "JOINT"

    # Choose on which ws to perform the fit in y space
    fitInYSpace = None    # Options: None, "BACKWARD", "FORWARD", "JOINT"

    # Perform bootstrap procedure
    # Independent of procedure and runFItInYSpace
    bootstrap = None  # Options: None, "BACKWARD", "FORWARD", "JOINT"


class BootstrapAnalysis:
    # Flag below controls whether or not analysis is run
    runAnalysis = True   

    # Choose whether to filter averages as done in original procedure
    filterAvg = True                 # True discards some unreasonable values of widths and intensities
    
    # Flags below control the plots to show
    plotRawWidthsIntensities = False
    plotMeanWidthsIntensities = False
    plotMeansEvolution = False
    plot2DHists = False 
    plotYFitHists = True


# Initialize classes and run script below
# Not for useers

start_time = time.time()

wsBackIC = LoadVesuvioBackParameters
wsFrontIC = LoadVesuvioFrontParameters  
bckwdIC = BackwardInitialConditions
fwdIC = ForwardInitialConditions
yFitIC = YSpaceFitInitialConditions
bootIC = BootstrapInitialConditions
userCtr = UserScriptControls

runScript(userCtr, scriptName, wsBackIC, wsFrontIC, bckwdIC, fwdIC, yFitIC, bootIC)

end_time = time.time()
print("\nRunning time: ", end_time-start_time, " seconds")

analysisIC = BootstrapAnalysis

runAnalysisOfStoredBootstrap(bckwdIC, fwdIC, yFitIC, bootIC, analysisIC)
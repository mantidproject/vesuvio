
import time
import numpy as np
from pathlib import Path
from vesuvio_analysis.core_functions.bootstrap_analysis import runAnalysisOfStoredBootstrap
from vesuvio_analysis.core_functions.run_script import runScript

scriptName =  Path(__file__).name.split(".")[0]  # Take out .py
experimentPath = Path(__file__).absolute().parent / "experiments" / scriptName  # Path to experiments/sample
ipFilesPath = Path(__file__).absolute().parent / "vesuvio_analysis" / "ip_files"


class LoadVesuvioBackParameters:
    runs='36517-36556'              # The numbers of the runs to be analysed
    empty_runs='34038-34045'                # The numbers of the empty runs to be subtracted
    spectra='3-134'                            # Spectra to be analysed
    mode = 'DoubleDifference'
    ipfile=ipFilesPath / "ip2018_3.par" 

    subEmptyFromRaw = True         # Flag to control wether empty ws gets subtracted from raw
    scaleEmpty = 1       # None or scaling factor
    scaleRaw = 1  

class LoadVesuvioFrontParameters:
    runs='36517-36556'                       # The numbers of the runs to be analysed
    empty_runs='34038-34045'                 # The numbers of the empty runs to be subtracted
    spectra='135-182'                        # Spectra to be analysed
    mode='SingleDifference'
    ipfile=ipFilesPath / "ip2018_3.par"

    subEmptyFromRaw = False         # Flag to control wether empty ws gets subtracted from raw
    scaleEmpty = 1       # None or scaling factor 
    scaleRaw = 1

class GeneralInitialConditions:
    """Used to define initial conditions shared by both Back and Forward scattering"""
    
    transmission_guess =  0.92        # Experimental value from VesuvioTransmission
    multiple_scattering_order, number_of_events = 2, 1.e5
    # Sample slab parameters
    vertical_width, horizontal_width, thickness = 0.1, 0.1, 0.001  # Expressed in meters


class BackwardInitialConditions(GeneralInitialConditions):
    # InstrParsPath = ipFilesPath / "ip2018_3.par" 

    HToMass0Ratio = None   # Set to zero or None when H is not present
    HToMassIdx = 0

    # Masses, instrument parameters and initial fitting parameters
    masses = np.array([2.015, 12, 14, 27])

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

    noOfMSIterations = 1     #4
    firstSpec = 3    #3
    lastSpec = 134    #134

    maskedSpecAllNo = np.array([18, 34, 42, 43, 59, 60, 62, 118, 119, 133])

    # Boolean Flags to control script
    MSCorrectionFlag = True
    GammaCorrectionFlag = False

    # # Parameters of workspaces in input_ws
    tof_binning='50,1.,420'                    # Binning of ToF spectra


class ForwardInitialConditions(GeneralInitialConditions):
    # InstrParsPath = ipFilesPath / "ip2018_3.par" 

    masses = np.array([2.015, 12, 14, 27]) 
  
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
    
    noOfMSIterations = 4 #2   #4
    firstSpec = 135   #135
    lastSpec = 182  #182

    # Boolean Flags to control script
    MSCorrectionFlag = True
    GammaCorrectionFlag = True

    maskedSpecAllNo = np.array([180])

    tof_binning="110,1.,430"                 # Binning of ToF spectra
 

# This class inherits all of the atributes in ForwardInitialConditions
class YSpaceFitInitialConditions:
    showPlots = True
    symmetrisationFlag = True
    rebinParametersForYSpaceFit = "-25, 0.5, 25"    # Needs to be symetric
    fitModel = "GC_C4_C6"     # Options: 'SINGLE_GAUSSIAN', 'GC_C4', 'GC_C6', 'GC_C4_C6'
    globalFit = "MINUIT"
    forceManualMinos = True
    nGlobalFitGroups = 4   


class BootstrapInitialConditions:
    runningJackknife = True
    nSamples = 500
    skipMSIterations = False
    runningTest = False
    userConfirmation = True


class UserScriptControls:
    # Choose main procedure to run
    procedure = "BACKWARD"   # Options: "BACKWARD", "FORWARD", "JOINT"

    # Choose on which ws to perform the fit in y space
    fitInYSpace = "BACKWARD"    # Options: None, "BACKWARD", "FORWARD", "JOINT"

    # Perform bootstrap procedure
    # Independent of procedure and runFItInYSpace
    bootstrap = None   # Options: None, "BACKWARD", "FORWARD", "JOINT"


class BootstrapAnalysis:
    # Flag below controls whether or not analysis is run
    runAnalysis = False

    # Choose whether to filter averages as done in original procedure
    filterAvg = True                 # True discards some unreasonable values of widths and intensities
    
    # Flags below control the plots to show
    plotRawWidthsIntensities = True
    plotMeanWidthsIntensities = True
    plotMeansEvolution = True
    plot2DHists = True
    plotYFitHists = True


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
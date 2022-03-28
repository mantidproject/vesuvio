from vesuvio_analysis.core_functions.procedures import runIndependentIterativeProcedure, extractNCPFromWorkspaces
import time
import numpy as np
from pathlib import Path


currentPath = Path(__file__).absolute().parent 
ipFilesPath = currentPath / "vesuvio_analysis" / "ip_files"

comparisonPath = currentPath / "comparison_with_old_script"

inputWSPath = comparisonPath / "input_ws"
outputPath = comparisonPath / "original_and_current_data" / "current_data"

backWsRawPath = inputWSPath / "starch_80_RD_raw_backward.nxs"
backWsEmptyPath = inputWSPath / "starch_80_RD_empty_backward.nxs"
frontWsRawPath = inputWSPath / "starch_80_RD_raw_forward.nxs"
frontWsEmptyPath = None

forwardSavePath = outputPath / "4iter_forward_GM_MS.npz"
backSavePath = outputPath / "4iter_backward_MS.npz"

ipFileFrontPath = ipFilesPath / "ip2018_3.par"  
ipFileBackPath = ipFilesPath / "ip2018_3.par"  
# ipFileBackPath = ipFilesPath / "ip2019.par"  


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
    masses = np.array([12, 16, 27]).astype(float)
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

    noOfMSIterations = 4     
    firstSpec = 3    
    lastSpec = 134    

    maskedSpecAllNo = np.array([18, 34, 42, 43, 59, 60, 62, 118, 119, 133])

    # Boolean Flags to control script
    MSCorrectionFlag = True
    GammaCorrectionFlag = False

    # # Parameters of workspaces in input_ws
    name = "starch_80_RD_backward_"
    tof_binning='275.,1.,420'                    # Binning of ToF spectra
    mode='DoubleDifference'

    # Parameters below are not to be changed
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

    noOfMSIterations = 4   
    firstSpec = 144
    lastSpec = 182

    # Boolean Flags to control script
    MSCorrectionFlag = True
    GammaCorrectionFlag = True

    maskedSpecAllNo = np.array([173, 174, 179])

    name = "starch_80_RD_forward_"
    tof_binning="110,1.,430"                 # Binning of ToF spectra
    mode='SingleDifference'

    # Parameters below are not to be changed

    # Consider only the masked spectra between first and last spectrum
    maskedSpecNo = maskedSpecAllNo[
        (maskedSpecAllNo >= firstSpec) & (maskedSpecAllNo <= lastSpec)
    ]
    maskedDetectorIdx = maskedSpecNo - firstSpec


bckwdIC = BackwardInitialConditions
fwdIC = ForwardInitialConditions


start_time = time.time()

# Run forward and backward independently, results are being stored in current_data folder
runIndependentIterativeProcedure(fwdIC)
runIndependentIterativeProcedure(bckwdIC)


end_time = time.time()
print("\nRunning time: ", end_time-start_time, " seconds")

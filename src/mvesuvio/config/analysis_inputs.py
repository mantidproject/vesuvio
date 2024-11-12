import numpy as np
from dataclasses import dataclass


@dataclass
class SampleParameters:
    # Sample slab parameters, expressed in meters
    vertical_width = 0.1 
    horizontal_width = 0.1 
    thickness = 0.001


@dataclass
class BackwardInitialConditions(SampleParameters):
    runs = "43066-43076"
    empty_runs = "41876-41923" 
    mode = "DoubleDifference"
    ipfile = "ip2019.par"

    subEmptyFromRaw = True  # Flag to control wether empty ws gets subtracted from raw
    scaleEmpty = 1  # None or scaling factor
    scaleRaw = 1
    # instrParsFile = "ip2018_3.par"

    HToMassIdxRatio = 19.0620008206  # Set to zero when H is not present

    masses = np.array([12, 16, 27])

    # Intensities, NCP widths, NCP centers
    initPars = np.array([1, 12, 0.0, 1, 12, 0.0, 1, 12.5, 0.0])
    bounds = np.array(
        [
            [0, None],
            [8, 16],
            [-3, 1],
            [0, None],
            [8, 16],
            [-3, 1],
            [0, None],
            [11, 14],
            [-3, 1],
        ]
    )
    constraints = ()

    noOfMSIterations = 0  # 4
    firstSpec = 3  # 3
    lastSpec = 134  # 134

    maskedSpecAllNo = np.array([18, 34, 42, 43, 59, 60, 62, 118, 119, 133])

    # Boolean Flags to control script
    MSCorrectionFlag = True
    GammaCorrectionFlag = False

    # # Parameters of workspaces in input_ws
    tofBinning = "275.,1.,420"  # Binning of ToF spectra
    maskTOFRange = None

    transmission_guess = 0.8537  # Experimental value from VesuvioTransmission
    multiple_scattering_order = 2
    number_of_events = 1.0e5


@dataclass
class ForwardInitialConditions(SampleParameters):

    runs = "43066-43076"
    empty_runs = "43868-43911"
    mode = "SingleDifference"
    ipfile = "ip2018_3.par"

    subEmptyFromRaw = False  # Flag to control wether empty ws gets subtracted from raw
    scaleEmpty = 1  # None or scaling factor
    scaleRaw = 1

    masses = np.array([1.0079, 12, 16, 27])

    # Intensities, NCP widths, NCP centers
    initPars = np.array([1, 4.7, 0, 1, 12.71, 0.0, 1, 8.76, 0.0, 1, 13.897, 0.0])
    bounds = np.array(
        [
            [0, None],
            [3, 6],
            [-3, 1],
            [0, None],
            [12.71, 12.71],
            [-3, 1],
            [0, None],
            [8.76, 8.76],
            [-3, 1],
            [0, None],
            [13.897, 13.897],
            [-3, 1],
        ]
    )
    constraints = ()

    noOfMSIterations = 0  # 4
    firstSpec = 144  # 144
    lastSpec = 182  # 182

    # Boolean Flags to control script
    MSCorrectionFlag = True
    GammaCorrectionFlag = True

    maskedSpecAllNo = np.array([173, 174, 179])

    tofBinning = "110,1,430"  # Binning of ToF spectra
    maskTOFRange = None

    transmission_guess = 0.8537  # Experimental value from VesuvioTransmission
    multiple_scattering_order = 2
    number_of_events = 1.0e5


@dataclass
class YSpaceFitInitialConditions:
    showPlots = True
    symmetrisationFlag = True
    rebinParametersForYSpaceFit = "-25, 0.5, 25"  # Needs to be symetric
    fitModel = "SINGLE_GAUSSIAN"  # Options: 'SINGLE_GAUSSIAN', 'GC_C4', 'GC_C6', 'GC_C4_C6', 'DOUBLE_WELL', 'ANSIO_GAUSSIAN', 'Gaussian3D'
    runMinos = True
    globalFit = True  # Performs global fit with Minuit by default
    nGlobalFitGroups = 4  # Number or string "ALL"
    maskTypeProcedure = "NAN"  # Options: 'NCP', 'NAN', None


@dataclass
class UserScriptControls:
    runRoutine = True

    # Choose main procedure to run
    procedure = "FORWARD"  # Options: None, "BACKWARD", "FORWARD", "JOINT"

    # Choose on which ws to perform the fit in y space
    fitInYSpace = "FORWARD"   #"FORWARD"  # Options: None, "BACKWARD", "FORWARD", "JOINT"


########################
### END OF USER EDIT ###
########################


if (__name__ == "__main__") or (__name__ == "mantidqt.widgets.codeeditor.execution"):
    import mvesuvio
    from pathlib import Path
    mvesuvio.set_config(inputs_file=Path(__file__))
    mvesuvio.run()

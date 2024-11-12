import numpy as np


class GeneralInitialConditions:
    transmission_guess = 0.8537  # Experimental value from VesuvioTransmission
    multiple_scattering_order, number_of_events = 2, 1.0e5
    vertical_width, horizontal_width, thickness = 0.1, 0.1, 0.001  # Expressed in meters


class BackwardInitialConditions(GeneralInitialConditions):
    run_this_scattering_type = False
    fit_in_y_space = False 

    ipfile = "ip2019.par"

    runs = "43066-43076"  # 77K         # The numbers of the runs to be analysed
    empty_runs = (
        "41876-41923"  # 77K         # The numbers of the empty runs to be subtracted
    )
    spectra = "3-134"  # Spectra to be analysed
    mode = "DoubleDifference"
    subEmptyFromRaw = True  # Flag to control wether empty ws gets subtracted from raw
    scaleEmpty = 1  # None or scaling factor
    scaleRaw = 1

    HToMassIdxRatio = 19.0620008206  # Set to zero or None when H is not present
    massIdx = 0
    masses = np.array([12, 16, 27])
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
    noOfMSIterations = 3  # 4
    firstSpec = 3  # 3
    lastSpec = 134  # 134
    maskedSpecAllNo = np.array([18, 34, 42, 43, 59, 60, 62, 118, 119, 133])
    MSCorrectionFlag = True
    GammaCorrectionFlag = False
    tofBinning = "275.,1.,420"  # Binning of ToF spectra
    maskTOFRange = None


class ForwardInitialConditions(GeneralInitialConditions):
    run_this_scattering_type = True
    fit_in_y_space = False

    ipfile = "ip2018_3.par"

    runs = "43066-43076"  # 100K        # The numbers of the runs to be analysed
    empty_runs = (
        "43868-43911"  # 100K        # The numbers of the empty runs to be subtracted
    )
    spectra = "144-182"  # Spectra to be analysed
    mode = "SingleDifference"
    subEmptyFromRaw = False  # Flag to control wether empty ws gets subtracted from raw
    scaleEmpty = 1  # None or scaling factor
    scaleRaw = 1

    HToMassIdxRatio = 0     # New way to ignore ratio

    masses = np.array([1.0079, 12, 16, 27])
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
    noOfMSIterations = 3  # 4
    firstSpec = 144  # 144
    lastSpec = 182  # 182
    MSCorrectionFlag = True
    GammaCorrectionFlag = True
    maskedSpecAllNo = np.array([173, 174, 179])
    tofBinning = "110,1,430"  # Binning of ToF spectra
    maskTOFRange = None


class YSpaceFitInitialConditions:
    showPlots = False
    symmetrisationFlag = True
    rebinParametersForYSpaceFit = "-20, 0.5, 20"  # Needs to be symetric
    fitModel = "SINGLE_GAUSSIAN"
    runMinos = True
    globalFit = None
    nGlobalFitGroups = 4
    maskTypeProcedure = None


class UserScriptControls:
    runRoutine = True
    procedure = "FORWARD"  # Options: None, "BACKWARD", "FORWARD", "JOINT"
    fitInYSpace = None  # Options: None, "BACKWARD", "FORWARD", "JOINT"

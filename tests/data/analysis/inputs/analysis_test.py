
class SampleParameters:
    transmission_guess = 0.8537  # Experimental value from VesuvioTransmission
    multiple_scattering_order, number_of_events = 2, 1.0e5
    vertical_width, horizontal_width, thickness = 0.1, 0.1, 0.001  # Expressed in meters


class BackwardAnalysisInputs(SampleParameters):
    run_this_scattering_type = False
    fit_in_y_space = False 
    ipfile = "ip2019.par"
    runs = "43066-43076"  
    empty_runs = "41876-41923"  
    spectra = "3-134"
    mode = "DoubleDifference"
    tofBinning = "275.,1.,420"  # Binning of ToF spectra
    maskTOFRange = None
    maskedSpecAllNo = [18, 34, 42, 43, 59, 60, 62, 118, 119, 133]
    firstSpec = 3  # 3
    lastSpec = 134  # 134
    subEmptyFromRaw = True 
    scaleEmpty = 1 
    scaleRaw = 1

    masses = [12, 16, 27]
    initPars = [1, 12, 0.0, 1, 12, 0.0, 1, 12.5, 0.0]
    bounds =[
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
    constraints = ()

    noOfMSIterations = 3  # 4
    MSCorrectionFlag = True
    HToMassIdxRatio = 19.0620008206  # Set to zero or None when H is not present
    GammaCorrectionFlag = False


class ForwardAnalysisInputs(SampleParameters):
    run_this_scattering_type = True
    fit_in_y_space = False

    ipfile = "ip2018_3.par"
    runs = "43066-43076"
    empty_runs = "43868-43911"
    spectra = "144-182"
    mode = "SingleDifference"
    tofBinning = "110,1,430"  # Binning of ToF spectra
    maskTOFRange = None
    maskedSpecAllNo = [173, 174, 179]
    firstSpec = 144  # 144
    lastSpec = 182  # 182
    subEmptyFromRaw = False 
    scaleEmpty = 1 
    scaleRaw = 1


    masses = [1.0079, 12, 16, 27]
    initPars = [1, 4.7, 0, 1, 12.71, 0.0, 1, 8.76, 0.0, 1, 13.897, 0.0]
    bounds =[
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
    constraints = ()
    noOfMSIterations = 1  # 4
    MSCorrectionFlag = True
    GammaCorrectionFlag = True


class YSpaceFitInputs:
    showPlots = False
    symmetrisationFlag = True
    subtractFSE = False
    rebinParametersForYSpaceFit = "-20, 0.5, 20"  # Needs to be symetric
    fitModel = "SINGLE_GAUSSIAN"
    runMinos = True
    globalFit = None
    nGlobalFitGroups = 4
    maskTypeProcedure = None

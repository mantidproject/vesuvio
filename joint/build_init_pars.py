import numpy as np
from pathlib import Path
repoPath = Path(__file__).absolute().parent  # Path to the repository

testingCleaning = True
if testingCleaning:     
    pathForTesting = repoPath / "tests" / "cleaning"  
    forwardScatteringSavePath = pathForTesting / "current_forward.npz" 
    backScatteringSavePath = pathForTesting / "current_backward.npz" 
else:
    forwardScatteringSavePath = repoPath / "tests" / "fixatures" / "4iter_forward_GB_MS_opt.npz" 
    backScatteringSavePath = repoPath / "tests" / "fixatures" / "4iter_backward_MS_opt.npz"


class BackwardInitialConditions:
    # Multiscaterring Correction Parameters
    HToMass0Ratio = 19.0620008206

    backScatteringSavePath = backScatteringSavePath



    transmission_guess =  0.8537        # Experimental value from VesuvioTransmission
    multiple_scattering_order, number_of_events = 2, 1.e5   
    hydrogen_peak = True                 # Hydrogen multiple scattering
    
    # Sample slab parameters
    vertical_width, horizontal_width, thickness = 0.1, 0.1, 0.001  # Expressed in meters


    modeRunning = "BACKWARD"

        # Parameters to load Raw and Empty Workspaces
    userWsRawPath = r"./input_ws/starch_80_RD_raw_backward.nxs"
    userWsEmptyPath = r"./input_ws/starch_80_RD_empty_backward.nxs"
    name = "starch_80_RD_backward_"
    runs='43066-43076'  # 77K             # The numbers of the runs to be analysed
    empty_runs='41876-41923'   # 77K             # The numbers of the empty runs to be subtracted
    spectra='3-134'                            # Spectra to be analysed
    tof_binning='275.,1.,420'                    # Binning of ToF spectra
    mode='DoubleDifference'
    ipfile=r'./ip2019.par' 

        # Masses, instrument parameters and initial fitting parameters
    masses = np.array([12, 16, 27])
    noOfMasses = len(masses)
    InstrParsPath = repoPath / 'ip2018_3.par'

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

        # Boolean Flags to control script
    loadWsFromUserPathFlag = True
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

    forwardScatteringSavePath = forwardScatteringSavePath

    HToMass0Ratio = None

    transmission_guess =  0.8537        # Experimental value from VesuvioTransmission
    multiple_scattering_order, number_of_events = 2, 1.e5   
    hydrogen_peak = True                 # Hydrogen multiple scattering
    
    # Sample slab parameters
    vertical_width, horizontal_width, thickness = 0.1, 0.1, 0.001  # Expressed in meters

    modeRunning = "FORWARD"  # Used to control MS correction

    userWsRawPath = r"./input_ws/starch_80_RD_raw_forward.nxs"
    userWsEmptyPath = r"./input_ws/starch_80_RD_raw_forward.nxs"

    name = "starch_80_RD_forward_"
    runs='43066-43076'         # 100K        # The numbers of the runs to be analysed
    empty_runs='43868-43911'   # 100K        # The numbers of the empty runs to be subtracted
    spectra='144-182'                        # Spectra to be analysed
    tof_binning="110,1.,430"                 # Binning of ToF spectra
    mode='SingleDifference'
    ipfile=r'./ip2018_3.par'

    masses = np.array([1.0079, 12, 16, 27]) 
    noOfMasses = len(masses)
    InstrParsPath = repoPath / 'ip2018_3.par'

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

    noOfMSIterations = 2     #4
    firstSpec = 164   #144
    lastSpec = 175    #182

    # Boolean Flags to control script
    loadWsFromUserPathFlag = True
    scaleParsFlag = False
    MSCorrectionFlag = True
    GammaCorrectionFlag = True

    # Parameters to control fit in Y-Space
    symmetrisationFlag = True
    symmetriseHProfileUsingAveragesFlag = True      # When False, use mirror sym
    rebinParametersForYSpaceFit = "-20, 0.5, 20"    # Needs to be symetric
    singleGaussFitToHProfile = True      # When False, use Hermite expansion
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

bckwdIC = BackwardInitialConditions
fwdIC = ForwardInitialConditions

print(bckwdIC.masses.size)

# class InitialConditions:

    # # Initialize object to use methods
    # def __init__(:
    #     return None

    # # Multiscaterring Correction Parameters
    # HToMass0Ratio = 19.0620008206

    # transmission_guess =  0.8537        # Experimental value from VesuvioTransmission
    # multiple_scattering_order, number_of_events = 2, 1.e5   
    # hydrogen_peak = True                 # Hydrogen multiple scattering
    
    # # Sample slab parameters
    # vertical_width, horizontal_width, thickness = 0.1, 0.1, 0.001  # Expressed in meters
  
    # modeRunning = "None"     # Stores wether is running forward or backward

    # # Paths to save results for back and forward scattering
    # testingCleaning = True
    # if testingCleaning:     
    #     pathForTesting = repoPath / "tests" / "cleaning"  
    #     forwardScatteringSavePath = pathForTesting / "current_forward.npz" 
    #     backScatteringSavePath = pathForTesting / "current_backward.npz" 
    # else:
    #     forwardScatteringSavePath = repoPath / "tests" / "fixatures" / "4iter_forward_GB_MS_opt.npz" 
    #     backScatteringSavePath = repoPath / "tests" / "fixatures" / "4iter_backward_MS_opt.npz"
    

    # def setBackscatteringInitialConditions(:
    #     modeRunning = "BACKWARD"

    #     # Parameters to load Raw and Empty Workspaces
    #     userWsRawPath = r"./input_ws/starch_80_RD_raw_backward.nxs"
    #     userWsEmptyPath = r"./input_ws/starch_80_RD_empty_backward.nxs"

    #     name = "starch_80_RD_backward_"
    #     runs='43066-43076'  # 77K             # The numbers of the runs to be analysed
    #     empty_runs='41876-41923'   # 77K             # The numbers of the empty runs to be subtracted
    #     spectra='3-134'                            # Spectra to be analysed
    #     tof_binning='275.,1.,420'                    # Binning of ToF spectra
    #     mode='DoubleDifference'
    #     ipfile=r'./ip2019.par' 

    #     # Masses, instrument parameters and initial fitting parameters
    #     masses = np.array([12, 16, 27])
    #     noOfMasses = len(masses)
    #     InstrParsPath = repoPath / 'ip2018_3.par'

    #     initPars = np.array([ 
    #     # Intensities, NCP widths, NCP centers   
    #         1, 12, 0.,    
    #         1, 12, 0.,   
    #         1, 12.5, 0.    
    #     ])
    #     bounds = np.array([
    #         [0, np.nan], [8, 16], [-3, 1],
    #         [0, np.nan], [8, 16], [-3, 1],
    #         [0, np.nan], [11, 14], [-3, 1]
    #     ])
    #     constraints = ()

    #     noOfMSIterations = 4     #4
    #     firstSpec = 3    #3
    #     lastSpec = 134    #134

    #     # Boolean Flags to control script
    #     loadWsFromUserPathFlag = True
    #     scaleParsFlag = False
    #     MSCorrectionFlag = True
    #     GammaCorrectionFlag = False
    #     maskedSpecAllNo = np.array([18, 34, 42, 43, 59, 60, 62, 118, 119, 133])

    #     # Parameters below are not to be changed
    #     firstSpecIdx = 0
    #     lastSpecIdx = lastSpec - firstSpec

    #     # Consider only the masked spectra between first and last spectrum
    #     maskedSpecNo = maskedSpecAllNo[
    #         (maskedSpecAllNo >= firstSpec) & (maskedSpecAllNo <= lastSpec)
    #     ]
    #     maskedDetectorIdx = maskedSpecNo - firstSpec

    #     # Set scaling factors for the fitting parameters, default is ones
    #     scalingFactors = np.ones(initPars.shape)
    #     if scaleParsFlag:        # Scale fitting parameters using initial values
    #             initPars[2::3] = np.ones((1, noOfMasses))  # Main problem is that zeros have to be replaced by non zeros
    #             scalingFactors = 1 / initPars



    # def setForwardScatteringInitialConditions(:
    #     modeRunning = "FORWARD"  # Used to control MS correction

    #     userWsRawPath = r"./input_ws/starch_80_RD_raw_forward.nxs"
    #     userWsEmptyPath = r"./input_ws/starch_80_RD_raw_forward.nxs"

    #     name = "starch_80_RD_forward_"
    #     runs='43066-43076'         # 100K        # The numbers of the runs to be analysed
    #     empty_runs='43868-43911'   # 100K        # The numbers of the empty runs to be subtracted
    #     spectra='144-182'                        # Spectra to be analysed
    #     tof_binning="110,1.,430"                 # Binning of ToF spectra
    #     mode='SingleDifference'
    #     ipfile=r'./ip2018_3.par'

    #     masses = np.array([1.0079, 12, 16, 27]) 
    #     noOfMasses = len(masses)
    #     InstrParsPath = repoPath / 'ip2018_3.par'

    #     initPars = np.array([ 
    #     # Intensities, NCP widths, NCP centers  
    #         1, 4.7, 0, 
    #         1, 12.71, 0.,    
    #         1, 8.76, 0.,   
    #         1, 13.897, 0.    
    #     ])
    #     bounds = np.array([
    #         [0, np.nan], [3, 6], [-3, 1],
    #         [0, np.nan], [12.71, 12.71], [-3, 1],
    #         [0, np.nan], [8.76, 8.76], [-3, 1],
    #         [0, np.nan], [13.897, 13.897], [-3, 1]
    #     ])
    #     constraints = ()

    #     noOfMSIterations = 2     #4
    #     firstSpec = 164   #144
    #     lastSpec = 175    #182

    #     # Boolean Flags to control script
    #     loadWsFromUserPathFlag = True
    #     scaleParsFlag = False
    #     MSCorrectionFlag = True
    #     GammaCorrectionFlag = True

    #     # Parameters to control fit in Y-Space
    #     symmetrisationFlag = True
    #     symmetriseHProfileUsingAveragesFlag = True      # When False, use mirror sym
    #     rebinParametersForYSpaceFit = "-20, 0.5, 20"    # Needs to be symetric
    #     singleGaussFitToHProfile = True      # When False, use Hermite expansion
    #     maskedSpecAllNo = np.array([173, 174, 179])

    #     # Parameters below are not to be changed
    #     firstSpecIdx = 0
    #     lastSpecIdx = lastSpec - firstSpec

    #     # Consider only the masked spectra between first and last spectrum
    #     maskedSpecNo = maskedSpecAllNo[
    #         (maskedSpecAllNo >= firstSpec) & (maskedSpecAllNo <= lastSpec)
    #     ]
    #     maskedDetectorIdx = maskedSpecNo - firstSpec

    #     # Set scaling factors for the fitting parameters, default is ones
    #     scalingFactors = np.ones(initPars.shape)
    #     if scaleParsFlag:        # Scale fitting parameters using initial values
    #             initPars[2::3] = np.ones((1, noOfMasses))  # Main problem is that zeros have to be replaced by non zeros
    #             scalingFactors = 1 / initPars
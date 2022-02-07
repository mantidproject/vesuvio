import numpy as np
from pathlib import Path
experimentPath = Path(__file__).absolute().parent  # Path to the repository

testingCleaning = True
if testingCleaning:     
    cleaningPath = experimentPath / "output" / "testing" / "cleaning"


    forwardScatteringSavePath = cleaningPath / "current_forward.npz" 
    backScatteringSavePath = cleaningPath / "current_backward.npz" 
    ySpaceFitSavePath = cleaningPath / "current_yspacefit.npz"

else:
    outputPath = experimentPath / "output" / "testing" / "original" / "current_data"

    forwardScatteringSavePath = outputPath / "4iter_forward_GM_MS.npz"
    backScatteringSavePath = outputPath / "4iter_backward_MS.npz"
    ySpaceFitSavePath = outputPath / "4iter_yspacefit.npz"


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


class BackwardInitialConditions:
    # Multiscaterring Correction Parameters
    HToMass0Ratio = 19.0620008206

    resultsSavePath = backScatteringSavePath

    transmission_guess =  0.8537        # Experimental value from VesuvioTransmission
    multiple_scattering_order, number_of_events = 2, 1.e5   
    hydrogen_peak = True                 # Hydrogen multiple scattering
    
    # Sample slab parameters
    vertical_width, horizontal_width, thickness = 0.1, 0.1, 0.001  # Expressed in meters

    modeRunning = "BACKWARD"

    # Parameters to Load Raw and Empty Workspaces
    userWsRawPath = str(backWsRawPath)
    userWsEmptyPath = str(backWsEmptyPath)
    name = "starch_80_RD_backward_"
    runs='43066-43076'  # 77K             # The numbers of the runs to be analysed
    empty_runs='41876-41923'   # 77K             # The numbers of the empty runs to be subtracted
    spectra='3-134'                            # Spectra to be analysed
    tof_binning='275.,1.,420'                    # Binning of ToF spectra
    mode='DoubleDifference'
    # ipfile='./ip2019.par'

    # Masses, instrument parameters and initial fitting parameters
    masses = np.array([12, 16, 27])
    noOfMasses = len(masses)
    InstrParsPath = ipFilePath

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

    noOfMSIterations = 2     #4
    firstSpec = 3    #3
    lastSpec = 134    #134

    # Boolean Flags to control script
    # loadWsFromUserPathFlag = True
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

    resultsSavePath = forwardScatteringSavePath
    ySpaceFitSavePath = ySpaceFitSavePath

    HToMass0Ratio = None

    transmission_guess =  0.8537        # Experimental value from VesuvioTransmission
    multiple_scattering_order, number_of_events = 2, 1.e5   
    hydrogen_peak = True                 # Hydrogen multiple scattering
    
    # Sample slab parameters
    vertical_width, horizontal_width, thickness = 0.1, 0.1, 0.001  # Expressed in meters

    modeRunning = "FORWARD"  # Used to control MS correction

    userWsRawPath = str(frontWsRawPath)
    userWsEmptyPath = str(frontWsEmptyPath)

    name = "starch_80_RD_forward_"
    runs='43066-43076'         # 100K        # The numbers of the runs to be analysed
    empty_runs='43868-43911'   # 100K        # The numbers of the empty runs to be subtracted
    spectra='144-182'                        # Spectra to be analysed
    tof_binning="110,1.,430"                 # Binning of ToF spectra
    mode='SingleDifference'
    # ipfile='./ip2018_3.par'

    masses = np.array([1.0079, 12, 16, 27]) 
    noOfMasses = len(masses)
    InstrParsPath = ipFilePath

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
    # loadWsFromUserPathFlag = True
    scaleParsFlag = False
    MSCorrectionFlag = True
    GammaCorrectionFlag = True

    # Parameters to control fit in Y-Space
    # symmetrisationFlag = True
    # symmetriseHProfileUsingAveragesFlag = True      # When False, use mirror sym
    # rebinParametersForYSpaceFit = "-20, 0.5, 20"    # Needs to be symetric
    # singleGaussFitToHProfile = True      # When False, use Hermite expansion
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


# Make a child class for the parameters of yspace fitting
# This class inherits all of the atributes in ForwardInitialConditions
class YSpaceFitInitialConditions(ForwardInitialConditions):
    ySpaceFitSavePath = ySpaceFitSavePath

    symmetrisationFlag = True
    symmetriseHProfileUsingAveragesFlag = True      # When False, use mirror sym
    rebinParametersForYSpaceFit = "-20, 0.5, 20"    # Needs to be symetric
    singleGaussFitToHProfile = True      # When False, use Hermite expansion
    

bckwdIC = BackwardInitialConditions
fwdIC = ForwardInitialConditions
yfitIC = YSpaceFitInitialConditions

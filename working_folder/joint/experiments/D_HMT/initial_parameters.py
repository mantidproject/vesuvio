import numpy as np
from pathlib import Path
experimentPath = Path(__file__).absolute().parent  # Path to the repository

testingCleaning = False
if testingCleaning:     
    cleaningPath = experimentPath / "output" / "testing" / "cleaning"


    forwardScatteringSavePath = cleaningPath / "current_forward.npz" 
    backScatteringSavePath = cleaningPath / "current_backward.npz" 
    ySpaceFitSavePath = cleaningPath / "current_yspacefit.npz"

else:
    outputPath = experimentPath / "output"

    forwardScatteringSavePath = outputPath / "1iter_forward_GM_MS.npz"
    backScatteringSavePath = outputPath / "1iter_backward_MS.npz"
    ySpaceFitSavePath = outputPath / "1iter_yspacefit.npz"


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
    HToMass0Ratio = None

    resultsSavePath = backScatteringSavePath

    transmission_guess =  0.92        # Experimental value from VesuvioTransmission
    multiple_scattering_order, number_of_events = 2, 1.e6   
    hydrogen_peak = False                 # Hydrogen multiple scattering
    
    # Sample slab parameters
    vertical_width, horizontal_width, thickness = 0.1, 0.1, 0.001  # Expressed in meters

    modeRunning = "BACKWARD"

    # Parameters to Load Raw and Empty Workspaces
    userWsRawPath = str(backWsRawPath)
    userWsEmptyPath = str(backWsEmptyPath)

    name = "DHMT_300K_backward_"
    runs='36517-36556'              # The numbers of the runs to be analysed
    empty_runs='34038-34045'                # The numbers of the empty runs to be subtracted
    spectra='3-134'                            # Spectra to be analysed
    tof_binning='100.,1.,420'                    # Binning of ToF spectra
    mode = 'DoubleDifference'

    # Masses, instrument parameters and initial fitting parameters
    masses = np.array([2.015, 12, 14, 27])
    noOfMasses = len(masses)
    InstrParsPath = ipFilePath

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

    # Boolean Flags to control script
    # loadWsFromUserPathFlag = True
    scaleParsFlag = False
    MSCorrectionFlag = True
    GammaCorrectionFlag = False
    maskedSpecAllNo = np.array([18, 34, 42, 43, 62])

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

    transmission_guess =  0.92       # Experimental value from VesuvioTransmission
    multiple_scattering_order, number_of_events = 2, 1.e6   
    hydrogen_peak = True                 # Hydrogen multiple scattering
    
    # Sample slab parameters
    vertical_width, horizontal_width, thickness = 0.1, 0.1, 0.001  # Expressed in meters

    modeRunning = "FORWARD"  # Used to control MS correction

    userWsRawPath = str(frontWsRawPath)
    userWsEmptyPath = str(frontWsEmptyPath)

    name = "DHMT_300K_RD_forward_"
    runs='36517-36556'                       # The numbers of the runs to be analysed
    empty_runs='34038-34045'                 # The numbers of the empty runs to be subtracted
    spectra='135-182'                        # Spectra to be analysed
    tof_binning="110,1.,430"                 # Binning of ToF spectra
    mode='SingleDifference'

    masses = np.array([2.015, 12, 14, 27]) 
    noOfMasses = len(masses)
    InstrParsPath = ipFilePath

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


    noOfMSIterations = 1   #4
    firstSpec = 135 #164   #144
    lastSpec = 182 #175    #182

    # Boolean Flags to control script
    # loadWsFromUserPathFlag = True
    scaleParsFlag = False
    MSCorrectionFlag = True
    GammaCorrectionFlag = True

    # Parameters to control fit in Y-Space
    symmetrisationFlag = True
    symmetriseHProfileUsingAveragesFlag = True      # When False, use mirror sym
    rebinParametersForYSpaceFit = "-20, 0.5, 20"    # Needs to be symetric
    singleGaussFitToHProfile = True      # When False, use Hermite expansion
    maskedSpecAllNo = np.array([180])

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

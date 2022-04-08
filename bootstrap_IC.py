from vesuvio_analysis.core_functions.fit_in_yspace import fitInYSpaceProcedure
from vesuvio_analysis.core_functions.procedures import runIndependentIterativeProcedure, runJointBackAndForwardProcedure, extractNCPFromWorkspaces
# from vesuvio_analysis.directories_helpers import IODirectoriesForSample, loadWsFromLoadVesuvio
from vesuvio_analysis.core_functions.bootstrap import runBootstrap
from vesuvio_analysis.ICHelpers import completeICFromInputs
from mantid.api import AnalysisDataService, mtd
import time
import numpy as np
from pathlib import Path


scriptName =  Path(__file__).name.split(".")[0]  # Take out .py
experimentPath = Path(__file__).absolute().parent / "experiments" / scriptName  # Path to the repository
ipFilesPath = Path(__file__).absolute().parent / "vesuvio_analysis" / "ip_files"


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


class GeneralInitialConditions:
    """Used to define initial conditions shared by both Back and Forward scattering"""
    
    transmission_guess =  0.8537        # Experimental value from VesuvioTransmission
    multiple_scattering_order, number_of_events = 2, 1.e5
    # Sample slab parameters
    vertical_width, horizontal_width, thickness = 0.1, 0.1, 0.001  # Expressed in meters


class BackwardInitialConditions(GeneralInitialConditions):
    InstrParsPath = ipFilesPath / "ip2018_3.par" 

    HToMass0Ratio = 19.0620008206  # Set to zero or None when H is not present

    # Masses, instrument parameters and initial fitting parameters
    masses = np.array([12, 16, 27])
    # noOfMasses = len(masses)

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

    maskedSpecAllNo = np.array([18, 34, 42, 43, 59, 60, 62, 118, 119, 133])

    # Boolean Flags to control script
    MSCorrectionFlag = True
    GammaCorrectionFlag = False

    # # Parameters of workspaces in input_ws
    tof_binning='275.,1.,420'                    # Binning of ToF spectra



class ForwardInitialConditions(GeneralInitialConditions):
    InstrParsPath = ipFilesPath / "ip2018_3.par" 

    masses = np.array([1.0079, 12, 16, 27]) 
    # noOfMasses = len(masses)

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
    firstSpec = 144   #144
    lastSpec = 154    #182

    # Boolean Flags to control script
    MSCorrectionFlag = True
    GammaCorrectionFlag = True

    maskedSpecAllNo = np.array([173, 174, 179])

    tof_binning="110,1,430"                 # Binning of ToF spectra
 

# This class inherits all of the atributes in ForwardInitialConditions
class YSpaceFitInitialConditions(ForwardInitialConditions):
    # ySpaceFitSavePath = ySpaceFitSavePath

    symmetrisationFlag = True
    rebinParametersForYSpaceFit = "-25, 0.5, 25"    # Needs to be symetric
    singleGaussFitToHProfile = False      # When False, use Hermite expansion
    globalFitFlag = True
    forceManualMinos = False
    nGlobalFitGroups = 4


class bootstrapInitialConditions:
    speedQuick = False
    nSamples = 3


icWSBack = LoadVesuvioBackParameters
icWSFront = LoadVesuvioFrontParameters  

bckwdIC = BackwardInitialConditions
fwdIC = ForwardInitialConditions
yfitIC = YSpaceFitInitialConditions

bootIC = bootstrapInitialConditions

completeICFromInputs(fwdIC, scriptName, icWSFront, bootIC)
completeICFromInputs(bckwdIC, scriptName, icWSBack, bootIC)


start_time = time.time()
# Interactive section 

runIndependentIterativeProcedure(bckwdIC)

# runBootstrap(bckwdIC, bootIC)
# bootIC.speedQuick = False
# runBootstrap(bckwdIC, bootIC)


# End of iteractive section
end_time = time.time()
print("\nRunning time: ", end_time-start_time, " seconds")

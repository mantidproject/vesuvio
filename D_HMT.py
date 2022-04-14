from vesuvio_analysis.core_functions.fit_in_yspace import fitInYSpaceProcedure
from vesuvio_analysis.core_functions.procedures import runJointBackAndForwardProcedure, runIndependentIterativeProcedure
from vesuvio_analysis.core_functions.bootstrap import runBootstrap
from vesuvio_analysis.ICHelpers import completeICFromInputs
from mantid.api import AnalysisDataService, mtd
import time
import numpy as np
from pathlib import Path

scriptName =  Path(__file__).name.split(".")[0]  # Take out .py
experimentPath = Path(__file__).absolute().parent / "experiments" / scriptName  # Path to experiments/sample
ipFilesPath = Path(__file__).absolute().parent / "vesuvio_analysis" / "ip_files"


class LoadVesuvioBackParameters:
    runs='36517-36556'              # The numbers of the runs to be analysed
    empty_runs='34038-34045'                # The numbers of the empty runs to be subtracted
    spectra='3-134'                            # Spectra to be analysed
    mode = 'DoubleDifference'
    ipfile=str(ipFilesPath / "ip2018_3.par")   


class LoadVesuvioFrontParameters:
    runs='36517-36556'                       # The numbers of the runs to be analysed
    empty_runs='34038-34045'                 # The numbers of the empty runs to be subtracted
    spectra='135-182'                        # Spectra to be analysed
    mode='SingleDifference'
    ipfile=str(ipFilesPath / "ip2018_3.par") 


class GeneralInitialConditions:
    """Used to define initial conditions shared by both Back and Forward scattering"""
    
    transmission_guess =  0.92        # Experimental value from VesuvioTransmission
    multiple_scattering_order, number_of_events = 2, 1.e5
    # Sample slab parameters
    vertical_width, horizontal_width, thickness = 0.1, 0.1, 0.001  # Expressed in meters


class BackwardInitialConditions(GeneralInitialConditions):
    InstrParsPath = ipFilesPath / "ip2018_3.par" 

    # TODO: Automatically prevent MS from adding H even if ratio!=None
    HToMass0Ratio = None   # Set to zero or None when H is not present

    # Masses, instrument parameters and initial fitting parameters
    masses = np.array([2.015, 12, 14, 27])
    # noOfMasses = len(masses)

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

    noOfMSIterations = 2     #4
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

    # HToMass0Ratio = None

    masses = np.array([2.015, 12, 14, 27]) 
    # noOfMasses = len(masses)

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
    
    noOfMSIterations = 1 #2   #4
    firstSpec = 135   #135
    lastSpec = 155  #182

    # Boolean Flags to control script
    MSCorrectionFlag = True
    GammaCorrectionFlag = True

    maskedSpecAllNo = np.array([180])

    tof_binning="110,1.,430"                 # Binning of ToF spectra
 

# This class inherits all of the atributes in ForwardInitialConditions
class YSpaceFitInitialConditions:
    showPlots = True
    symmetrisationFlag = True
    rebinParametersForYSpaceFit = "-30, 0.5, 30"    # Needs to be symetric
    singleGaussFitToHProfile = True    # When False, use Hermite expansion
    globalFitFlag = True
    forceManualMinos = True
    nGlobalFitGroups = 4   


class bootstrapInitialConditions:
    speedQuick = False
    nSamples = 10
    ySpaceFit = True


icWSBack = LoadVesuvioBackParameters
icWSFront = LoadVesuvioFrontParameters  

bckwdIC = BackwardInitialConditions
fwdIC = ForwardInitialConditions
yfitIC = YSpaceFitInitialConditions

bootIC = bootstrapInitialConditions

completeICFromInputs(fwdIC, scriptName, icWSFront, bootIC)
completeICFromInputs(bckwdIC, scriptName, icWSBack, bootIC)


start_time = time.time()
# Start of interactive section 

# wsName = f"{fwdIC.name}{fwdIC.noOfMSIterations-1}"   #"DHMT_300K_RD_forward_"+str(fwdIC.noOfMSIterations-1)
# if wsName in mtd:
#     wsFinal = mtd[wsName]
#     allNCP = extractNCPFromWorkspaces(wsFinal)     # Seems that it is not working
# else:
#     wsFinal, forwardScatteringResults = runIndependentIterativeProcedure(fwdIC)
#     lastIterationNCP = forwardScatteringResults.all_ncp_for_each_mass[-1]
#     allNCP = lastIterationNCP

# assert ~np.all(allNCP==0), "NCP extraction not working!"

# print("\nFitting workspace ", wsFinal.name(), " in Y Space.")
# fitInYSpaceProcedure(yfitIC, wsFinal, allNCP)

# wsFinal, scatteringResults = runIndependentIterativeProcedure(fwdIC)
# fitInYSpaceProcedure(yfitIC, fwdIC, wsFinal)


# Run Bootstrap procedures
runBootstrap(fwdIC, bootIC, yfitIC)
# bootIC.speedQuick = False
# runBootstrap(fwdIC, bootIC, yfitIC)



# End of iteractive section
end_time = time.time()
print("\nRunning time: ", end_time-start_time, " seconds")

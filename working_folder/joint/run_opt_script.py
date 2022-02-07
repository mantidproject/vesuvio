
# Set desired initial conditions file
from experiments.starch_80_RD.initial_parameters import bckwdIC, fwdIC, yfitIC
# from experiments.D_HMT.initial_parameters import bckwdIC, fwdIC, yfitIC

from analysis_functions import iterativeFitForDataReduction
from mantid.api import AnalysisDataService, mtd
from fit_in_yspace import fitInYSpaceProcedure
import time
import numpy as np


def runIndependentIterativeProcedure(IC):
    """Runs the iterative fitting of NCP.
    input: Backward or Forward scattering initial conditions object
    output: Final workspace that was fitted, object with results arrays"""

    AnalysisDataService.clear()
    wsFinal, ncpFitResultsObject = iterativeFitForDataReduction(IC)
    return wsFinal, ncpFitResultsObject


# def runOnlyBackScattering(bckwdIC):
#     AnalysisDataService.clear()
#     wsFinal, backScatteringResults = iterativeFitForDataReduction(bckwdIC)


# def runOnlyForwardScattering(fwdIC):
#     AnalysisDataService.clear()
#     wsFinal, forwardScatteringResults = iterativeFitForDataReduction(fwdIC)
#     fitInYSpaceProcedure(fwdIC, wsFinal, forwardScatteringResults.all_ncp_for_each_mass[-1])
 

def runSequenceForKnownRatio(bckwdIC, fwdIC):
    AnalysisDataService.clear()
    # If H to first mass ratio is known, can run MS correction for backscattering
    # Back scattering produces precise results for widhts and intensity ratios for non-H masses
    wsFinal, backScatteringResults = iterativeFitForDataReduction(bckwdIC)
    setInitFwdParsFromBackResults(backScatteringResults, bckwdIC.HToMass0Ratio, fwdIC)
    wsFinal, forwardScatteringResults = iterativeFitForDataReduction(fwdIC)
    return wsFinal, forwardScatteringResults
    # fitInYSpaceProcedure(fwdIC, wsFinal, forwardScatteringResults.all_ncp_for_each_mass[-1])


def runSequenceRatioNotKnown(bckwdIC, fwdIC):
    # Run preliminary forward with a good guess for the widths of non-H masses
    wsFinal, forwardScatteringResults = iterativeFitForDataReduction(fwdIC)
    for i in range(2):    # Loop until convergence is achieved
        AnalysisDataService.clear()    # Clears all Workspaces
        # Get first estimate of H to mass0 ratio
        fwdMeanIntensityRatios = forwardScatteringResults.all_mean_intensities[-1] 
        bckwdIC.HToMass0Ratio = fwdMeanIntensityRatios[0] / fwdMeanIntensityRatios[1]
        # Run backward procedure with this estimate
        wsFinal, backScatteringResults = iterativeFitForDataReduction(bckwdIC)
        setInitFwdParsFromBackResults(backScatteringResults, bckwdIC.HToMass0Ratio, fwdIC)
        wsFinal, forwardScatteringResults = iterativeFitForDataReduction(fwdIC)
    # fitInYSpaceProcedure(fwdIC, wsFinal, forwardScatteringResults.all_ncp_for_each_mass[-1])
    return wsFinal, forwardScatteringResults
 


def setInitFwdParsFromBackResults(backScatteringResults, HToMass0Ratio, fwdIC):
    """Takes widths and intensity ratios obtained in backscattering
    and uses them as the initial conditions for forward scattering """

    # Get widts and intensity ratios from backscattering results
    backMeanWidths = backScatteringResults.all_mean_widths[-1]
    backMeanIntensityRatios = backScatteringResults.all_mean_intensities[-1] 

    if fwdIC.masses[0] == 1.0079:
        HIntensity = HToMass0Ratio * backMeanIntensityRatios[0]
        initialFwdIntensityRatios = np.append([HIntensity], backMeanIntensityRatios)
        initialFwdIntensityRatios /= np.sum(initialFwdIntensityRatios)
        # Set starting conditions for forward scattering
        # Fix known widths and intensity ratios from back scattering results
        fwdIC.initPars[4::3] = backMeanWidths
        fwdIC.bounds[4::3] = backMeanWidths[:, np.newaxis] * np.ones((1,2))
        fwdIC.initPars[0::3] = initialFwdIntensityRatios

    else:
        fwdIC.initPars[1::3] = backMeanWidths
        # First width is set to vary
        fwdIC.bounds[4::3] = backMeanWidths[1:][:, np.newaxis] * np.ones((1,2))
        fwdIC.initPars[0::3] = backMeanIntensityRatios

    print("\nChanged initial conditions of forward scattering according to mean widhts and intensity ratios from backscattering.\n")

start_time = time.time()
# Interactive section 

wsFinal, fwdResults = runIndependentIterativeProcedure(fwdIC)
# wsFinal = mtd["starch_80_RD_forward_1"]

# Choose whether to get ncp from the workspace or from results object
# Useful when running in Mantid, can select workspace to fit 
# by assigning it to wsFinal
ncpFromWs = True
if ncpFromWs:  
    allNCP = mtd[wsFinal.name()+"_tof_fitted_profile_1"].extractY()[np.newaxis, :, :]
    for i in range(2, fwdIC.noOfMasses+1):
        ncpToAppend = mtd[wsFinal.name()+"_tof_fitted_profile_" + str(i)].extractY()[np.newaxis, :, :]
        allNCP = np.append(allNCP, ncpToAppend, axis=0)
else:
    lastIterationNCP = fwdResults.all_ncp_for_each_mass[-1]
    allNCP = lastIterationNCP

def switchFirstTwoAxis(A):
    return np.stack(np.split(A, len(A), axis=0), axis=2)[0]

allNCP = switchFirstTwoAxis(allNCP)
print("\nShape of all NCP: ", allNCP.shape)

print("\nFitting workspace in Y Space: ", wsFinal.name())

fitInYSpaceProcedure(yfitIC, wsFinal, allNCP)


# End of iteractive section
end_time = time.time()
print("\nRunning time: ", end_time-start_time, " seconds")

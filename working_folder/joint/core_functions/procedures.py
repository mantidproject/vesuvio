from .analysis_functions import iterativeFitForDataReduction, switchFirstTwoAxis
from mantid.api import AnalysisDataService, mtd
import numpy as np

def runIndependentIterativeProcedure(IC):
    """Runs the iterative fitting of NCP.
    input: Backward or Forward scattering initial conditions object
    output: Final workspace that was fitted, object with results arrays"""

    AnalysisDataService.clear()
    wsFinal, ncpFitResultsObject = iterativeFitForDataReduction(IC)
    return wsFinal, ncpFitResultsObject


def runJointBackAndForward(bckwdIC, fwdIC):
    """Used when H is not present.
    Runs backward scattering and uses the resulting widhts and intensity ratios
    to set initial forward scattering parameters."""
    # Clear all workspaces
    AnalysisDataService.clear()

    # Run backward scattering
    wsFinal, backScatteringResults = iterativeFitForDataReduction(bckwdIC)

    # Set initial parameters for forward scattering
    # No H present, set starting widths and intensity ratios to the resutls from backscattering
    backMeanWidths = backScatteringResults.all_mean_widths[-1]
    backMeanIntensityRatios = backScatteringResults.all_mean_intensities[-1] 

    fwdIC.initPars[1::3] = backMeanWidths       # Set widths
    fwdIC.bounds[4::3] = backMeanWidths[1:][:, np.newaxis] * np.ones((1,2))   # Fix all widhts except first
    fwdIC.initPars[0::3] = backMeanIntensityRatios    # Set intensity ratios
    print("\nChanged initial conditions of forward scattering according to mean widhts and intensity ratios from backscattering.\n")

    # Run forward scattering
    wsFinal, forwardScatteringResults = iterativeFitForDataReduction(fwdIC)  
    return wsFinal, forwardScatteringResults


def runSequenceForKnownHRatio(bckwdIC, fwdIC):
    """When H is present and H to first mass ratio is known.
    Runs backscattering and uses results + H to mass ratio to set up initial forward parameters."""
    AnalysisDataService.clear()
    # If H to first mass ratio is known, can run MS correction for backscattering
    # Back scattering produces precise results for widhts and intensity ratios for non-H masses
    wsFinal, backScatteringResults = iterativeFitForDataReduction(bckwdIC)
    setInitFwdParsFromBackResultsAndHRatio(backScatteringResults, bckwdIC.HToMass0Ratio, fwdIC)
    wsFinal, forwardScatteringResults = iterativeFitForDataReduction(fwdIC)
    return wsFinal, forwardScatteringResults


def runSequenceHRatioNotKnown(bckwdIC, fwdIC):
    # Run preliminary forward with a good guess for the widths of non-H masses
    wsFinal, forwardScatteringResults = iterativeFitForDataReduction(fwdIC)
    for i in range(2):    # Loop until convergence is achieved
        AnalysisDataService.clear()    # Clears all Workspaces
        # Get first estimate of H to mass0 ratio
        fwdMeanIntensityRatios = forwardScatteringResults.all_mean_intensities[-1] 
        bckwdIC.HToMass0Ratio = fwdMeanIntensityRatios[0] / fwdMeanIntensityRatios[1]
        # Run backward procedure with this estimate
        wsFinal, backScatteringResults = iterativeFitForDataReduction(bckwdIC)
        setInitFwdParsFromBackResultsAndHRatio(backScatteringResults, bckwdIC.HToMass0Ratio, fwdIC)
        wsFinal, forwardScatteringResults = iterativeFitForDataReduction(fwdIC)
    return wsFinal, forwardScatteringResults
 

def setInitFwdParsFromBackResultsAndHRatio(backScatteringResults, HToMass0Ratio, fwdIC):
    """Takes widths and intensity ratios obtained in backscattering, 
    H to mass ratios and sets the initial conditions for forward scattering."""

    # Get widts and intensity ratios from backscattering results
    backMeanWidths = backScatteringResults.all_mean_widths[-1]
    backMeanIntensityRatios = backScatteringResults.all_mean_intensities[-1] 

    HIntensity = HToMass0Ratio * backMeanIntensityRatios[0]
    initialFwdIntensityRatios = np.append([HIntensity], backMeanIntensityRatios)
    initialFwdIntensityRatios /= np.sum(initialFwdIntensityRatios)
    # Set starting conditions for forward scattering
    # Fix known widths and intensity ratios from back scattering results
    fwdIC.initPars[4::3] = backMeanWidths
    fwdIC.bounds[4::3] = backMeanWidths[:, np.newaxis] * np.ones((1,2))
    fwdIC.initPars[0::3] = initialFwdIntensityRatios
    print("\nChanged initial conditions of forward scattering according to mean widhts and intensity ratios from backscattering.\n")


def extractNCPFromWorkspaces(wsFinal):
    allNCP = mtd[wsFinal.name()+"_tof_fitted_profile_1"].extractY()[np.newaxis, :, :]
    i = 2
    while True:   # By default, looks for all ncp ws until it breaks
        try:
            ncpToAppend = mtd[wsFinal.name()+"_tof_fitted_profile_" + str(i)].extractY()[np.newaxis, :, :]
            allNCP = np.append(allNCP, ncpToAppend, axis=0)
            i += 1
        except KeyError:
            break
    allNCP = switchFirstTwoAxis(allNCP)
    return allNCP
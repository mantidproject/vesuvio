from build_init_pars import bckwdIC, fwdIC
from Optimized_joint_script import iterativeFitForDataReduction, printInitialParameters
from mantid.api import AnalysisDataService
from fit_in_yspace import fitInYSpaceProcedure
import time
import numpy as np


def runOnlyBackScattering(bckwdIC):
    AnalysisDataService.clear()
    # ic.setBackscatteringInitialConditions()
    printInitialParameters(bckwdIC)
    wsFinal, backScatteringResults = iterativeFitForDataReduction(bckwdIC)
    backScatteringResults.save() 


def runOnlyForwardScattering(fwdIC):
    AnalysisDataService.clear()
    # ic.setForwardScatteringInitialConditions()
    printInitialParameters(fwdIC)
    wsFinal, forwardScatteringResults = iterativeFitForDataReduction(fwdIC)
    fitInYSpaceProcedure(fwdIC, wsFinal, forwardScatteringResults.all_ncp_for_each_mass[-1])
    forwardScatteringResults.save()


def runSequenceForKnownRatio(bckwdIC, fwdIC):
    AnalysisDataService.clear()
    # If H to first mass ratio is known, can run MS correction for backscattering
    # Back scattering produces precise results for widhts and intensity ratios for non-H masses
    printInitialParameters(bckwdIC)
    wsFinal, backScatteringResults = iterativeFitForDataReduction(bckwdIC)
    backScatteringResults.save() 

    setInitFwdParsFromBackResults(backScatteringResults, bckwdIC.HToMass0Ratio, fwdIC)
    printInitialParameters(fwdIC)
    wsFinal, forwardScatteringResults = iterativeFitForDataReduction(fwdIC)
    fitInYSpaceProcedure(fwdIC, wsFinal, forwardScatteringResults.all_ncp_for_each_mass[-1])
    forwardScatteringResults.save()


def runSequenceRatioNotKnown(bckwdIC, fwdIC):
    # Run preliminary forward with a good guess for the widths of non-H masses
    # ic.setForwardScatteringInitialConditions()
    printInitialParameters(fwdIC)
    wsFinal, forwardScatteringResults = iterativeFitForDataReduction(fwdIC)
    for i in range(2):    # Loop until convergence is achieved
        AnalysisDataService.clear()    # Clears all Workspaces

        # Get first estimate of H to mass0 ratio
        fwdMeanIntensityRatios = forwardScatteringResults.all_mean_intensities[-1] 
        bckwdIC.HToMass0Ratio = fwdMeanIntensityRatios[0] / fwdMeanIntensityRatios[1]
       
        # Run backward procedure with this estimate
        # ic.setBackscatteringInitialConditions()
        printInitialParameters(bckwdIC)
        wsFinal, backScatteringResults = iterativeFitForDataReduction(bckwdIC)

        # ic.setForwardScatteringInitialConditions()
        setInitFwdParsFromBackResults(backScatteringResults, bckwdIC.HToMass0Ratio, fwdIC)
        printInitialParameters(fwdIC)
        wsFinal, forwardScatteringResults = iterativeFitForDataReduction(fwdIC)

    fitInYSpaceProcedure(fwdIC, wsFinal, forwardScatteringResults.all_ncp_for_each_mass[-1])
    backScatteringResults.save()
    forwardScatteringResults.save()


def setInitFwdParsFromBackResults(backScatteringResults, HToMass0Ratio, fwdIC):
    """Takes widths and intensity ratios obtained in backscattering
    and uses them as the initial conditions for forward scattering """

    # Get widts and intensity ratios from backscattering results
    backMeanWidths = backScatteringResults.all_mean_widths[-1]
    backMeanIntensityRatios = backScatteringResults.all_mean_intensities[-1] 

    HIntensity = HToMass0Ratio * backMeanIntensityRatios[0]
    initialFwdIntensityRatios = np.append([HIntensity], backMeanIntensityRatios)
    initialFwdIntensityRatios /= np.sum(initialFwdIntensityRatios)

    # Set starting conditions for forward scattering
    # Fix known widths and intensity ratios from back scattering results
    fwdIC.initPars[4::3] = backMeanWidths
    fwdIC.initPars[0::3] = initialFwdIntensityRatios
    fwdIC.bounds[4::3] = backMeanWidths[:, np.newaxis] * np.ones((1,2))
    # Fix the intensity ratios 
    # ic.bounds[0::3] = initialFwdIntensityRatios[:, np.newaxis] * np.ones((1,2)) 

    print("\nChanged initial conditions of forward scattering \
        according to mean widhts and intensity ratios from backscattering.\n")


start_time = time.time()

runOnlyForwardScattering(fwdIC)

end_time = time.time()
print("\nRunning time: ", end_time-start_time, " seconds")

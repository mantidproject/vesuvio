from core_functions.fit_in_yspace import fitInYSpaceProcedure
from starch_80_RD import ForwardInitialConditions
from mantid.simpleapi import Load
from mantid.api import AnalysisDataService
from pathlib import Path
import numpy as np
experimentPath = Path(__file__).absolute().parent / "experiments" / "starch_80_RD"  # Path to the repository

AnalysisDataService.clear()

wsPath = experimentPath / "ws_test_yspace_fit.nxs"
wsFinal = Load(str(wsPath))

experimentPath = Path(__file__).absolute().parent / "experiments" / "starch_80_RD"  # Path to the repository
cleaningPath = experimentPath / "output" / "testing" / "cleaning"


oriPath = cleaningPath / "starter_forward.npz"
AllNCP = np.load(oriPath)["all_ncp_for_each_mass"][-1]



ySpaceFitSavePath = cleaningPath / "current_yspacefit.npz"

class YSpaceFitInitialConditions(ForwardInitialConditions):
    ySpaceFitSavePath = ySpaceFitSavePath

    symmetrisationFlag = True
    symmetriseHProfileUsingAveragesFlag = True      # When False, use mirror sym
    rebinParametersForYSpaceFit = "-20, 0.5, 20"    # Needs to be symetric
    resolutionRebinPars = "-20, 0.5, 20" 
    singleGaussFitToHProfile = True      # When False, use Hermite expansion
    globalFitFlag = True
    forceManualMinos = False
    nGlobalFitGroups = 4
 
yfitIC = YSpaceFitInitialConditions


fitInYSpaceProcedure(yfitIC, wsFinal, AllNCP)
from core_functions.fit_in_yspace import fitInYSpaceProcedure
from starch_80_RD import yfitIC
from mantid.simpleapi import Load
from pathlib import Path
import numpy as np
experimentPath = Path(__file__).absolute().parent / "experiments" / "starch_80_RD"  # Path to the repository

wsPath = experimentPath / "ws_test_yspace_fit.nxs"
wsFinal = Load(str(wsPath))

experimentPath = Path(__file__).absolute().parent / "experiments" / "starch_80_RD"  # Path to the repository
cleaningPath = experimentPath / "output" / "testing" / "cleaning"


oriPath = cleaningPath / "starter_forward.npz"
AllNCP = np.load(oriPath)["all_ncp_for_each_mass"][-1]

fitInYSpaceProcedure(yfitIC, wsFinal, AllNCP)
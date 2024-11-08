from mvesuvio.util.process_inputs import (
    completeICFromInputs,
    completeYFitIC,
)
from mvesuvio.analysis_fitting import fitInYSpaceProcedure
from mvesuvio.analysis_routines import (
    runIndependentIterativeProcedure,
    runJointBackAndForwardProcedure,
    runPreProcToEstHRatio,
)
from mvesuvio.util import handle_config

from mantid.api import mtd
import numpy as np

import time
from pathlib import Path
import importlib
import sys


class Runner:
    def __init__(self, yes_to_all=False, running_tests=False) -> None:
        self.yes_to_all = yes_to_all
        self.running_tests = running_tests
        self.inputs_path = Path(handle_config.read_config_var("caching.inputs"))
        self.setup()

    def setup(self):
        
        ai = self.import_from_inputs()

        self.wsBackIC = ai.LoadVesuvioBackParameters
        self.wsFrontIC = ai.LoadVesuvioFrontParameters
        self.bckwdIC = ai.BackwardInitialConditions
        self.fwdIC = ai.ForwardInitialConditions
        self.yFitIC = ai.YSpaceFitInitialConditions
        self.userCtr = ai.UserScriptControls

        # Set extra attributes from user attributes
        completeICFromInputs(self.fwdIC, self.wsFrontIC)
        completeICFromInputs(self.bckwdIC, self.wsBackIC)
        completeYFitIC(self.yFitIC)
        checkInputs(self.userCtr)

        # Names of workspaces to check if they exist to do fitting
        self.ws_to_fit_y_space = []
        self.classes_to_fit_y_space = []
        for mode, i_cls in zip(["BACKWARD", "FORWARD"], [self.bckwdIC, self.fwdIC]):
            if (self.userCtr.fitInYSpace == mode) | (self.userCtr.fitInYSpace == "JOINT"):
                self.ws_to_fit_y_space.append(i_cls.name + '_' + str(i_cls.noOfMSIterations))
                self.classes_to_fit_y_space.append(i_cls)

        self.analysis_result = None
        self.fitting_result = None


    def import_from_inputs(self):
        name = "analysis_inputs"
        spec = importlib.util.spec_from_file_location(name, self.inputs_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module


    def run(self):
        start_time = time.time()

        if not self.userCtr.runRoutine:
            return
        # Default workflow for procedure + fit in y space

        # If any ws for y fit already loaded
        wsInMtd = [ws in mtd for ws in self.ws_to_fit_y_space]  # Bool list
        if (len(wsInMtd) > 0) and all(wsInMtd):
            self.runAnalysisFitting()
            return self.analysis_result, self.fitting_result  

        checkUserClearWS(self.yes_to_all)  # Check if user is OK with cleaning all workspaces

        self.runAnalysisRoutine()
        self.runAnalysisFitting()

        end_time = time.time()
        print("\nRunning time: ", end_time - start_time, " seconds")

        # Return results used only in tests
        return self.analysis_result, self.fitting_result  


    def runAnalysisFitting(self):
        for wsName, i_cls in zip(self.ws_to_fit_y_space, self.classes_to_fit_y_space):
            self.fitting_result = fitInYSpaceProcedure(self.yFitIC, i_cls, wsName)
        return


    def runAnalysisRoutine(self):
        routine_type = self.userCtr.procedure

        if routine_type is None:
            return

        if (routine_type == "BACKWARD") | (routine_type== "JOINT"):

            if isHPresent(self.fwdIC.masses) & (self.bckwdIC.HToMassIdxRatio==0):
                runPreProcToEstHRatio(self.bckwdIC, self.fwdIC)
                return

            assert isHPresent(self.fwdIC.masses) != (
                self.bckwdIC.HToMassIdxRatio==0 
            ), "When H is not present, HToMassIdxRatio has to be set to None"

        if routine_type == "BACKWARD":
            self.analysis_result = runIndependentIterativeProcedure(self.bckwdIC, running_tests=self.running_tests)
        if routine_type == "FORWARD":
            self.analysis_result = runIndependentIterativeProcedure(self.fwdIC, running_tests=self.running_tests)
        if routine_type == "JOINT":
            self.analysis_result = runJointBackAndForwardProcedure(self.bckwdIC, self.fwdIC)
        return 


def checkUserClearWS(yes_to_all=False):
    """If any workspace is loaded, check if user is sure to start new procedure."""

    if not yes_to_all and len(mtd) != 0:
        userInput = input(
            "This action will clean all current workspaces to start anew. Proceed? (y/n): "
        )
        if (userInput == "y") | (userInput == "Y"):
            pass
        else:
            raise KeyboardInterrupt("Run of procedure canceled.")
    return


def checkInputs(crtIC):
    try:
        if ~crtIC.runRoutine:
            return
    except AttributeError:
        if ~crtIC.runBootstrap:
            return

    for flag in [crtIC.procedure, crtIC.fitInYSpace]:
        assert (
            (flag == "BACKWARD")
            | (flag == "FORWARD")
            | (flag == "JOINT")
            | (flag is None)
        ), "Option not recognized."

    if (crtIC.procedure != "JOINT") & (crtIC.fitInYSpace is not None):
        assert crtIC.procedure == crtIC.fitInYSpace


def isHPresent(masses) -> bool:
    Hmask = np.abs(masses - 1) / 1 < 0.1  # H mass whithin 10% of 1 au

    if np.any(Hmask):  # H present
        print("\nH mass detected.\n")
        assert (
            len(Hmask) > 1
        ), "When H is only mass present, run independent forward procedure, not joint."
        assert Hmask[0], "H mass needs to be the first mass in masses and initPars."
        assert sum(Hmask) == 1, "More than one mass very close to H were detected."
        return True
    else:
        return False

if __name__ == "__main__":
    Runner().run()

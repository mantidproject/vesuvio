from mvesuvio.util.process_inputs import (
    buildFinalWSName,
    completeICFromInputs,
    completeYFitIC,
)
from mvesuvio.analysis_fitting import fitInYSpaceProcedure
from mvesuvio.analysis_routines import (
    runIndependentIterativeProcedure,
    runJointBackAndForwardProcedure,
    runPreProcToEstHRatio,
)

from mantid.api import mtd
import numpy as np

def runRoutine(
    userCtr,
    wsBackIC,
    wsFrontIC,
    bckwdIC,
    fwdIC,
    yFitIC,
    yes_to_all=False,
    running_tests=False
):
    # Set extra attributes from user attributes
    completeICFromInputs(fwdIC, wsFrontIC)
    completeICFromInputs(bckwdIC, wsBackIC)
    completeYFitIC(yFitIC)
    checkInputs(userCtr)

    def runProcedure():
        proc = userCtr.procedure  # Shorthad to make it easier to read

        if proc is None:
            return

        if (proc == "BACKWARD") | (proc == "JOINT"):

            if isHPresent(fwdIC.masses) & (bckwdIC.HToMassIdxRatio==0):
                runPreProcToEstHRatio(bckwdIC, fwdIC)
                return

            assert isHPresent(fwdIC.masses) != (
                bckwdIC.HToMassIdxRatio==0 
            ), "When H is not present, HToMassIdxRatio has to be set to None"

        if proc == "BACKWARD":
            res = runIndependentIterativeProcedure(bckwdIC, running_tests=running_tests)
        if proc == "FORWARD":
            res = runIndependentIterativeProcedure(fwdIC, running_tests=running_tests)
        if proc == "JOINT":
            res = runJointBackAndForwardProcedure(bckwdIC, fwdIC)
        return res

    # Names of workspaces to be fitted in y space
    wsNames = []
    ICs = []
    for mode, IC in zip(["BACKWARD", "FORWARD"], [bckwdIC, fwdIC]):
        if (userCtr.fitInYSpace == mode) | (userCtr.fitInYSpace == "JOINT"):
            wsNames.append(IC.name + '_' + str(IC.noOfMSIterations))
            ICs.append(IC)

    # Default workflow for procedure + fit in y space
    if userCtr.runRoutine:
        # Check if final ws are loaded:
        wsInMtd = [ws in mtd for ws in wsNames]  # Bool list
        if (len(wsInMtd) > 0) and all(
            wsInMtd
        ):  # When wsName is empty list, loop doesn't run
            for wsName, IC in zip(wsNames, ICs):
                resYFit = fitInYSpaceProcedure(yFitIC, IC, wsName)
            return None, resYFit  # To match return below.

        checkUserClearWS(yes_to_all)  # Check if user is OK with cleaning all workspaces
        res = runProcedure()

        resYFit = None
        for wsName, IC in zip(wsNames, ICs):
            resYFit = fitInYSpaceProcedure(yFitIC, IC, wsName)

        return res, resYFit  # Return results used only in tests


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

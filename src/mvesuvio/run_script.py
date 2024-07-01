from mvesuvio.ICHelpers import (
    buildFinalWSName,
    completeICFromInputs,
    completeYFitIC,
)
from mvesuvio.fit_in_yspace import fitInYSpaceProcedure
from mvesuvio.procedures import (
    runIndependentIterativeProcedure,
    runJointBackAndForwardProcedure,
    runPreProcToEstHRatio,
    createTableWSHRatios,
    isHPresent,
)
from mantid.api import mtd

def runScript(
    userCtr,
    wsBackIC,
    wsFrontIC,
    bckwdIC,
    fwdIC,
    yFitIC,
    yes_to_all=False,
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

        ranPreliminary = False
        if (proc == "BACKWARD") | (proc == "JOINT"):
            if isHPresent(fwdIC.masses) & (bckwdIC.HToMassIdxRatio is None):
                HRatios, massIdxs = runPreProcToEstHRatio(
                    bckwdIC, fwdIC
                )  # Sets H ratio to bckwdIC automatically
                ranPreliminary = True
            assert isHPresent(fwdIC.masses) != (
                bckwdIC.HToMassIdxRatio is None
            ), "When H is not present, HToMassIdxRatio has to be set to None"

        if proc == "BACKWARD":
            res = runIndependentIterativeProcedure(bckwdIC)
        if proc == "FORWARD":
            res = runIndependentIterativeProcedure(fwdIC)
        if proc == "JOINT":
            res = runJointBackAndForwardProcedure(bckwdIC, fwdIC)

        # If preliminary procedure ran, make TableWS with H ratios values
        if ranPreliminary:
            createTableWSHRatios(HRatios, massIdxs)
        return res

    # Names of workspaces to be fitted in y space
    wsNames = []
    ICs = []
    for mode, IC in zip(["BACKWARD", "FORWARD"], [bckwdIC, fwdIC]):
        if (userCtr.fitInYSpace == mode) | (userCtr.fitInYSpace == "JOINT"):
            wsNames.append(buildFinalWSName(mode, IC))
            ICs.append(IC)

    # Default workflow for procedure + fit in y space
    if userCtr.runRoutine:
        # Check if final ws are loaded:
        wsInMtd = [ws in mtd for ws in wsNames]  # Bool list
        if (len(wsInMtd) > 0) and all(
            wsInMtd
        ):  # When wsName is empty list, loop doesn't run
            for wsName, IC in zip(wsNames, ICs):
                resYFit = fitInYSpaceProcedure(yFitIC, IC, mtd[wsName])
            return None, resYFit  # To match return below.

        checkUserClearWS(yes_to_all)  # Check if user is OK with cleaning all workspaces
        res = runProcedure()

        resYFit = None
        for wsName, IC in zip(wsNames, ICs):
            resYFit = fitInYSpaceProcedure(yFitIC, IC, mtd[wsName])

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

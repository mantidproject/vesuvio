
from vesuvio_analysis.ICHelpers import completeICFromInputs
from vesuvio_analysis.core_functions.bootstrap import runIndependentBootstrap, runJointBootstrap
from vesuvio_analysis.core_functions.fit_in_yspace import fitInYSpaceProcedure
from vesuvio_analysis.core_functions.procedures import runIndependentIterativeProcedure, runJointBackAndForwardProcedure
from mantid.api import mtd


def runScript(userCtr, scriptName, wsBackIC, wsFrontIC, bckwdIC, fwdIC, yFitIC, bootIC):

    # Set extra attributes from user attributes
    completeICFromInputs(fwdIC, scriptName, wsFrontIC)
    completeICFromInputs(bckwdIC, scriptName, wsBackIC)


    if userCtr.procedure == "BACKWARD":
        assert userCtr.procedure == userCtr.fitInYSpace, "For isolated forward and backward, procedure needs to match fitInYSpace."
        def runProcedure():
            return runIndependentIterativeProcedure(bckwdIC)

    elif userCtr.procedure == "FORWARD":
        assert userCtr.procedure == userCtr.fitInYSpace, "For isolated forward and backward, procedure needs to match fitInYSpace."
        def runProcedure():
            return runIndependentIterativeProcedure(fwdIC)
    
    elif userCtr.procedure == "JOINT":
        def runProcedure():
            return runJointBackAndForwardProcedure(bckwdIC, fwdIC)
    else:
        raise ValueError("Procedure not recognized.")


    if userCtr.fitInYSpace == "BACKWARD":
        wsNames = buildFinalWSNames(scriptName, ["BACKWARD"], [bckwdIC])
        ICs = [bckwdIC]

    elif userCtr.fitInYSpace == "FORWARD":
        wsNames = buildFinalWSNames(scriptName, ["FORWARD"], [fwdIC])
        ICs = [fwdIC]

    elif userCtr.fitInYSpace == "JOINT":
        wsNames = buildFinalWSNames(scriptName, ["BACKWARD", "FORWARD"], [bckwdIC, fwdIC])
        ICs = [bckwdIC, fwdIC]
    else:
        raise ValueError("fitInYSpace not recognized.")


    if userCtr.bootstrap == None:
        pass

    elif userCtr.bootstrap == "BACKWARD":
        runIndependentBootstrap(bckwdIC, bootIC, yFitIC)
        return

    elif userCtr.bootstrap == "FORWARD":
        runIndependentBootstrap(fwdIC, bootIC, yFitIC)
        return

    elif userCtr.bootstrap == "JOINT":
        runJointBootstrap(bckwdIC, fwdIC, bootIC, yFitIC)
        return
    else:
        raise ValueError("Bootstrap option not recognized.")


    # Check if final ws are loaded:
    wsInMtd = [ws in mtd for ws in wsNames]   # List of bool

    # If final ws are already loaded
    if all(wsInMtd):
        for wsName, IC in zip(wsNames, ICs):
            fitInYSpaceProcedure(yFitIC, IC, mtd[wsName])
        return
    
    runProcedure()

    for wsName, IC in zip(wsNames, ICs):
        fitInYSpaceProcedure(yFitIC, IC, mtd[wsName])
    return


def buildFinalWSNames(scriptName: str, procedures: list, inputIC: list):
    wsNames = []
    for proc, IC in zip(procedures, inputIC):
        # Format of corrected ws from last iteration
        name = scriptName + "_" + proc + "_" + str(IC.noOfMSIterations-1)
        wsNames.append(name)
    return wsNames

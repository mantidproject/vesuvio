
from vesuvio_analysis.ICHelpers import completeICFromInputs
from vesuvio_analysis.core_functions.bootstrap import runIndependentBootstrap, runJointBootstrap
from vesuvio_analysis.core_functions.fit_in_yspace import fitInYSpaceProcedure
from vesuvio_analysis.core_functions.procedures import runIndependentIterativeProcedure, runJointBackAndForwardProcedure
from mantid.api import mtd


def runScript(userCtr, scriptName, wsBackIC, wsFrontIC, bckwdIC, fwdIC, yFitIC, bootIC):

    # Set extra attributes from user attributes
    completeICFromInputs(fwdIC, scriptName, wsFrontIC)
    completeICFromInputs(bckwdIC, scriptName, wsBackIC)


    if userCtr.procedure == None:
         def runProcedure():
            return None 
    
    elif userCtr.procedure == "BACKWARD":
        if userCtr.fitInYSpace != None:
            assert userCtr.procedure == userCtr.fitInYSpace, "For isolated forward and backward, procedure needs to match fitInYSpace."
        def runProcedure():
            return runIndependentIterativeProcedure(bckwdIC)

    elif userCtr.procedure == "FORWARD":
        if userCtr.fitInYSpace != None:
            assert userCtr.procedure == userCtr.fitInYSpace, "For isolated forward and backward, procedure needs to match fitInYSpace."
        def runProcedure():
            return runIndependentIterativeProcedure(fwdIC)
    
    elif userCtr.procedure == "JOINT":
        def runProcedure():
            return runJointBackAndForwardProcedure(bckwdIC, fwdIC)
    else:
        raise ValueError("Procedure option not recognized.")


    if userCtr.fitInYSpace == None:
        wsNames = []
        ICs = []

    elif userCtr.fitInYSpace == "BACKWARD":
        wsNames = buildFinalWSNames(scriptName, ["BACKWARD"], [bckwdIC])
        ICs = [bckwdIC]

    elif userCtr.fitInYSpace == "FORWARD":
        wsNames = buildFinalWSNames(scriptName, ["FORWARD"], [fwdIC])
        ICs = [fwdIC]

    elif userCtr.fitInYSpace == "JOINT":
        wsNames = buildFinalWSNames(scriptName, ["BACKWARD", "FORWARD"], [bckwdIC, fwdIC])
        ICs = [bckwdIC, fwdIC]
    else:
        raise ValueError("fitInYSpace option not recognized.")


    # If bootstrap is not None, run bootstrap procedure and finish
    if userCtr.bootstrap == None:
        pass

    elif userCtr.bootstrap == "BACKWARD":
        return runIndependentBootstrap(bckwdIC, bootIC, yFitIC), None

    elif userCtr.bootstrap == "FORWARD":
        return runIndependentBootstrap(fwdIC, bootIC, yFitIC), None

    elif userCtr.bootstrap == "JOINT":
        return runJointBootstrap(bckwdIC, fwdIC, bootIC, yFitIC), None
    else:
        raise ValueError("Bootstrap option not recognized.")

    
    # Default workflow for procedure + fit in y space

    # Check if final ws are loaded:
    wsInMtd = [ws in mtd for ws in wsNames]   # List of bool

    # If final ws are already loaded
    if (len(wsInMtd)>0) and all(wsInMtd):       # When wsName is empty list, loop doesn't run
        for wsName, IC in zip(wsNames, ICs):  
            resYFit = fitInYSpaceProcedure(yFitIC, IC, mtd[wsName])
        return None, resYFit       # To match return below. 
    
    res = runProcedure()

    resYFit = None
    for wsName, IC in zip(wsNames, ICs):
        resYFit = fitInYSpaceProcedure(yFitIC, IC, mtd[wsName])
    
    return res, resYFit   # Return results used only in tests


def buildFinalWSNames(scriptName: str, procedures: list, inputIC: list):
    wsNames = []
    for proc, IC in zip(procedures, inputIC):
        # Format of corrected ws from last iteration
        name = scriptName + "_" + proc + "_" + str(IC.noOfMSIterations-1)
        wsNames.append(name)
    return wsNames

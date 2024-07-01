
# Script created to store original bootstrap routine 
# Does not run, only to be used as a reference for
# future work
from mvesuvio.ICHelpers import completeBootIC
from .bootstrap import runBootstrap
from mantid.api import mtd

def runScript(
    bckwdIC,
    fwdIC,
    yFitIC,
    bootIC,
):
    completeBootIC(bootIC, bckwdIC, fwdIC, yFitIC)

    # If bootstrap is not None, run bootstrap procedure and finish
    if bootIC.runBootstrap:
        assert (
            (bootIC.procedure == "FORWARD")
            | (bootIC.procedure == "BACKWARD")
            | (bootIC.procedure == "JOINT")
        ), "Invalid Bootstrap procedure."
        return runBootstrap(bckwdIC, fwdIC, bootIC, yFitIC), None



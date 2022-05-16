
from mantid.simpleapi import LoadVesuvio, SaveNexus
from pathlib import Path
import numpy as np
currentPath = Path(__file__).absolute().parent
experimentsPath = currentPath / ".." / "experiments"


def completeICFromInputs(ic, scriptName, icWS):
    """Assigns new methods to the initial conditions class from the inputs of that class"""

    assert ic.lastSpec > ic.firstSpec, "Last spectrum needs to be bigger than first spectrum"
    assert ((ic.lastSpec<135) & (ic.firstSpec<135)) | ((ic.lastSpec>=135) & (ic.firstSpec>=135)), "First and last spec need to be both in Back or Front scattering."

    if ic.lastSpec < 135:
        ic.modeRunning = "BACKWARD"
    elif ic.firstSpec >= 134:
        ic.modeRunning = "FORWARD"
    else:
        raise ValueError("Invalid first and last spectra input.")

    ic.name = scriptName+"_"+ic.modeRunning+"_"

    ic.masses = ic.masses.astype(float)
    ic.noOfMasses = len(ic.masses)

    ic.maskedSpecNo = ic.maskedSpecAllNo[(ic.maskedSpecAllNo>=ic.firstSpec) & (ic.maskedSpecAllNo<=ic.lastSpec)]
    ic.maskedDetectorIdx = ic.maskedSpecNo - ic.firstSpec

    ic.mode = icWS.mode
    # Sort out input and output paths
    inputDirsForSample(ic, scriptName, icWS)
    setOutputDirsForSample(ic, scriptName)
    
    # Do not run bootstrap sample, by default
    ic.runningSampleWS = False

    # Store script name
    ic.scriptName = scriptName
    return 


def inputDirsForSample(ic, sampleName, icWS):
    inputWSPath = experimentsPath / sampleName / "input_ws"
    inputWSPath.mkdir(parents=True, exist_ok=True)

    wsPresent = False
    for wsPath in inputWSPath.iterdir():
        keywords = wsPath.name.split(".")[0].split("_")

        if ic.modeRunning == "BACKWARD":
            modeName = "backward"
        else:
            modeName = "forward"

        for key in keywords:
            if key == modeName:
                wsPresent = True

    if not wsPresent:
        loadWsFromLoadVesuvio(icWS, inputWSPath, sampleName)

    for wsPath in inputWSPath.iterdir():

        keywords = wsPath.name.split(".")[0].split("_")

        if ic.modeRunning == "BACKWARD":
            if "raw" in keywords and "backward" in keywords:
                ic.userWsRawPath = str(wsPath)          
            if "empty" in keywords and "backward" in keywords:
                ic.userWsEmptyPath = str(wsPath)

        if ic.modeRunning == "FORWARD":
            if "raw" in keywords and "forward" in keywords:
                ic.userWsRawPath = str(wsPath)          
            if "empty" in keywords and "forward" in keywords:
                ic.userWsEmptyPath = str(wsPath)      
    return


def setOutputDirsForSample(ic, sampleName):
    outputPath = experimentsPath / sampleName / "output_npz_for_testing"
    outputPath.mkdir(parents=True, exist_ok=True)

    # Build Filename based on ic
    corr = ""
    if ic.MSCorrectionFlag & (ic.noOfMSIterations>1):
        corr+="_MS"
    if ic.GammaCorrectionFlag & (ic.noOfMSIterations>1):
        corr+="_GC"

    fileName = f"spec_{ic.firstSpec}-{ic.lastSpec}_iter_{ic.noOfMSIterations}{corr}"+".npz"
    fileNameYSpace = fileName + "_ySpaceFit"+".npz"

    ic.resultsSavePath = outputPath / fileName
    ic.ySpaceFitSavePath = outputPath / fileNameYSpace
    return


def loadWsFromLoadVesuvio(ic, inputWSPath, sampleName):
    
    print(f"\nLoading and storing workspace sample runs: {ic.runs}\n")

    if int(ic.spectra.split("-")[1])<135:
        runningType = "backward"
    elif int(ic.spectra.split("-")[0])>=135:
        runningType = "forward"
    else:
        print("Problem in loading workspace spectra.")

    rawVesuvio = LoadVesuvio(
        Filename=ic.runs, 
        SpectrumList=ic.spectra, 
        Mode=ic.mode,
        InstrumentParFile=ic.ipfile, 
        OutputWorkspace=sampleName+'_raw_'+runningType
        )

    rawName = rawVesuvio.name() + ".nxs"
    rawPath = inputWSPath / rawName
    SaveNexus(rawVesuvio, str(rawPath))
    print("Raw workspace stored locally.")

    emptyVesuvio = LoadVesuvio(
        Filename=ic.empty_runs, 
        SpectrumList=ic.spectra, 
        Mode=ic.mode,
        InstrumentParFile=ic.ipfile, 
        OutputWorkspace=sampleName+'_empty_'+runningType
        )

    emptyName = emptyVesuvio.name() + ".nxs"
    emptyPath = inputWSPath / emptyName
    SaveNexus(emptyVesuvio, str(emptyPath))
    print("Empty workspace stored locally.")
    return 

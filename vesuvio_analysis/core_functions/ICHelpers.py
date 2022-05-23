
from mantid.simpleapi import LoadVesuvio, SaveNexus
from pathlib import Path
import numpy as np
currentPath = Path(__file__).absolute().parent
experimentsPath = currentPath / ".."/ ".." / "experiments"


def completeICFromInputs(IC, scriptName, wsIC):
    """Assigns new methods to the initial conditions class from the inputs of that class"""

    assert IC.lastSpec > IC.firstSpec, "Last spectrum needs to be bigger than first spectrum"
    assert ((IC.lastSpec<135) & (IC.firstSpec<135)) | ((IC.lastSpec>=135) & (IC.firstSpec>=135)), "First and last spec need to be both in Back or Front scattering."

    if IC.lastSpec < 135:
        IC.modeRunning = "BACKWARD"
    elif IC.firstSpec >= 134:
        IC.modeRunning = "FORWARD"
    else:
        raise ValueError("Invalid first and last spectra input.")

    IC.name = scriptName+"_"+IC.modeRunning+"_"

    IC.masses = IC.masses.astype(float)
    IC.noOfMasses = len(IC.masses)

    IC.maskedSpecNo = IC.maskedSpecAllNo[(IC.maskedSpecAllNo>=IC.firstSpec) & (IC.maskedSpecAllNo<=IC.lastSpec)]
    IC.maskedDetectorIdx = IC.maskedSpecNo - IC.firstSpec

    IC.mode = wsIC.mode
    # Sort out input and output paths
    inputDirsForSample(IC, scriptName, wsIC)
    setOutputDirsForSample(IC, scriptName)
    
    # Do not run bootstrap sample, by default
    IC.runningSampleWS = False

    # Store script name
    IC.scriptName = scriptName
    return 


def inputDirsForSample(IC, sampleName, wsIC):
    inputWSPath = experimentsPath / sampleName / "input_ws"
    inputWSPath.mkdir(parents=True, exist_ok=True)

    wsPresent = False
    for wsPath in inputWSPath.iterdir():
        keywords = wsPath.name.split(".")[0].split("_")

        if IC.modeRunning == "BACKWARD":
            modeName = "backward"
        else:
            modeName = "forward"

        for key in keywords:
            if key == modeName:
                wsPresent = True

    if not wsPresent:
        loadWsFromLoadVesuvio(wsIC, inputWSPath, sampleName)

    for wsPath in inputWSPath.iterdir():

        keywords = wsPath.name.split(".")[0].split("_")

        if IC.modeRunning == "BACKWARD":
            if "raw" in keywords and "backward" in keywords:
                IC.userWsRawPath = str(wsPath)          
            if "empty" in keywords and "backward" in keywords:
                IC.userWsEmptyPath = str(wsPath)

        if IC.modeRunning == "FORWARD":
            if "raw" in keywords and "forward" in keywords:
                IC.userWsRawPath = str(wsPath)          
            if "empty" in keywords and "forward" in keywords:
                IC.userWsEmptyPath = str(wsPath)      
    return


def setOutputDirsForSample(IC, sampleName):
    outputPath = experimentsPath / sampleName / "output_npz_for_testing"
    outputPath.mkdir(parents=True, exist_ok=True)

    # Build Filename based on ic
    corr = ""
    if IC.MSCorrectionFlag & (IC.noOfMSIterations>1):
        corr+="_MS"
    if IC.GammaCorrectionFlag & (IC.noOfMSIterations>1):
        corr+="_GC"

    fileName = f"spec_{IC.firstSpec}-{IC.lastSpec}_iter_{IC.noOfMSIterations}{corr}"+".npz"
    fileNameYSpace = fileName + "_ySpaceFit"+".npz"

    IC.resultsSavePath = outputPath / fileName
    IC.ySpaceFitSavePath = outputPath / fileNameYSpace
    return


def loadWsFromLoadVesuvio(IC, inputWSPath, sampleName):
    
    print(f"\nLoading and storing workspace sample runs: {IC.runs}\n")

    if int(IC.spectra.split("-")[1])<135:
        runningType = "backward"
    elif int(IC.spectra.split("-")[0])>=135:
        runningType = "forward"
    else:
        print("Problem in loading workspace spectra.")

    rawVesuvio = LoadVesuvio(
        Filename=IC.runs, 
        SpectrumList=IC.spectra, 
        Mode=IC.mode,
        InstrumentParFile=IC.ipfile, 
        OutputWorkspace=sampleName+'_raw_'+runningType
        )

    rawName = rawVesuvio.name() + ".nxs"
    rawPath = inputWSPath / rawName
    SaveNexus(rawVesuvio, str(rawPath))
    print("Raw workspace stored locally.")

    emptyVesuvio = LoadVesuvio(
        Filename=IC.empty_runs, 
        SpectrumList=IC.spectra, 
        Mode=IC.mode,
        InstrumentParFile=IC.ipfile, 
        OutputWorkspace=sampleName+'_empty_'+runningType
        )

    emptyName = emptyVesuvio.name() + ".nxs"
    emptyPath = inputWSPath / emptyName
    SaveNexus(emptyVesuvio, str(emptyPath))
    print("Empty workspace stored locally.")
    return 

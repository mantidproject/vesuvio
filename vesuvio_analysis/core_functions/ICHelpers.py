
from unittest.loader import VALID_MODULE_NAME
from mantid.simpleapi import LoadVesuvio, SaveNexus
from pathlib import Path
import numpy as np
currentPath = Path(__file__).absolute().parent
experimentsPath = currentPath / ".."/ ".." / "experiments"


def completeICFromInputs(IC, scriptName, wsIC):
    """Assigns new methods to the initial conditions class from the inputs of that class"""

    assert IC.lastSpec > IC.firstSpec, "Last spectrum needs to be bigger than first spectrum"
    assert ((IC.lastSpec<135) & (IC.firstSpec<135)) | ((IC.lastSpec>=135) & (IC.firstSpec>=135)), "First and last spec need to be both in Back or Front scattering."

    if IC.lastSpec <= 134:
        IC.modeRunning = "BACKWARD"
    elif IC.firstSpec >= 135:
        IC.modeRunning = "FORWARD"
    else:
        raise ValueError("Invalid first and last spectra input.")

    IC.name = scriptName+"_"+IC.modeRunning+"_"

    IC.masses = IC.masses.astype(float)
    IC.noOfMasses = len(IC.masses)

    IC.maskedSpecNo = IC.maskedSpecAllNo[(IC.maskedSpecAllNo>=IC.firstSpec) & (IC.maskedSpecAllNo<=IC.lastSpec)]
    IC.maskedDetectorIdx = IC.maskedSpecNo - IC.firstSpec

    # Extract some attributes from wsIC
    IC.mode = wsIC.mode
    IC.subEmptyFromRaw = wsIC.subEmptyFromRaw
    IC.scaleEmpty = wsIC.scaleEmpty
    IC.scaleRaw = wsIC.scaleRaw
    
    # When attribute InstrParsPath is not present, set it equal to path from wsIC
    try:    
        reading = IC.InstrParsPath    # If present, leave it unaltered
    except AttributeError:
        IC.InstrParsPath = wsIC.ipfile

    # Sort out input and output paths
    rawPath, emptyPath = inputDirsForSample(wsIC, scriptName)
    if not(rawPath.is_file()) or not(emptyPath.is_file()):    
        print(f"\nWorkspaces not found.\nSaving Workspaces:\n{rawPath.name}\n{emptyPath.name}")
        saveWSFromLoadVesuvio(wsIC, rawPath, emptyPath)
    
    IC.userWsRawPath = rawPath
    IC.userWsEmptyPath = emptyPath

    setOutputDirsForSample(IC, scriptName)
    
    # Do not run bootstrap sample, by default
    IC.runningSampleWS = False

    # Store script name
    IC.scriptName = scriptName

    # Default not running preliminary procedure to estimate HToMass0Ratio
    IC.runningPreliminary = False
    

    # Set directories for figures
    return 


def inputDirsForSample(wsIC, sampleName):
    inputWSPath = experimentsPath / sampleName / "input_ws"
    inputWSPath.mkdir(parents=True, exist_ok=True)

    if int(wsIC.spectra.split("-")[1])<135:
        runningMode = "backward"
    elif int(wsIC.spectra.split("-")[0])>=135:
        runningMode = "forward"
    else:
        print("Problem in loading workspaces: invalid range of spectra.")

    rawWSName = sampleName + "_" + "raw" + "_" + runningMode + ".nxs"
    emptyWSName = sampleName + "_" + "empty" + "_" + runningMode + ".nxs"

    rawPath = inputWSPath / rawWSName
    emptyPath = inputWSPath / emptyWSName
    return rawPath, emptyPath


def setOutputDirsForSample(IC, sampleName):
    outputPath = experimentsPath / sampleName / "output_npz_for_testing"
    outputPath.mkdir(parents=True, exist_ok=True)

    # Build Filename based on ic
    corr = ""
    if IC.MSCorrectionFlag & (IC.noOfMSIterations>0):
        corr+="_MS"
    if IC.GammaCorrectionFlag & (IC.noOfMSIterations>0):
        corr+="_GC"

    fileName = f"spec_{IC.firstSpec}-{IC.lastSpec}_iter_{IC.noOfMSIterations}{corr}"+".npz"
    fileNameYSpace = fileName + "_ySpaceFit"+".npz"

    IC.resultsSavePath = outputPath / fileName
    IC.ySpaceFitSavePath = outputPath / fileNameYSpace
    return


def saveWSFromLoadVesuvio(wsIC, rawPath, emptyPath):
    
    print(f"\nLoading and storing workspace sample runs: {wsIC.runs}\n")

    rawVesuvio = LoadVesuvio(
        Filename=wsIC.runs, 
        SpectrumList=wsIC.spectra, 
        Mode=wsIC.mode,
        InstrumentParFile=str(wsIC.ipfile), 
        OutputWorkspace=rawPath.name
        )

    SaveNexus(rawVesuvio, str(rawPath))
    print("\nRaw workspace stored locally.\n")

    emptyVesuvio = LoadVesuvio(
        Filename=wsIC.empty_runs, 
        SpectrumList=wsIC.spectra, 
        Mode=wsIC.mode,
        InstrumentParFile=str(wsIC.ipfile), 
        OutputWorkspace=emptyPath.name
        )

    SaveNexus(emptyVesuvio, str(emptyPath))
    print("\nEmpty workspace stored locally.\n")
    return 

def completeBootIC(bootIC, inputIC, userCtr):

    try:    # Assume it is not running a test if atribute is not found
        reading = bootIC.runningTest
    except AttributeError:
        bootIC.runningTest = False

    setBootstrapDirs(inputIC, bootIC, userCtr)
    return


def setBootstrapDirs(inputIC: list, bootIC, userCtr):
    """Form bootstrap output data paths"""

    # Select script name and experiments path
    sampleName = inputIC[0].scriptName   # Name of sample currently running
    experimentsPath = currentPath/".."/".."/"experiments"

    # Make bootstrap and jackknife data directories
    if bootIC.runningJackknife:
        bootPath = experimentsPath / sampleName / "jackknife_data"
    else:
        bootPath = experimentsPath / sampleName / "bootstrap_data"
    bootPath.mkdir(exist_ok=True)

    # Folders for skipped and unskipped MS
    if bootIC.skipMSIterations:
        speedPath = bootPath / "skip_MS_corrections"
    else:
        speedPath = bootPath / "with_MS_corrections"
    speedPath.mkdir(exist_ok=True)

    # Create subfolders dependin no procedure running
    if (userCtr.bootstrap == "FORWARD") | (userCtr.bootstrap=="BACKWARD"):
        Path(speedPath / userCtr.bootstrap).mkdir(exist_ok=True)
    elif userCtr.bootstrap=="JOINT":
        if bootIC.runningJackknife:
            Path(speedPath / "FORWARD").mkdir(exist_ok=True)
            Path(speedPath / "BACKWARD").mkdir(exist_ok=True)
        else:
            Path(speedPath / "JOINT").mkdir(exist_ok=True)
    else:
        raise ValueError("Procedure in UserScriptControls not recognized.")


    # if (userCtr.bootstrap == "FORWARD") | (userCtr.bootstrap=="BACKWARD") | (userCtr.bootstrap=="JOINT"):
    #     finalPath = speedPath / userCtr.bootstrap
    # else:
    #     raise ValueError("Procedure in UserScriptControls not recognized.")
    # finalPath.mkdir(exist_ok=True)

    for IC in inputIC:    # Make save paths for .npz files

        nSamples = bootIC.nSamples
        if bootIC.runningJackknife: 
            nSamples = 3 if bootIC.runningTest else noOfHistsFromTOFBinning(IC)

        # Build Filename based on ic
        corr = ""
        if IC.MSCorrectionFlag & (IC.noOfMSIterations>0):
            corr+="_MS"
        if IC.GammaCorrectionFlag & (IC.noOfMSIterations>0):
            corr+="_GC"

        fileName = f"spec_{IC.firstSpec}-{IC.lastSpec}_iter_{IC.noOfMSIterations}{corr}"
        bootName = fileName + f"_nsampl_{nSamples}"+".npz"
        bootNameYFit = fileName + "_ySpaceFit" + f"_nsampl_{nSamples}"+".npz"

        if bootIC.runningJackknife:
            IC.bootSavePath = speedPath / IC.modeRunning / bootName          # works because modeRunning has same strings as procedure
            IC.bootYFitSavePath = speedPath / IC.modeRunning / bootNameYFit
        else:
            IC.bootSavePath = speedPath / userCtr.bootstrap / bootName          # works because modeRunning has same strings as procedure
            IC.bootYFitSavePath = speedPath / userCtr.bootstrap / bootNameYFit
    return 

def noOfHistsFromTOFBinning(IC):
    start, spacing, end = [int(float(s)) for s in IC.tofBinning.split(",")]  # Convert first to float and then to int because of decimal points
    return int((end-start)/spacing) - 1 # To account for last column being ignored


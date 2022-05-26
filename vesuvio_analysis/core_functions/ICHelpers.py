
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
    
    # When attribute InstrParsPath is not present, set it equal to path from wsIC
    try:    
        reading = IC.InstrParsPath    # If present, leave it unaltered
    except AttributeError:
        IC.InstrParsPath = wsIC.ipfile

    # Sort out input and output paths
    #TODO: Confirm that last cleaning is working
    # inputDirsForSample(IC, scriptName, wsIC)
    rawPath, emptyPath = inputDirsForSample(wsIC, scriptName)
    if not(rawPath.is_file()): # or not(emptyPath.is_file()):    # Temporary while starch_80_RD doesnt have emtpy saved
        print(f"\nWorkspaces not found.\nSaving Workspaces:\n{rawPath.name}\n{emptyPath.name}")
        saveWSFromLoadVesuvio(wsIC, rawPath, emptyPath)
    
    IC.userWsRawPath = rawPath
    IC.userWsEmptyPath = emptyPath

    setOutputDirsForSample(IC, scriptName)
    
    # Do not run bootstrap sample, by default
    IC.runningSampleWS = False

    # Store script name
    IC.scriptName = scriptName
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


# def inputDirsForSample(IC, sampleName, wsIC):
#     inputWSPath = experimentsPath / sampleName / "input_ws"
#     inputWSPath.mkdir(parents=True, exist_ok=True)

#     wsPresent = False
#     for wsPath in inputWSPath.iterdir():
#         keywords = wsPath.name.split(".")[0].split("_")

#         if IC.modeRunning == "BACKWARD":
#             modeName = "backward"
#         else:
#             modeName = "forward"

#         for key in keywords:
#             if key == modeName:
#                 wsPresent = True

#     if not wsPresent:
#         saveWSFromLoadVesuvio(wsIC, inputWSPath, sampleName)

#     for wsPath in inputWSPath.iterdir():

#         keywords = wsPath.name.split(".")[0].split("_")

#         if IC.modeRunning == "BACKWARD":
#             if "raw" in keywords and "backward" in keywords:
#                 IC.userWsRawPath = str(wsPath)          
#             if "empty" in keywords and "backward" in keywords:
#                 IC.userWsEmptyPath = str(wsPath)

#         elif IC.modeRunning == "FORWARD":
#             if "raw" in keywords and "forward" in keywords:
#                 IC.userWsRawPath = str(wsPath)          
#             if "empty" in keywords and "forward" in keywords:
#                 IC.userWsEmptyPath = str(wsPath)  
#         else:
#             raise ValueError("Mode running not recognized.")
#     return


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


def saveWSFromLoadVesuvio(wsIC, rawPath, emptyPath):
    
    print(f"\nLoading and storing workspace sample runs: {wsIC.runs}\n")

    # if int(wsIC.spectra.split("-")[1])<135:
    #     runningType = "backward"
    # elif int(wsIC.spectra.split("-")[0])>=135:
    #     runningType = "forward"
    # else:
    #     print("Problem in loading workspaces: invalid range of spectra.")

    rawVesuvio = LoadVesuvio(
        Filename=wsIC.runs, 
        SpectrumList=wsIC.spectra, 
        Mode=wsIC.mode,
        InstrumentParFile=str(wsIC.ipfile), 
        OutputWorkspace=rawPath.name
        )

    # rawName = rawVesuvio.name() + ".nxs"
    # rawPath = inputWSPath / rawName
    SaveNexus(rawVesuvio, str(rawPath))
    print("\nRaw workspace stored locally.\n")

    emptyVesuvio = LoadVesuvio(
        Filename=wsIC.empty_runs, 
        SpectrumList=wsIC.spectra, 
        Mode=wsIC.mode,
        InstrumentParFile=str(wsIC.ipfile), 
        OutputWorkspace=emptyPath.name
        )

    # emptyName = emptyVesuvio.name() + ".nxs"
    # emptyPath = inputWSPath / emptyName
    SaveNexus(emptyVesuvio, str(emptyPath))
    print("\nEmpty workspace stored locally.\n")
    return 

from mantid.simpleapi import LoadVesuvio, SaveNexus
from pathlib import Path
experimentsPath = Path(__file__).absolute().parent


def IODirectoriesForSample(sampleName, wsp):
    samplePath = experimentsPath / sampleName
    inputWSPath  = samplePath / "input_ws"
    outputPath = samplePath / "output_npz_for_testing"

    if ~inputWSPath.exists():   # TODO: introduce check on whether it contains files

        inputWSPath.mkdir(parents=True)
        outputPath.mkdir(parents=True)

        loadRawAndEmptyWsFromLoadVesuvio(wsp, inputWSPath)

    # Input paths
    backWsRawPath = None
    frontWsRawPath = None
    frontWsEmptyPath = None
    backWsEmptyPath = None

    for wsPath in inputWSPath.iterdir():

        keywords = wsPath.name.split(".")[0].split("_")

        if "raw" in keywords and "backward" in keywords:
            backWsRawPath = wsPath 
        if "raw" in keywords and "forward" in keywords:
            frontWsRawPath = wsPath
        if "empty" in keywords and "forward" in keywords:
            frontWsEmptyPath = wsPath 
        if "empty" in keywords and "backward" in keywords:
            backWsEmptyPath = wsPath 
 
    assert (backWsRawPath!=None) | (frontWsRawPath!=None), "No raw ws detected, usage: wsName_raw_backward.nxs"

    # Output paths
    forwardSavePath = outputPath / "current_forward.npz" 
    backSavePath = outputPath / "current_backward.npz" 
    ySpaceFitSavePath = outputPath / "current_yspacefit.npz"

    return [backWsRawPath, frontWsRawPath, backWsEmptyPath, frontWsEmptyPath], [forwardSavePath, backSavePath, ySpaceFitSavePath]



def loadRawAndEmptyWsFromLoadVesuvio(ic, inputWSPath):
    
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
        OutputWorkspace=ic.name+'raw_'+runningType
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
        OutputWorkspace=ic.name+'empty_'+runningType
        )

    emptyName = emptyVesuvio.name() + ".nxs"
    emptyPath = inputWSPath / emptyName
    SaveNexus(emptyVesuvio, str(emptyPath))
    print("Empty workspace stored locally.")

    return 



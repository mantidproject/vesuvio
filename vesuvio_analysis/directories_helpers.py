from mantid.simpleapi import LoadVesuvio, SaveNexus
from pathlib import Path
currentPath = Path(__file__).absolute().parent
experimentsPath = currentPath / ".." / "experiments"


# def inputDirsForSample(sampleName):
#     inputWSPath = experimentsPath / sampleName / "input_ws"

#     if not inputWSPath.exists():  
#         inputWSPath.mkdir(parents=True)

#     backWsRawPath = None
#     frontWsRawPath = None
#     frontWsEmptyPath = None
#     backWsEmptyPath = None

#     for wsPath in inputWSPath.iterdir():

#         keywords = wsPath.name.split(".")[0].split("_")

#         if "raw" in keywords and "backward" in keywords:
#             backWsRawPath = wsPath 
#         if "raw" in keywords and "forward" in keywords:
#             frontWsRawPath = wsPath
#         if "empty" in keywords and "forward" in keywords:
#             frontWsEmptyPath = wsPath 
#         if "empty" in keywords and "backward" in keywords:
#             backWsEmptyPath = wsPath  


#     return inputWSPath, [backWsRawPath, frontWsRawPath, backWsEmptyPath, frontWsEmptyPath]


# def outputDirsForSample(sampleName):
#     outputPath = experimentsPath / sampleName / "output_npz_for_testing"

#     if not outputPath.exists():
#         outputPath.mkdir(parents=True)

#     # Output paths
#     forwardSavePath = outputPath / "current_forward.npz" 
#     backSavePath = outputPath / "current_backward.npz" 
#     ySpaceFitSavePath = outputPath / "current_yspacefit.npz"

#     return [forwardSavePath, backSavePath, ySpaceFitSavePath]


# def checkInputPaths(inputWSPath):
#     backWsRawPath = None
#     frontWsRawPath = None
#     frontWsEmptyPath = None
#     backWsEmptyPath = None

#     for wsPath in inputWSPath.iterdir():

#         keywords = wsPath.name.split(".")[0].split("_")

#         if "raw" in keywords and "backward" in keywords:
#             backWsRawPath = wsPath 
#         if "raw" in keywords and "forward" in keywords:
#             frontWsRawPath = wsPath
#         if "empty" in keywords and "forward" in keywords:
#             frontWsEmptyPath = wsPath 
#         if "empty" in keywords and "backward" in keywords:
#             backWsEmptyPath = wsPath  


#     return [backWsRawPath, frontWsRawPath, backWsEmptyPath, frontWsEmptyPath]


# def loadWsFromLoadVesuvio(ic, inputWSPath, sampleName):
    
#     print(f"\nLoading and storing workspace sample runs: {ic.runs}\n")

#     if int(ic.spectra.split("-")[1])<135:
#         runningType = "backward"
#     elif int(ic.spectra.split("-")[0])>=135:
#         runningType = "forward"
#     else:
#         print("Problem in loading workspace spectra.")

#     rawVesuvio = LoadVesuvio(
#         Filename=ic.runs, 
#         SpectrumList=ic.spectra, 
#         Mode=ic.mode,
#         InstrumentParFile=ic.ipfile, 
#         OutputWorkspace=sampleName+'_raw_'+runningType
#         )

#     rawName = rawVesuvio.name() + ".nxs"
#     rawPath = inputWSPath / rawName
#     SaveNexus(rawVesuvio, str(rawPath))
#     print("Raw workspace stored locally.")

#     emptyVesuvio = LoadVesuvio(
#         Filename=ic.empty_runs, 
#         SpectrumList=ic.spectra, 
#         Mode=ic.mode,
#         InstrumentParFile=ic.ipfile, 
#         OutputWorkspace=sampleName+'_empty_'+runningType
#         )

#     emptyName = emptyVesuvio.name() + ".nxs"
#     emptyPath = inputWSPath / emptyName
#     SaveNexus(emptyVesuvio, str(emptyPath))
#     print("Empty workspace stored locally.")

#     return 



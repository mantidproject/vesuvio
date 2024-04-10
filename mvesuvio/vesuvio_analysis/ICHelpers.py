from mantid.simpleapi import Load, LoadVesuvio, SaveNexus, DeleteWorkspace
from pathlib import Path
from mvesuvio.scripts import handle_config
from mantid.kernel import logger
import ntpath


def _get_expr_path():
    inputsPath = Path(handle_config.read_config_var("caching.inputs"))
    scriptName = handle_config.get_script_name()
    experimentPath = inputsPath.parent / scriptName
    return experimentPath


def completeICFromInputs(IC, wsIC):
    """Assigns new methods to the initial conditions class from the inputs of that class"""
    scriptName = handle_config.get_script_name()

    assert (
        IC.lastSpec > IC.firstSpec
    ), "Last spectrum needs to be bigger than first spectrum"
    assert ((IC.lastSpec < 135) & (IC.firstSpec < 135)) | (
        (IC.lastSpec >= 135) & (IC.firstSpec >= 135)
    ), "First and last spec need to be both in Back or Front scattering."

    if IC.lastSpec <= 134:
        IC.modeRunning = "BACKWARD"
    elif IC.firstSpec >= 135:
        IC.modeRunning = "FORWARD"
    else:
        raise ValueError("Invalid first and last spectra input.")

    IC.name = scriptName + "_" + IC.modeRunning + "_"

    IC.masses = IC.masses.astype(float)
    IC.noOfMasses = len(IC.masses)

    IC.maskedSpecNo = IC.maskedSpecAllNo[
        (IC.maskedSpecAllNo >= IC.firstSpec) & (IC.maskedSpecAllNo <= IC.lastSpec)
    ]
    IC.maskedDetectorIdx = IC.maskedSpecNo - IC.firstSpec

    # Extract some attributes from wsIC
    IC.mode = wsIC.mode
    IC.subEmptyFromRaw = wsIC.subEmptyFromRaw
    IC.scaleEmpty = wsIC.scaleEmpty
    IC.scaleRaw = wsIC.scaleRaw

    # When attribute InstrParsPath is not present, set it equal to path from wsIC
    try:
        IC.InstrParsPath  # If present, leave it unaltered
    except AttributeError:
        IC.InstrParsPath = wsIC.ipfile

    # Sort out input and output paths
    rawPath, emptyPath = setInputWSForSample(wsIC)

    IC.userWsRawPath = rawPath
    IC.userWsEmptyPath = emptyPath

    setOutputDirsForSample(IC)

    # Do not run bootstrap sample, by default
    IC.runningSampleWS = False

    # Store script name
    IC.scriptName = scriptName

    # Default not running preliminary procedure to estimate HToMass0Ratio
    IC.runningPreliminary = False

    # Create default of not running original version with histogram data
    try:
        IC.runHistData
    except AttributeError:
        IC.runHistData = False

    # Norm voigt except when comparing with tests
    try:
        IC.normVoigt
    except AttributeError:
        IC.normVoigt = True

    return


def setInputWSForSample(wsIC):
    experimentPath = _get_expr_path()
    scriptName = handle_config.get_script_name()

    inputWSPath = experimentPath / "input_ws"
    inputWSPath.mkdir(parents=True, exist_ok=True)

    runningMode = getRunningMode(wsIC)

    rawWSName = scriptName + "_" + "raw" + "_" + runningMode + ".nxs"
    emptyWSName = scriptName + "_" + "empty" + "_" + runningMode + ".nxs"

    rawPath = inputWSPath / rawWSName
    emptyPath = inputWSPath / emptyWSName

    if not wsHistoryMatchesInputs(wsIC.runs, wsIC.mode, wsIC.ipfile, rawPath):
        saveWSFromLoadVesuvio(wsIC.runs, wsIC.mode, wsIC.ipfile, rawPath)

    if not wsHistoryMatchesInputs(wsIC.empty_runs, wsIC.mode, wsIC.ipfile, emptyPath):
        saveWSFromLoadVesuvio(wsIC.empty_runs, wsIC.mode, wsIC.ipfile, emptyPath)

    return rawPath, emptyPath


def getRunningMode(wsIC):
    if wsIC.__class__.__name__ == "LoadVesuvioBackParameters":
        runningMode = "backward"
    elif wsIC.__class__.__name__ == "LoadVesuvioFrontParameters":
        runningMode = "forward"
    else:
        raise ValueError(
            f"Input class for loading workspace not valid: {wsIC.__class__.__name__}"
        )
    return runningMode


def setOutputDirsForSample(IC):
    experimentPath = _get_expr_path()
    outputPath = experimentPath / "output_files"
    outputPath.mkdir(parents=True, exist_ok=True)

    # Build Filename based on ic
    corr = ""
    if IC.GammaCorrectionFlag & (IC.noOfMSIterations > 0):
        corr += "_GC"
    if IC.MSCorrectionFlag & (IC.noOfMSIterations > 0):
        corr += "_MS"

    fileName = (
        f"spec_{IC.firstSpec}-{IC.lastSpec}_iter_{IC.noOfMSIterations}{corr}" + ".npz"
    )
    fileNameYSpace = fileName + "_ySpaceFit" + ".npz"

    IC.resultsSavePath = outputPath / fileName
    IC.ySpaceFitSavePath = outputPath / fileNameYSpace

    # Set directories for figures
    figSavePath = experimentPath / "figures"
    figSavePath.mkdir(exist_ok=True)
    IC.figSavePath = figSavePath
    return


def wsHistoryMatchesInputs(runs, mode, ipfile, localPath):
    if not (localPath.is_file()):
        logger.notice("Cached workspace not found")
        return False
    local_ws = Load(Filename=str(localPath))
    ws_history = local_ws.getHistory()
    metadata = ws_history.getAlgorithmHistory(0)

    saved_runs = metadata.getPropertyValue("Filename")
    if saved_runs != runs:
        logger.notice(
            f"Filename in saved workspace did not match: {saved_runs} and {runs}"
        )
        return False

    saved_mode = metadata.getPropertyValue("Mode")
    if saved_mode != mode:
        logger.notice(f"Mode in saved workspace did not match: {saved_mode} and {mode}")
        return False

    saved_ipfile_name = ntpath.basename(metadata.getPropertyValue("InstrumentParFile"))
    if saved_ipfile_name != ipfile.name:
        logger.notice(
            f"IP files in saved workspace did not match: {saved_ipfile_name} and {ipfile.name}"
        )
        return False

    print("\nLocally saved workspace metadata matched with analysis inputs.\n")
    DeleteWorkspace(local_ws)
    return True


def saveWSFromLoadVesuvio(runs, mode, ipfile, localPath):
    if "backward" in localPath.name:
        spectra = "3-134"
    elif "forward" in localPath.name:
        spectra = "135-198"
    else:
        raise ValueError(f"Invalid name to save workspace: {localPath.name}")

    vesuvio_ws = LoadVesuvio(
        Filename=runs,
        SpectrumList=spectra,
        Mode=mode,
        InstrumentParFile=str(ipfile),
        OutputWorkspace=localPath.name,
        LoadLogFiles=False,
    )

    SaveNexus(vesuvio_ws, str(localPath.absolute()))
    print(f"Workspace saved locally at: {localPath.absolute()}")
    return


def completeBootIC(bootIC, bckwdIC, fwdIC, yFitIC):
    if not (bootIC.runBootstrap):
        return

    try:  # Assume it is not running a test if atribute is not found
        bootIC.runningTest
    except AttributeError:
        bootIC.runningTest = False

    setBootstrapDirs(bckwdIC, fwdIC, bootIC, yFitIC)
    return


def setBootstrapDirs(bckwdIC, fwdIC, bootIC, yFitIC):
    """Form bootstrap output data paths"""
    experimentPath = _get_expr_path()
    scriptName = handle_config.get_script_name()

    # Select script name and experiments path
    sampleName = bckwdIC.scriptName  # Name of sample currently running

    # Used to store running times required to estimate Bootstrap total run time.
    bootIC.runTimesPath = experimentPath / "running_times.txt"

    # Make bootstrap and jackknife data directories
    if bootIC.bootstrapType == "JACKKNIFE":
        bootPath = experimentPath / "jackknife_data"
    else:
        bootPath = experimentPath / "bootstrap_data"
    bootPath.mkdir(exist_ok=True)

    # Folders for skipped and unskipped MS
    if bootIC.skipMSIterations:
        dataPath = bootPath / "skip_MS_corrections"
    else:
        dataPath = bootPath / "with_MS_corrections"
    dataPath.mkdir(exist_ok=True)

    # Create text file for logs
    logFilePath = dataPath / "data_files_log.txt"
    if not (logFilePath.is_file()):
        with open(logFilePath, "w") as txtFile:
            txtFile.write(header_string())

    for IC in [bckwdIC, fwdIC]:  # Make save paths for .npz files
        bootName, bootNameYFit = genBootFilesName(IC, bootIC)

        IC.bootSavePath = (
            dataPath / bootName
        )  # works because modeRunning has same strings as procedure
        IC.bootYFitSavePath = dataPath / bootNameYFit

        IC.logFilePath = logFilePath
        IC.bootSavePathLog = logString(bootName, IC, yFitIC, bootIC, isYFit=False)
        IC.bootYFitSavePathLog = logString(
            bootNameYFit, IC, yFitIC, bootIC, isYFit=True
        )
    return


def genBootFilesName(IC, bootIC):
    """Generates save file name for either BACKWARD or FORWARD class"""

    nSamples = bootIC.nSamples
    if bootIC.bootstrapType == "JACKKNIFE":
        nSamples = 3 if bootIC.runningTest else noOfHistsFromTOFBinning(IC)

    # Build Filename based on ic
    corr = ""
    if IC.MSCorrectionFlag & (IC.noOfMSIterations > 0):
        corr += "_MS"
    if IC.GammaCorrectionFlag & (IC.noOfMSIterations > 0):
        corr += "_GC"

    fileName = f"spec_{IC.firstSpec}-{IC.lastSpec}_iter_{IC.noOfMSIterations}{corr}"
    bootName = fileName + f"_nsampl_{nSamples}" + ".npz"
    bootNameYFit = fileName + "_ySpaceFit" + f"_nsampl_{nSamples}" + ".npz"
    return bootName, bootNameYFit


def header_string():
    return """
    This file contains some information about each data file in the folder.
    ncp data file: boot type | procedure | tof binning | masked tof range.
    yspace fit data file: boot type | procedure | symmetrisation | rebin pars | fit model | mask type
    """


def logString(bootDataName, IC, yFitIC, bootIC, isYFit):
    if isYFit:
        log = (
            bootDataName
            + " : "
            + bootIC.bootstrapType
            + " | "
            + str(bootIC.fitInYSpace)
            + " | "
            + str(yFitIC.symmetrisationFlag)
            + " | "
            + yFitIC.rebinParametersForYSpaceFit
            + " | "
            + yFitIC.fitModel
            + " | "
            + str(yFitIC.maskTypeProcedure)
        )
    else:
        log = (
            bootDataName
            + " : "
            + bootIC.bootstrapType
            + " | "
            + str(bootIC.procedure)
            + " | "
            + IC.tofBinning
            + " | "
            + str(IC.maskTOFRange)
        )
    return log


def noOfHistsFromTOFBinning(IC):
    # Convert first to float and then to int because of decimal points
    start, spacing, end = [int(float(s)) for s in IC.tofBinning.split(",")]
    return int((end - start) / spacing) - 1  # To account for last column being ignored


def buildFinalWSName(procedure: str, IC):
    scriptName = handle_config.get_script_name()
    # Format of corrected ws from last iteration
    name = scriptName + "_" + procedure + "_" + str(IC.noOfMSIterations)
    return name


def completeYFitIC(yFitIC):
    experimentPath = _get_expr_path()
    # Set directories for figures
    figSavePath = experimentPath / "figures"
    figSavePath.mkdir(exist_ok=True)
    yFitIC.figSavePath = figSavePath
    return

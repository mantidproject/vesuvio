from mvesuvio.util import handle_config

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



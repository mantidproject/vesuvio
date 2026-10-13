from mvesuvio.util.files_manager import FilesManager
from mvesuvio.calibrate_analysis import EVSCalibrationAnalysis
from mvesuvio.calibrate_fit import EVSCalibrationFit
from mantid.api import AlgorithmManager, AlgorithmFactory, WorkspaceGroup
from mantid.simpleapi import LoadVesuvio, LoadRaw, ConvertToDistribution, mtd
from mvesuvio.util.calibration_helpers import EVSGlobals
from mvesuvio.globals import SharedParameterFitType, Mode
import numpy as np
from numpy.typing import NDArray
from typing import Literal


class EVSCalibrationAnalysisInputs:
    # Sample run numbers to fit peaks to (cannot be an empty str)
    # Samples: str = [17087, 17088]
    Samples: str = [17087]

    # Run numbers to use as a background (can be empty)
    Background: str = [17086]

    # Filename of the instrument parameter file.
    InstrumentParameterFile: str = str(FilesManager.get_instrument_parameters_dir() / "IP0005.par")

    # Mass of the sample in amu to be used when calculating energy.
    # Default is Pb: 207.19
    Mass: float = 63.546  # Mass of copper in amu

    # List of d-spacings used to estimate the positions of peaks in TOF.
    DSpacings: NDArray[np.float64] = np.array([2.0865, 1.807, 1.278, 1.0897])  # d-spacings of a copper samp

    # Value at which to fix E1 value and E1 error (form: E1 value, E1 Error).
    # If no input is provided, values will be calculated
    # Default is []
    E1FixedValueAndError: list = []

    # List of detectors to be marked as invalid (3-198),
    # to be excluded from analysis calculations.
    # Default is []
    InvalidDetectors: list = []

    # Number of iterations to perform.
    # Default is 2
    Iterations: int = 1

    # Calculate shared parameters using an individual and/or global fit
    # Default is SharedParameterFitType.INDIVIDUAL
    SharedParameterFitType: Literal["Individual", "Shared", "Both"] = SharedParameterFitType.INDIVIDUAL.value

    # Whether to create output from fitting.
    # Default is False
    CreateOutput: bool = False

    # Whether to calculate L0 or just use the values from the parameter file.
    # Default is False
    CalculateL0: bool = False

    # Whether to save the output as an IP file. This file will use the
    # same name as the OutputWorkspace and will be saved to the default
    # save directory.
    # Default is False
    CreateIPFile: bool = False

    # Name to call the output workspace
    # Default is ""
    OutputWorkspace: str = "copper_analysis_test"

    # Mode to load files with. This is passed to the LoadVesuvio algorithm.
    # Default is FoilOut
    Mode: Literal["SingleDifference", "DoubleDifference", "ThickDifference", "FoilOut", "FoilIn", "FoilInOut"] = Mode.FOIL_OUT.value

    # Spectrum range to use.
    # EVSGlobals.DETECTOR_RANGE [3, 198]
    SpectrumRange: list[int] = EVSGlobals.DETECTOR_RANGE


########################
### END OF USER EDIT ###
########################


def main() -> None:
    # Load the samples and the background
    calibration_inputs_dir = FilesManager.get_calibration_inputs_dir()
    file_prefix = "EVS"
    samples_group, background_group = _load_files(calibration_inputs_dir, file_prefix)

    AlgorithmFactory.subscribe(EVSCalibrationAnalysis)
    AlgorithmFactory.subscribe(EVSCalibrationFit)

    alg = AlgorithmManager.createUnmanaged("EVSCalibrationAnalysis")
    alg.initialize()

    alg.setProperty("Samples", samples_group)
    alg.setProperty("Background", background_group)
    alg.setProperty("InstrumentParameterFile", EVSCalibrationAnalysisInputs.InstrumentParameterFile)
    alg.setProperty("Mass", EVSCalibrationAnalysisInputs.Mass)
    alg.setProperty("DSpacings", EVSCalibrationAnalysisInputs.DSpacings)
    alg.setProperty("E1FixedValueAndError", EVSCalibrationAnalysisInputs.E1FixedValueAndError)
    alg.setProperty("InvalidDetectors", EVSCalibrationAnalysisInputs.InvalidDetectors)
    alg.setProperty("Iterations", EVSCalibrationAnalysisInputs.Iterations)
    alg.setProperty("SharedParameterFitType", EVSCalibrationAnalysisInputs.SharedParameterFitType)
    alg.setProperty("CreateOutput", EVSCalibrationAnalysisInputs.CreateOutput)
    alg.setProperty("CalculateL0", EVSCalibrationAnalysisInputs.CalculateL0)
    alg.setProperty("CreateIPFile", EVSCalibrationAnalysisInputs.CreateIPFile)
    alg.setProperty("OutputWorkspace", EVSCalibrationAnalysisInputs.OutputWorkspace)

    alg.execute()


def _load_files(path, prefix):
    """
    Loads the samples and backgrounds into group workspaces.

    @param path - the path to the files.
    @param prefix - the prefix of files.
    """
    samples_group = WorkspaceGroup()
    background_group = WorkspaceGroup()

    for sample in EVSCalibrationAnalysisInputs.Samples:
        _load_file(path, sample, EVSCalibrationAnalysisInputs.SpectrumRange, EVSCalibrationAnalysisInputs.Mode, prefix)
        samples_group.addWorkspace(mtd[prefix + str(sample)])

    for background in EVSCalibrationAnalysisInputs.Background:
        _load_file(path, background, EVSCalibrationAnalysisInputs.SpectrumRange, EVSCalibrationAnalysisInputs.Mode, prefix)
        background_group.addWorkspace(mtd[prefix + str(background)])

    return samples_group, background_group


def _load_file(path, run_number, spec_list, mode, prefix):
    """
    Loads the run number into a workspace.

    This will attempt to use LoadVesuvio, but will fall back on LoadRaw if LoadVesuvio fails.
    @param path - the path to the files.
    @param run_number - the run to load.
    @param spec_list - the spect list to load.
    @param mode - the mode to use when loading files.
    @param prefix - the prefix of files.
    """
    output = prefix + str(run_number)
    file = output + ".raw"
    file_path = str(path / file)
    try:
        LoadVesuvio(
            Filename=file_path,
            Mode=mode,
            OutputWorkspace=output,
            SpectrumList=f"{spec_list[0]}-{spec_list[-1]}",
            EnableLogging=False,
        )
    except RuntimeError:
        LoadRaw(
            file_path,
            OutputWorkspace=output,
            SpectrumMin=spec_list[0],
            SpectrumMax=spec_list[-1],
            EnableLogging=False,
        )
        ConvertToDistribution(output, EnableLogging=False)


if (__name__ == "__main__") or (__name__ == "mantidqt.widgets.codeeditor.execution"):
    main()

from mvesuvio.calibrate_analysis import EVSCalibrationAnalysis
from mvesuvio.globals import SharedParameterFitType
from mantid.kernel import logger
import numpy as np
from numpy.typing import NDArray
from typing import Literal


class EVSCalibrationAnalysisInputs:
    # Sample run numbers to fit peaks to (cannot be an empty str)
    Samples: str = "17087-17088"

    # Run numbers to use as a background (can be empty)
    Background: str = "17086"

    # Filename of the instrument parameter file.
    InstrumentParameterFile: str = "IP0005.par"

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
    Iterations: int = 2

    # Calculate shared parameters using an individual and/or global fit
    # Default is SharedParameterFitType.INDIVIDUAL
    SharedParameterFitType: Literal["Individual", "Shared", "Both"] = SharedParameterFitType.INDIVIDUAL

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
    OutputWorkspace = ""


########################
### END OF USER EDIT ###
########################


def main() -> None:
    logger.information("Starting the EVSCalibrationAnalysis algorithm")
    EVSCalibrationAnalysis(
        Samples=EVSCalibrationAnalysisInputs.Samples,
        Background=EVSCalibrationAnalysisInputs.Background,
        InstrumentParameterFile=EVSCalibrationAnalysisInputs.InstrumentParameterFile,
        Mass=EVSCalibrationAnalysisInputs.Mass,
        DSpacings=EVSCalibrationAnalysisInputs.DSpacings,
        E1FixedValueAndError=EVSCalibrationAnalysisInputs.E1FixedValueAndError,
        InvalidDetectors=EVSCalibrationAnalysisInputs.InvalidDetectors,
        Iterations=EVSCalibrationAnalysisInputs.Iterations,
        SharedParameterFitType=EVSCalibrationAnalysisInputs.SharedParameterFitType,
        CreateOutput=EVSCalibrationAnalysisInputs.CreateOutput,
        CalculateL0=EVSCalibrationAnalysisInputs.CalculateL0,
        CreateIPFile=EVSCalibrationAnalysisInputs.CreateIPFile,
        OutputWorkspace=EVSCalibrationAnalysisInputs.OutputWorkspace,
    )


if (__name__ == "__main__") or (__name__ == "mantidqt.widgets.codeeditor.execution"):
    main()

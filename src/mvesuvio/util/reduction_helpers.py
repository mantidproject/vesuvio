from __future__ import annotations

from typing import TYPE_CHECKING

from mantid.simpleapi import (
    Load,
    CropWorkspace,
    MaskDetectors,
    CreateEmptyTableWorkspace,
    DeleteWorkspace,
    SaveNexus,
    LoadVesuvio,
)
from mantid.kernel import logger
import numpy as np
from pathlib import Path

from mvesuvio import globals
from mvesuvio.util.files_manager import FilesManager
from mvesuvio.analysis_reduction import VesuvioAnalysisRoutine
import dill  # To convert constraints to string
from mantid.api import AlgorithmFactory, AlgorithmManager

if TYPE_CHECKING:
    from mvesuvio.config.run_reduction import BackwardAnalysisInputs, ForwardAnalysisInputs

import ntpath


def init_analysis_algorithm(ws, inputs_class: BackwardAnalysisInputs | ForwardAnalysisInputs):
    profiles_table = create_profiles_table(ws.name() + "_initial_parameters", inputs_class)
    instrument_parameters_file = str(FilesManager.get_instrument_parameters_dir() / inputs_class.instrument_parameters_file)
    kwargs = {
        "InputWorkspace": ws.name(),
        "InputProfiles": profiles_table.name(),
        "InstrumentParametersFile": instrument_parameters_file,
        "ChosenMassIndex": inputs_class.chosen_mass_index if hasattr(inputs_class, "chosen_mass_index") else 0,
        "HRatioToChosenMass": inputs_class.intensity_ratio_of_hydrogen_to_chosen_mass
        if hasattr(inputs_class, "intensity_ratio_of_hydrogen_to_chosen_mass")
        else 0,
        "NumberOfIterations": int(inputs_class.number_of_iterations_for_corrections),
        "InvalidDetectors": convert_to_list_of_spectrum_numbers(inputs_class.mask_detectors),
        "MultipleScatteringCorrection": inputs_class.do_multiple_scattering_correction,
        "SampleShapeXml": inputs_class.sample_shape_xml,
        "GammaCorrection": inputs_class.do_gamma_correction,
        "ModeRunning": "BACKWARD" if inputs_class.__name__ in ["BackwardAnalysisInputs"] else "FORWARD",
        "TransmissionGuess": inputs_class.transmission_guess,
        "MultipleScatteringOrder": int(inputs_class.multiple_scattering_order),
        "NumberOfEvents": int(inputs_class.multiple_scattering_number_of_events),
        "Constraints": str(dill.dumps(inputs_class.constraints)),
        "ResultsPath": str(FilesManager.get_outputs_reduction_dir().absolute()),
        "MinimalOutputFiles": inputs_class.minimal_output,
        "OutputMeansTable": " Final_Means",
    }

    AlgorithmFactory.subscribe(VesuvioAnalysisRoutine)
    alg = AlgorithmManager.createUnmanaged("VesuvioAnalysisRoutine")
    alg.initialize()
    alg.setProperties(kwargs)
    return alg


def crop_and_mask_workspace(ws, inputs_class: BackwardAnalysisInputs | ForwardAnalysisInputs):
    """Returns cloned and cropped workspace with modified name"""

    first_detector, last_detector = [int(s) for s in inputs_class.detectors.split("-")]

    # Read initial Spectrum number
    ws_first_detector = ws.getSpectrumNumbers()[0]
    assert first_detector >= ws_first_detector, "Can't crop workspace, firstSpec < first spectrum in workspace."
    first_idx = first_detector - ws_first_detector
    last_idx = last_detector - ws_first_detector

    ws_cropped = CropWorkspace(
        InputWorkspace=ws,
        StartWorkspaceIndex=first_idx,
        EndWorkspaceIndex=last_idx,
        OutputWorkspace=ws.name(),
    )
    mask_time_of_flight_bins_with_zeros(ws_cropped, inputs_class.mask_time_of_flight_range)  # Used to mask resonance peaks
    MaskDetectors(Workspace=ws_cropped, SpectraList=inputs_class.mask_detectors)
    return ws_cropped


def load_and_save_input_ws_if_not_on_path(inputs_class: BackwardAnalysisInputs | ForwardAnalysisInputs) -> tuple[Path, Path]:
    if inputs_class.__name__ in ["BackwardAnalysisInputs"]:
        raw_path = FilesManager.get_inputs_ws_dir() / FilesManager.get_backward_raw_filename()
        empty_path = FilesManager.get_inputs_ws_dir() / FilesManager.get_backward_empty_filename()

    elif inputs_class.__name__ in ["ForwardAnalysisInputs"]:
        raw_path = FilesManager.get_inputs_ws_dir() / FilesManager.get_forward_raw_filename()
        empty_path = FilesManager.get_inputs_ws_dir() / FilesManager.get_forward_empty_filename()
    else:
        raise ValueError(f"Input class for workspace not valid: {inputs_class.__name__}")

    if not ws_history_matches_inputs(inputs_class.runs, inputs_class.mode, inputs_class.instrument_parameters_file, raw_path):
        save_ws_from_load_vesuvio(
            inputs_class.runs,
            inputs_class.mode,
            str(FilesManager.get_instrument_parameters_dir() / inputs_class.instrument_parameters_file),
            raw_path,
        )

    if not ws_history_matches_inputs(inputs_class.empty_runs, inputs_class.mode, inputs_class.instrument_parameters_file, empty_path):
        save_ws_from_load_vesuvio(
            inputs_class.empty_runs,
            inputs_class.mode,
            str(FilesManager.get_instrument_parameters_dir() / inputs_class.instrument_parameters_file),
            empty_path,
        )

    return raw_path, empty_path


def convert_to_list_of_spectrum_numbers(detectors):
    if isinstance(detectors, str):
        detector_ranges = [r.split("-") for r in detectors.replace(" ", "").split(",")]
        return [i for r in detector_ranges for i in range(int(r[0]), int(r[-1]) + 1)]

    if isinstance(detectors, list) or isinstance(detectors, np.ndarray):
        return [int(d) for d in detectors]

    raise ValueError("Type not recognized: Masked detectors should be string, list or array.")


def ws_history_matches_inputs(runs, mode, ipfile, ws_path):
    if not (ws_path.is_file()):
        logger.notice(f"Cached workspace not found at {ws_path}")
        return False

    ws = Load(Filename=str(ws_path))
    ws_history = ws.getHistory()
    metadata = ws_history.getAlgorithmHistory(0)

    saved_runs = metadata.getPropertyValue("Filename")
    if saved_runs != runs:
        logger.notice(f"Filename in saved workspace did not match: {saved_runs} and {runs}")
        return False

    saved_mode = metadata.getPropertyValue("Mode")
    if saved_mode != mode:
        logger.notice(f"Mode in saved workspace did not match: {saved_mode} and {mode}")
        return False

    saved_ipfile_name = ntpath.basename(metadata.getPropertyValue("InstrumentParFile"))
    if saved_ipfile_name != ipfile:
        logger.notice(f"IP files in saved workspace did not match: {saved_ipfile_name} and {ipfile}")
        return False

    logger.notice("\nLocally saved workspace metadata matched with analysis inputs.\n")
    DeleteWorkspace(ws)
    return True


def save_ws_from_load_vesuvio(runs, mode, ipfile, ws_path):
    if globals.BACKWARD_TAG in ws_path.stem:
        spectra = "3-134"
    elif globals.FORWARD_TAG in ws_path.stem:
        spectra = "135-198"
    else:
        raise ValueError(f"Invalid name to save workspace: {ws_path.name}")

    vesuvio_ws = LoadVesuvio(
        Filename=runs,
        SpectrumList=spectra,
        Mode=mode,
        InstrumentParFile=str(ipfile),
        OutputWorkspace=ws_path.name,
        LoadLogFiles=False,
    )

    SaveNexus(vesuvio_ws, Filename=str(ws_path.absolute()))
    print(f"Workspace saved locally at: {ws_path.absolute()}")
    return


def mask_time_of_flight_bins_with_zeros(ws, maskTOFRange):
    """
    Masks a given TOF range on ws with zeros on dataY.
    Leaves errors dataE unchanged, as they are used by later treatments.
    Used to mask resonance peaks.
    """

    if maskTOFRange is None:
        return

    dataX, dataY, dataE = extractWS(ws)

    ranges = [r.split("-") for r in maskTOFRange.replace(" ", "").split(",")]
    for r in ranges:
        mask = (dataX >= float(r[0])) & (dataX <= float(r[-1]))
        dataY[mask] = 0

    pass_data_into_ws(dataX, dataY, dataE, ws)
    return


def extractWS(ws):
    """Directly extracts data from workspace into arrays"""
    return ws.extractX(), ws.extractY(), ws.extractE()


def pass_data_into_ws(dataX, dataY, dataE, ws):
    "Modifies ws data to input data"
    for i in range(ws.getNumberHistograms()):
        ws.dataX(i)[:] = dataX[i, :]
        ws.dataY(i)[:] = dataY[i, :]
        ws.dataE(i)[:] = dataE[i, :]
    return ws


def create_profiles_table(name, ai):
    table = CreateEmptyTableWorkspace(OutputWorkspace=name)
    table.addColumn(type="str", name="label")
    table.addColumn(type="float", name="mass")
    table.addColumn(type="float", name="intensity")
    table.addColumn(type="float", name="intensity_lb")
    table.addColumn(type="float", name="intensity_ub")
    table.addColumn(type="float", name="width")
    table.addColumn(type="float", name="width_lb")
    table.addColumn(type="float", name="width_ub")
    table.addColumn(type="float", name="center")
    table.addColumn(type="float", name="center_lb")
    table.addColumn(type="float", name="center_ub")

    def wrapb(bound):
        # Literally to just account for NoneType
        if bound is None:
            return np.inf
        return bound

    for mass, intensity, width, center, intensity_bound, width_bound, center_bound in zip(
        ai.masses,
        ai.initial_fitting_parameters[::3],
        ai.initial_fitting_parameters[1::3],
        ai.initial_fitting_parameters[2::3],
        ai.fitting_bounds[::3],
        ai.fitting_bounds[1::3],
        ai.fitting_bounds[2::3],
    ):
        table.addRow(
            [
                str(float(mass)),
                float(mass),
                float(intensity),
                float(wrapb(intensity_bound[0])),
                float(wrapb(intensity_bound[1])),
                float(width),
                float(wrapb(width_bound[0])),
                float(wrapb(width_bound[1])),
                float(center),
                float(wrapb(center_bound[0])),
                float(wrapb(center_bound[1])),
            ]
        )

    print_table_workspace(table)
    return table


def print_table_workspace(table, precision=3):
    table_dict = table.toDict()
    # Convert floats into strings
    for key, values in table_dict.items():
        new_column = [int(item) if (isinstance(item, float) and item.is_integer()) else item for item in values]
        table_dict[key] = [f"{item:.{precision}f}" if isinstance(item, float) else str(item) for item in new_column]

    max_spacing = [max([len(item) for item in values] + [len(key)]) for key, values in table_dict.items()]
    header = "|" + "|".join(f"{item}{' ' * (spacing - len(item))}" for item, spacing in zip(table_dict.keys(), max_spacing)) + "|"
    logger.notice(f"Table {table.name()}:")
    logger.notice(" " + "-" * (len(header) - 2) + " ")
    logger.notice(header)
    for i in range(table.rowCount()):
        table_row = "|".join(
            f"{values[i]}{' ' * (spacing - len(str(values[i])))}" for values, spacing in zip(table_dict.values(), max_spacing)
        )
        logger.notice("|" + table_row + "|")
    logger.notice(" " + "-" * (len(header) - 2) + " ")
    return

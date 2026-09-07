from __future__ import annotations

from typing import TYPE_CHECKING

from mantid import AnalysisDataService
from mantid.simpleapi import (
    Load,
    CropWorkspace,
    MaskDetectors,
    CreateEmptyTableWorkspace,
    DeleteWorkspace,
    SaveNexus,
    SaveAscii,
    LoadVesuvio,
)
from mantid.kernel import logger
import numpy as np
import math
from pathlib import Path

from mvesuvio import globals
from mvesuvio.util.files_manager import FilesManager
from mvesuvio.analysis_reduction import VesuvioAnalysisRoutine
import dill  # To convert constraints to string
from mantid.api import AlgorithmFactory, AlgorithmManager
from mantid.simpleapi import mtd, RenameWorkspace

if TYPE_CHECKING:
    from mvesuvio.config.run_reduction import BackwardAnalysisInputs, ForwardAnalysisInputs

import ntpath
import re


def make_summarised_log_file() -> None:
    pattern = re.compile(r"^\d{4}-\d{2}-\d{2}")
    try:
        with open(FilesManager.get_mantid_log_file(), "r") as infile, open(FilesManager.get_summarised_log_file(), "w") as outfile:
            for line in infile:
                if "VesuvioAnalysisRoutine" in line:
                    outfile.write(line)

                if "Notice Python" in line:  # For Fitting notices
                    outfile.write(line)

                if not pattern.match(line):
                    outfile.write(line)
    except OSError:
        logger.error("Mantid log file not available. Unable to produce a summarized log file for this routine.")
    return


def _get_scattering_type(inputs_class) -> str:
    """Infer scattering type from user inputs without depending on concrete class symbols."""
    name = str(getattr(inputs_class, "name", "")).lower()
    class_name = getattr(inputs_class, "__name__", "").lower()
    mode = str(getattr(inputs_class, "mode", "")).lower().replace(" ", "")

    if name.startswith("back") or "backward" in class_name or mode == "doubledifference":
        return "backward"

    if name.startswith("front") or "forward" in class_name or mode == "singledifference":
        return "forward"

    raise ValueError(f"Could not infer scattering type for input class: {inputs_class}")


def run_estimate_h_ratio(back_alg, front_alg, back_masses, back_chosen_mass_index, rtol, max_iter):
    """
    Used when H is present and H to first mass ratio is not known.
    Preliminary forward scattering is run to get rough estimate of H to first mass ratio.
    Runs iterative procedure with alternating back and forward scattering.
    """
    assert back_alg.getPropertyValue("ModeRunning") == "BACKWARD"
    assert front_alg.getPropertyValue("ModeRunning") == "FORWARD"

    # Ask for explicit consent from user to start procedure
    try:
        userInput = input("\nHydrogen intensity ratio to lowest mass is not set. Press Enter to start estimate procedure.")
        if not userInput == "":
            raise EOFError

    except EOFError:
        logger.error("Estimation of Hydrogen intensity ratio interrupted.")
        return

    chosen_mass = back_masses[back_chosen_mass_index]

    table_h_ratios = CreateEmptyTableWorkspace(OutputWorkspace="hydrogen_intensity_ratios_estimates")
    table_h_ratios.addColumn(type="float", name="Hydrogen intensity ratio to chosen mass at each iteration")

    front_alg.execute()

    means_table = mtd[front_alg.getPropertyValue("OutputMeansTable")]
    current_ratio = calculate_h_ratio(means_table, chosen_mass)

    table_h_ratios.addRow([current_ratio])
    previous_ratio = np.nan

    for _ in range(max_iter):
        if (
            current_ratio is not None
            and previous_ratio is not None
            and math.isclose(current_ratio, previous_ratio, rel_tol=rtol, abs_tol=1e-12)
        ):
            break

        back_alg.setProperty("HRatioToChosenMass", current_ratio)
        execute_joint_algorithms(back_alg, front_alg)

        previous_ratio = current_ratio

        means_table = mtd[front_alg.getPropertyValue("OutputMeansTable")]
        current_ratio = calculate_h_ratio(means_table, chosen_mass)

        table_h_ratios.addRow([current_ratio])

        SaveAscii(table_h_ratios.name(), str(FilesManager.get_outputs_dir() / table_h_ratios.name()))

    logger.notice("\nProcedute to estimate Hydrogen ratio finished.\n")
    print_table_workspace(table_h_ratios)
    return table_h_ratios


def calculate_h_ratio(means_table, chosen_mass):
    masses = means_table.column("mass")
    intensities = np.array(means_table.column("mean_intensity"))

    if not np.isclose(min(masses), 1, atol=0.1):  # Hydrogen not present
        return None

    # Hydrogen present, assumes its lowest mass
    return intensities[np.argmin(masses)] / intensities[np.argmax(np.isclose(masses, chosen_mass, atol=0.01))]


def h_ratio_is_zero_when_h_present(back_inputs: type[BackwardAnalysisInputs], front_inputs: type[ForwardAnalysisInputs]):
    if back_inputs.run_this_scattering_type:
        if is_hydrogen_present(front_inputs.masses):
            if back_inputs.intensity_ratio_of_hydrogen_to_chosen_mass == 0:
                return True
        else:
            logger.warning("Ignoring Hydrogen ratio because not detected in masses.")
            back_inputs.intensity_ratio_of_hydrogen_to_chosen_mass = 0
    return False


def is_hydrogen_present(masses) -> bool:
    Hmask = np.abs(np.array(masses) - 1) / 1 < 0.1  # H mass whithin 10% of 1 au

    if ~np.any(Hmask):  # H not present
        return False

    print("\nH mass detected.\n")
    assert len(Hmask) > 1, "When H is only mass present, run independent forward procedure, not joint."
    assert Hmask[0], "H mass needs to be the first mass in masses and initPars."
    assert sum(Hmask) == 1, "More than one mass very close to H were detected."
    return True


def execute_joint_algorithms(back_alg, front_alg):
    assert back_alg.getPropertyValue("ModeRunning") == "BACKWARD"
    assert front_alg.getPropertyValue("ModeRunning") == "FORWARD"

    back_alg.execute()

    incoming_means_table = mtd[back_alg.getPropertyValue("OutputMeansTable")]
    h_ratio = back_alg.getProperty("HRatioToChosenMass").value

    assert incoming_means_table is not None, "Means table from backward routine not correctly accessed."
    assert h_ratio is not None, "H ratio from backward routine not correctly accesssed."

    receiving_profiles_table = mtd[front_alg.getPropertyValue("InputProfiles")]

    fixed_profiles_table = fix_profile_parameters(incoming_means_table, receiving_profiles_table, h_ratio)

    # Update original profiles table
    RenameWorkspace(fixed_profiles_table, receiving_profiles_table.name())
    print_table_workspace(mtd[receiving_profiles_table.name()])
    # Even if the name is the same, need to trigger update
    front_alg.setPropertyValue("InputProfiles", receiving_profiles_table.name())

    front_alg.execute()
    return


def fix_profile_parameters(incoming_means_table, receiving_profiles_table, h_ratio):
    means_dict = _convert_table_to_dict(incoming_means_table)
    profiles_dict = _convert_table_to_dict(receiving_profiles_table)

    # Set intensities
    for p in profiles_dict.values():
        if np.isclose(p["mass"], 1, atol=0.1):  # Hydrogen present
            p["intensity"] = h_ratio * _get_lightest_profile(means_dict)["mean_intensity"]
            continue
        p["intensity"] = means_dict[p["label"]]["mean_intensity"]

    # Normalise intensities
    sum_intensities = sum([p["intensity"] for p in profiles_dict.values()])
    for p in profiles_dict.values():
        p["intensity"] /= sum_intensities

    # Set widths
    for p in profiles_dict.values():
        try:
            p["width"] = means_dict[p["label"]]["mean_width"]
        except KeyError:
            continue

    # Fix all widths except lightest mass
    for p in profiles_dict.values():
        if p == _get_lightest_profile(profiles_dict):
            continue
        p["width_lb"] = p["width"]
        p["width_ub"] = p["width"]

    result_profiles_table = _convert_dict_to_table(profiles_dict)
    return result_profiles_table


def _convert_table_to_dict(table):
    result_dict = {}
    for i in range(table.rowCount()):
        row_dict = table.row(i)
        result_dict[row_dict["label"]] = row_dict
    return result_dict


def _convert_dict_to_table(m_dict):
    table = CreateEmptyTableWorkspace()
    for p in m_dict.values():
        if table.columnCount() == 0:
            for key, value in p.items():
                value_type = "str" if isinstance(value, str) else "float"
                table.addColumn(value_type, key)

        table.addRow(p)
    return table


def _get_lightest_profile(p_dict):
    profiles = [p for p in p_dict.values()]
    masses = [p["mass"] for p in p_dict.values()]
    return profiles[np.argmin(masses)]


def init_analysis_algorithm(ws_name: str, inputs_class: type[BackwardAnalysisInputs] | type[ForwardAnalysisInputs]):
    # Skip if workspace not found in ADS
    if not AnalysisDataService.doesExist(ws_name):
        logger.warning(f"\n{ws_name} not found in ADS: Skipping cropping and masking ...\n")
        return

    profiles_table = create_profiles_table(ws_name + "_initial_parameters", inputs_class)
    instrument_parameters_file = str(FilesManager.get_instrument_parameters_dir() / inputs_class.instrument_parameters_file)

    scattering_type = _get_scattering_type(inputs_class)

    if scattering_type == "forward":
        inputs_class.chosen_mass_index = 0
        inputs_class.intensity_ratio_of_hydrogen_to_chosen_mass = 0

    kwargs = {
        "InputWorkspace": ws_name,
        "InputProfiles": profiles_table.name(),
        "InstrumentParametersFile": instrument_parameters_file,
        "ChosenMassIndex": inputs_class.chosen_mass_index,
        "HRatioToChosenMass": inputs_class.intensity_ratio_of_hydrogen_to_chosen_mass,
        "NumberOfIterations": int(inputs_class.number_of_iterations_for_corrections),
        "InvalidDetectors": convert_to_list_of_spectrum_numbers(inputs_class.mask_detectors),
        "MultipleScatteringCorrection": inputs_class.do_multiple_scattering_correction,
        "SampleShapeXml": inputs_class.sample_shape_xml,
        "GammaCorrection": inputs_class.do_gamma_correction,
        "ModeRunning": "BACKWARD" if scattering_type == "backward" else "FORWARD",
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


def crop_and_mask_workspace(ws_name, inputs_class: type[BackwardAnalysisInputs] | type[ForwardAnalysisInputs]):
    """Returns cloned and cropped workspace with modified name"""

    # Skip if workspace not found in ADS
    if not AnalysisDataService.doesExist(ws_name):
        logger.warning(f"\n{ws_name} not found in ADS: Skipping cropping and masking ...\n")
        return

    first_detector, last_detector = [int(s) for s in inputs_class.detectors.split("-")]

    # Read initial Spectrum number
    ws = mtd[ws_name]
    ws_first_detector = ws.getSpectrumNumbers()[0]
    assert first_detector >= ws_first_detector, "Can't crop workspace, firstSpec < first spectrum in workspace."
    first_idx = first_detector - ws_first_detector
    last_idx = last_detector - ws_first_detector

    ws_cropped = CropWorkspace(
        InputWorkspace=ws_name,
        StartWorkspaceIndex=first_idx,
        EndWorkspaceIndex=last_idx,
        OutputWorkspace=ws_name,
    )
    mask_time_of_flight_bins_with_zeros(ws_cropped, inputs_class.mask_time_of_flight_range)  # Used to mask resonance peaks
    MaskDetectors(Workspace=ws_cropped, SpectraList=inputs_class.mask_detectors)
    return ws_cropped


def load_and_save_input_ws_if_not_on_path(
    inputs_class: type[BackwardAnalysisInputs] | type[ForwardAnalysisInputs],
) -> tuple[Path, Path]:
    scattering_type = _get_scattering_type(inputs_class)

    if scattering_type == "backward":
        raw_path = FilesManager.get_inputs_ws_dir() / FilesManager.get_backward_raw_filename()
        empty_path = FilesManager.get_inputs_ws_dir() / FilesManager.get_backward_empty_filename()

    elif scattering_type == "forward":
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

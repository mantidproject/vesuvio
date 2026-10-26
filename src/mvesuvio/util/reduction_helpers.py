from __future__ import annotations

from typing import TYPE_CHECKING

from mantid import AnalysisDataService
from mantid.simpleapi import (
    mtd,
    Load,
    RenameWorkspace,
    SumSpectra,
    CropWorkspace,
    MaskDetectors,
    CreateEmptyTableWorkspace,
    DeleteWorkspace,
    SaveNexus,
    LoadVesuvio,
    Divide,
    Rebin,
    ConvertToYSpace,
    Integration,
    CreateWorkspace,
    CloneWorkspace,
)
from mantid.kernel import logger
import numpy as np
import math
from pathlib import Path

from mvesuvio.globals import ModeRunning, ScatteringType, Tags, Masking
from mvesuvio.util.files_manager import FilesManager
from mvesuvio.util import general_helpers
from mvesuvio.util.general_helpers import pass_data_into_ws, print_table_workspace, extractWS
from mvesuvio.util.constraints_transport import serialize_constraints
from mvesuvio.util import fitting_helpers
from mvesuvio.analysis_reduction import VesuvioAnalysisRoutine
from mantid.api import AlgorithmFactory, AlgorithmManager

if TYPE_CHECKING:
    from mvesuvio.default_config.experiment_template.run_reduction import BackwardAnalysisInputs, ForwardAnalysisInputs

import ntpath


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

    scattering_type = general_helpers._get_scattering_type(inputs_class)

    if scattering_type == ScatteringType.FORWARD:
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
        "GammaCorrection": inputs_class.do_gamma_correction,
        "ModeRunning": ModeRunning.BACKWARD.value if scattering_type == ScatteringType.BACKWARD else ModeRunning.FORWARD.value,
        "TransmissionGuess": inputs_class.transmission_guess,
        "MultipleScatteringOrder": int(inputs_class.multiple_scattering_order),
        "NumberOfEvents": int(inputs_class.multiple_scattering_number_of_events),
        "Constraints": serialize_constraints(inputs_class.constraints),
        "ResultsPath": str(FilesManager.get_reduction_outputs_dir().absolute()),
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


def load_input_ws(input_class: type[BackwardAnalysisInputs] | type[ForwardAnalysisInputs]):
    raw_path, empty_path = store_input_ws_if_not_on_path(input_class)

    raw_name = raw_path.stem
    empty_name = empty_path.stem

    Load(Filename=str(raw_path), OutputWorkspace=raw_name)
    Load(Filename=str(empty_path), OutputWorkspace=empty_name)

    SumSpectra(InputWorkspace=raw_name, OutputWorkspace=raw_name + "_sum")
    SumSpectra(InputWorkspace=empty_name, OutputWorkspace=empty_name + "_sum")

    return raw_name, empty_name


def store_input_ws_if_not_on_path(
    inputs_class: type[BackwardAnalysisInputs] | type[ForwardAnalysisInputs],
) -> tuple[Path, Path]:
    scattering_type = general_helpers._get_scattering_type(inputs_class)

    if scattering_type == ScatteringType.BACKWARD:
        raw_path = FilesManager.get_reduction_inputs_dir() / FilesManager.get_backward_raw_filename()
        empty_path = FilesManager.get_reduction_inputs_dir() / FilesManager.get_backward_empty_filename()

    elif scattering_type == ScatteringType.FORWARD:
        raw_path = FilesManager.get_reduction_inputs_dir() / FilesManager.get_forward_raw_filename()
        empty_path = FilesManager.get_reduction_inputs_dir() / FilesManager.get_forward_empty_filename()
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
    if Tags.BACKWARD in ws_path.stem:
        spectra = "3-134"
    elif Tags.FORWARD in ws_path.stem:
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


def save_fitting_input_workspaces(
    analysis_inputs: type[BackwardAnalysisInputs] | type[ForwardAnalysisInputs] | None,
) -> None:
    fitting_inputs_dir = FilesManager.get_fitting_inputs_dir()
    fitting_inputs_dir.mkdir(parents=True, exist_ok=True)
    if analysis_inputs is None:
        logger.warning("Could not save derived fitting workspaces because the inputs class was not provided.")
        return
    if not analysis_inputs.run_this_scattering_type:
        return

    iteration = str(analysis_inputs.number_of_iterations_for_corrections)
    overwrite_workspace = general_helpers.get_workspace_name_if_path(analysis_inputs.overwrite_analysis_input_workspace)
    if overwrite_workspace and AnalysisDataService.doesExist(overwrite_workspace):
        base_workspace_name = overwrite_workspace
    else:
        base_workspace_name = str(analysis_inputs.name_of_subtracted_workspace)

    workspace_name = f"{base_workspace_name}_{iteration}"
    ncp_group_name = f"{workspace_name}_ncp_group"

    if not AnalysisDataService.doesExist(workspace_name) or not AnalysisDataService.doesExist(ncp_group_name):
        logger.warning(f"Could not save derived fitting workspaces because expected reduction outputs are missing: {workspace_name}.")
        return

    ws_to_fit = mtd[workspace_name]
    ws_to_fit_ncps = mtd[ncp_group_name]

    ws_resolution = fitting_helpers.calculate_resolution(
        min(analysis_inputs.masses),
        ws_to_fit,
        analysis_inputs.range_for_rebinning_in_y_space,
    )

    ws_lighest_data, ws_lighest_ncp = fitting_helpers.isolate_lighest_mass_data(
        ws_to_fit,
        ws_to_fit_ncps,
        analysis_inputs.subtract_calculated_fse_from_data,
    )

    ws_joy, ws_joy_avg = ySpaceReduction(ws_lighest_data, ws_lighest_ncp, analysis_inputs)
    if analysis_inputs.do_symmetrisation:
        ws_joy_avg = symmetrizeWs(ws_joy_avg)

    SaveNexus(ws_resolution, str(fitting_inputs_dir / f"{workspace_name}_ws_resolution.nxs"))
    SaveNexus(ws_joy, str(fitting_inputs_dir / f"{workspace_name}_joy.nxs"))
    SaveNexus(ws_joy_avg, str(fitting_inputs_dir / f"{workspace_name}_joy_avg.nxs"))


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


def ySpaceReduction(wsTOF, ws_ncp, ic):
    """Seperate procedures depending on masking specified."""
    mass0 = ic.masses[0]
    ncp = ws_ncp.extractY()

    rebinPars = ic.range_for_rebinning_in_y_space

    if np.any(np.all(wsTOF.extractY() == 0, axis=0)):  # Masked columns present
        if ic.mask_zeros_with == Masking.NAN:
            # Build special workspace to store accumulated points
            wsJoY = convertToYSpace(wsTOF, mass0)
            xp = buildXRangeFromRebinPars(ic)
            wsJoYB = dataXBining(wsJoY, xp)  # Unusual ws with several dataY points per each dataX point

            # Need normalisation values from NCP masked workspace
            wsTOFNCP = replaceZerosWithNCP(wsTOF, ncp)
            wsJoYNCP = convertToYSpace(wsTOFNCP, mass0)
            wsJoYNCPN, wsJoYInt = rebinAndNorm(wsJoYNCP, rebinPars)

            # Normalize spectra of specieal workspace
            wsJoYN = Divide(wsJoYB, wsJoYInt, OutputWorkspace=wsJoYB.name() + "_norm")
            wsJoYAvg = weightedAvgXBins(wsJoYN, xp)
            return wsJoYN, wsJoYAvg

        elif ic.mask_zeros_with == Masking.NCP:
            wsTOF = replaceZerosWithNCP(wsTOF, ncp)

        else:
            raise ValueError(
                """
            Masked TOF bins were found but no valid procedure in y-space fit was selected.
            Options: 'nan', 'ncp'
            """
            )

    wsJoY = convertToYSpace(wsTOF, mass0)
    wsJoYN, wsJoYI = rebinAndNorm(wsJoY, rebinPars)
    wsJoYAvg = weightedAvgCols(wsJoYN)
    return wsJoYN, wsJoYAvg


def convertToYSpace(wsTOF, mass0):
    wsJoY = ConvertToYSpace(wsTOF, Mass=mass0, OutputWorkspace=wsTOF.name() + "_joy")
    return wsJoY


def rebinAndNorm(wsJoY, rebinPars):
    wsJoYR = Rebin(
        InputWorkspace=wsJoY,
        Params=rebinPars,
        FullBinsOnly=True,
        OutputWorkspace=wsJoY.name() + "_rebin",
    )
    wsJoYInt = Integration(wsJoYR, OutputWorkspace=wsJoYR.name() + "_integrated")
    wsJoYNorm = Divide(wsJoYR, wsJoYInt, OutputWorkspace=wsJoYR.name() + "_norm")
    return wsJoYNorm, wsJoYInt


def replaceZerosWithNCP(ws, ncp):
    """
    Replaces columns of bins with zeros on dataY with NCP provided.
    """
    dataX, dataY, dataE = extractWS(ws)
    mask = np.all(dataY == 0, axis=0)  # Masked Cols

    dataY[:, mask] = ncp[:, mask[: ncp.shape[1]]]  # mask of ncp adjusted for last col present or not

    wsMasked = CloneWorkspace(ws, OutputWorkspace=ws.name() + "_NCPMasked")
    pass_data_into_ws(dataX, dataY, dataE, wsMasked)
    SumSpectra(wsMasked, OutputWorkspace=wsMasked.name() + "_Sum")
    return wsMasked


def buildXRangeFromRebinPars(yFitIC):
    # Range used in case mask is set to NAN
    first, step, last = [float(s) for s in yFitIC.range_for_rebinning_in_y_space.split(",")]
    xp = np.arange(first, last, step) + step / 2  # Correction to match Mantid range
    return xp


def dataXBining(ws, xp):
    """
    Changes dataX of a workspace to values in range of bin centers xp.
    Same as shifting dataY values to closest bin center.
    Output ws has several dataY values per dataX point.
    """

    assert np.min(xp[:-1] - xp[1:]) == np.max(xp[:-1] - xp[1:]), "Bin widths need to be the same."
    step = xp[1] - xp[0]  # Calculate step from first two numbers
    # Form bins with xp being the centers
    bins = np.append(xp, [xp[-1] + step]) - step / 2

    dataX, dataY, dataE = extractWS(ws)
    # Loop below changes only the values of DataX
    for i, x in enumerate(dataX):
        # Select only valid range xr
        mask = (x < np.min(bins)) | (x > np.max(bins))
        xr = x[~mask]

        idxs = np.digitize(xr, bins)
        newXR = np.array([xp[idx] for idx in idxs - 1])  # Bin idx 1 refers to first bin ie idx 0 of centers

        # Pad invalid values with nans
        newX = x
        newX[mask] = np.nan  # Cannot use 0 as to not be confused with a dataX value
        newX[~mask] = newXR
        dataX[i] = newX  # Update DataX

    # Mask DataE values in same places as DataY values
    dataE[dataY == 0] = 0

    wsXBins = CloneWorkspace(ws, OutputWorkspace=ws.name() + "_XBinned")
    wsXBins = pass_data_into_ws(dataX, dataY, dataE, wsXBins)
    return wsXBins


def weightedAvgXBins(wsXBins, xp):
    """Weighted average on ws where dataY points are grouped per dataX bin centers."""
    dataX, dataY, dataE = extractWS(wsXBins)

    meansY, meansE = weightedAvgXBinsArr(dataX, dataY, dataE, xp)

    wsYSpaceAvg = CreateWorkspace(
        DataX=xp,
        DataY=meansY,
        DataE=meansE,
        NSpec=1,
        OutputWorkspace=wsXBins.name() + "_wavg",
    )
    return wsYSpaceAvg


def weightedAvgXBinsArr(dataX, dataY, dataE, xp):
    """
    Weighted Average on arrays where several dataY points correspond to a single dataX point.
    xp is the range over which to perform the average.
    dataX points can only take values in xp.
    Ignores any zero or NaN value.
    """
    meansY = np.zeros(len(xp))
    meansE = np.zeros(len(xp))

    for i in range(len(xp)):
        # Perform weighted average over all dataY and dataE values with the same xp[i]
        # Change shape to column to match weighted average function
        pointMask = dataX == xp[i]
        allY = dataY[pointMask][:, np.newaxis]
        allE = dataE[pointMask][:, np.newaxis]

        # If no points were found for a given abcissae
        if np.sum(pointMask) == 0:
            mY, mE = 0, 0  # Mask with zeros

        # If one point was found, set to that point
        elif np.sum(pointMask) == 1:
            mY, mE = allY.flatten(), allE.flatten()

        # Weighted avg over all spectra and several points per spectra
        else:
            # Case of bootstrap replica with no errors
            if np.all(dataE == 0):
                mY = avgArr(allY)
                mE = 0

            # Default for most cases
            else:
                mY, mE = weightedAvgArr(allY, allE)  # Outputs masked values as zeros

        # DataY and DataE should never reach NaN, but safeguard in case they do
        if (mE == np.nan) | (mY == np.nan):
            mY, mE = 0, 0

        meansY[i] = mY
        meansE[i] = mE

    return meansY, meansE


def weightedAvgCols(wsYSpace):
    """Returns ws with weighted avg of columns of input ws"""
    dataX, dataY, dataE = extractWS(wsYSpace)
    if np.all(dataE == 0):  # Bootstrap case where errors are not used
        meanY = avgArr(dataY)
        meanE = np.zeros(meanY.shape)
    else:
        meanY, meanE = weightedAvgArr(dataY, dataE)
    wsYSpaceAvg = CreateWorkspace(
        DataX=dataX[0, :],
        DataY=meanY,
        DataE=meanE,
        NSpec=1,
        OutputWorkspace=wsYSpace.name() + "_wavg",
    )
    return wsYSpaceAvg


def avgArr(dataYO):
    """
    Average over columns of 2D dataY.
    Ignores any zero values as being masked.
    """

    assert len(dataYO) > 1, "Averaging needs more than one element."

    dataY = dataYO.copy()
    dataY[dataY == 0] = np.nan
    meanY = np.nanmean(dataY, axis=0)
    meanY[meanY == np.nan] = 0

    assert np.all(np.all(dataYO == 0, axis=0) == (meanY == 0)), "Columns of zeros should give zero."
    return meanY


def weightedAvgArr(dataYO, dataEO):
    """
    Weighted average over columns of 2D arrays.
    Ignores any zero or NaN value.
    """

    # Run some tests
    assert dataYO.shape == dataEO.shape, "Y and E arrays should have same shape for weighted average."
    assert np.all((dataYO == 0) == (dataEO == 0)), (
        f"Masked zeros should match in DataY and DataE: {np.argwhere((dataYO == 0) != (dataEO == 0))}"
    )
    assert np.all(np.isnan(dataYO) == np.isnan(dataEO)), "Masked nans should match in DataY and DataE."
    assert len(dataYO) > 1, "Weighted average needs more than one element to be performed."

    dataY = dataYO.copy()  # Copy arrays not to change original data
    dataE = dataEO.copy()

    # Ignore invalid data by changing zeros to nans
    # If data is already masked with nans, it remains unaltered
    zerosMask = dataY == 0
    dataY[zerosMask] = np.nan
    dataE[zerosMask] = np.nan

    meanY = np.nansum(dataY / np.square(dataE), axis=0) / np.nansum(1 / np.square(dataE), axis=0)
    meanE = np.sqrt(1 / np.nansum(1 / np.square(dataE), axis=0))

    # Change invalid data back to original masking format with zeros
    nanInfMask = (meanE == np.inf) | (meanE == np.nan) | (meanY == np.nan)
    meanY[nanInfMask] = 0
    meanE[nanInfMask] = 0

    # Test that columns of zeros are left unchanged
    assert np.all((meanY == 0) == (meanE == 0)), "Weighted avg output should have masks in the same DataY and DataE."
    assert np.all((np.all(dataYO == 0, axis=0) | np.all(np.isnan(dataYO), axis=0)) == (meanY == 0)), "Masked cols should be ignored."

    return meanY, meanE


def symmetrizeWs(avgYSpace):
    """
    Symmetrizes workspace after weighted average.
    Needs to have symmetric binning.
    """

    dataX, dataY, dataE = extractWS(avgYSpace)

    if np.all(dataE == 0):
        dataYS = symArr(dataY)
        dataES = np.zeros(dataYS.shape)
    else:
        dataYS, dataES = weightedSymArr(dataY, dataE)

    wsSym = CloneWorkspace(avgYSpace, OutputWorkspace=avgYSpace.name() + "_sym")
    wsSym = pass_data_into_ws(dataX, dataYS, dataES, wsSym)
    return wsSym


def symArr(dataYO):
    """
    Performs averaging between two oposite points.
    Takes only one 2D array.
    Any zero gets assigned the oposite value.
    """

    assert len(dataYO.shape) == 2, "Symmetrization is written for 2D arrays."
    dataY = dataYO.copy()  # Copy arrays not to risk changing original data
    coMask = dataY == 0
    dataY[coMask] = np.nan

    yFlip = np.flip(dataY, axis=1)

    dataYS = np.nanmean(np.stack((dataY, yFlip)), axis=0)  # Normal avg between two numbers, cut-offs get ignored

    dataYS[dataYS == np.nan] = 0
    np.testing.assert_array_equal(dataYS, np.flip(dataYS, axis=1)), f"Symmetrisation failed in {np.argwhere(dataYS != np.flip(dataYS))}"
    np.testing.assert_allclose(dataYS[coMask], np.flip(dataYO, axis=1)[coMask])
    return dataYS


def weightedSymArr(dataYO, dataEO):
    """
    Performs Inverse variance weighting between two oposite points.
    When one of the points is a cut-off and the other is a valid point,
    the final value will be the valid point.
    """
    assert len(dataYO.shape) == 2, "Symmetrization is written for 2D arrays."
    assert np.all((dataYO == 0) == (dataEO == 0)), "Masked values should have zeros on both dataY and dataE."

    dataY = dataYO.copy()  # Copy arrays not to risk changing original data
    dataE = dataEO.copy()

    cutOffMask = dataY == 0
    # Change values of yerr to leave cut-offs unchanged during symmetrisation
    dataE[cutOffMask] = np.full(np.sum(cutOffMask), np.inf)

    yFlip = np.flip(dataY, axis=1)
    eFlip = np.flip(dataE, axis=1)

    # Inverse variance weighting
    dataYS = (dataY / dataE**2 + yFlip / eFlip**2) / (1 / dataE**2 + 1 / eFlip**2)
    dataES = 1 / np.sqrt(1 / dataE**2 + 1 / eFlip**2)

    # Deal with effects from previously changing dataE=np.inf
    nanInfMask = (dataES == np.inf) | (dataES == np.nan) | (dataYS == np.nan)
    dataYS[nanInfMask] = 0
    dataES[nanInfMask] = 0

    # Test that arrays are symmetrised
    np.testing.assert_array_equal(dataYS, np.flip(dataYS, axis=1)), f"Symmetrisation failed in {np.argwhere(dataYS != np.flip(dataYS))}"
    np.testing.assert_array_equal(dataES, np.flip(dataES, axis=1)), f"Symmetrisation failed in {np.argwhere(dataES != np.flip(dataES))}"

    # Test that cut-offs were not included in the symmetrisation
    np.testing.assert_allclose(dataYS[cutOffMask], np.flip(dataYO, axis=1)[cutOffMask])
    np.testing.assert_allclose(dataES[cutOffMask], np.flip(dataEO, axis=1)[cutOffMask])

    return dataYS, dataES

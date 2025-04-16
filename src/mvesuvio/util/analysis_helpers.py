
from mantid.simpleapi import Load, Rebin, Scale, SumSpectra, Minus, CropWorkspace, \
                            MaskDetectors, CreateWorkspace, CreateEmptyTableWorkspace, \
                            DeleteWorkspace, SaveNexus, LoadVesuvio, mtd, VesuvioResolution, \
                            AppendSpectra, RenameWorkspace, CloneWorkspace
from mantid.kernel import logger
import numpy as np
import numbers

from mvesuvio.util import handle_config

import ntpath

def isolate_lighest_mass_data(initial_ws, ws_group_ncp, subtract_fse=True):
    # NOTE: Minus() is not used so it doesn't change dataE

    ws_ncp_names = [n for n in ws_group_ncp.getNames() if n.endswith("ncp")]
    masses = [float(n.split('_')[-2]) for n in ws_ncp_names if 'total' not in n]
    ws_total_ncp_name = [n for n in ws_ncp_names if n.endswith('_total_ncp')][0]
    ws_lighest_ncp_name = ws_ncp_names[masses.index(min(masses))]
    ws_lighest_ncp = mtd[ws_lighest_ncp_name]
    ws_total_ncp = mtd[ws_total_ncp_name]
    suffix = "_m0"

    # Main subtraction
    isolated_data_y = initial_ws.extractY() - (ws_total_ncp.extractY() - ws_lighest_ncp.extractY())

    if subtract_fse:
        suffix += "_-fse"
        ws_lighest_fse = mtd[ws_lighest_ncp_name.replace('ncp', 'fse')]

        isolated_data_y -= ws_lighest_fse.extractY()

        # Subtract from fitted ncp
        # TODO: Find a better solution later
        ws_lighest_ncp_y = ws_lighest_ncp.extractY()
        ws_lighest_ncp_y -= ws_lighest_fse.extractY()
        ws_lighest_ncp = CloneWorkspace(ws_lighest_ncp, OutputWorkspace=ws_lighest_ncp.name()+"_-fse")
        write_data_y_into_ws(ws_lighest_ncp_y, ws_lighest_ncp)
        SumSpectra(ws_lighest_ncp.name(), OutputWorkspace=ws_lighest_ncp.name() + "_sum")

    # TODO: Find a better way to propagate masked values
    isolated_data_y[initial_ws.extractY() == 0] = 0
    ws_lighest_data = CloneWorkspace(initial_ws, OutputWorkspace=initial_ws.name()+suffix)
    write_data_y_into_ws(isolated_data_y, ws_lighest_data)
    SumSpectra(ws_lighest_data.name(), OutputWorkspace=ws_lighest_data.name() + "_sum")

    return ws_lighest_data, ws_lighest_ncp


def write_data_y_into_ws(data_y, ws):
    for i in range(ws.getNumberHistograms()):
        ws.dataY(i)[:] = data_y[i, :] 
    return


def calculate_resolution(mass, ws, rebin_range):

    resName = ws.name() + f"_resolution"
    for index in range(ws.getNumberHistograms()):
        VesuvioResolution(
            Workspace=ws, WorkspaceIndex=index, Mass=mass, OutputWorkspaceYSpace="tmp"
        )
        Rebin(
            InputWorkspace="tmp",
            Params=rebin_range,
            OutputWorkspace="tmp",
        )

        if index == 0:  # Ensures that workspace has desired units
            RenameWorkspace("tmp", resName)
        else:
            AppendSpectra(resName, "tmp", OutputWorkspace=resName)

    masked_idx = [ws.spectrumInfo().isMasked(i) for i in range(ws.getNumberHistograms())]
    MaskDetectors(resName, WorkspaceIndexList=np.flatnonzero(masked_idx))
    DeleteWorkspace("tmp")
    return mtd[resName]


def pass_data_into_ws(dataX, dataY, dataE, ws):
    "Modifies ws data to input data"
    for i in range(ws.getNumberHistograms()):
        ws.dataX(i)[:] = dataX[i, :]
        ws.dataY(i)[:] = dataY[i, :]
        ws.dataE(i)[:] = dataE[i, :]
    return ws


def print_table_workspace(table, precision=3):
    table_dict = table.toDict()
    # Convert floats into strings 
    for key, values in table_dict.items():
        new_column = [int(item) if (isinstance(item, float) and item.is_integer()) else item for item in values]
        table_dict[key] = [f"{item:.{precision}f}" if isinstance(item, float) else str(item) for item in new_column]

    max_spacing = [max([len(item) for item in values] + [len(key)]) for key, values in table_dict.items()]
    header = "|" + "|".join(f"{item}{' '*(spacing-len(item))}" for item, spacing in zip(table_dict.keys(), max_spacing)) + "|"
    logger.notice(f"Table {table.name()}:")
    logger.notice(' '+'-'*(len(header)-2)+' ')
    logger.notice(header)
    for i in range(table.rowCount()):
        table_row = "|".join(f"{values[i]}{' '*(spacing-len(str(values[i])))}" for values, spacing in zip(table_dict.values(), max_spacing))
        logger.notice("|" + table_row + "|")
    logger.notice(' '+'-'*(len(header)-2)+' ')
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
        if bound == None:
            return np.inf
        return bound

    for mass, intensity, width, center, intensity_bound, width_bound, center_bound in zip(
        ai.masses, ai.initial_fitting_parameters[::3], ai.initial_fitting_parameters[1::3], ai.initial_fitting_parameters[2::3],
        ai.fitting_bounds[::3], ai.fitting_bounds[1::3], ai.fitting_bounds[2::3]
    ):
        table.addRow(
            [str(float(mass)), float(mass), float(intensity), float(wrapb(intensity_bound[0])), float(wrapb(intensity_bound[1])),
             float(width), float(wrapb(width_bound[0])), float(wrapb(width_bound[1])), float(center), float(wrapb(center_bound[0])), float(wrapb(center_bound[1]))]
        )
    return table


def create_table_for_hydrogen_to_mass_ratios():
    table = CreateEmptyTableWorkspace(
        OutputWorkspace="hydrogen_intensity_ratios_estimates"
    )
    table.addColumn(type="float", name="Hydrogen intensity ratio to lowest mass at each iteration")
    return table


def is_hydrogen_present(masses) -> bool:
    Hmask = np.abs(np.array(masses) - 1) / 1 < 0.1  # H mass whithin 10% of 1 au

    if ~np.any(Hmask):  # H not present
        return False

    print("\nH mass detected.\n")
    assert (
        len(Hmask) > 1
    ), "When H is only mass present, run independent forward procedure, not joint."
    assert Hmask[0], "H mass needs to be the first mass in masses and initPars."
    assert sum(Hmask) == 1, "More than one mass very close to H were detected."
    return True


def ws_history_matches_inputs(runs, mode, ipfile, ws_path):

    if not (ws_path.is_file()):
        logger.notice(f"Cached workspace not found at {ws_path}")
        return False

    ws = Load(Filename=str(ws_path))
    ws_history = ws.getHistory()
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
    if saved_ipfile_name != ipfile:
        logger.notice(
            f"IP files in saved workspace did not match: {saved_ipfile_name} and {ipfile}"
        )
        return False

    logger.notice("\nLocally saved workspace metadata matched with analysis inputs.\n")
    DeleteWorkspace(ws)
    return True


def save_ws_from_load_vesuvio(runs, mode, ipfile, ws_path):

    if "backward" in ws_path.name:
        spectra = "3-134"
    elif "forward" in ws_path.name:
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


def name_for_starting_ws(load_ai):
    name_suffix = scattering_type(load_ai, shorthand=True) 
    name = handle_config.get_script_name() + "_" + name_suffix
    return name


def scattering_type(load_ai, shorthand=False):
    if load_ai.__name__ in ["BackwardAnalysisInputs"]:
        scatteringType = "BACKWARD"
        if shorthand:
            scatteringType = "bckwd"
    elif load_ai.__name__ in ["ForwardAnalysisInputs"]:
        scatteringType = "FORWARD"
        if shorthand:
            scatteringType = "fwd"
    else:
        raise ValueError(
            f"Input class for workspace not valid: {load_ai.__name__}"
        )
    return scatteringType 


def load_raw_and_empty_from_path(userWsRawPath, userWsEmptyPath, 
                                  tofBinning, name, scaleRaw, scaleEmpty, subEmptyFromRaw):
    print("\nLoading local workspaces ...\n")
    Load(Filename=str(userWsRawPath), OutputWorkspace=name + "_raw")
    Rebin(
        InputWorkspace=name + "_raw",
        Params=tofBinning,
        OutputWorkspace=name + "_raw",
    )

    assert (isinstance(scaleRaw, numbers.Real)), "Scaling factor of raw ws needs to be float or int."
    Scale(
        InputWorkspace=name + "_raw",
        OutputWorkspace=name + "_raw",
        Factor=str(scaleRaw),
    )

    SumSpectra(InputWorkspace=name + "_raw", OutputWorkspace=name + "_raw" + "_sum")
    wsToBeFitted = mtd[name+"_raw"]

    if subEmptyFromRaw:
        Load(Filename=str(userWsEmptyPath), OutputWorkspace=name + "_empty")
        Rebin(
            InputWorkspace=name + "_empty",
            Params=tofBinning,
            OutputWorkspace=name + "_empty",
        )

        assert (isinstance(scaleEmpty, float)) | (
            isinstance(scaleEmpty, int)
        ), "Scaling factor of empty ws needs to be float or int"
        Scale(
            InputWorkspace=name + "_empty",
            OutputWorkspace=name + "_empty",
            Factor=str(scaleEmpty),
        )

        SumSpectra(
            InputWorkspace=name + "_empty", OutputWorkspace=name + "_empty" + "_sum"
        )

        wsToBeFitted = Minus(
            LHSWorkspace=name + "_raw",
            RHSWorkspace=name + "_empty",
            OutputWorkspace=name + "_raw_-empty",
        )
    return wsToBeFitted


def cropAndMaskWorkspace(ws, firstSpec, lastSpec, maskedDetectors, maskTOFRange):
    """Returns cloned and cropped workspace with modified name"""
    # Read initial Spectrum number
    wsFirstSpec = ws.getSpectrumNumbers()[0]
    assert (
        firstSpec >= wsFirstSpec
    ), "Can't crop workspace, firstSpec < first spectrum in workspace."

    initialIdx = firstSpec - wsFirstSpec
    lastIdx = lastSpec - wsFirstSpec

    newWsName = ws.name().split("_raw")[0]  # Retrieve original name
    wsCrop = CropWorkspace(
        InputWorkspace=ws,
        StartWorkspaceIndex=initialIdx,
        EndWorkspaceIndex=lastIdx,
        OutputWorkspace=newWsName,
    )

    mask_time_of_flight_bins_with_zeros(wsCrop, maskTOFRange)  # Used to mask resonance peaks

    MaskDetectors(Workspace=wsCrop, SpectraList=maskedDetectors)
    return wsCrop


def mask_time_of_flight_bins_with_zeros(ws, maskTOFRange):
    """
    Masks a given TOF range on ws with zeros on dataY.
    Leaves errors dataE unchanged, as they are used by later treatments.
    Used to mask resonance peaks.
    """

    if maskTOFRange is None:
        return

    dataX, dataY, dataE = extractWS(ws)
    start, end = [float(s) for s in maskTOFRange.split("-")]
    assert (
        start <= end
    ), "Start value for masking needs to be smaller or equal than end."
    mask = (dataX >= start) & (dataX <= end)  # TOF region to mask

    dataY[mask] = 0

    pass_data_into_ws(dataX, dataY, dataE, ws)
    return


def extractWS(ws):
    """Directly extracts data from workspace into arrays"""
    return ws.extractX(), ws.extractY(), ws.extractE()


def numerical_third_derivative(x, y):
    k6 = (- y[:, 12:] + y[:, :-12]) * 1
    k5 = (+ y[:, 11:-1] - y[:, 1:-11]) * 24
    k4 = (- y[:, 10:-2] + y[:, 2:-10]) * 192
    k3 = (+ y[:, 9:-3] - y[:, 3:-9]) * 488
    k2 = (+ y[:, 8:-4] - y[:, 4:-8]) * 387
    k1 = (- y[:, 7:-5] + y[:, 5:-7]) * 1584

    dev = k1 + k2 + k3 + k4 + k5 + k6
    dev /= np.power(x[:, 7:-5] - x[:, 6:-6], 3)
    dev /= 12**3
    return dev


def load_resolution(instrument_params):
    """Resolution of parameters to propagate into TOF resolution
    Output: matrix with each parameter in each column"""
    spectra = instrument_params[:, 0]
    L = len(spectra)
    # For spec no below 135, back scattering detectors, mode is double difference
    # For spec no 135 or above, front scattering detectors, mode is single difference
    dE1 = np.where(spectra < 135, 88.7, 73)  # meV, STD
    dE1_lorz = np.where(spectra < 135, 40.3, 24)  # meV, HFHM
    dTOF = np.repeat(0.37, L)  # us
    dTheta = np.repeat(0.016, L)  # rad
    dL0 = np.repeat(0.021, L)  # meters
    dL1 = np.repeat(0.023, L)  # meters

    resolutionPars = np.vstack((dE1, dTOF, dTheta, dL0, dL1, dE1_lorz)).transpose()
    return resolutionPars


def load_instrument_params(ip_file, spectrum_list):

    first_spec = min(spectrum_list)
    last_spec = max(spectrum_list)
    data = np.loadtxt(ip_file, dtype=str)[1:].astype(float)
    spectra = data[:, 0]

    select_rows = np.where((spectra >= first_spec) & (spectra <= last_spec))
    return data[select_rows]


def fix_profile_parameters(incoming_means_table, receiving_profiles_table, h_ratio):
    means_dict = _convert_table_to_dict(incoming_means_table)
    profiles_dict = _convert_table_to_dict(receiving_profiles_table)

    # Set intensities
    for p in profiles_dict.values():
        if np.isclose(p['mass'], 1, atol=0.1):    # Hydrogen present
            p['intensity'] = h_ratio * _get_lightest_profile(means_dict)['mean_intensity']
            continue
        p['intensity'] = means_dict[p['label']]['mean_intensity']

    # Normalise intensities
    sum_intensities = sum([p['intensity'] for p in profiles_dict.values()])
    for p in profiles_dict.values():
        p['intensity'] /= sum_intensities
        
    # Set widths
    for p in profiles_dict.values():
        try:
            p['width'] = means_dict[p['label']]['mean_width']
        except KeyError:
            continue

    # Fix all widths except lightest mass
    for p in profiles_dict.values():
        if p == _get_lightest_profile(profiles_dict):
            continue
        p['width_lb'] = p['width']
        p['width_ub'] = p['width']

    result_profiles_table = _convert_dict_to_table(profiles_dict)
    return result_profiles_table


def _convert_table_to_dict(table):
    result_dict = {}
    for i in range(table.rowCount()):
        row_dict = table.row(i) 
        result_dict[row_dict['label']] = row_dict
    return result_dict


def _convert_dict_to_table(m_dict):
    table = CreateEmptyTableWorkspace()
    for p in m_dict.values():
        if table.columnCount() == 0:
            for key, value in p.items():
                value_type = 'str' if isinstance(value, str) else 'float'
                table.addColumn(value_type, key)

        table.addRow(p)
    return table


def _get_lightest_profile(p_dict):
    profiles = [p for p in p_dict.values()]
    masses = [p['mass'] for p in p_dict.values()]
    return profiles[np.argmin(masses)]


def calculate_h_ratio(means_table):

    masses = means_table.column("mass")
    intensities = np.array(means_table.column("mean_intensity"))

    if not np.isclose(min(masses), 1, atol=0.1):    # Hydrogen not present
        return None
    
    # Hydrogen present 
    sorted_intensities = intensities[np.argsort(masses)]

    return sorted_intensities[0] / sorted_intensities[1] 


def extend_range_of_array(arr, n_columns):
    arr = arr.copy()
    left_extend = arr[:, :n_columns] + (arr[:, 0] - arr[:, n_columns]).reshape(-1, 1)
    right_extend = arr[:, -n_columns:] + (arr[:, -1] - arr[:, -n_columns-1]).reshape(-1, 1)
    return np.concatenate([left_extend, arr, right_extend], axis=-1)


def make_gamma_correction_input_string(masses, mean_widths, mean_intensity_ratios):
    profiles = ""
    for mass, width, intensity in zip(masses, mean_widths, mean_intensity_ratios):
        profiles += (
            "name=GaussianComptonProfile,Mass="
            + str(mass)
            + ",Width="
            + str(width)
            + ",Intensity="
            + str(intensity)
            + ";"
        )
    logger.notice("\nThe sample properties for Gamma Correction are:\n\n" + \
            str(profiles).replace(';', '\n\n').replace(',', '\n'))
    return profiles


def make_multiple_scattering_input_string(masses, meanWidths, meanIntensityRatios):
    atomic_properties_list = np.vstack([masses, meanIntensityRatios, meanWidths]).transpose().flatten().tolist()
    logger.notice(
        "\nSample properties for multiple scattering correction:\n\n" + \
        "mass   intensity   width\n" + \
        str(np.array(atomic_properties_list).reshape(-1, 3)).replace('[', '').replace(']', '') + "\n"
    )
    return atomic_properties_list 

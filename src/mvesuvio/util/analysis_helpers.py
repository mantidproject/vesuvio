
from mantid.simpleapi import Load, Rebin, Scale, SumSpectra, Minus, CropWorkspace, \
                            MaskDetectors, CreateWorkspace, CreateEmptyTableWorkspace, \
                            DeleteWorkspace, SaveNexus, LoadVesuvio, mtd
from mantid.kernel import logger
import numpy as np
import numbers

from mvesuvio.analysis_fitting import passDataIntoWS
from mvesuvio.util import handle_config

import ntpath


def create_profiles_table(name, ai):
    table = CreateEmptyTableWorkspace(OutputWorkspace=name)
    table.addColumn(type="str", name="label")
    table.addColumn(type="float", name="mass")
    table.addColumn(type="float", name="intensity")
    table.addColumn(type="str", name="intensity_bounds")
    table.addColumn(type="float", name="width")
    table.addColumn(type="str", name="width_bounds")
    table.addColumn(type="float", name="center")
    table.addColumn(type="str", name="center_bounds")
    for mass, intensity, width, center, intensity_bound, width_bound, center_bound in zip(
        ai.masses, ai.initPars[::3], ai.initPars[1::3], ai.initPars[2::3],
        ai.bounds[::3], ai.bounds[1::3], ai.bounds[2::3]
    ):
        table.addRow(
            [str(float(mass)), float(mass), float(intensity), str(list(intensity_bound)),
            float(width), str(list(width_bound)), float(center), str(list(center_bound))]
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
        logger.notice("Cached workspace not found")
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

    print("\nLocally saved workspace metadata matched with analysis inputs.\n")
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

    SaveNexus(vesuvio_ws, str(ws_path.absolute()))
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


def loadRawAndEmptyWsFromUserPath(userWsRawPath, userWsEmptyPath, 
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
            OutputWorkspace=name + "_raw_minus_empty",
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

    maskBinsWithZeros(wsCrop, maskTOFRange)  # Used to mask resonance peaks

    MaskDetectors(Workspace=wsCrop, SpectraList=maskedDetectors)
    return wsCrop


def maskBinsWithZeros(ws, maskTOFRange):
    """
    Masks a given TOF range on ws with zeros on dataY.
    Leaves errors dataE unchanged, as they are used by later treatments.
    Used to mask resonance peaks.
    """

    if maskTOFRange is None:
        return

    dataX, dataY, dataE = extractWS(ws)
    start, end = [int(s) for s in maskTOFRange.split(",")]
    assert (
        start <= end
    ), "Start value for masking needs to be smaller or equal than end."
    mask = (dataX >= start) & (dataX <= end)  # TOF region to mask

    dataY[mask] = 0

    passDataIntoWS(dataX, dataY, dataE, ws)
    return


def extractWS(ws):
    """Directly extracts data from workspace into arrays"""
    return ws.extractX(), ws.extractY(), ws.extractE()


def loadConstants():
    """Output: the mass of the neutron, final energy of neutrons (selected by gold foil),
    factor to change energies into velocities, final velocity of neutron and hbar"""
    mN = 1.008  # a.m.u.
    Ef = 4906.0  # meV
    en_to_vel = 4.3737 * 1.0e-4
    vf = np.sqrt(Ef) * en_to_vel  # m/us
    hbar = 2.0445
    constants = (mN, Ef, en_to_vel, vf, hbar)
    return constants


def numericalThirdDerivative(x, y):
    k6 = (- y[:, 12:] + y[:, :-12]) * 1
    k5 = (+ y[:, 11:-1] - y[:, 1:-11]) * 24
    k4 = (- y[:, 10:-2] + y[:, 2:-10]) * 192
    k3 = (+ y[:, 9:-3] - y[:, 3:-9]) * 488
    k2 = (+ y[:, 8:-4] - y[:, 4:-8]) * 387
    k1 = (- y[:, 7:-5] + y[:, 5:-7]) * 1584

    dev = k1 + k2 + k3 + k4 + k5 + k6
    dev /= np.power(x[:, 7:-5] - x[:, 6:-6], 3)
    dev /= 12**3

    derivative = np.zeros_like(y)
    # Padded with zeros left and right to return array with same shape
    derivative[:, 6:-6] = dev
    return derivative


def createWS(dataX, dataY, dataE, wsName, parentWorkspace=None):
    ws = CreateWorkspace(
        DataX=dataX.flatten(),
        DataY=dataY.flatten(),
        DataE=dataE.flatten(),
        Nspec=len(dataY),
        OutputWorkspace=wsName,
        ParentWorkspace=parentWorkspace
    )
    return ws


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
        p['width_bounds'] = str([p['width'] , p['width']])

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


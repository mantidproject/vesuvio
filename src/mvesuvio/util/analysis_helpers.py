
from mantid.simpleapi import Load, Rebin, Scale, SumSpectra, Minus, CropWorkspace, \
                            MaskDetectors, CreateWorkspace, CreateEmptyTableWorkspace, \
                            DeleteWorkspace, SaveNexus, LoadVesuvio, mtd, VesuvioResolution, \
                            AppendSpectra, RenameWorkspace, CloneWorkspace, Divide, Integration, ConvertToYSpace
from mantid.kernel import logger
import numpy as np
import numbers

from mvesuvio.util import handle_config

import ntpath

def isolate_lighest_mass_data(initial_ws, ws_group_ncp, subtract_fse=True):
    # NOTE: Minus() is not used so it doesn't change dataE

    ws_ncp_names = ws_group_ncp.getNames()
    masses = [float(n.split('_')[-2]) for n in ws_ncp_names if 'total' not in n]
    ws_name_lightest_profile = ws_ncp_names[masses.index(min(masses))]
    ws_name_profiles = [n for n in ws_ncp_names if n.endswith('_total_ncp')][0]

    ws_lighest_data = CloneWorkspace(initial_ws, OutputWorkspace=initial_ws.name()+"_m0")

    isolated_data_y = ws_lighest_data.extractY() - (mtd[ws_name_profiles].extractY() - mtd[ws_name_lightest_profile].extractY())

    for i in range(ws_lighest_data.getNumberHistograms()):
        ws_lighest_data.dataY(i)[:] = isolated_data_y[i, :] 
    SumSpectra(ws_lighest_data.name(), OutputWorkspace=ws_lighest_data.name() + "_sum")

    if subtract_fse:

        isolated_data_y -= mtd[ws_name_lightest_profile.replace('ncp', 'fse')].extractY()
        ws_lighest_data = CloneWorkspace(ws_lighest_data, OutputWorkspace=ws_lighest_data.name()+"_fse")

        for i in range(ws_lighest_data.getNumberHistograms()):
            ws_lighest_data.dataY(i)[:] = isolated_data_y[i, :] 
        SumSpectra(ws_lighest_data.name(), OutputWorkspace=ws_lighest_data.name() + "_sum")

    return ws_lighest_data, mtd[ws_name_lightest_profile]


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
    table.addColumn(type="str", name="intensity_bounds")
    table.addColumn(type="float", name="width")
    table.addColumn(type="str", name="width_bounds")
    table.addColumn(type="float", name="center")
    table.addColumn(type="str", name="center_bounds")

    def bounds_to_str(bounds):
        return " : ".join([str(i) for i in list(bounds)])

    for mass, intensity, width, center, intensity_bound, width_bound, center_bound in zip(
        ai.masses, ai.initial_fitting_parameters[::3], ai.initial_fitting_parameters[1::3], ai.initial_fitting_parameters[2::3],
        ai.fitting_bounds[::3], ai.fitting_bounds[1::3], ai.fitting_bounds[2::3]
    ):
        table.addRow(
            [str(float(mass)), float(mass), float(intensity), bounds_to_str(intensity_bound),
             float(width), bounds_to_str(width_bound), float(center), bounds_to_str(center_bound)]
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


def extend_range_of_array(arr, n_columns):
    arr = arr.copy()
    left_extend = arr[:, :n_columns] + (arr[:, 0] - arr[:, n_columns]).reshape(-1, 1)
    right_extend = arr[:, -n_columns:] + (arr[:, -1] - arr[:, -n_columns-1]).reshape(-1, 1)
    return np.concatenate([left_extend, arr, right_extend], axis=-1)


# TODO: Sort out whathever is going on with this reduction
# And the rest of the functions until the end of the page

def ySpaceReduction(ws_data, ws_ncp, mass, ic):
    """Reduction for workspace corresponding to a single mass profile"""

    # mass0 = ic.masses[0]
    # profiles_table = mtd[ic.name + '_initial_parameters']
    # lightest_mass_str = profiles_table.column('label')[np.argmin(profiles_table.column('mass'))]
    # ws_name_lightest_mass = ic.name + '_' + str(ic.number_of_iterations_for_corrections) + '_' + lightest_mass_str + '_ncp'

    ncp = ws_ncp.extractY()

    rebinPars = ic.range_for_rebinning_in_y_space

    if np.any(np.all(ws_data.extractY() == 0, axis=0)):  # Masked columns present
        if ic.mask_zeros_with == "nan":
            # Build special workspace to store accumulated points
            wsJoY = convertToYSpace(ws_data, mass)
            xp = buildXRangeFromRebinPars(ic)
            wsJoYB = dataXBining(
                wsJoY, xp
            )  # Unusual ws with several dataY points per each dataX point

            # Need normalisation values from NCP masked workspace
            wsTOFNCP = replaceZerosWithNCP(ws_data, ncp)
            wsJoYNCP = convertToYSpace(wsTOFNCP, mass)
            wsJoYNCPN, wsJoYInt = rebinAndNorm(wsJoYNCP, rebinPars)

            # Normalize spectra of specieal workspace
            wsJoYN = Divide(
                wsJoYB, wsJoYInt, OutputWorkspace=wsJoYB.name() + "_norm"
            )
            wsJoYAvg = weightedAvgXBins(wsJoYN, xp)
            return wsJoYN, wsJoYAvg

        elif ic.mask_zeros_with == "ncp":
            ws_data = replaceZerosWithNCP(ws_data, ncp)

        else:
            raise ValueError(
                """
            Masked TOF bins were found but no valid procedure in y-space fit was selected.
            Options: 'nan', 'ncp'
            """
            )

    wsJoY = convertToYSpace(ws_data, mass)
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

    dataY[:, mask] = ncp[
        :, mask[: ncp.shape[1]]
    ]  # mask of ncp adjusted for last col present or not

    wsMasked = CloneWorkspace(ws, OutputWorkspace=ws.name() + "_NCPMasked")
    pass_data_into_ws(dataX, dataY, dataE, wsMasked)
    SumSpectra(wsMasked, OutputWorkspace=wsMasked.name() + "_Sum")
    return wsMasked


def buildXRangeFromRebinPars(yFitIC):
    # Range used in case mask is set to NAN
    first, step, last = [
        float(s) for s in yFitIC.range_for_rebinning_in_y_space.split(",")
    ]
    xp = np.arange(first, last, step) + step / 2  # Correction to match Mantid range
    return xp


def dataXBining(ws, xp):
    """
    Changes dataX of a workspace to values in range of bin centers xp.
    Same as shifting dataY values to closest bin center.
    Output ws has several dataY values per dataX point.
    """

    assert np.min(xp[:-1] - xp[1:]) == np.max(
        xp[:-1] - xp[1:]
    ), "Bin widths need to be the same."
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
        newXR = np.array(
            [xp[idx] for idx in idxs - 1]
        )  # Bin idx 1 refers to first bin ie idx 0 of centers

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

    assert np.all(
        np.all(dataYO == 0, axis=0) == (meanY == 0)
    ), "Columns of zeros should give zero."
    return meanY


def weightedAvgArr(dataYO, dataEO):
    """
    Weighted average over columns of 2D arrays.
    Ignores any zero or NaN value.
    """

    # Run some tests
    assert (
        dataYO.shape == dataEO.shape
    ), "Y and E arrays should have same shape for weighted average."
    assert np.all(
        (dataYO == 0) == (dataEO == 0)
    ), f"Masked zeros should match in DataY and DataE: {np.argwhere((dataYO==0)!=(dataEO==0))}"
    assert np.all(
        np.isnan(dataYO) == np.isnan(dataEO)
    ), "Masked nans should match in DataY and DataE."
    assert (
        len(dataYO) > 1
    ), "Weighted average needs more than one element to be performed."

    dataY = dataYO.copy()  # Copy arrays not to change original data
    dataE = dataEO.copy()

    # Ignore invalid data by changing zeros to nans
    # If data is already masked with nans, it remains unaltered
    zerosMask = dataY == 0
    dataY[zerosMask] = np.nan
    dataE[zerosMask] = np.nan

    meanY = np.nansum(dataY / np.square(dataE), axis=0) / np.nansum(
        1 / np.square(dataE), axis=0
    )
    meanE = np.sqrt(1 / np.nansum(1 / np.square(dataE), axis=0))

    # Change invalid data back to original masking format with zeros
    nanInfMask = (meanE == np.inf) | (meanE == np.nan) | (meanY == np.nan)
    meanY[nanInfMask] = 0
    meanE[nanInfMask] = 0

    # Test that columns of zeros are left unchanged
    assert np.all(
        (meanY == 0) == (meanE == 0)
    ), "Weighted avg output should have masks in the same DataY and DataE."
    assert np.all(
        (np.all(dataYO == 0, axis=0) | np.all(np.isnan(dataYO), axis=0)) == (meanY == 0)
    ), "Masked cols should be ignored."

    return meanY, meanE


def normalise_workspace(ws_name):
    """Updates workspace with the normalised version."""
    tmp_norm = Integration(ws_name)
    Divide(LHSWorkspace=ws_name, RHSWorkspace=tmp_norm, OutputWorkspace=ws_name)
    DeleteWorkspace("tmp_norm")


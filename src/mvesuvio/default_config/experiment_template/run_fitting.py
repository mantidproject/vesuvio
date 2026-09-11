from mvesuvio.analysis_fitting import FitInYSpace
from mvesuvio.util.files_manager import FilesManager
from mantid.api import AnalysisDataService
from mantid.kernel import logger
from mantid.simpleapi import Load, SaveAscii, mtd
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .run_reduction import BackwardAnalysisInputs, ForwardAnalysisInputs
else:
    try:
        # Preferred import path when this module is executed as part of the package.
        from .run_reduction import BackwardAnalysisInputs, ForwardAnalysisInputs
    except ImportError:
        # Fallback for direct/script execution: load sibling run_reduction.py by file path.
        import importlib.util
        from pathlib import Path

        _run_reduction_path = Path(__file__).resolve().parent / "run_reduction.py"
        _module_name = "mvesuvio.default_config.experiment_template._local_run_reduction"
        _spec = importlib.util.spec_from_file_location(_module_name, _run_reduction_path)
        if _spec is None or _spec.loader is None:
            raise ImportError(f"Could not load run_reduction module from {_run_reduction_path}")

        _run_reduction = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_run_reduction)

        BackwardAnalysisInputs = _run_reduction.BackwardAnalysisInputs
        ForwardAnalysisInputs = _run_reduction.ForwardAnalysisInputs


class BackwardFittingInputs(BackwardAnalysisInputs):
    run_this_fitting_type = False
    show_plots = True
    do_symmetrisation = False
    subtract_calculated_fse_from_data = True
    range_for_rebinning_in_y_space = "-25, 0.5, 25"  # Needs to be symetric, usually bounds = 10 x lowest mass (a.m.u.)
    # Fitting model options
    # 'gauss': Single Gaussian
    # 'gauss_cntr': Single Gaussian with fixed center at zero
    # 'gcc4': Gram-Charlier with C4 parameter
    # 'gcc4_cntr': Gram-Charlier with C4 parameter with fixed center at zero
    # 'gcc6': Gram-Charlier with C6 parameter
    # 'gcc6_cntr': Gram-Charlier with C6 parameter with fixed center at zero
    # 'gcc4c6': Gram-Charlier with C4 and C6 parameter
    # 'gcc4c6_cntr': Gram-Charlier with C4 and C6 parameter and fixed center at zero
    # 'doublewell': Double Well function
    # 'gauss2d': Anisotropic Gaussian
    # 'gauss3d': 3-Dimensional Gaussian
    fitting_model = "gauss"
    run_minos = True
    do_global_fit = True  # Performs global fit with Minuit by default
    # Number of groups of detectors to perform global (simultaneous) fit on
    # Either an integer less than the number of detectors
    # or option 'all', which does not form groups and fits all spectra simultaneously and individualy
    number_of_global_fit_groups = 4
    # Type of masking
    # 'nan': Zeros in workspace being fit are ignored
    # 'ncp': Zeros in workspace being fit are replaced by the fitted neutron compton profile
    mask_zeros_with = "nan"


class ForwardFittingInputs(ForwardAnalysisInputs):
    run_this_fitting_type = True
    show_plots = True
    do_symmetrisation = False
    subtract_calculated_fse_from_data = True
    range_for_rebinning_in_y_space = "-25, 0.5, 25"  # Needs to be symetric, usually bounds = 10 x lowest mass (a.m.u.)
    # Fitting model options
    # 'gauss': Single Gaussian
    # 'gauss_cntr': Single Gaussian with fixed center at zero
    # 'gcc4': Gram-Charlier with C4 parameter
    # 'gcc4_cntr': Gram-Charlier with C4 parameter with fixed center at zero
    # 'gcc6': Gram-Charlier with C6 parameter
    # 'gcc6_cntr': Gram-Charlier with C6 parameter with fixed center at zero
    # 'gcc4c6': Gram-Charlier with C4 and C6 parameter
    # 'gcc4c6_cntr': Gram-Charlier with C4 and C6 parameter and fixed center at zero
    # 'doublewell': Double Well function
    # 'gauss2d': Anisotropic Gaussian
    # 'gauss3d': 3-Dimensional Gaussian
    fitting_model = "gauss"
    run_minos = True
    do_global_fit = True  # Performs global fit with Minuit by default
    # Number of groups of detectors to perform global (simultaneous) fit on
    # Either an integer less than the number of detectors
    # or option 'all', which does not form groups and fits all spectra simultaneously and individualy
    number_of_global_fit_groups = 4
    # Type of masking
    # 'nan': Zeros in workspace being fit are ignored
    # 'ncp': Zeros in workspace being fit are replaced by the fitted neutron compton profile
    mask_zeros_with = "nan"


def load_saved_fitting_input_workspaces() -> None:
    fitting_inputs_dir = FilesManager.get_fitting_inputs_dir()
    if not fitting_inputs_dir.exists():
        logger.notice(f"No saved fitting input workspaces found in {fitting_inputs_dir}. Skipping workspace reload.")
        return

    workspace_files = sorted(fitting_inputs_dir.glob("*.nxs"))
    logger.notice(f"Loading saved fitting input workspaces from {fitting_inputs_dir} ({len(workspace_files)} file(s)).")
    for workspace_file in workspace_files:
        workspace_name = workspace_file.stem
        if AnalysisDataService.doesExist(workspace_name):
            logger.notice(f"Workspace {workspace_name} already loaded in ADS; reusing existing workspace.")
            continue
        logger.notice(f"Loading workspace {workspace_name} from {workspace_file.name}.")
        Load(Filename=str(workspace_file), OutputWorkspace=workspace_name)


def run_y_space_reduction_and_fit(fitting_inputs: type[BackwardFittingInputs] | type[ForwardFittingInputs]) -> bool:
    iteration = str(fitting_inputs.number_of_iterations_for_corrections)
    ws_name_candidates = [f"{fitting_inputs.name}_{iteration}", fitting_inputs.name]

    for candidate in ws_name_candidates:
        resolution_name = f"{candidate}_ws_resolution"
        lightest_data_name = f"{candidate}_ws_lighest_data"
        lightest_ncp_name = f"{candidate}_ws_lighest_ncp"
        if (
            AnalysisDataService.doesExist(resolution_name)
            and AnalysisDataService.doesExist(lightest_data_name)
            and AnalysisDataService.doesExist(lightest_ncp_name)
        ):
            fitting_directory = FilesManager.get_fitting_outputs_dir()
            fitting_directory.mkdir(parents=True, exist_ok=True)
            SaveAscii(resolution_name, str(fitting_directory / resolution_name))

            FitInYSpace(
                fitting_inputs,
                mtd[lightest_data_name],
                mtd[lightest_ncp_name],
                mtd[resolution_name],
                outputs_dir=fitting_directory,
            ).run()
            return True

    logger.warning(
        "Could not find expected derived workspaces "
        + ", ".join(f"({name}_ws_resolution, {name}_ws_lighest_data, {name}_ws_lighest_ncp)" for name in ws_name_candidates)
        + ". Skipping fitting in Y-Space."
    )
    return False


def run_fitting(back_ws_to_fit: str = "", front_ws_to_fit: str = "") -> bool:
    load_saved_fitting_input_workspaces()

    # Optional names let users fit already-loaded workspaces without re-running reduction.
    if back_ws_to_fit:
        BackwardFittingInputs.name = str(back_ws_to_fit)
    if front_ws_to_fit:
        ForwardFittingInputs.name = str(front_ws_to_fit)

    success = False
    if BackwardFittingInputs.run_this_fitting_type:
        success |= run_y_space_reduction_and_fit(BackwardFittingInputs)
    if ForwardFittingInputs.run_this_fitting_type:
        success |= run_y_space_reduction_and_fit(ForwardFittingInputs)
    return success


if (__name__ == "__main__") or (__name__ == "mantidqt.widgets.codeeditor.execution"):
    BACK_WS_TO_FIT = globals().get("BACK_WS_TO_FIT", "")
    FRONT_WS_TO_FIT = globals().get("FRONT_WS_TO_FIT", "")
    run_fitting(back_ws_to_fit=BACK_WS_TO_FIT, front_ws_to_fit=FRONT_WS_TO_FIT)

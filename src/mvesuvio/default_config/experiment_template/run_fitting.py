from typing import TYPE_CHECKING

from mantid.api import AnalysisDataService
from mantid.kernel import logger
from mantid.simpleapi import SaveAscii, mtd

from mvesuvio.analysis_fitting import FitInYSpace
from mvesuvio.default_config.experiment_template.fitting_models import register_fitting_models
from mvesuvio.globals import CORRECTED_WORKSPACE_SUFFIX, JOY_AVG_WORKSPACE_SUFFIX, JOY_WORKSPACE_SUFFIX, RESOLUTION_WORKSPACE_SUFFIX
from mvesuvio.util import general_helpers
from mvesuvio.util.files_manager import FilesManager
from mvesuvio.util.model_manager import model_manager

register_fitting_models()

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
    # Fitting model options
    # "GAUSS": Single Gaussian
    # "GAUSS_CNTR": Single Gaussian with fixed center at zero
    # "GCC4": Gram-Charlier with C4 parameter
    # "GCC4_CNTR": Gram-Charlier with C4 parameter with fixed center at zero
    # "GCC6": Gram-Charlier with C6 parameter
    # "GCC6_CNTR": Gram-Charlier with C6 parameter with fixed center at zero
    # "GCC4C6": Gram-Charlier with C4 and C6 parameter
    # "GCC4C6_CNTR": Gram-Charlier with C4 and C6 parameter and fixed center at zero
    # "DOUBLEWELL": Double Well function
    # "GAUSS2D": Anisotropic Gaussian
    # "GAUSS3D": 3-Dimensional Gaussian
    fitting_model = model_manager.name("GAUSS")
    run_minos = True
    do_global_fit = True  # Performs global fit with Minuit by default
    # Number of groups of detectors to perform global (simultaneous) fit on
    # Either an integer less than the number of detectors
    # or option 'all', which does not form groups and fits all spectra simultaneously and individualy
    number_of_global_fit_groups = 4


class ForwardFittingInputs(ForwardAnalysisInputs):
    run_this_fitting_type = True
    show_plots = True
    # Fitting model options
    # "GAUSS": Single Gaussian
    # "GAUSS_CNTR": Single Gaussian with fixed center at zero
    # "GCC4": Gram-Charlier with C4 parameter
    # "GCC4_CNTR": Gram-Charlier with C4 parameter with fixed center at zero
    # "GCC6": Gram-Charlier with C6 parameter
    # "GCC6_CNTR": Gram-Charlier with C6 parameter with fixed center at zero
    # "GCC4C6": Gram-Charlier with C4 and C6 parameter
    # "GCC4C6_CNTR": Gram-Charlier with C4 and C6 parameter and fixed center at zero
    # "DOUBLEWELL": Double Well function
    # "GAUSS2D": Anisotropic Gaussian
    # "GAUSS3D": 3-Dimensional Gaussian
    fitting_model = model_manager.name("GAUSS")
    run_minos = True
    do_global_fit = True  # Performs global fit with Minuit by default
    # Number of groups of detectors to perform global (simultaneous) fit on
    # Either an integer less than the number of detectors
    # or option 'all', which does not form groups and fits all spectra simultaneously and individualy
    number_of_global_fit_groups = 4


def load_saved_fitting_input_workspaces() -> set[str]:
    return general_helpers.load_saved_fitting_input_workspaces(
        FilesManager.get_fse_correction_outputs_dir(), (RESOLUTION_WORKSPACE_SUFFIX, JOY_WORKSPACE_SUFFIX, JOY_AVG_WORKSPACE_SUFFIX)
    )


def run_y_space_reduction_and_fit(
    fitting_inputs: type[BackwardFittingInputs] | type[ForwardFittingInputs],
    available_workspace_names: set[str] | None = None,
) -> bool:
    ws_name_candidates = general_helpers.get_workspace_name_candidates(fitting_inputs)

    for candidate in ws_name_candidates:
        candidate += CORRECTED_WORKSPACE_SUFFIX
        resolution_name = candidate + RESOLUTION_WORKSPACE_SUFFIX
        joy_name = candidate + JOY_WORKSPACE_SUFFIX
        joy_avg_name = candidate + JOY_AVG_WORKSPACE_SUFFIX
        names = (resolution_name, joy_name, joy_avg_name)
        if (available_workspace_names is None or all(name in available_workspace_names for name in names)) and all(
            AnalysisDataService.doesExist(name) for name in names
        ):
            fitting_directory = FilesManager.get_fitting_outputs_dir()
            fitting_directory.mkdir(parents=True, exist_ok=True)
            SaveAscii(resolution_name, str(fitting_directory / resolution_name))

            FitInYSpace(
                fitting_inputs,
                mtd[joy_name],
                mtd[joy_avg_name],
                mtd[resolution_name],
                outputs_dir=fitting_directory,
            ).run()
            return True

    logger.warning(
        "Could not find expected derived workspaces "
        + ", ".join(
            f"({name}{CORRECTED_WORKSPACE_SUFFIX}{RESOLUTION_WORKSPACE_SUFFIX}, "
            f"{name}{CORRECTED_WORKSPACE_SUFFIX}{JOY_WORKSPACE_SUFFIX}, "
            f"{name}{CORRECTED_WORKSPACE_SUFFIX}{JOY_AVG_WORKSPACE_SUFFIX})"
            for name in ws_name_candidates
        )
        + ". Skipping fitting in Y-Space."
    )
    return False


def run_fitting() -> None:
    available_workspace_names = load_saved_fitting_input_workspaces()
    if BackwardFittingInputs.run_this_fitting_type:
        run_y_space_reduction_and_fit(BackwardFittingInputs, available_workspace_names)
    if ForwardFittingInputs.run_this_fitting_type:
        run_y_space_reduction_and_fit(ForwardFittingInputs, available_workspace_names)
    general_helpers.make_summarised_log_file()


if (__name__ == "__main__") or (__name__ == "mantidqt.widgets.codeeditor.execution"):
    run_fitting()

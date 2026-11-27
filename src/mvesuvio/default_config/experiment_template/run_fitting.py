from typing import TYPE_CHECKING

from mantid.api import AnalysisDataService
from mantid.kernel import logger
from mantid.simpleapi import SaveAscii, mtd

from mvesuvio.analysis_fitting import FitInYSpace
from mvesuvio.globals import CORRECTED_WORKSPACE_SUFFIX, JOY_AVG_WORKSPACE_SUFFIX, JOY_WORKSPACE_SUFFIX, RESOLUTION_WORKSPACE_SUFFIX
from mvesuvio.util import general_helpers
from mvesuvio.util.files_manager import FilesManager

if TYPE_CHECKING or __package__:
    from .fitting_models import FittingModel, register_fitting_models
    from .run_reduction import BackwardAnalysisInputs, ForwardAnalysisInputs
else:
    _fitting_models, _run_reduction = general_helpers.load_experiment_modules(__file__)
    FittingModel = _fitting_models.FittingModel
    register_fitting_models = _fitting_models.register_fitting_models
    BackwardAnalysisInputs = _run_reduction.BackwardAnalysisInputs
    ForwardAnalysisInputs = _run_reduction.ForwardAnalysisInputs


register_fitting_models()


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
    fitting_model = FittingModel.GAUSS
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
    fitting_model = FittingModel.GAUSS
    run_minos = True
    do_global_fit = True  # Performs global fit with Minuit by default
    # Number of groups of detectors to perform global (simultaneous) fit on
    # Either an integer less than the number of detectors
    # or option 'all', which does not form groups and fits all spectra simultaneously and individualy
    number_of_global_fit_groups = 4


def run_y_space_reduction_and_fit(
    fitting_inputs: type[BackwardFittingInputs] | type[ForwardFittingInputs],
) -> bool:
    ws_name_candidates = general_helpers.get_workspace_name_candidates(fitting_inputs)
    candidate_names = []
    for base_candidate in ws_name_candidates:
        candidate = base_candidate + CORRECTED_WORKSPACE_SUFFIX
        candidate_names.append(
            (
                candidate,
                (
                    candidate + RESOLUTION_WORKSPACE_SUFFIX,
                    candidate + JOY_WORKSPACE_SUFFIX,
                    candidate + JOY_AVG_WORKSPACE_SUFFIX,
                ),
            )
        )

    if not any(all(AnalysisDataService.doesExist(name) for name in names) for _, names in candidate_names):
        output_directory = FilesManager.get_fse_correction_outputs_dir()
        for base_candidate, (_, names) in zip(ws_name_candidates, candidate_names):
            candidate_directory = output_directory / base_candidate
            if general_helpers.load_saved_fitting_input_workspaces(candidate_directory, required_workspace_names=names):
                break

    for candidate, names in candidate_names:
        resolution_name, joy_name, joy_avg_name = names
        if all(AnalysisDataService.doesExist(name) for name in names):
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
    if BackwardFittingInputs.run_this_fitting_type:
        run_y_space_reduction_and_fit(BackwardFittingInputs)
    if ForwardFittingInputs.run_this_fitting_type:
        run_y_space_reduction_and_fit(ForwardFittingInputs)
    general_helpers.make_summarised_log_file()


if (__name__ == "__main__") or (__name__ == "mantidqt.widgets.codeeditor.execution"):
    run_fitting()

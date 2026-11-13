from typing import TYPE_CHECKING

import numpy as np
from mantid.api import AnalysisDataService, Workspace
from mantid.kernel import logger
from mantid.simpleapi import mtd
from scipy.optimize import NonlinearConstraint, OptimizeResult, minimize

from mvesuvio.default_config.experiment_template.fitting_models import register_fitting_models
from mvesuvio.util import fitting_helpers, general_helpers
from mvesuvio.util.files_manager import FilesManager
from mvesuvio.util.fse_correction_helpers import WORKSPACE_SUFFIX, FSEFitWith, save_fse_fit_results, subtract_fse_contribution
from mvesuvio.util.general_helpers import extractWS
from mvesuvio.util.global_fit_builder import FSE_INTENSITY_PARAMETER, GlobalFitWithFSEBuilder
from mvesuvio.util.model_manager import model_manager

register_fitting_models()


if TYPE_CHECKING:
    from .run_reduction import BackwardAnalysisInputs, ForwardAnalysisInputs
else:
    try:
        from .run_reduction import BackwardAnalysisInputs, ForwardAnalysisInputs
    except ImportError:
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


class BackwardFSEFittingInputs(BackwardAnalysisInputs):
    run_this_fitting_type = False
    fitting_model = model_manager.name("GAUSS")
    fit_fse_with = FSEFitWith.FREE_SHARED
    initial_fse_width = 5.0
    tie_fse_width_to_model_param = "sigma"


class ForwardFSEFittingInputs(ForwardAnalysisInputs):
    run_this_fitting_type = True
    fitting_model = model_manager.name("GAUSS")
    fit_fse_with = FSEFitWith.TIED_TO_MODEL
    initial_fse_width = 3.0
    tie_fse_width_to_model_param = "sigma"


def run_fse_fitting(
    ws_resolution: Workspace,
    ws_joy: Workspace,
    ws_q: Workspace,
    fitting_inputs: type[BackwardFSEFittingInputs] | type[ForwardFSEFittingInputs],
) -> tuple[GlobalFitWithFSEBuilder, OptimizeResult]:
    data_x, data_y, data_e = extractWS(ws_joy)
    _, res_y, _ = extractWS(ws_resolution)
    q = ws_q.extractY()

    definition = model_manager.get_model(fitting_inputs.fitting_model)
    initial_parameters = dict(definition.default_parameters)
    shared_parameters = tuple(definition.shared_parameters)
    fit_fse_with = FSEFitWith(fitting_inputs.fit_fse_with)
    model_width_parameter = None
    if fit_fse_with in (FSEFitWith.FREE_SHARED, FSEFitWith.FREE_UNSHARED):
        initial_parameters[FSE_INTENSITY_PARAMETER] = fitting_inputs.initial_fse_width**4 / 3
        if fit_fse_with is FSEFitWith.FREE_SHARED:
            shared_parameters += (FSE_INTENSITY_PARAMETER,)
    elif fit_fse_with is FSEFitWith.TIED_TO_MODEL:
        definition.parameter(fitting_inputs.tie_fse_width_to_model_param)
        model_width_parameter = fitting_inputs.tie_fse_width_to_model_param

    builder = GlobalFitWithFSEBuilder(
        data_x,
        data_y,
        data_e,
        res_y,
        definition.function,
        initial_parameters,
        shared_parameters,
        q=q,
        model_width_parameter=model_width_parameter,
    )

    result = minimize(
        builder.cost_function,
        builder.initial_vector(),
        method="SLSQP",
        constraints=[NonlinearConstraint(builder.constraint_function, 0, np.inf)],
        bounds=builder.optimizer_bounds(),
        options={"maxiter": 2000, "ftol": 1e-10, "disp": True},
    )
    covariance = builder.parameter_covariance(result.x)
    degrees_of_freedom = max(int(np.count_nonzero(builder.valid)) - result.x.size, 1)
    reduced_chi2 = builder.cost_function(result.x) / degrees_of_freedom
    parameter_names, parameter_values, parameter_errors = builder.reporting_parameters(result.x, covariance)
    fitting_helpers.create_table_for_fse_fit_parameters(
        ws_joy.name() + f"_{WORKSPACE_SUFFIX[fit_fse_with]}",
        fitting_inputs.fitting_model,
        parameter_names,
        parameter_values,
        parameter_errors,
        reduced_chi2,
    )
    _, _, ws_fse = save_fse_fit_results(ws_joy, builder, result, fit_fse_with, covariance)
    subtract_fse_contribution(ws_joy, ws_fse, fit_fse_with)
    return builder, result


def run_fse_correction_for_inputs(
    fitting_inputs: type[BackwardFSEFittingInputs] | type[ForwardFSEFittingInputs],
) -> None:
    workspace_name_candidates = general_helpers.get_workspace_name_candidates(fitting_inputs)
    found_candidate = False
    for candidate in workspace_name_candidates:
        resolution_name = f"{candidate}_ws_resolution"
        joy_name = f"{candidate}_joy"
        q_name = f"{candidate}_q"
        if all(AnalysisDataService.doesExist(name) for name in (resolution_name, joy_name, q_name)):
            found_candidate = True
            run_fse_fitting(mtd[resolution_name], mtd[joy_name], mtd[q_name], fitting_inputs)

    if not found_candidate:
        logger.warning(
            "Could not find expected derived workspaces "
            + ", ".join(f"({name}_ws_resolution, {name}_joy, {name}_q)" for name in workspace_name_candidates)
            + ". Skipping FSE correction."
        )


def run_fse_correction() -> None:
    general_helpers.load_saved_fitting_input_workspaces(FilesManager.get_fse_correction_inputs_dir())
    for fitting_inputs in (BackwardFSEFittingInputs, ForwardFSEFittingInputs):
        if fitting_inputs.run_this_fitting_type:
            run_fse_correction_for_inputs(fitting_inputs)
    general_helpers.make_summarised_log_file()


if (__name__ == "__main__") or (__name__ == "mantidqt.widgets.codeeditor.execution"):
    run_fse_correction()

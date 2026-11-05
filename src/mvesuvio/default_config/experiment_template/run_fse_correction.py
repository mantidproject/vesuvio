from typing import TYPE_CHECKING

import numpy as np
from mantid.api import AnalysisDataService, Workspace
from mantid.kernel import logger
from mantid.simpleapi import CloneWorkspace, SumSpectra, mtd
from scipy.optimize import NonlinearConstraint, OptimizeResult, minimize

from mvesuvio.default_config.experiment_template.fitting_models import register_fitting_models
from mvesuvio.util import general_helpers
from mvesuvio.util.files_manager import FilesManager
from mvesuvio.util.general_helpers import extractWS
from mvesuvio.util.global_fit_builder import GlobalFitBuilder
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


class ForwardFSEFittingInputs(ForwardAnalysisInputs):
    run_this_fitting_type = True
    fitting_model = model_manager.name("GAUSS")


def save_fse_fit_results(ws_joy: Workspace, builder: GlobalFitBuilder, result: OptimizeResult) -> tuple[Workspace, Workspace]:
    """Store the fitted curves (with errors) and the residuals (without errors) in clones of ws_joy."""
    fit_y = builder.convolved_model(result.x)
    fit_e = builder.fit_curve_errors(result.x)
    # Masked points were not fitted, so they carry no residual
    residuals = np.where(builder.valid, builder.data_y - fit_y, 0.0)

    ws_fit = CloneWorkspace(ws_joy, OutputWorkspace=ws_joy.name() + "_fse_fit")
    general_helpers.pass_data_into_ws(builder.data_x, fit_y, fit_e, ws_fit)

    ws_residuals = CloneWorkspace(ws_joy, OutputWorkspace=ws_joy.name() + "_fse_residuals")
    general_helpers.pass_data_into_ws(builder.data_x, residuals, np.zeros_like(residuals), ws_residuals)

    SumSpectra(InputWorkspace=ws_joy, OutputWorkspace=ws_joy.name() + "_sum")
    SumSpectra(InputWorkspace=ws_fit, OutputWorkspace=ws_fit.name() + "_sum")
    SumSpectra(InputWorkspace=ws_residuals, OutputWorkspace=ws_residuals.name() + "_sum")
    return ws_fit, ws_residuals


def run_fse_fitting(
    ws_resolution: Workspace,
    ws_joy: Workspace,
    fitting_inputs: type[BackwardFSEFittingInputs] | type[ForwardFSEFittingInputs],
) -> tuple[GlobalFitBuilder, OptimizeResult]:
    data_x, data_y, data_e = extractWS(ws_joy)
    _, res_y, _ = extractWS(ws_resolution)

    definition = model_manager.get_model(fitting_inputs.fitting_model)
    builder = GlobalFitBuilder(
        data_x,
        data_y,
        data_e,
        res_y,
        definition.function,
        definition.default_parameters,
        definition.shared_parameters,
    )

    result = minimize(
        builder.cost_function,
        builder.initial_vector(),
        method="SLSQP",
        constraints=[NonlinearConstraint(builder.constraint_function, 0, np.inf)],
        options={"maxiter": 2000, "ftol": 1e-10, "disp": True},
    )
    save_fse_fit_results(ws_joy, builder, result)
    return builder, result


def run_fse_correction_for_inputs(
    fitting_inputs: type[BackwardFSEFittingInputs] | type[ForwardFSEFittingInputs],
) -> None:
    workspace_name_candidates = general_helpers.get_workspace_name_candidates(fitting_inputs)
    found_candidate = False
    for candidate in workspace_name_candidates:
        resolution_name = f"{candidate}_ws_resolution"
        joy_name = f"{candidate}_joy"
        if AnalysisDataService.doesExist(resolution_name) and AnalysisDataService.doesExist(joy_name):
            found_candidate = True
            run_fse_fitting(mtd[resolution_name], mtd[joy_name], fitting_inputs)

    if not found_candidate:
        logger.warning(
            "Could not find expected derived workspaces "
            + ", ".join(f"({name}_ws_resolution, {name}_joy)" for name in workspace_name_candidates)
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

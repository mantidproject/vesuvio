import numpy as np
from mantid.api import Workspace
from mantid.simpleapi import CloneWorkspace, Minus, SumSpectra
from scipy.optimize import OptimizeResult

from mvesuvio.globals import (
    FSE_CONTRIBUTION_WORKSPACE_SUFFIX,
    FSE_FIT_WORKSPACE_SUFFIX,
    FSE_RESIDUALS_WORKSPACE_SUFFIX,
    SUM_WORKSPACE_SUFFIX,
    WORKSPACE_FSE_SUFFIX,
    FSEFitWith,
)
from mvesuvio.util import general_helpers
from mvesuvio.util.global_fit_builder import GlobalFitWithFSEBuilder


def save_fse_fit_results(
    ws_joy: Workspace,
    builder: GlobalFitWithFSEBuilder,
    result: OptimizeResult,
    fit_fse_with: FSEFitWith,
    covariance: np.ndarray,
) -> tuple[Workspace, Workspace, Workspace]:
    """Store the fitted curves, the FSE contribution (both with errors) and the residuals (without errors) in clones of ws_joy."""
    workspace_suffix = WORKSPACE_FSE_SUFFIX[fit_fse_with]
    fit_y = builder.convolved_model(result.x)
    fit_e = builder.fit_curve_errors(result.x, covariance)
    # Masked points were not fitted, so they carry no residual
    residuals = np.where(builder.valid, builder.data_y - fit_y, 0.0)

    ws_fit = CloneWorkspace(ws_joy, OutputWorkspace=ws_joy.name() + f"_{workspace_suffix}{FSE_FIT_WORKSPACE_SUFFIX}")
    general_helpers.pass_data_into_ws(builder.data_x, fit_y, fit_e, ws_fit)

    ws_residuals = CloneWorkspace(ws_joy, OutputWorkspace=ws_joy.name() + f"_{workspace_suffix}{FSE_RESIDUALS_WORKSPACE_SUFFIX}")
    general_helpers.pass_data_into_ws(builder.data_x, residuals, np.zeros_like(residuals), ws_residuals)

    ws_fse = CloneWorkspace(ws_joy, OutputWorkspace=ws_joy.name() + f"_{workspace_suffix}{FSE_CONTRIBUTION_WORKSPACE_SUFFIX}")
    fse_y = builder.fse_contribution(result.x)
    fse_e = builder.fse_contribution_errors(result.x, covariance)
    general_helpers.pass_data_into_ws(builder.data_x, fse_y, fse_e, ws_fse)

    SumSpectra(InputWorkspace=ws_joy, OutputWorkspace=ws_joy.name() + SUM_WORKSPACE_SUFFIX)
    SumSpectra(InputWorkspace=ws_fit, OutputWorkspace=ws_fit.name() + SUM_WORKSPACE_SUFFIX)
    SumSpectra(InputWorkspace=ws_residuals, OutputWorkspace=ws_residuals.name() + SUM_WORKSPACE_SUFFIX)
    SumSpectra(InputWorkspace=ws_fse, OutputWorkspace=ws_fse.name() + SUM_WORKSPACE_SUFFIX)
    return ws_fit, ws_residuals, ws_fse


def subtract_fse_contribution(ws_joy: Workspace, ws_fse: Workspace, output_workspace: str) -> Workspace:
    """Subtract the fitted FSE contribution from the data, errors are added in quadrature."""
    ws_corrected = Minus(
        LHSWorkspace=ws_joy,
        RHSWorkspace=ws_fse,
        OutputWorkspace=output_workspace,
    )
    SumSpectra(InputWorkspace=ws_corrected, OutputWorkspace=ws_corrected.name() + SUM_WORKSPACE_SUFFIX)
    return ws_corrected

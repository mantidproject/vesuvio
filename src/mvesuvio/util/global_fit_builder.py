from collections.abc import Callable, Iterable, Mapping
from typing import Any

import jacobi
import numpy as np
from scipy import signal

from mvesuvio.util.analysis_helpers import numerical_third_derivative

BACKGROUND_PARAMETER = "y0"
FSE_PARAMETER = "fse_intensity"
# The finite difference stencil of the third derivative is not defined on this many points at each edge
THIRD_DERIVATIVE_EDGE = 6


class GlobalFitBuilder:
    """Builds the cost function of a simultaneous fit over all rows (detectors) of the data.

    Parameters in ``shared_params`` take a single value common to all rows, the remaining
    parameters (plus a ``y0`` constant background) are fitted independently for each row.
    The parameter vector is laid out as the shared parameters, followed by, for each unshared
    parameter, one value per row. The model is convolved with the resolution of each row before
    being compared to the data.

    If ``fse_intensity`` is in ``initial_params`` it is not passed to the model. Instead, the third derivative
    of the model is added to it as a final state effect (FSE) term, ``-fse_intensity * d3 model / dx3``, before
    the convolution. Like any other parameter, ``fse_intensity`` is shared when listed in ``shared_params``.
    """

    def __init__(
        self,
        data_x: np.ndarray,
        data_y: np.ndarray,
        data_e: np.ndarray,
        res_y: np.ndarray,
        model: Callable[..., Any],
        initial_params: Mapping[str, float],
        shared_params: Iterable[str] = (),
    ) -> None:
        self.data_x = np.asarray(data_x, dtype=float)
        self.data_y = np.asarray(data_y, dtype=float)
        self.data_e = np.asarray(data_e, dtype=float)
        self.res_y = np.asarray(res_y, dtype=float)
        self.model = model
        self.initial_params = dict(initial_params)
        self.shared_params = tuple(shared_params)

        self._validate_inputs()

        self.has_fse = FSE_PARAMETER in self.initial_params
        self.n_spectra = self.data_x.shape[0]
        self.unshared_params = (BACKGROUND_PARAMETER,) + tuple(name for name in self.initial_params if name not in self.shared_params)
        # Points with zero or non finite error (e.g. masked bins where y and e are zero) are ignored
        self.valid = np.isfinite(self.data_y) & np.isfinite(self.data_e) & (self.data_e != 0)
        grids = [self._build_convolution_grid(x, res) for x, res in zip(self.data_x, self.res_y)]
        self._x_delta = np.array([delta for delta, _ in grids])[:, np.newaxis]
        self._res_dense = np.stack([res for _, res in grids])
        self._safe_e = np.where(self.valid, self.data_e, 1.0)
        self._safe_y = np.where(self.valid, self.data_y, 0.0)

    def _validate_inputs(self) -> None:
        if self.data_x.ndim != 2:
            raise ValueError("data_x, data_y, data_e and res_y must be 2D arrays with one row per detector.")
        if not (self.data_x.shape == self.data_y.shape == self.data_e.shape == self.res_y.shape):
            raise ValueError("data_x, data_y, data_e and res_y must all have the same shape.")
        missing = [name for name in self.shared_params if name not in self.initial_params]
        if missing:
            raise ValueError(f"Shared parameters are not in the initial parameters: {', '.join(missing)}")
        if BACKGROUND_PARAMETER in self.initial_params:
            raise ValueError(f"{BACKGROUND_PARAMETER!r} is reserved for the background of the convolved model.")
        if FSE_PARAMETER in self.initial_params and self.data_x.shape[1] <= 2 * THIRD_DERIVATIVE_EDGE:
            raise ValueError("Not enough points to compute the third derivative for the FSE term.")

    @staticmethod
    def _build_convolution_grid(x: np.ndarray, res: np.ndarray) -> tuple[float, np.ndarray]:
        """Return the step and the resolution sampled on an odd, symmetric grid so its peak is at the center."""
        if not np.isclose(np.min(x), -np.max(x)):
            raise ValueError("The resolution needs to be defined on a symmetric x range.")
        n_points = x.size if x.size % 2 else x.size + 1
        x_dense = np.linspace(np.min(x), np.max(x), n_points)
        return x_dense[1] - x_dense[0], np.interp(x_dense, x, res)

    @property
    def parameter_names(self) -> list[str]:
        """Names of the entries of the parameter vector, unshared ones are suffixed with the row index."""
        names = list(self.shared_params)
        for name in self.unshared_params:
            names.extend(f"{name}{i}" for i in range(self.n_spectra))
        return names

    def initial_vector(self) -> np.ndarray:
        """Starting parameter vector for the minimizer."""
        shared = [self.initial_params[name] for name in self.shared_params]
        unshared = []
        for name in self.unshared_params:
            default = self.initial_params.get(name, 0.0)
            unshared.extend([default] * self.n_spectra)
        return np.array(shared + unshared, dtype=float)

    def unpack(self, vector: np.ndarray) -> dict[str, Any]:
        """Split the parameter vector into shared scalars and unshared arrays with one value per row (y0 included)."""
        vector = np.asarray(vector, dtype=float)
        expected = len(self.shared_params) + len(self.unshared_params) * self.n_spectra
        if vector.size != expected:
            raise ValueError(f"Expected {expected} parameters, got {vector.size}.")

        n_shared = len(self.shared_params)
        params = dict(zip(self.shared_params, vector[:n_shared]))
        unshared = vector[n_shared:].reshape(len(self.unshared_params), self.n_spectra)
        params.update(zip(self.unshared_params, unshared))
        return params

    def _evaluate_rows(self, params: dict[str, Any]) -> tuple[np.ndarray, np.ndarray | None]:
        """Model and FSE term (None without FSE) of every row, before the convolution."""
        params = dict(params)
        params.pop(BACKGROUND_PARAMETER)
        fse_intensity = params.pop(FSE_PARAMETER, None)
        # The model is evaluated per row because not all models accept array parameters
        model_y = np.stack(
            [
                self.model(x, **{name: value[i] if np.ndim(value) else value for name, value in params.items()})
                for i, x in enumerate(self.data_x)
            ]
        )
        if fse_intensity is None:
            return model_y, None

        # The edges are left at zero, where the third derivative is not defined
        third_derivative = np.zeros_like(model_y)
        edge = THIRD_DERIVATIVE_EDGE
        third_derivative[:, edge:-edge] = numerical_third_derivative(self.data_x, model_y)
        intensity = np.reshape(fse_intensity, (-1, 1)) if np.ndim(fse_intensity) else fse_intensity
        return model_y, -intensity * third_derivative

    def _convolve(self, model_y: np.ndarray) -> np.ndarray:
        return signal.fftconvolve(model_y, self._res_dense, mode="same", axes=1) * self._x_delta

    def convolved_model(self, vector: np.ndarray) -> np.ndarray:
        """Model (plus FSE term) of every row convolved with its resolution, as an array shaped like data_y."""
        params = self.unpack(vector)
        model_y, fse_y = self._evaluate_rows(params)
        if fse_y is not None:
            model_y = model_y + fse_y
        return params[BACKGROUND_PARAMETER][:, np.newaxis] + self._convolve(model_y)

    def fse_contribution(self, vector: np.ndarray) -> np.ndarray:
        """Convolved FSE term of every row, as an array shaped like data_y."""
        if not self.has_fse:
            raise ValueError(f"The fit has no FSE term, add {FSE_PARAMETER!r} to the initial parameters.")
        _, fse_y = self._evaluate_rows(self.unpack(vector))
        return self._convolve(fse_y)

    def row_costs(self, vector: np.ndarray) -> np.ndarray:
        """Least squares cost of each row, ignoring the masked points."""
        residuals = (self._safe_y - self.convolved_model(vector)) / self._safe_e
        return np.sum(np.where(self.valid, residuals, 0.0) ** 2, axis=1)

    def weighted_residuals(self, vector: np.ndarray) -> np.ndarray:
        """Residuals divided by the errors of all valid points, concatenated into a 1D array."""
        return ((self._safe_y - self.convolved_model(vector)) / self._safe_e)[self.valid]

    def cost_function(self, vector: np.ndarray) -> float:
        """Total cost: the sum of the cost of each row given the shared and unshared parameters in the vector."""
        return float(np.sum(self.row_costs(vector)))

    def constraint_function(self, vector: np.ndarray) -> np.ndarray:
        """Fitted y values of all rows over their full x range, concatenated into a 1D array."""
        return self.convolved_model(vector).ravel()

    def parameter_covariance(self, parameters: np.ndarray) -> np.ndarray:
        """Covariance of the fitted parameters, estimated from the Jacobian J of the weighted residuals.

        Computed as (J^T J)^-1 scaled by the reduced chi2.
        """
        parameters = np.asarray(parameters, dtype=float)
        jacobian_residuals, _ = jacobi.jacobi(self.weighted_residuals, parameters)
        n_dof = max(jacobian_residuals.shape[0] - parameters.size, 1)
        reduced_chi2 = np.sum(self.weighted_residuals(parameters) ** 2) / n_dof
        return np.linalg.pinv(jacobian_residuals.T @ jacobian_residuals) * reduced_chi2

    def curve_errors(
        self,
        curve: Callable[[np.ndarray], np.ndarray],
        parameters: np.ndarray,
        covariance: np.ndarray | None = None,
    ) -> np.ndarray:
        """Propagate the uncertainty of the fitted parameters onto curve(parameters), shaped like data_y.

        Only the diagonal of the propagated covariance, J_curve C J_curve^T, is computed, because the full
        matrix over all detectors and bins would be too large. Pass the covariance to avoid recomputing it.
        """
        parameters = np.asarray(parameters, dtype=float)
        if covariance is None:
            covariance = self.parameter_covariance(parameters)
        jacobian_curve, _ = jacobi.jacobi(lambda vector: curve(vector).ravel(), parameters)
        variance = np.einsum("ij,jk,ik->i", jacobian_curve, covariance, jacobian_curve)
        return np.sqrt(np.clip(variance, 0, None)).reshape(self.data_y.shape)

    def fit_curve_errors(self, parameters: np.ndarray, covariance: np.ndarray | None = None) -> np.ndarray:
        """Uncertainty of the fitted y values, shaped like data_y."""
        return self.curve_errors(self.convolved_model, parameters, covariance)

    def fse_contribution_errors(self, parameters: np.ndarray, covariance: np.ndarray | None = None) -> np.ndarray:
        """Uncertainty of the convolved FSE term, shaped like data_y."""
        return self.curve_errors(self.fse_contribution, parameters, covariance)

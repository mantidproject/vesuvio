from collections.abc import Callable, Iterable, Mapping
from typing import Any

import jacobi
import numpy as np
from scipy import signal

BACKGROUND_PARAMETER = "y0"


class GlobalFitBuilder:
    """Builds the cost function of a simultaneous fit over all rows (detectors) of the data.

    Parameters in ``shared_params`` take a single value common to all rows, the remaining
    parameters (plus a ``y0`` constant background) are fitted independently for each row.
    The parameter vector is laid out as the shared parameters, followed by, for each unshared
    parameter, one value per row. The model is convolved with the resolution of each row before
    being compared to the data.
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

    def convolved_model(self, vector: np.ndarray) -> np.ndarray:
        """Model of every row convolved with its resolution, as an array shaped like data_y."""
        params = self.unpack(vector)
        background = params.pop(BACKGROUND_PARAMETER)
        # The model is evaluated per row because not all models accept array parameters
        model_y = np.stack(
            [
                self.model(x, **{name: value[i] if np.ndim(value) else value for name, value in params.items()})
                for i, x in enumerate(self.data_x)
            ]
        )
        convolved = signal.fftconvolve(model_y, self._res_dense, mode="same", axes=1)
        return background[:, np.newaxis] + convolved * self._x_delta

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

    def fit_curve_errors(self, parameters: np.ndarray) -> np.ndarray:
        """Propagate the uncertainty of the fitted parameters onto the fitted y values, shaped like data_y.

        The covariance of the parameters is estimated from the Jacobian J of the weighted residuals as
        (J^T J)^-1 scaled by the reduced chi2. Only the diagonal of the propagated covariance, J_fit C J_fit^T,
        is computed, because the full matrix over all detectors and bins would be too large.
        """
        parameters = np.asarray(parameters, dtype=float)
        jacobian_residuals, _ = jacobi.jacobi(self.weighted_residuals, parameters)
        n_dof = max(jacobian_residuals.shape[0] - parameters.size, 1)
        reduced_chi2 = np.sum(self.weighted_residuals(parameters) ** 2) / n_dof
        covariance = np.linalg.pinv(jacobian_residuals.T @ jacobian_residuals) * reduced_chi2

        jacobian_fit, _ = jacobi.jacobi(self.constraint_function, parameters)
        variance = np.einsum("ij,jk,ik->i", jacobian_fit, covariance, jacobian_fit)
        return np.sqrt(np.clip(variance, 0, None)).reshape(self.data_y.shape)

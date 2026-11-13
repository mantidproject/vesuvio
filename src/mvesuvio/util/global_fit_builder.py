from collections.abc import Callable, Iterable, Mapping
from typing import Any

import jacobi
import numpy as np
from scipy import fft

from mvesuvio.util.analysis_helpers import numerical_third_derivative

BACKGROUND_PARAMETER = "y"
FSE_WIDTH_PARAMETER = "fse_width"
FSE_INTENSITY_PARAMETER = "fse_intensity"
# The finite difference stencil of the third derivative is not defined on this many points at each edge
THIRD_DERIVATIVE_EDGE = 6


class GlobalFitBuilder:
    """Builds the cost function of a simultaneous fit over all rows (detectors) of the data.

    Parameters in ``shared_params`` take a single value common to all rows, the remaining
    parameters (plus a ``y`` constant background) are fitted independently for each row.
    The parameter vector is laid out as the shared parameters, followed by, for each unshared
    parameter, one value per row. The model is convolved with the resolution of each row before
    being compared to the data. ``VECTORISED`` selects models that accept 2D x arrays and
    row-wise parameters shaped as ``(n_rows, 1)``.

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
        VECTORISED: bool = False,
    ) -> None:
        self.data_x = np.asarray(data_x, dtype=float)
        self.data_y = np.asarray(data_y, dtype=float)
        self.data_e = np.asarray(data_e, dtype=float)
        self.res_y = np.asarray(res_y, dtype=float)
        self.model = model
        if not isinstance(VECTORISED, bool):
            raise TypeError("VECTORISED must be a bool.")
        self.VECTORISED = VECTORISED
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
        self._convolution_full_length = self.data_x.shape[1] + self._res_dense.shape[1] - 1
        self._convolution_fft_length = fft.next_fast_len(self._convolution_full_length, real=True)
        self._convolution_same_start = (self._res_dense.shape[1] - 1) // 2
        self._res_fft = fft.rfft(self._res_dense, n=self._convolution_fft_length, axis=1)
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

    def reporting_parameters(self, params_vector: np.ndarray, covariance: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Return parameter names, values, and errors."""
        params_vector = np.asarray(params_vector, dtype=float)
        covariance = np.asarray(covariance, dtype=float)
        errors = np.sqrt(np.clip(np.diag(covariance), 0, None))
        return self.parameter_names, params_vector.copy(), errors

    def optimizer_bounds(self) -> list[tuple[float | None, float | None]]:
        """Return unbounded limits for all parameters."""
        return [(None, None) for _ in self.parameter_names]

    def initial_vector(self) -> np.ndarray:
        """Starting parameter vector for the minimizer."""
        shared = [self.initial_params[name] for name in self.shared_params]
        unshared = []
        for name in self.unshared_params:
            default = self.initial_params.get(name, 0.0)
            unshared.extend([default] * self.n_spectra)
        return np.array(shared + unshared, dtype=float)

    def unpack(self, params_vector: np.ndarray) -> dict[str, Any]:
        """Split the parameter vector into shared scalars and unshared arrays with one value per row (y included)."""
        # An existing float64 ndarray is returned unchanged; other inputs are converted.
        params_vector = np.asarray(params_vector, dtype=float)
        expected = len(self.shared_params) + len(self.unshared_params) * self.n_spectra
        if params_vector.size != expected:
            raise ValueError(f"Expected {expected} parameters, got {params_vector.size}.")

        n_shared = len(self.shared_params)
        params_dict = dict(zip(self.shared_params, params_vector[:n_shared]))
        unshared = params_vector[n_shared:].reshape(len(self.unshared_params), self.n_spectra)
        params_dict.update(zip(self.unshared_params, unshared))
        return params_dict

    def _evaluate_rows(self, params_dict: dict[str, Any]) -> np.ndarray:
        """Evaluate the model for every row, before convolution."""
        # Background is added after convolution; keep it in params_dict for the caller.
        model_params = {name: value for name, value in params_dict.items() if name != BACKGROUND_PARAMETER}
        if self.VECTORISED:
            row_parameters = {name: np.asarray(value)[:, np.newaxis] if np.ndim(value) else value for name, value in model_params.items()}
            return self.model(self.data_x, **row_parameters)

        return np.stack(
            [
                self.model(
                    x,
                    **{name: value[i] if np.ndim(value) else value for name, value in model_params.items()},
                )
                for i, x in enumerate(self.data_x)
            ]
        )

    def _convolve(self, model_y: np.ndarray) -> np.ndarray:
        # Reuse the fixed resolution FFT, then crop the inverse transform to "same" length.
        # This cached-FFT path was fastest in a benchmarked 132x161 workload.
        model_fft = fft.rfft(model_y, n=self._convolution_fft_length, axis=1)
        full = fft.irfft(model_fft * self._res_fft, n=self._convolution_fft_length, axis=1)
        full = full[:, : self._convolution_full_length]
        start = self._convolution_same_start
        convolved = full[:, start : start + self.data_x.shape[1]]
        return convolved * self._x_delta

    def convolved_model(self, params_vector: np.ndarray) -> np.ndarray:
        """Model of every row convolved with its resolution, as an array shaped like data_y."""
        params_dict = self.unpack(params_vector)
        model_y = self._evaluate_rows(params_dict)
        return params_dict[BACKGROUND_PARAMETER][:, np.newaxis] + self._convolve(model_y)

    def row_costs(self, params_vector: np.ndarray) -> np.ndarray:
        """Least squares cost of each row, ignoring the masked points."""
        residuals = (self._safe_y - self.convolved_model(params_vector)) / self._safe_e
        return np.sum(np.where(self.valid, residuals, 0.0) ** 2, axis=1)

    def weighted_residuals(self, params_vector: np.ndarray) -> np.ndarray:
        """Residuals divided by the errors of all valid points, concatenated into a 1D array."""
        return ((self._safe_y - self.convolved_model(params_vector)) / self._safe_e)[self.valid]

    def cost_function(self, params_vector: np.ndarray) -> float:
        """Total cost: the sum of the cost of each row given the shared and unshared parameters in the vector."""
        return float(np.sum(self.row_costs(params_vector)))

    def constraint_function(self, params_vector: np.ndarray) -> np.ndarray:
        """Fitted y values of all rows over their full x range, concatenated into a 1D array."""
        return self.convolved_model(params_vector).ravel()

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
        jacobian_curve, _ = jacobi.jacobi(lambda params_vector: curve(params_vector).ravel(), parameters)
        variance = np.einsum("ij,jk,ik->i", jacobian_curve, covariance, jacobian_curve)
        return np.sqrt(np.clip(variance, 0, None)).reshape(self.data_y.shape)

    def fit_curve_errors(self, parameters: np.ndarray, covariance: np.ndarray | None = None) -> np.ndarray:
        """Uncertainty of the fitted y values, shaped like data_y."""
        return self.curve_errors(self.convolved_model, parameters, covariance)


class GlobalFitWithFSEBuilder(GlobalFitBuilder):
    """Build a global fit including a final state effect (FSE) correction.

    A free ``fse_intensity`` parameter scales ``-d3 model / dx3 / q`` before
    convolution and is reported as a width. Alternatively, set
    ``model_width_parameter`` to a model parameter used as the FSE width, with
    intensity ``width**4 / 3``. If both are supplied, the free intensity takes
    precedence. One of these parameters and ``q`` are required.
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
        q: np.ndarray | None = None,
        model_width_parameter: str | None = None,
        VECTORISED: bool = False,
    ) -> None:
        self.q = None if q is None else np.asarray(q, dtype=float)
        self.model_width_parameter = model_width_parameter
        super().__init__(data_x, data_y, data_e, res_y, model, initial_params, shared_params, VECTORISED=VECTORISED)
        spacing = self.data_x[:, 7:-5] - self.data_x[:, 6:-6]
        self._third_derivative_scale = 1.0 / (np.power(spacing, 3) * 12**3)
        assert self.q is not None  # Validated by _validate_inputs.
        # Bins where q is zero or not finite get no FSE term.
        usable_q = np.isfinite(self.q) & (self.q != 0)
        self._inverse_q = np.divide(1.0, self.q, out=np.zeros_like(self.q), where=usable_q)

    def _validate_inputs(self) -> None:
        super()._validate_inputs()
        if self.q is None:
            raise ValueError("q is required for an FSE fit.")
        if self.q.shape != self.data_y.shape:
            raise ValueError("q must have the same shape as data_y.")
        if FSE_INTENSITY_PARAMETER not in self.initial_params and self.model_width_parameter is None:
            raise ValueError(f"An FSE fit requires {FSE_INTENSITY_PARAMETER!r} or a model_width_parameter.")
        if self.model_width_parameter is not None and self.model_width_parameter not in self.initial_params:
            raise ValueError(f"Model width parameter {self.model_width_parameter!r} is not in the model parameters.")
        if self.data_x.shape[1] <= 2 * THIRD_DERIVATIVE_EDGE:
            raise ValueError("Not enough points to compute the third derivative for the FSE term.")

    def reporting_parameters(self, params_vector: np.ndarray, covariance: np.ndarray) -> tuple[list[str], np.ndarray, np.ndarray]:
        """Report free FSE intensity as width, propagating its error."""
        names, values, errors = super().reporting_parameters(params_vector, covariance)
        indices = self._free_intensity_indices()
        for row, index in enumerate(indices):
            names[index] = FSE_WIDTH_PARAMETER + ("" if FSE_INTENSITY_PARAMETER in self.shared_params else str(row))
            intensity = values[index]
            if intensity < 0:
                raise ValueError("FSE intensity must be non-negative to report a real width.")
            values[index] = (3 * intensity) ** 0.25
            derivative = np.inf if intensity == 0 else 0.75 * (3 * intensity) ** (-0.75)
            errors[index] *= derivative
        return names, values, errors

    def _free_intensity_indices(self) -> range:
        if FSE_INTENSITY_PARAMETER not in self.initial_params:
            return range(0)
        if FSE_INTENSITY_PARAMETER in self.shared_params:
            index = self.shared_params.index(FSE_INTENSITY_PARAMETER)
            return range(index, index + 1)
        index = len(self.shared_params) + self.unshared_params.index(FSE_INTENSITY_PARAMETER) * self.n_spectra
        return range(index, index + self.n_spectra)

    def optimizer_bounds(self) -> list[tuple[float | None, float | None]]:
        """Keep free FSE intensity non-negative for a real width transform."""
        bounds = super().optimizer_bounds()
        for index in self._free_intensity_indices():
            bounds[index] = (0.0, None)
        return bounds

    def _model_and_fse_rows(self, params_dict: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        params_dict = dict(params_dict)
        if FSE_INTENSITY_PARAMETER in params_dict:
            intensity = params_dict.pop(FSE_INTENSITY_PARAMETER)
        else:
            assert self.model_width_parameter is not None  # Validated by _validate_inputs.
            intensity = np.asarray(params_dict[self.model_width_parameter]) ** 4 / 3
        model_y = super()._evaluate_rows(params_dict)
        # The edges are left at zero, where the third derivative is not defined.
        third_derivative = np.zeros_like(model_y)
        edge = THIRD_DERIVATIVE_EDGE
        third_derivative[:, edge:-edge] = numerical_third_derivative(self.data_x, model_y, self._third_derivative_scale)
        intensity = np.reshape(intensity, (-1, 1)) if np.ndim(intensity) else intensity
        return model_y, -intensity * third_derivative * self._inverse_q

    def _evaluate_rows(self, params_dict: dict[str, Any]) -> np.ndarray:
        model_y, fse_y = self._model_and_fse_rows(params_dict)
        return model_y + fse_y

    def fse_contribution(self, params_vector: np.ndarray) -> np.ndarray:
        """Convolved FSE term of every row, as an array shaped like data_y."""
        _, fse_y = self._model_and_fse_rows(self.unpack(params_vector))
        return self._convolve(fse_y)

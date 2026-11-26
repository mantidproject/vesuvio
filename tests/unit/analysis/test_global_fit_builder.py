import unittest

import numpy as np
from scipy import signal
from scipy.optimize import NonlinearConstraint, minimize

from mvesuvio.default_config.experiment_template.fitting_models import _gauss
from mvesuvio.util.global_fit_builder import FSE_INTENSITY_PARAMETER, GlobalFitBuilder, GlobalFitWithFSEBuilder


class TestGlobalFitBuilder(unittest.TestCase):
    def setUp(self):
        self.n_spectra = 3
        x = np.linspace(-20, 20, 81)
        self.data_x = np.tile(x, (self.n_spectra, 1))
        self.res_y = np.tile(_gauss(x, 1, 0, 1.0), (self.n_spectra, 1))
        self.amplitudes = [1.0, 2.0, 3.0]
        self.true_sigma = 5.0
        self.total_sigma = np.hypot(self.true_sigma, 1.0)
        self.data_y = np.array([_gauss(x, a, 0, self.total_sigma) for a in self.amplitudes])
        self.data_e = np.full_like(self.data_y, 0.01)
        self.initial = {"A": 1.5, "x0": 0.0, "sigma": 4.0}

    def build(self, data_y=None, data_e=None, VECTORISED=False):
        return GlobalFitBuilder(
            self.data_x,
            self.data_y if data_y is None else data_y,
            self.data_e if data_e is None else data_e,
            self.res_y,
            _gauss,
            self.initial,
            ["sigma"],
            VECTORISED=VECTORISED,
        )

    def test_parameter_layout(self):
        builder = self.build()
        self.assertEqual(builder.parameter_names[0], "sigma")
        self.assertEqual(len(builder.parameter_names), 1 + 3 * self.n_spectra)
        self.assertEqual(builder.initial_vector().size, len(builder.parameter_names))

    def test_initial_vector_uses_shared_and_per_row_defaults_in_parameter_order(self):
        builder = self.build()
        self.assertEqual(builder.parameter_names, ["sigma", "y0", "y1", "y2", "A0", "A1", "A2", "x00", "x01", "x02"])
        np.testing.assert_array_equal(builder.initial_vector(), [4, 0, 0, 0, 1.5, 1.5, 1.5, 0, 0, 0])

    def test_unpack_shares_parameters(self):
        builder = self.build()
        vector = np.arange(builder.initial_vector().size, dtype=float)
        params_dict = builder.unpack(params_vector=vector)
        self.assertEqual(params_dict["sigma"], 0.0)
        np.testing.assert_array_equal(params_dict["y"], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(params_dict["A"], [4.0, 5.0, 6.0])
        np.testing.assert_array_equal(params_dict["x0"], [7.0, 8.0, 9.0])

    def test_unpack_rejects_wrong_number_of_parameters(self):
        builder = self.build()
        with self.assertRaisesRegex(ValueError, "Expected 10 parameters, got 9"):
            builder.unpack(np.zeros(9))

    def test_convolution_grid_interpolates_even_length_and_rejects_asymmetry(self):
        delta, resolution = GlobalFitBuilder._build_convolution_grid(np.array([-3, -1, 1, 3]), np.array([0, 2, 4, 0]))
        self.assertEqual(delta, 1.5)
        np.testing.assert_allclose(resolution, [0, 1.5, 3, 3, 0])
        delta, resolution = GlobalFitBuilder._build_convolution_grid(np.array([-2, -1, 0, 1, 2]), np.array([0, 1, 3, 1, 0]))
        self.assertEqual(delta, 1)
        np.testing.assert_array_equal(resolution, [0, 1, 3, 1, 0])
        with self.assertRaisesRegex(ValueError, "symmetric x range"):
            GlobalFitBuilder._build_convolution_grid(np.array([-2, 0, 3]), np.ones(3))

    def test_convolution_matches_row_by_row_direct_convolution(self):
        builder = self.build()
        vector = builder.initial_vector()
        expected = []
        for i in range(self.n_spectra):
            model_y = _gauss(self.data_x[i], 1.5, 0.0, 4.0)
            expected.append(signal.convolve(model_y, builder._res_dense[i], mode="same", method="direct") * builder._x_delta[i, 0])
        np.testing.assert_allclose(builder.convolved_model(vector), np.array(expected), atol=1e-10)

    def test_cached_convolution_matches_fftconvolve_for_even_and_odd_lengths(self):
        for n_points in (80, 81):
            x = np.tile(np.linspace(-20, 20, n_points), (self.n_spectra, 1))
            resolution = np.array([_gauss(row, 1.0, 0.0, sigma) for row, sigma in zip(x, (0.8, 1.0, 1.2))])
            builder = GlobalFitBuilder(
                x,
                np.zeros_like(x),
                np.ones_like(x),
                resolution,
                _gauss,
                self.initial,
            )
            model_y = np.array([_gauss(row, amplitude, 0.0, 4.0) for row, amplitude in zip(x, self.amplitudes)])
            expected = signal.fftconvolve(model_y, builder._res_dense, mode="same", axes=1) * builder._x_delta

            np.testing.assert_allclose(builder._convolve(model_y), expected, rtol=2e-13, atol=2e-13)

    def test_evaluate_rows_preserves_background_and_excludes_it_from_model(self):
        builder = self.build()
        params_dict = builder.unpack(builder.initial_vector())
        params_dict["y"][:] = [1.0, 2.0, 3.0]
        rows = builder._evaluate_rows(params_dict)
        np.testing.assert_array_equal(params_dict["y"], [1.0, 2.0, 3.0])
        for i in range(self.n_spectra):
            np.testing.assert_allclose(rows[i], _gauss(self.data_x[i], 1.5, 0.0, 4.0))

    def test_evaluate_rows_vectorises_x_and_row_parameters(self):
        builder = self.build(VECTORISED=True)
        params_dict = builder.unpack(builder.initial_vector())
        params_dict["A"][:] = self.amplitudes

        rows = builder._evaluate_rows(params_dict)

        for i in range(self.n_spectra):
            np.testing.assert_allclose(rows[i], _gauss(self.data_x[i], self.amplitudes[i], 0.0, 4.0))

    def test_cost_is_zero_at_true_parameters(self):
        builder = self.build()
        vector = builder.initial_vector()
        vector[0] = self.true_sigma
        names = builder.parameter_names
        for i, a in enumerate(self.amplitudes):
            vector[names.index(f"A{i}")] = a
        self.assertAlmostEqual(builder.cost_function(vector), 0.0, places=3)

    def test_masked_points_are_ignored(self):
        data_y = self.data_y.copy()
        data_e = self.data_e.copy()
        data_y[1, 10:15] = 0
        data_e[1, 10:15] = 0
        builder = self.build(data_y, data_e)
        self.assertFalse(builder.valid[1, 10:15].any())
        self.assertTrue(np.isfinite(builder.cost_function(builder.initial_vector())))

        data_y[1, 10:15] = 1e6  # Changing masked points must not change the cost
        other = self.build(data_y, data_e)
        vector = builder.initial_vector()
        self.assertEqual(builder.cost_function(vector), other.cost_function(vector))

    def test_minimize_recovers_shared_parameter(self):
        builder = self.build()
        result = minimize(builder.cost_function, builder.initial_vector(), method="L-BFGS-B")
        self.assertAlmostEqual(result.x[0], self.true_sigma, places=1)

    def test_constraint_is_concatenated_fit(self):
        builder = self.build()
        vector = builder.initial_vector()
        constraint = builder.constraint_function(vector)
        self.assertEqual(constraint.shape, (self.n_spectra * self.data_x.shape[1],))
        np.testing.assert_array_equal(constraint, builder.convolved_model(vector).ravel())

    def test_slsqp_with_positive_constraint_keeps_fit_non_negative(self):
        lowered_y = self.data_y - 0.2  # Pushes the data (and an unconstrained fit) well below zero
        builder = self.build(data_y=lowered_y)

        unconstrained = minimize(builder.cost_function, builder.initial_vector(), method="SLSQP")
        self.assertLess(builder.constraint_function(unconstrained.x).min(), -1e-2)

        constrained = minimize(
            builder.cost_function,
            builder.initial_vector(),
            method="SLSQP",
            constraints=[NonlinearConstraint(builder.constraint_function, 0, np.inf)],
        )
        # SLSQP satisfies the constraint only up to a small tolerance
        self.assertGreaterEqual(builder.constraint_function(constrained.x).min(), -1e-3)

        self.assertFalse(np.allclose(constrained.x, unconstrained.x, rtol=1e-3, atol=1e-3))
        self.assertGreater(constrained.fun, unconstrained.fun)

    def test_weighted_residuals_match_cost(self):
        builder = self.build()
        vector = builder.initial_vector()
        self.assertAlmostEqual(np.sum(builder.weighted_residuals(vector) ** 2), builder.cost_function(vector))

    def test_row_costs_and_weighted_residuals_ignore_invalid_points(self):
        data_y = np.zeros_like(self.data_y)
        data_e = np.zeros_like(self.data_e)
        data_y[0, :3] = [5, 3, np.nan]
        data_e[0, :3] = [2, 1, 1]
        data_y[1, :3] = [8, -1, 100]
        data_e[1, :3] = [2, 1, np.inf]
        data_y[2, 0] = 1
        data_e[2, 0] = 1
        builder = self.build(data_y, data_e)
        vector = builder.initial_vector()
        for i in range(self.n_spectra):
            vector[builder.parameter_names.index(f"y{i}")] = i + 1
            vector[builder.parameter_names.index(f"A{i}")] = 0

        np.testing.assert_allclose(builder.weighted_residuals(vector), [2, 2, 3, -3, -2])
        np.testing.assert_allclose(builder.row_costs(vector), [8, 18, 4])
        self.assertAlmostEqual(builder.cost_function(vector), 30)

    def test_parameter_covariance_matches_linear_least_squares(self):
        x = np.tile(np.arange(-2, 3, dtype=float), (2, 1))
        resolution = np.tile([0, 0, 1, 0, 0], (2, 1))
        residual = np.array([1, -2, 2, -2, 1])
        data_y = np.array([2 * x[0] + 1 + residual, 2 * x[1] + 3 - residual])
        builder = GlobalFitBuilder(x, data_y, np.ones_like(x), resolution, lambda x, slope: slope * x, {"slope": 2}, ["slope"])

        covariance = builder.parameter_covariance(np.array([2, 1, 3]))

        np.testing.assert_allclose(covariance, np.diag([0.2, 0.8, 0.8]), rtol=1e-6, atol=1e-9)

    def test_curve_errors_propagate_correlated_parameter_uncertainties(self):
        x = np.tile(np.arange(-2, 3, dtype=float), (2, 1))
        resolution = np.tile([0, 0, 1, 0, 0], (2, 1))
        builder = GlobalFitBuilder(x, np.zeros_like(x), np.ones_like(x), resolution, lambda x, slope: slope * x, {"slope": 2}, ["slope"])
        parameters = np.array([2, 1, 3])
        covariance = np.array([[4, 1, 0], [1, 9, 0.5], [0, 0.5, 16]])
        expected = np.sqrt([4 * x[0] ** 2 + 2 * x[0] + 9, 4 * x[1] ** 2 + 16])

        np.testing.assert_allclose(builder.curve_errors(builder.convolved_model, parameters, covariance), expected, rtol=1e-6)
        np.testing.assert_allclose(builder.fit_curve_errors(parameters, covariance), expected, rtol=1e-6)

    def test_fit_curve_errors_are_finite_and_small_for_good_fit(self):
        builder = self.build()
        result = minimize(builder.cost_function, builder.initial_vector(), method="SLSQP")
        errors = builder.fit_curve_errors(result.x)
        self.assertEqual(errors.shape, self.data_y.shape)
        self.assertTrue(np.all(np.isfinite(errors)))
        self.assertTrue(np.all(errors >= 0))
        self.assertLess(errors.max(), 0.1 * builder.convolved_model(result.x).max())

    def test_invalid_inputs_raise(self):
        with self.assertRaises(ValueError):
            GlobalFitBuilder(self.data_x, self.data_y[:, :-1], self.data_e, self.res_y, _gauss, self.initial, ["sigma"])
        with self.assertRaises(ValueError):
            GlobalFitBuilder(self.data_x, self.data_y, self.data_e, self.res_y, _gauss, self.initial, ["missing"])

    def test_validate_inputs_rejects_reserved_background_parameter(self):
        builder = self.build()
        builder.initial_params["y"] = 1.0
        with self.assertRaisesRegex(ValueError, "'y' is reserved"):
            builder._validate_inputs()

    def test_fse_named_parameter_is_an_ordinary_model_parameter(self):
        def model(x, fse_intensity):
            return np.full_like(x, fse_intensity)

        builder = GlobalFitBuilder(self.data_x, self.data_y, self.data_e, self.res_y, model, {"fse_intensity": -2.0}, ["fse_intensity"])
        vector = builder.initial_vector()
        names, values, errors = builder.reporting_parameters(vector, np.eye(vector.size))
        self.assertEqual(names[0], "fse_intensity")
        self.assertEqual(values[0], -2.0)
        np.testing.assert_array_equal(errors, np.ones(vector.size))
        self.assertTrue(all(bound == (None, None) for bound in builder.optimizer_bounds()))
        np.testing.assert_allclose(builder.convolved_model(vector), builder._convolve(np.full_like(self.data_x, -2.0)))


class TestGlobalFitBuilderFSE(unittest.TestCase):
    def setUp(self):
        self.n_spectra = 3
        x = np.linspace(-20, 20, 161)
        self.x = x
        self.data_x = np.tile(x, (self.n_spectra, 1))
        self.res_y = np.tile(_gauss(x, 1, 0, 1.0), (self.n_spectra, 1))
        self.sigma = 4.0
        self.true_fse_intensity = 3.0
        self.true_width = (3 * self.true_fse_intensity) ** 0.25
        self.amplitudes = [1.0, 2.0, 3.0]

        def deriv3(a):
            return a * (3 * x / self.sigma**4 - x**3 / self.sigma**6) * _gauss(x, 1, 0, self.sigma)

        # Analytical third derivative of the gaussian
        self.expected_fse = np.array([-self.true_fse_intensity * deriv3(a) for a in self.amplitudes])
        data = np.array([_gauss(x, a, 0, self.sigma) for a in self.amplitudes]) + self.expected_fse
        self.data_y = np.array([np.convolve(row, self.res_y[0], mode="same") * (x[1] - x[0]) for row in data])
        self.data_e = np.full_like(self.data_y, 0.001)

    def build(self, shared, width_initial=1.0, q=None, model_width_parameter=None, VECTORISED=False):
        initial = {"A": 1.5, "x0": 0.0, "sigma": self.sigma}
        if model_width_parameter is None:
            initial[FSE_INTENSITY_PARAMETER] = width_initial**4 / 3
            shared = [FSE_INTENSITY_PARAMETER if name == "fse_width" else name for name in shared]
        q = np.ones_like(self.data_y) if q is None else q
        return GlobalFitWithFSEBuilder(
            self.data_x,
            self.data_y,
            self.data_e,
            self.res_y,
            _gauss,
            initial,
            shared,
            q=q,
            model_width_parameter=model_width_parameter,
            VECTORISED=VECTORISED,
        )

    def test_vectorised_fse_cost_matches_sequential(self):
        sequential = self.build(["sigma", "fse_width"])
        vectorised = self.build(["sigma", "fse_width"], VECTORISED=True)

        np.testing.assert_allclose(
            vectorised.cost_function(vectorised.initial_vector()),
            sequential.cost_function(sequential.initial_vector()),
            rtol=1e-14,
            atol=1e-14,
        )

    def test_model_and_fse_rows_match_analytic_gaussian_derivative(self):
        q = np.tile(np.array([1.0, 2.0, 3.0])[:, np.newaxis], (1, self.x.size))
        q[:, 60:63] = [0, np.nan, np.inf]
        expected_model = np.array([_gauss(self.x, a, 0, self.sigma) for a in self.amplitudes])

        for tied in (False, True):
            for vectorised in (False, True):
                with self.subTest(tied=tied, vectorised=vectorised):
                    builder = self.build(["sigma"], q=q, model_width_parameter="sigma" if tied else None, VECTORISED=vectorised)
                    params = builder.unpack(builder.initial_vector())
                    params["A"][:] = self.amplitudes
                    params["y"][:] = [1, 2, 3]
                    if tied:
                        expected_fse = self.expected_fse * (self.sigma**4 / 9) / np.array([1, 2, 3])[:, np.newaxis]
                    else:
                        params[FSE_INTENSITY_PARAMETER][:] = [3, 6, 9]
                        expected_fse = self.expected_fse.copy()
                    original = {name: np.copy(value) for name, value in params.items()}
                    expected_fse[:, :6] = 0
                    expected_fse[:, -6:] = 0
                    expected_fse[:, 60:63] = 0

                    model, fse = builder._model_and_fse_rows(params)

                    np.testing.assert_allclose(model, expected_model)
                    np.testing.assert_allclose(fse, expected_fse, atol=1e-4)
                    np.testing.assert_array_equal(fse[:, :6], 0)
                    np.testing.assert_array_equal(fse[:, -6:], 0)
                    np.testing.assert_array_equal(fse[:, 60:63], 0)
                    np.testing.assert_allclose(builder._evaluate_rows(params), expected_model + expected_fse, atol=1e-4)
                    self.assertEqual(params.keys(), original.keys())
                    for name in params:
                        np.testing.assert_array_equal(params[name], original[name])

    def test_free_intensity_indices_follow_parameter_layout(self):
        shared = self.build(["sigma", "fse_width"])
        unshared = self.build(["sigma"])
        tied = self.build(["sigma"], model_width_parameter="sigma")

        self.assertEqual(list(shared._free_intensity_indices()), [1])
        self.assertEqual(list(unshared._free_intensity_indices()), [10, 11, 12])
        self.assertEqual(list(tied._free_intensity_indices()), [])

    def test_fse_validate_inputs_checks_q_shape(self):
        builder = self.build(["sigma", "fse_width"])
        builder.q = np.ones_like(self.data_y[:, :-1])
        with self.assertRaisesRegex(ValueError, "q must have the same shape"):
            builder._validate_inputs()

    def test_fse_term_is_third_derivative_of_model(self):
        builder = self.build(["sigma", "fse_width"])
        vector = builder.initial_vector()
        names = builder.parameter_names
        vector[names.index(FSE_INTENSITY_PARAMETER)] = self.true_fse_intensity
        vector[names.index("sigma")] = self.sigma
        for i, a in enumerate(self.amplitudes):
            vector[names.index(f"A{i}")] = a
        convolved_fse = builder.fse_contribution(params_vector=vector)
        for i in range(self.n_spectra):
            expected = np.convolve(self.expected_fse[i], self.res_y[i], mode="same") * (self.x[1] - self.x[0])
            np.testing.assert_allclose(convolved_fse[i, 10:-10], expected[10:-10], atol=1e-4)

    def test_fse_term_is_divided_by_q_of_each_row(self):
        q = np.array([np.full(self.x.size, factor) for factor in (0.5, 2.0, 3.0)])
        builder = self.build(["sigma", "fse_width"], q=q)
        reference = self.build(["sigma", "fse_width"])
        np.testing.assert_array_equal(builder.q, q)
        vector = builder.initial_vector()
        np.testing.assert_allclose(builder.fse_contribution(vector), reference.fse_contribution(vector) / q, atol=1e-12)

    def test_fse_term_is_zero_where_q_is_zero(self):
        q = np.ones_like(self.data_y)
        q[:, 80] = 0.0
        builder = self.build(["sigma", "fse_width"], q=q)
        fse = builder.fse_contribution(builder.initial_vector())
        self.assertTrue(np.all(np.isfinite(fse)))

    def test_fse_without_q_or_with_wrong_shape_raises(self):
        initial = {"A": 1.5, "x0": 0.0, "sigma": self.sigma, FSE_INTENSITY_PARAMETER: 1.0}
        with self.assertRaisesRegex(ValueError, "q is required"):
            GlobalFitWithFSEBuilder(self.data_x, self.data_y, self.data_e, self.res_y, _gauss, initial, ["sigma"])
        with self.assertRaisesRegex(ValueError, "q must have the same shape"):
            GlobalFitWithFSEBuilder(self.data_x, self.data_y, self.data_e, self.res_y, _gauss, initial, ["sigma"], self.data_y[:, :-1])
        tied_initial = {"A": 1.5, "x0": 0.0, "sigma": self.sigma}
        with self.assertRaisesRegex(ValueError, "q is required"):
            GlobalFitWithFSEBuilder(
                self.data_x,
                self.data_y,
                self.data_e,
                self.res_y,
                _gauss,
                tied_initial,
                ["sigma"],
                model_width_parameter="sigma",
            )

    def test_fse_requires_enough_points(self):
        x = self.data_x[:, 74:86]
        initial = {"A": 1.5, "x0": 0.0, "sigma": self.sigma, FSE_INTENSITY_PARAMETER: 1.0}
        with self.assertRaisesRegex(ValueError, "Not enough points"):
            GlobalFitWithFSEBuilder(x, x, x, x, _gauss, initial, ["sigma"], np.ones_like(x))

    def test_zero_intensity_matches_fit_without_fse(self):
        with_fse = self.build(["sigma", "fse_width"], width_initial=0.0)
        without = GlobalFitBuilder(
            self.data_x, self.data_y, self.data_e, self.res_y, _gauss, {"A": 1.5, "x0": 0.0, "sigma": self.sigma}, ["sigma"]
        )
        np.testing.assert_allclose(with_fse.convolved_model(with_fse.initial_vector()), without.convolved_model(without.initial_vector()))

    def test_fit_is_model_plus_convolved_fse_and_background(self):
        builder = self.build(["sigma", "fse_width"])
        without = GlobalFitBuilder(
            self.data_x,
            self.data_y,
            self.data_e,
            self.res_y,
            _gauss,
            {"A": 1.5, "x0": 0.0, "sigma": self.sigma},
            ["sigma"],
        )
        vector = builder.initial_vector()
        base_vector = without.initial_vector()
        for i in range(self.n_spectra):
            vector[builder.parameter_names.index(f"y{i}")] = 0.1 * (i + 1)
            base_vector[without.parameter_names.index(f"y{i}")] = 0.1 * (i + 1)
        np.testing.assert_allclose(
            builder.convolved_model(vector),
            without.convolved_model(base_vector) + builder.fse_contribution(vector),
            atol=1e-12,
        )

    def test_fse_width_sharing(self):
        shared = self.build(["sigma", "fse_width"])
        self.assertEqual(shared.parameter_names[:2], ["sigma", FSE_INTENSITY_PARAMETER])
        unshared = self.build(["sigma"])
        self.assertEqual(len(unshared.parameter_names), len(shared.parameter_names) + self.n_spectra - 1)
        self.assertIn("fse_intensity2", unshared.parameter_names)

    def test_fse_width_can_be_tied_to_a_model_parameter(self):
        tied = self.build(["sigma"], model_width_parameter="sigma")
        untied = self.build(["sigma", "fse_width"], width_initial=self.sigma)
        tied_vector = tied.initial_vector()
        untied_vector = untied.initial_vector()
        untied_vector[untied.parameter_names.index(FSE_INTENSITY_PARAMETER)] = self.sigma**4 / 3

        self.assertEqual(tied.model_width_parameter, "sigma")
        self.assertNotIn(FSE_INTENSITY_PARAMETER, tied.parameter_names)
        np.testing.assert_allclose(tied.fse_contribution(tied_vector), untied.fse_contribution(untied_vector))

    def test_unshared_tied_fse_width_matches_unshared_free_intensity(self):
        tied = self.build([], model_width_parameter="sigma")
        untied = self.build([], width_initial=self.sigma)
        tied_vector = tied.initial_vector()
        untied_vector = untied.initial_vector()
        for i, width in enumerate((3.0, 4.0, 5.0)):
            tied_vector[tied.parameter_names.index(f"sigma{i}")] = width
            untied_vector[untied.parameter_names.index(f"sigma{i}")] = width
            untied_vector[untied.parameter_names.index(f"{FSE_INTENSITY_PARAMETER}{i}")] = width**4 / 3
        np.testing.assert_allclose(tied.fse_contribution(tied_vector), untied.fse_contribution(untied_vector))
        np.testing.assert_allclose(tied.convolved_model(tied_vector), untied.convolved_model(untied_vector))

    def test_free_intensity_takes_precedence_over_model_width(self):
        for is_shared in (True, False):
            with self.subTest(shared=is_shared):
                initial = {"A": 1.5, "x0": 0.0, "sigma": self.sigma, FSE_INTENSITY_PARAMETER: 1.0}
                shared = ["sigma"] + ([FSE_INTENSITY_PARAMETER] if is_shared else [])
                builder = GlobalFitWithFSEBuilder(
                    self.data_x,
                    self.data_y,
                    self.data_e,
                    self.res_y,
                    _gauss,
                    initial,
                    shared,
                    q=np.ones_like(self.data_y),
                    model_width_parameter="sigma",
                )
                reference = GlobalFitWithFSEBuilder(
                    self.data_x,
                    self.data_y,
                    self.data_e,
                    self.res_y,
                    _gauss,
                    initial,
                    shared,
                    q=np.ones_like(self.data_y),
                )
                vector = builder.initial_vector()
                intensity_names = (
                    [FSE_INTENSITY_PARAMETER] if is_shared else [f"{FSE_INTENSITY_PARAMETER}{i}" for i in range(self.n_spectra)]
                )
                for i, name in enumerate(intensity_names):
                    vector[builder.parameter_names.index(name)] = i + 2.0
                np.testing.assert_allclose(builder.fse_contribution(vector), reference.fse_contribution(vector))
                np.testing.assert_allclose(builder.convolved_model(vector), reference.convolved_model(vector))
                names, values, errors = builder.reporting_parameters(vector, np.eye(vector.size))
                ref_names, ref_values, ref_errors = reference.reporting_parameters(vector, np.eye(vector.size))
                self.assertEqual(names, ref_names)
                np.testing.assert_array_equal(values, ref_values)
                np.testing.assert_array_equal(errors, ref_errors)
                for name in intensity_names:
                    self.assertEqual(builder.optimizer_bounds()[builder.parameter_names.index(name)], (0.0, None))

    def test_free_fse_intensity_is_reported_as_width_with_propagated_error(self):
        builder = self.build(["sigma", "fse_width"])
        vector = builder.initial_vector()
        intensity_index = builder.parameter_names.index(FSE_INTENSITY_PARAMETER)
        vector[intensity_index] = self.true_fse_intensity
        covariance = np.eye(vector.size)
        covariance[intensity_index, intensity_index] = 0.04

        names, values, errors = builder.reporting_parameters(vector, covariance)

        self.assertEqual(names[intensity_index], "fse_width")
        self.assertAlmostEqual(values[intensity_index], self.true_width)
        expected_error = 0.2 * 0.75 * (3 * self.true_fse_intensity) ** (-0.75)
        self.assertAlmostEqual(errors[intensity_index], expected_error)

    def test_unshared_intensities_are_reported_as_individual_widths(self):
        builder = self.build(["sigma"])
        vector = builder.initial_vector()
        indices = [builder.parameter_names.index(f"{FSE_INTENSITY_PARAMETER}{i}") for i in range(self.n_spectra)]
        vector[indices] = [1 / 3, 16 / 3, 81 / 3]
        original = vector.copy()
        covariance = np.eye(vector.size)
        covariance[indices, indices] = [0.04, 0.09, 0.16]

        names, values, errors = builder.reporting_parameters(vector, covariance)

        self.assertEqual([names[i] for i in indices], ["fse_width0", "fse_width1", "fse_width2"])
        np.testing.assert_allclose(values[indices], [1, 2, 3])
        np.testing.assert_allclose(errors[indices], [0.15, 0.028125, 0.4 / 36])
        np.testing.assert_array_equal(vector, original)

    def test_fse_intensity_optimizer_bounds_are_non_negative(self):
        shared = self.build(["sigma", "fse_width"])
        shared_index = shared.parameter_names.index(FSE_INTENSITY_PARAMETER)
        self.assertEqual(shared.optimizer_bounds()[shared_index], (0.0, None))

        unshared = self.build(["sigma"])
        for index, name in enumerate(unshared.parameter_names):
            expected = (0.0, None) if name.startswith(FSE_INTENSITY_PARAMETER) else (None, None)
            self.assertEqual(unshared.optimizer_bounds()[index], expected)

    def test_similarly_named_model_parameter_is_not_transformed_or_bounded(self):
        def model(x, fse_intensity_extra):
            return np.full_like(x, fse_intensity_extra)

        builder = GlobalFitWithFSEBuilder(
            self.data_x,
            self.data_y,
            self.data_e,
            self.res_y,
            model,
            {"fse_intensity_extra": -2.0, FSE_INTENSITY_PARAMETER: 1.0},
            ["fse_intensity_extra"],
            np.ones_like(self.data_y),
        )
        vector = builder.initial_vector()
        names, values, _ = builder.reporting_parameters(vector, np.eye(vector.size))
        index = names.index("fse_intensity_extra")
        self.assertEqual(values[index], -2.0)
        self.assertEqual(builder.optimizer_bounds()[index], (None, None))
        self.assertTrue(np.all(np.isfinite(builder.convolved_model(vector))))

    def test_tied_fse_width_requires_a_model_parameter(self):
        with self.assertRaisesRegex(ValueError, "is not in the model parameters"):
            self.build(["sigma"], model_width_parameter="missing")

    def test_fit_recovers_fse_width(self):
        for shared in (["sigma", "fse_width"], ["sigma"]):
            builder = self.build(shared)
            result = minimize(
                builder.cost_function,
                builder.initial_vector(),
                method="SLSQP",
                bounds=builder.optimizer_bounds(),
                options={"maxiter": 500},
            )
            fitted_intensity = builder.unpack(result.x)[FSE_INTENSITY_PARAMETER]
            fitted_width = (3 * fitted_intensity) ** 0.25
            np.testing.assert_allclose(fitted_width, self.true_width, rtol=0.1)

    def test_fse_builder_requires_an_fse_parameter(self):
        initial = {"A": 1, "x0": 0, "sigma": 4}
        with self.assertRaisesRegex(ValueError, "fse_intensity.*model_width_parameter"):
            GlobalFitWithFSEBuilder(
                self.data_x, self.data_y, self.data_e, self.res_y, _gauss, initial, ["sigma"], q=np.ones_like(self.data_y)
            )
        with self.assertRaisesRegex(ValueError, "q is required"):
            GlobalFitWithFSEBuilder(self.data_x, self.data_y, self.data_e, self.res_y, _gauss, initial, ["sigma"])


if __name__ == "__main__":
    unittest.main()

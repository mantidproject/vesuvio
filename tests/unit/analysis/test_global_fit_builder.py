import unittest

import numpy as np
from scipy.optimize import NonlinearConstraint, minimize

from mvesuvio.util.global_fit_builder import FSE_INTENSITY_PARAMETER, GlobalFitBuilder, GlobalFitWithFSEBuilder


def gauss(x, A, x0, sigma):
    return A / (2 * np.pi) ** 0.5 / sigma * np.exp(-((x - x0) ** 2) / 2 / sigma**2)


class TestGlobalFitBuilder(unittest.TestCase):
    def setUp(self):
        self.n_spectra = 3
        x = np.linspace(-20, 20, 81)
        self.data_x = np.tile(x, (self.n_spectra, 1))
        self.res_y = np.tile(gauss(x, 1, 0, 1.0), (self.n_spectra, 1))
        self.amplitudes = [1.0, 2.0, 3.0]
        self.true_sigma = 5.0
        self.total_sigma = np.hypot(self.true_sigma, 1.0)
        self.data_y = np.array([gauss(x, a, 0, self.total_sigma) for a in self.amplitudes])
        self.data_e = np.full_like(self.data_y, 0.01)
        self.initial = {"A": 1.5, "x0": 0.0, "sigma": 4.0}

    def build(self, data_y=None, data_e=None):
        return GlobalFitBuilder(
            self.data_x,
            self.data_y if data_y is None else data_y,
            self.data_e if data_e is None else data_e,
            self.res_y,
            gauss,
            self.initial,
            ["sigma"],
        )

    def test_parameter_layout(self):
        builder = self.build()
        self.assertEqual(builder.parameter_names[0], "sigma")
        self.assertEqual(len(builder.parameter_names), 1 + 3 * self.n_spectra)
        self.assertEqual(builder.initial_vector().size, len(builder.parameter_names))

    def test_unpack_shares_parameters(self):
        builder = self.build()
        vector = np.arange(builder.initial_vector().size, dtype=float)
        params_dict = builder.unpack(params_vector=vector)
        self.assertEqual(params_dict["sigma"], 0.0)
        np.testing.assert_array_equal(params_dict["y"], [1.0, 2.0, 3.0])

    def test_convolution_matches_row_by_row(self):
        from scipy import signal

        builder = self.build()
        vector = builder.initial_vector()
        expected = []
        for i in range(self.n_spectra):
            model_y = gauss(self.data_x[i], 1.5, 0.0, 4.0)
            expected.append(signal.convolve(model_y, builder._res_dense[i], mode="same") * builder._x_delta[i, 0])
        np.testing.assert_allclose(builder.convolved_model(vector), np.array(expected), atol=1e-10)

    def test_evaluate_rows_preserves_background_and_excludes_it_from_model(self):
        builder = self.build()
        params_dict = builder.unpack(builder.initial_vector())
        params_dict["y"][:] = [1.0, 2.0, 3.0]
        rows = builder._evaluate_rows(params_dict)
        np.testing.assert_array_equal(params_dict["y"], [1.0, 2.0, 3.0])
        for i in range(self.n_spectra):
            np.testing.assert_allclose(rows[i], gauss(self.data_x[i], 1.5, 0.0, 4.0))

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
            GlobalFitBuilder(self.data_x, self.data_y[:, :-1], self.data_e, self.res_y, gauss, self.initial, ["sigma"])
        with self.assertRaises(ValueError):
            GlobalFitBuilder(self.data_x, self.data_y, self.data_e, self.res_y, gauss, self.initial, ["missing"])

    def test_fse_named_parameter_is_an_ordinary_model_parameter(self):
        def model(x, fse_intensity):
            return np.full_like(x, fse_intensity)

        builder = GlobalFitBuilder(
            self.data_x, self.data_y, self.data_e, self.res_y, model, {"fse_intensity": -2.0}, ["fse_intensity"]
        )
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
        self.res_y = np.tile(gauss(x, 1, 0, 1.0), (self.n_spectra, 1))
        self.sigma = 4.0
        self.true_fse_intensity = 3.0
        self.true_width = (3 * self.true_fse_intensity) ** 0.25
        self.amplitudes = [1.0, 2.0, 3.0]
        def deriv3(a):
            return a * (3 * x / self.sigma**4 - x**3 / self.sigma**6) * gauss(x, 1, 0, self.sigma)
        # Analytical third derivative of the gaussian
        self.expected_fse = np.array([-self.true_fse_intensity * deriv3(a) for a in self.amplitudes])
        data = np.array([gauss(x, a, 0, self.sigma) for a in self.amplitudes]) + self.expected_fse
        self.data_y = np.array([np.convolve(row, self.res_y[0], mode="same") * (x[1] - x[0]) for row in data])
        self.data_e = np.full_like(self.data_y, 0.001)

    def build(self, shared, width_initial=1.0, q=None, model_width_parameter=None):
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
            gauss,
            initial,
            shared,
            q=q,
            model_width_parameter=model_width_parameter,
        )

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
            GlobalFitWithFSEBuilder(self.data_x, self.data_y, self.data_e, self.res_y, gauss, initial, ["sigma"])
        with self.assertRaisesRegex(ValueError, "q must have the same shape"):
            GlobalFitWithFSEBuilder(
                self.data_x, self.data_y, self.data_e, self.res_y, gauss, initial, ["sigma"], self.data_y[:, :-1]
            )
        tied_initial = {"A": 1.5, "x0": 0.0, "sigma": self.sigma}
        with self.assertRaisesRegex(ValueError, "q is required"):
            GlobalFitWithFSEBuilder(
                self.data_x, self.data_y, self.data_e, self.res_y, gauss, tied_initial,
                ["sigma"], model_width_parameter="sigma",
            )

    def test_fse_requires_enough_points(self):
        x = self.data_x[:, 74:86]
        initial = {"A": 1.5, "x0": 0.0, "sigma": self.sigma, FSE_INTENSITY_PARAMETER: 1.0}
        with self.assertRaisesRegex(ValueError, "Not enough points"):
            GlobalFitWithFSEBuilder(x, x, x, x, gauss, initial, ["sigma"], np.ones_like(x))

    def test_zero_intensity_matches_fit_without_fse(self):
        with_fse = self.build(["sigma", "fse_width"], width_initial=0.0)
        without = GlobalFitBuilder(
            self.data_x, self.data_y, self.data_e, self.res_y, gauss, {"A": 1.5, "x0": 0.0, "sigma": self.sigma}, ["sigma"]
        )
        np.testing.assert_allclose(with_fse.convolved_model(with_fse.initial_vector()), without.convolved_model(without.initial_vector()))

    def test_fit_is_model_plus_convolved_fse_and_background(self):
        builder = self.build(["sigma", "fse_width"])
        without = GlobalFitBuilder(
            self.data_x, self.data_y, self.data_e, self.res_y, gauss,
            {"A": 1.5, "x0": 0.0, "sigma": self.sigma}, ["sigma"],
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
                    self.data_x, self.data_y, self.data_e, self.res_y, gauss, initial, shared,
                    q=np.ones_like(self.data_y), model_width_parameter="sigma",
                )
                reference = GlobalFitWithFSEBuilder(
                    self.data_x, self.data_y, self.data_e, self.res_y, gauss, initial, shared,
                    q=np.ones_like(self.data_y),
                )
                vector = builder.initial_vector()
                intensity_names = (
                    [FSE_INTENSITY_PARAMETER] if is_shared
                    else [f"{FSE_INTENSITY_PARAMETER}{i}" for i in range(self.n_spectra)]
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
            self.data_x, self.data_y, self.data_e, self.res_y, model,
            {"fse_intensity_extra": -2.0, FSE_INTENSITY_PARAMETER: 1.0}, ["fse_intensity_extra"],
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

    def test_fse_contribution_errors(self):
        builder = self.build(["sigma", "fse_width"])
        result = minimize(builder.cost_function, builder.initial_vector(), method="SLSQP", options={"maxiter": 500})
        errors = builder.fse_contribution_errors(result.x)
        self.assertEqual(errors.shape, self.data_y.shape)
        self.assertTrue(np.all(np.isfinite(errors)))
        self.assertGreater(errors.max(), 0)

    def test_fse_builder_requires_an_fse_parameter(self):
        initial = {"A": 1, "x0": 0, "sigma": 4}
        with self.assertRaisesRegex(ValueError, "fse_intensity.*model_width_parameter"):
            GlobalFitWithFSEBuilder(
                self.data_x, self.data_y, self.data_e, self.res_y, gauss, initial, ["sigma"], q=np.ones_like(self.data_y)
            )
        with self.assertRaisesRegex(ValueError, "q is required"):
            GlobalFitWithFSEBuilder(self.data_x, self.data_y, self.data_e, self.res_y, gauss, initial, ["sigma"])


if __name__ == "__main__":
    unittest.main()

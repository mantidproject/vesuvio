import unittest

import numpy as np
from scipy.optimize import NonlinearConstraint, minimize

from mvesuvio.util.global_fit_builder import GlobalFitBuilder


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
        params = builder.unpack(vector)
        self.assertEqual(params["sigma"], 0.0)
        np.testing.assert_array_equal(params["y0"], [1.0, 2.0, 3.0])

    def test_convolution_matches_row_by_row(self):
        from scipy import signal

        builder = self.build()
        vector = builder.initial_vector()
        expected = []
        for i in range(self.n_spectra):
            model_y = gauss(self.data_x[i], 1.5, 0.0, 4.0)
            expected.append(signal.convolve(model_y, builder._res_dense[i], mode="same") * builder._x_delta[i, 0])
        np.testing.assert_allclose(builder.convolved_model(vector), np.array(expected), atol=1e-10)

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


if __name__ == "__main__":
    unittest.main()

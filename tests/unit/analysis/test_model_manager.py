import inspect
import unittest
from unittest.mock import patch

import numpy as np

from mvesuvio.analysis_fitting import fitProfileMantidFit, selectModelAndPars
from mvesuvio.default_config.experiment_template.fitting_models import FittingModel, register_fitting_models
from mvesuvio.util.model_manager import FittingModel as ModelDefinition
from mvesuvio.util.model_manager import ModelManager, model_manager

register_fitting_models()


class TestModelManager(unittest.TestCase):
    def test_register_fitting_models_is_cached(self):
        model_names = model_manager.get_model_names()
        cache_before = register_fitting_models.cache_info()

        with patch.object(model_manager, "subscribe") as subscribe:
            register_fitting_models()
            register_fitting_models()

        cache_after = register_fitting_models.cache_info()
        subscribe.assert_not_called()
        self.assertEqual(model_manager.get_model_names(), model_names)
        self.assertGreater(cache_after.hits, cache_before.hits)

    def test_default_models_are_registered_with_ordered_parameters(self):
        model_names = model_manager.get_model_names()
        enum_model_names = tuple(sorted(model.value for model in FittingModel))
        self.assertEqual(model_names, enum_model_names)
        self.assertTrue(model_manager.get_model("GAUSS").VECTORISED)
        self.assertFalse(model_manager.get_model("DOUBLEWELL").VECTORISED)
        self.assertFalse(model_manager.get_model("GAUSS2D").VECTORISED)
        for model_name in model_names:
            model, default_parameters, shared_parameters = selectModelAndPars(model_name)
            self.assertEqual(list(default_parameters), list(inspect.signature(model).parameters)[1:])
            if shared_parameters:
                self.assertEqual(list(inspect.signature(model).parameters)[-len(shared_parameters) :], shared_parameters)

    def test_model_lookup_returns_independent_parameter_values(self):
        model_name = FittingModel.GAUSS
        self.assertEqual(model_name, "GAUSS")
        self.assertIsInstance(model_name, str)
        registered_model = model_manager.get_model(model_name)
        self.assertIsInstance(registered_model.default_parameters, dict)
        registered_model.default_parameters["sigma"] = 99

        registered_model = model_manager.get_model(model_name)

        self.assertEqual(registered_model.default_parameters["sigma"], 5)
        self.assertEqual(registered_model.parameter("sigma"), 5)
        np.testing.assert_allclose(
            registered_model.function(np.array([0.0]), **registered_model.default_parameters),
            [1 / np.sqrt(2 * np.pi * 25)],
        )

    def test_vectorised_models_accept_2d_x_and_match_row_evaluation(self):
        x = np.tile(np.linspace(-10, 10, 21), (2, 1))
        for model_name in model_manager.get_model_names():
            definition = model_manager.get_model(model_name)
            if not definition.VECTORISED:
                continue
            row_parameters = {
                name: value if name in definition.shared_parameters else np.full((2, 1), value)
                for name, value in definition.default_parameters.items()
            }

            actual = definition.function(x, **row_parameters)
            expected = np.stack([definition.function(x[row], **definition.default_parameters) for row in range(x.shape[0])])

            np.testing.assert_allclose(actual, expected)

    def test_model_lookup_rejects_unsubscribed_models(self):
        with self.assertRaisesRegex(ValueError, "is not subscribed"):
            model_manager.get_model("NOT_SUBSCRIBED")

    def test_subscription_registers_a_string_model_name(self):
        manager = ModelManager()

        def custom_model(x, A, scale):
            return A * x / scale

        model_name = manager.subscribe("UNIT_TEST_CUSTOM_MODEL", custom_model, {"A": 1, "scale": 2}, ("scale",), VECTORISED=False)

        self.assertEqual(model_name, "UNIT_TEST_CUSTOM_MODEL")
        self.assertEqual(manager.get_model(model_name).shared_parameters, ("scale",))
        self.assertFalse(manager.get_model(model_name).VECTORISED)

    def test_subscription_requires_a_boolean_vectorisation_flag(self):
        manager = ModelManager()

        def model(x, A):
            return A * x

        with self.assertRaisesRegex(TypeError, "VECTORISED must be a bool"):
            manager.subscribe("UNIT_TEST_INVALID_VECTORISED_MODEL", model, {"A": 1}, VECTORISED=1)

    def test_subscription_rejects_duplicate_model_names(self):
        manager = ModelManager()

        def model(x, A):
            return A * x

        manager.subscribe("UNIT_TEST_DUPLICATE_MODEL", model, {"A": 1}, VECTORISED=False)

        with self.assertRaisesRegex(ValueError, "already subscribed"):
            manager.subscribe("UNIT_TEST_DUPLICATE_MODEL", model, {"A": 1}, VECTORISED=False)

    def test_subscription_rejects_unknown_shared_parameters(self):
        manager = ModelManager()

        def model(x, A):
            return A * x

        with self.assertRaisesRegex(ValueError, "not in the default parameters"):
            manager.subscribe("UNIT_TEST_INVALID_SHARED_MODEL", model, {"A": 1}, ("missing",), VECTORISED=False)

    def test_subscription_requires_an_unshared_amplitude_parameter(self):
        manager = ModelManager()

        def model_without_amplitude(x, scale):
            return x / scale

        with self.assertRaisesRegex(ValueError, "include an 'A' amplitude parameter"):
            manager.subscribe("UNIT_TEST_NO_AMPLITUDE_MODEL", model_without_amplitude, {"scale": 1}, VECTORISED=False)

        def model_with_shared_amplitude(x, A):
            return A * x

        with self.assertRaisesRegex(ValueError, "cannot be shared"):
            manager.subscribe("UNIT_TEST_SHARED_AMPLITUDE_MODEL", model_with_shared_amplitude, {"A": 1}, ("A",), VECTORISED=False)

    def test_custom_model_uses_registry_and_skips_mantid_fit(self):
        def custom_model(x, A):
            return A * x

        model_name = model_manager.subscribe("UNIT_TEST_MINUIT_ONLY_MODEL", custom_model, {"A": 1}, VECTORISED=False)
        self.addCleanup(model_manager._models.pop, model_name, None)

        model, parameters, shared_parameters = selectModelAndPars(model_name)

        self.assertIs(model, custom_model)
        self.assertEqual(parameters, {"A": 1})
        self.assertEqual(shared_parameters, [])

        class Inputs:
            fitting_model = model_name

        with patch("mvesuvio.analysis_fitting.logger.warning") as warning:
            fitProfileMantidFit(Inputs, None, None)

        warning.assert_called_once_with("Custom fitting model is not implemented in Mantid Fit. Skipping Mantid Fit ...")


class TestFittingModelParameter(unittest.TestCase):
    def setUp(self):
        self.model = ModelDefinition(lambda x, A, scale: x, {"A": 0, "scale": None}, (), False)

    def test_parameter_returns_the_default_value(self):
        self.assertEqual(self.model.parameter("A"), 0)
        self.assertIsNone(self.model.parameter("scale"))

    def test_parameter_rejects_unknown_name_and_lists_available_parameters(self):
        with self.assertRaises(ValueError) as caught:
            self.model.parameter("missing")

        self.assertEqual(
            str(caught.exception),
            "Parameter 'missing' is not defined for this model. Available parameters: A, scale",
        )
        self.assertIsInstance(caught.exception.__cause__, KeyError)


class TestModelManagerRegistry(unittest.TestCase):
    def setUp(self):
        self.manager = ModelManager()

    @staticmethod
    def model(x, A, offset, scale):
        return A * x / scale + offset

    def test_registration_copies_inputs_and_lookup_copies_defaults(self):
        defaults = {"A": 1, "offset": 0, "scale": 2}
        shared = ["scale"]
        self.manager.subscribe("model", self.model, defaults, shared, VECTORISED=True)
        defaults["scale"] = 10
        shared.append("offset")

        model = self.manager.get_model("model")
        self.assertIs(model.function, self.model)
        self.assertEqual(model.default_parameters, {"A": 1, "offset": 0, "scale": 2})
        self.assertEqual(model.shared_parameters, ("scale",))
        self.assertTrue(model.VECTORISED)
        self.assertIsInstance(model.default_parameters, dict)
        model.default_parameters["A"] = 99
        self.assertEqual(self.manager.get_model("model").parameter("A"), 1)

    def test_has_model_reports_registered_names(self):
        self.assertFalse(self.manager.has_model("beta"))
        defaults = {"A": 1, "offset": 0, "scale": 2}
        self.manager.subscribe("beta", self.model, defaults, VECTORISED=False)

        self.assertTrue(self.manager.has_model("beta"))
        self.assertFalse(self.manager.has_model("Beta"))
        self.assertFalse(self.manager.has_model("missing"))

    def test_names_are_sorted(self):
        self.assertEqual(self.manager.get_model_names(), ())
        defaults = {"A": 1, "offset": 0, "scale": 2}
        self.manager.subscribe("beta", self.model, defaults, VECTORISED=False)
        self.manager.subscribe("alpha", self.model, defaults, VECTORISED=False)

        self.assertEqual(self.manager.get_model_names(), ("alpha", "beta"))
        with self.assertRaisesRegex(ValueError, "Available options: alpha, beta"):
            self.manager.get_model("missing")
        with self.assertRaisesRegex(ValueError, "Available parameters: A, offset, scale"):
            self.manager.get_model("alpha").parameter("missing")

    def test_subscription_rejects_invalid_input_without_registering(self):
        defaults = {"A": 1, "offset": 0, "scale": 2}
        cases = (
            ("not callable", "model", None, defaults, (), TypeError, "must be callable"),
            ("non-string name", 1, self.model, defaults, (), TypeError, "name must be a string"),
            ("empty name", "", self.model, defaults, (), ValueError, "name cannot be empty"),
            ("whitespace name", "  ", self.model, defaults, (), ValueError, "name cannot be empty"),
            ("string shared", "model", self.model, defaults, "scale", TypeError, "not a string"),
            ("non-string default", "model", self.model, {"A": 1, 2: 3}, (), TypeError, "parameter names must be strings"),
            ("non-string shared", "model", self.model, defaults, (1,), TypeError, "parameter names must be strings"),
            ("duplicate shared", "model", self.model, defaults, ("scale", "scale"), ValueError, "must be unique"),
        )
        for label, name, function, parameters, shared, error, message in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(error, message):
                    self.manager.subscribe(name, function, parameters, shared, VECTORISED=False)
                self.assertEqual(self.manager.get_model_names(), ())

    def test_subscription_rejects_invalid_signatures_and_parameter_order(self):
        def no_parameters():
            pass

        def wrong_first_parameter(y, A):
            return A * y

        defaults = {"A": 1, "offset": 0, "scale": 2}
        cases = (
            ("empty signature", no_parameters, {"A": 1}, (), "x as its first parameter"),
            ("wrong first parameter", wrong_first_parameter, {"A": 1}, (), "x as its first parameter"),
            ("missing default", self.model, {"A": 1, "offset": 0}, (), "match the model signature in order"),
            ("extra default", self.model, {**defaults, "extra": 1}, (), "match the model signature in order"),
            ("reordered defaults", self.model, {"A": 1, "scale": 2, "offset": 0}, (), "match the model signature in order"),
            ("shared not last", self.model, defaults, ("offset",), "final model parameters"),
            ("shared in wrong order", self.model, defaults, ("scale", "offset"), "final model parameters"),
        )
        for label, function, parameters, shared, message in cases:
            with self.subTest(label=label):
                with self.assertRaisesRegex(ValueError, message):
                    self.manager.subscribe("model", function, parameters, shared, VECTORISED=False)
                self.assertFalse(self.manager.has_model("model"))

    def test_subscription_rejects_uninspectable_callable(self):
        for error in (TypeError("unsupported callable"), ValueError("invalid signature")):
            with self.subTest(error=type(error).__name__):
                with (
                    patch("mvesuvio.util.model_manager.signature", side_effect=error),
                    self.assertRaisesRegex(TypeError, "must have an inspectable signature") as caught,
                ):
                    self.manager.subscribe("model", self.model, {"A": 1, "offset": 0, "scale": 2}, VECTORISED=False)
                self.assertIs(caught.exception.__cause__, error)
                self.assertFalse(self.manager.has_model("model"))


if __name__ == "__main__":
    unittest.main()

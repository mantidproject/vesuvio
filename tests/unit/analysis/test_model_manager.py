import inspect
import unittest
from unittest.mock import patch

import numpy as np

from mvesuvio.analysis_fitting import fitProfileMantidFit, selectModelAndPars
from mvesuvio.default_config.experiment_template.fitting_models import register_fitting_models
from mvesuvio.util.model_manager import ModelManager, model_manager

register_fitting_models()


class TestModelManager(unittest.TestCase):
    def test_default_models_are_registered_with_ordered_parameters(self):
        model_names = model_manager.get_model_names()
        self.assertEqual(
            model_names,
            (
                "DOUBLEWELL",
                "GAUSS",
                "GAUSS2D",
                "GAUSS3D",
                "GAUSS_CNTR",
                "GCC4",
                "GCC4C6",
                "GCC4C6_CNTR",
                "GCC4_CNTR",
                "GCC6",
                "GCC6_CNTR",
            ),
        )
        for model_name in model_names:
            model, default_parameters, shared_parameters = selectModelAndPars(model_name)
            self.assertEqual(list(default_parameters), list(inspect.signature(model).parameters)[1:])
            if shared_parameters:
                self.assertEqual(list(inspect.signature(model).parameters)[-len(shared_parameters) :], shared_parameters)

    def test_model_lookup_returns_independent_parameter_values(self):
        model_name = model_manager.name("GAUSS")
        self.assertEqual(model_name, "GAUSS")
        registered_model = model_manager.get_model(model_name)
        parameters = dict(registered_model.default_parameters)
        parameters["sigma"] = 99

        registered_model = model_manager.get_model(model_name)

        self.assertEqual(registered_model.default_parameters["sigma"], 5)
        np.testing.assert_allclose(
            registered_model.function(np.array([0.0]), **registered_model.default_parameters),
            [1 / np.sqrt(2 * np.pi * 25)],
        )

    def test_name_rejects_unsubscribed_models(self):
        with self.assertRaisesRegex(ValueError, "is not subscribed"):
            model_manager.name("NOT_SUBSCRIBED")

    def test_subscription_registers_a_string_model_name(self):
        manager = ModelManager()

        def custom_model(x, A, scale):
            return A * x / scale

        model_name = manager.subscribe("UNIT_TEST_CUSTOM_MODEL", custom_model, {"A": 1, "scale": 2}, ("scale",))

        self.assertEqual(model_name, "UNIT_TEST_CUSTOM_MODEL")
        self.assertEqual(manager.get_model(model_name).shared_parameters, ("scale",))

    def test_subscription_rejects_duplicate_model_names(self):
        manager = ModelManager()

        def model(x, A):
            return A * x

        manager.subscribe("UNIT_TEST_DUPLICATE_MODEL", model, {"A": 1})

        with self.assertRaisesRegex(ValueError, "already subscribed"):
            manager.subscribe("UNIT_TEST_DUPLICATE_MODEL", model, {"A": 1})

    def test_subscription_rejects_unknown_shared_parameters(self):
        manager = ModelManager()

        def model(x, A):
            return A * x

        with self.assertRaisesRegex(ValueError, "not in the default parameters"):
            manager.subscribe("UNIT_TEST_INVALID_SHARED_MODEL", model, {"A": 1}, ("missing",))

    def test_subscription_requires_an_unshared_amplitude_parameter(self):
        manager = ModelManager()

        def model_without_amplitude(x, scale):
            return x / scale

        with self.assertRaisesRegex(ValueError, "include an 'A' amplitude parameter"):
            manager.subscribe("UNIT_TEST_NO_AMPLITUDE_MODEL", model_without_amplitude, {"scale": 1})

        def model_with_shared_amplitude(x, A):
            return A * x

        with self.assertRaisesRegex(ValueError, "cannot be shared"):
            manager.subscribe("UNIT_TEST_SHARED_AMPLITUDE_MODEL", model_with_shared_amplitude, {"A": 1}, ("A",))

    def test_custom_model_uses_registry_and_skips_mantid_fit(self):
        def custom_model(x, A):
            return A * x

        model_name = model_manager.subscribe("UNIT_TEST_MINUIT_ONLY_MODEL", custom_model, {"A": 1})
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


if __name__ == "__main__":
    unittest.main()

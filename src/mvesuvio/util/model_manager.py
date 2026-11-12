from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from inspect import signature
from typing import Any


@dataclass(frozen=True)
class FittingModel:
    function: Callable[..., Any]
    default_parameters: Mapping[str, Any]
    shared_parameters: tuple[str, ...]

    def parameter(self, name: str) -> Any:
        """Return a model parameter's default value, raising if it is not defined."""
        try:
            return self.default_parameters[name]
        except KeyError as error:
            options = ", ".join(self.default_parameters)
            raise ValueError(f"Parameter {name!r} is not defined for this model. Available parameters: {options}") from error


class ModelManager:
    def __init__(self) -> None:
        self._models: dict[str, FittingModel] = {}

    def subscribe(
        self,
        name: str,
        function: Callable[..., Any],
        default_parameters: Mapping[str, Any],
        shared_parameters: Sequence[str] = (),
    ) -> str:
        """Register a string-named model with x first, an A amplitude, ordered defaults, and shared parameters last."""
        if not callable(function):
            raise TypeError("Fitting model must be callable.")
        if not isinstance(name, str):
            raise TypeError("Fitting model name must be a string.")
        if not name.strip():
            raise ValueError("Fitting model name cannot be empty.")
        if isinstance(shared_parameters, str):
            raise TypeError("Shared fitting model parameters must be a sequence of names, not a string.")
        if name in self._models:
            raise ValueError(f"Fitting model {name!r} is already subscribed.")

        parameters = dict(default_parameters)
        shared = tuple(shared_parameters)
        if not all(isinstance(parameter, str) for parameter in parameters):
            raise TypeError("Fitting model parameter names must be strings.")
        if not all(isinstance(parameter, str) for parameter in shared):
            raise TypeError("Shared fitting model parameter names must be strings.")
        if len(set(shared)) != len(shared):
            raise ValueError("Shared fitting model parameters must be unique.")
        missing_parameters = [parameter for parameter in shared if parameter not in parameters]
        if missing_parameters:
            raise ValueError(f"Shared parameters are not in the default parameters: {', '.join(missing_parameters)}")
        if "A" not in parameters:
            raise ValueError("Fitting models must include an 'A' amplitude parameter.")
        if "A" in shared:
            raise ValueError("The 'A' amplitude parameter cannot be shared.")

        try:
            parameter_names = list(signature(function).parameters)
        except (TypeError, ValueError) as error:
            raise TypeError("Fitting model must have an inspectable signature.") from error
        if not parameter_names or parameter_names[0] != "x":
            raise ValueError("Fitting model must have x as its first parameter.")
        model_parameter_names = parameter_names[1:]
        if model_parameter_names != list(parameters):
            raise ValueError("Default parameter names must match the model signature in order.")
        if shared and model_parameter_names[-len(shared) :] != list(shared):
            raise ValueError("Shared parameters must be the final model parameters.")

        self._models[name] = FittingModel(function, parameters, shared)
        return name

    def name(self, model_name: str) -> str:
        """Return a subscribed model name, raising an error for unknown names."""
        if model_name not in self._models:
            options = ", ".join(self.get_model_names())
            raise ValueError(f"Fitting model {model_name!r} is not subscribed. Available options: {options}")

        return model_name

    def get_model(self, model_name: str) -> FittingModel:
        self.name(model_name)
        model = self._models[model_name]
        return FittingModel(model.function, dict(model.default_parameters), model.shared_parameters)

    def has_model(self, model_name: str) -> bool:
        return model_name in self._models

    def get_model_names(self) -> tuple[str, ...]:
        return tuple(sorted(self._models))


model_manager = ModelManager()

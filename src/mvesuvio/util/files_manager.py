from mvesuvio import globals
from mvesuvio.util import handle_config
from pathlib import Path
from mantid.kernel import ConfigService


class FilesManager:
    _experiment_dir: Path | None = None

    @classmethod
    def get_instrument_parameters_dir(cls) -> Path:
        return Path(handle_config.read_cached_var("caching.ipfolder"))

    @classmethod
    def get_experiment_dir(cls) -> Path:
        if cls._experiment_dir is not None:
            return cls._experiment_dir
        experiment_dir = Path(handle_config.read_cached_var("caching.inputs"))
        return experiment_dir

    @classmethod
    def set_experiment_dir(cls, path: str | Path) -> Path:
        cls._experiment_dir = Path(path)
        cls._experiment_dir.mkdir(parents=True, exist_ok=True)
        return cls._experiment_dir

    @classmethod
    def get_reduction_outputs_dir(cls) -> Path:
        reduction_outputs = cls.get_experiment_dir() / "reduction_outputs"
        reduction_outputs.mkdir(parents=True, exist_ok=True)
        return reduction_outputs

    @classmethod
    def get_reduction_inputs_dir(cls) -> Path:
        inputs_ws_dir = cls.get_experiment_dir() / "reduction_inputs"
        inputs_ws_dir.mkdir(parents=True, exist_ok=True)
        return inputs_ws_dir

    @classmethod
    def get_fitting_outputs_dir(cls) -> Path:
        fitting_outputs = cls.get_experiment_dir() / "fitting_outputs"
        fitting_outputs.mkdir(parents=True, exist_ok=True)
        return fitting_outputs

    @classmethod
    def get_fitting_inputs_dir(cls) -> Path:
        fitting_inputs_dir = cls.get_experiment_dir() / "fitting_inputs"
        fitting_inputs_dir.mkdir(parents=True, exist_ok=True)
        return fitting_inputs_dir

    @classmethod
    def get_backward_raw_filename(cls) -> str:
        return handle_config.get_experiment_name() + "_" + "raw" + "_" + globals.BACKWARD_TAG + ".nxs"

    @classmethod
    def get_backward_empty_filename(cls) -> str:
        return handle_config.get_experiment_name() + "_" + "empty" + "_" + globals.BACKWARD_TAG + ".nxs"

    @classmethod
    def get_forward_raw_filename(cls) -> str:
        return handle_config.get_experiment_name() + "_" + "raw" + "_" + globals.FORWARD_TAG + ".nxs"

    @classmethod
    def get_forward_empty_filename(cls) -> str:
        return handle_config.get_experiment_name() + "_" + "empty" + "_" + globals.FORWARD_TAG + ".nxs"

    @classmethod
    def get_mantid_log_file(cls) -> Path:
        return Path(ConfigService.getPropertiesDir(), "mantid.log")

    @classmethod
    def get_summarised_log_file(cls) -> Path:
        return cls.get_experiment_dir() / "summary.log"

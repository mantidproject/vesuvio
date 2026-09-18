from mvesuvio.globals import Tags
from mvesuvio.util import handle_config
from pathlib import Path
from mantid.kernel import ConfigService


class FilesManager:
    _experiment_dir: Path | None = None

    @staticmethod
    def get_instrument_parameters_dir() -> Path:
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
    def _get_experiment_subdir(cls, name: str) -> Path:
        subdir = cls.get_experiment_dir() / name
        subdir.mkdir(parents=True, exist_ok=True)
        return subdir

    @classmethod
    def get_reduction_outputs_dir(cls) -> Path:
        return cls._get_experiment_subdir("reduction_outputs")

    @classmethod
    def get_reduction_inputs_dir(cls) -> Path:
        return cls._get_experiment_subdir("reduction_inputs")

    @classmethod
    def get_fitting_outputs_dir(cls) -> Path:
        return cls._get_experiment_subdir("fitting_outputs")

    @classmethod
    def get_fitting_inputs_dir(cls) -> Path:
        return cls._get_experiment_subdir("fitting_inputs")

    @staticmethod
    def _get_detector_filename(tag: str, kind: str) -> str:
        return handle_config.get_experiment_name() + "_" + kind + "_" + tag + ".nxs"

    @staticmethod
    def get_backward_raw_filename() -> str:
        return FilesManager._get_detector_filename(Tags.Backward, "raw")

    @staticmethod
    def get_backward_empty_filename() -> str:
        return FilesManager._get_detector_filename(Tags.Backward, "empty")

    @staticmethod
    def get_forward_raw_filename() -> str:
        return FilesManager._get_detector_filename(Tags.Forward, "raw")

    @staticmethod
    def get_forward_empty_filename() -> str:
        return FilesManager._get_detector_filename(Tags.Forward, "empty")

    @staticmethod
    def get_mantid_log_file() -> Path:
        return Path(ConfigService.getPropertiesDir(), "mantid.log")

    @classmethod
    def get_summarised_log_file(cls) -> Path:
        return cls.get_experiment_dir() / "summary.log"

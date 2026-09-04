from mvesuvio import globals
from mvesuvio.util import handle_config
from pathlib import Path
from mantid.kernel import ConfigService


class FilesManager:
    _output_dir: Path | None = None
    _reduction_dir: Path | None = None
    _fitting_dir: Path | None = None

    @classmethod
    def get_instrument_parameters_dir(cls) -> Path:
        return Path(handle_config.read_cached_var("caching.ipfolder"))

    @classmethod
    def get_experiment_dir(cls) -> Path:
        inputs_script_path = Path(handle_config.read_cached_var("caching.inputs"))
        script_name = handle_config.get_script_name()
        return inputs_script_path.parent / script_name

    @classmethod
    def get_inputs_ws_dir(cls) -> Path:
        inputs_ws_dir = cls.get_experiment_dir() / "input_workspaces"
        inputs_ws_dir.mkdir(parents=True, exist_ok=True)
        return inputs_ws_dir

    @classmethod
    def set_outputs_dir(cls, path: str | Path) -> Path:
        cls._output_dir = Path(path)
        cls._output_dir.mkdir(parents=True, exist_ok=True)
        cls.set_outputs_reduction_dir(cls._output_dir / "reduction")
        cls.set_outputs_fitting_dir(cls._output_dir / "fitting")
        return cls._output_dir

    @classmethod
    def set_outputs_reduction_dir(cls, path: str | Path) -> Path:
        cls._reduction_dir = Path(path)
        cls._reduction_dir.mkdir(parents=True, exist_ok=True)
        return cls._reduction_dir

    @classmethod
    def set_outputs_fitting_dir(cls, path: str | Path) -> Path:
        cls._fitting_dir = Path(path)
        cls._fitting_dir.mkdir(parents=True, exist_ok=True)
        return cls._fitting_dir

    @classmethod
    def get_outputs_dir(cls) -> Path:
        if cls._output_dir is not None:
            return cls._output_dir
        return cls.get_experiment_dir() / "output_files"

    @classmethod
    def get_outputs_reduction_dir(cls) -> Path:
        if cls._reduction_dir is not None:
            return cls._reduction_dir
        return cls.get_outputs_dir() / "reduction"

    @classmethod
    def get_outputs_fitting_dir(cls) -> Path:
        if cls._fitting_dir is not None:
            return cls._fitting_dir
        return cls.get_outputs_dir() / "fitting"

    @classmethod
    def get_outputs_fitting_inputs_dir(cls) -> Path:
        fitting_inputs_dir = cls.get_outputs_fitting_dir() / "inputs"
        fitting_inputs_dir.mkdir(parents=True, exist_ok=True)
        return fitting_inputs_dir

    @classmethod
    def get_backward_raw_filename(cls) -> str:
        return handle_config.get_script_name() + "_" + "raw" + "_" + globals.BACKWARD_TAG + ".nxs"

    @classmethod
    def get_backward_empty_filename(cls) -> str:
        return handle_config.get_script_name() + "_" + "empty" + "_" + globals.BACKWARD_TAG + ".nxs"

    @classmethod
    def get_forward_raw_filename(cls) -> str:
        return handle_config.get_script_name() + "_" + "raw" + "_" + globals.FORWARD_TAG + ".nxs"

    @classmethod
    def get_forward_empty_filename(cls) -> str:
        return handle_config.get_script_name() + "_" + "empty" + "_" + globals.FORWARD_TAG + ".nxs"

    @classmethod
    def get_mantid_log_file(cls) -> Path:
        return Path(ConfigService.getPropertiesDir(), "mantid.log")

    @classmethod
    def get_summarised_log_file(cls) -> Path:
        return cls.get_outputs_dir() / "summary.log"

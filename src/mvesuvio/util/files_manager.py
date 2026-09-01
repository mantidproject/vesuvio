from mvesuvio import globals
from mvesuvio.util import handle_config
from pathlib import Path


class FilesManager:
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
    def get_outputs_dir(cls) -> Path:
        return cls.get_experiment_dir() / "output_files"

    @classmethod
    def get_outputs_reduction_dir(cls) -> Path:
        return cls.get_outputs_dir() / "reduction"

    @classmethod
    def get_outputs_fitting_dir(cls) -> Path:
        return cls.get_outputs_dir() / "fitting"

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

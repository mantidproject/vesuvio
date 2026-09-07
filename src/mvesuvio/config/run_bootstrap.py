from pathlib import Path
import runpy

import matplotlib.pyplot as plt
from mantid.api import AnalysisDataService
from mantid.simpleapi import Load
from mantid.kernel import logger

from mvesuvio.util import bootstrap_helpers
from mvesuvio.util.files_manager import FilesManager

# WARNING: Not ready to be used yet, contains basic functionality but lots of details still left to be sorted out.
# How to use this script:
# 1. Create a folder named "boot_inputs" next to this file, or change BOOTSTRAP_INPUTS_DIRECTORY to a different directory.
# 2. Inside that directory, provide a parent folder containing two subdirectories: one for backward files called "backward" and one for forward files called "forward".
# 3. Fill the "backward" and "forward" subdirectories with bootstrap workspaces, each workspace needs to end with a sample index!
# 3. Run this script from inside Manitd editor; it will load the matching workspaces for sample 1, 2, 3, ... by looking for filenames that end with the sample index.
# 4. The reduction step is then run once per sample using those loaded workspaces and writing results under a sibling "*_outputs" folder.
# This is intended as a bootstrap workflow for sample-by-sample reduction, not a fully polished user-facing CLI.

# Set this path before running this script.
BOOTSTRAP_INPUTS_DIRECTORY = Path(__file__).with_name("boot_inputs")
RUN_REDUCTION_PATH = Path(__file__).with_name("run_reduction.py")


def _run_reduction_with_injected_workspaces(back_ws_to_fit: str = "", front_ws_to_fit: str = ""):
    runpy.run_path(
        str(RUN_REDUCTION_PATH),
        run_name="__main__",
        init_globals={
            "BACK_WS_TO_FIT": back_ws_to_fit,
            "FRONT_WS_TO_FIT": front_ws_to_fit,
        },
    )


def _find_workspace_path_for_sample_index(directory: Path, sample_index: int) -> Path | None:
    sample_suffix = str(sample_index)
    return next(
        (
            path_obj
            for path_obj in sorted(directory.iterdir(), key=bootstrap_helpers.get_bootstrap_sample_sort_key)
            if path_obj.stem.endswith(sample_suffix)
        ),
        None,
    )


def run_bootstrap(bootstrap_inputs_directory: Path):
    input_dirs = bootstrap_helpers.get_bootstrap_input_directories(bootstrap_inputs_directory)
    if not input_dirs:
        return
    inputs_parent_path, inputs_backward_path, inputs_forward_path = input_dirs

    boot_outputs_dir_path = inputs_parent_path.parent / (inputs_parent_path.name + "_outputs")
    boot_outputs_dir_path.mkdir(exist_ok=True)

    sample_index = 1
    while True:
        back_ws_path = _find_workspace_path_for_sample_index(inputs_backward_path, sample_index)
        front_ws_path = _find_workspace_path_for_sample_index(inputs_forward_path, sample_index)

        if back_ws_path is None or front_ws_path is None:
            logger.warning(f"Could not find bootstrap workspaces ending in sample index {sample_index}. Stoping bootstrap procedure.")
            break

        # TODO: Replace "boot_" with sample name
        FilesManager.set_outputs_dir(Path(boot_outputs_dir_path, "boot_" + str(sample_index)))
        AnalysisDataService.clear()
        Load(Filename=str(back_ws_path), OutputWorkspace=back_ws_path.stem)
        Load(Filename=str(front_ws_path), OutputWorkspace=front_ws_path.stem)
        _run_reduction_with_injected_workspaces(back_ws_to_fit=back_ws_path.stem, front_ws_to_fit=front_ws_path.stem)
        plt.close("all")
        sample_index += 1
    return


if (__name__ == "__main__") or (__name__ == "mantidqt.widgets.codeeditor.execution"):
    run_bootstrap(BOOTSTRAP_INPUTS_DIRECTORY)

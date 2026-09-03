import re
from pathlib import Path
import runpy

import matplotlib.pyplot as plt
from mantid.api import AnalysisDataService
from mantid.simpleapi import Load
from mantid.kernel import logger

from mvesuvio.config.run_reduction import BackwardAnalysisInputs, ForwardAnalysisInputs
from mvesuvio.util import bootstrap_helpers
from mvesuvio.util import reduction_helpers


# Set this path before running this script.
BOOTSTRAP_INPUTS_DIRECTORY = Path(__file__).with_name("boot_inputs")
RUN_REDUCTION_PATH = Path(__file__).with_name("run_reduction.py")


def _run_reduction_with_injected_workspaces(back_ws_to_fit: str = "", front_ws_to_fit: str = ""):
    runpy.run_path(
        str(RUN_REDUCTION_PATH),
        init_globals={
            "BACK_WS_TO_FIT": back_ws_to_fit,
            "FRONT_WS_TO_FIT": front_ws_to_fit,
        },
    )


def run_bootstrap(bootstrap_inputs_directory: str):
    BackwardAnalysisInputs.minimal_output = True
    ForwardAnalysisInputs.minimal_output = True

    if reduction_helpers.h_ratio_is_zero_when_h_present(BackwardAnalysisInputs, ForwardAnalysisInputs):
        logger.error("Hydrogen ratio not set, run analysis on sample first before attempting bootstrap.")
        return

    input_dirs = bootstrap_helpers.get_bootstrap_input_directories(Path(bootstrap_inputs_directory))
    if not input_dirs:
        return

    inputs_parent_path, inputs_backward_path, inputs_forward_path = input_dirs

    def sorting_order(path_obj: Path):
        numeric_matches = re.findall(r"(\d+)", path_obj.stem)
        if numeric_matches:
            return (int(numeric_matches[-1]), path_obj.name)
        return (0, path_obj.name)

    boot_outputs_dir_path = inputs_parent_path.parent / (inputs_parent_path.name + "_outputs")
    boot_outputs_dir_path.mkdir(exist_ok=True)

    if BackwardAnalysisInputs.run_this_scattering_type and ForwardAnalysisInputs.run_this_scattering_type:
        procedure_output_dir_path = boot_outputs_dir_path / "joint"
        procedure_output_dir_path.mkdir(exist_ok=True)

        sample_pairs = bootstrap_helpers.pair_bootstrap_sample_paths(
            list(inputs_backward_path.iterdir()),
            list(inputs_forward_path.iterdir()),
        )
        for back_ws_path, front_ws_path in sample_pairs:
            common_prefix = bootstrap_helpers.get_common_prefix_of_bootstrap_sample_names(back_ws_path.stem, front_ws_path.stem)
            if not common_prefix:
                return

            sample_index = bootstrap_helpers.get_bootstrap_sample_sort_key(back_ws_path)[0]
            sample_output_directory = procedure_output_dir_path / (common_prefix + "_joint_" + str(sample_index))
            bootstrap_helpers.update_sample_inputs_outputs(
                back_inputs=BackwardAnalysisInputs,
                front_inputs=ForwardAnalysisInputs,
                back_ws_path=back_ws_path,
                front_ws_path=front_ws_path,
                output_path=sample_output_directory,
            )

            AnalysisDataService.clear()
            Load(Filename=str(back_ws_path), OutputWorkspace=back_ws_path.stem)
            Load(Filename=str(front_ws_path), OutputWorkspace=front_ws_path.stem)
            _run_reduction_with_injected_workspaces(back_ws_to_fit=back_ws_path.stem, front_ws_to_fit=front_ws_path.stem)
            plt.close("all")
        return

    if BackwardAnalysisInputs.run_this_scattering_type:
        procedure_output_dir_path = boot_outputs_dir_path / "backward"
        procedure_output_dir_path.mkdir(exist_ok=True)

        for back_ws_path in sorted(inputs_backward_path.iterdir(), key=sorting_order):
            sample_index = bootstrap_helpers.get_bootstrap_sample_sort_key(back_ws_path)[0]
            sample_output_directory = procedure_output_dir_path / (back_ws_path.stem.split("_")[0] + "_bckwd_" + str(sample_index))
            bootstrap_helpers.update_sample_inputs_outputs(
                back_inputs=BackwardAnalysisInputs,
                front_inputs=ForwardAnalysisInputs,
                back_ws_path=back_ws_path,
                front_ws_path=None,
                output_path=sample_output_directory,
            )

            AnalysisDataService.clear()
            Load(Filename=str(back_ws_path), OutputWorkspace=back_ws_path.stem)
            _run_reduction_with_injected_workspaces(back_ws_to_fit=back_ws_path.stem)
            plt.close("all")
        return

    if ForwardAnalysisInputs.run_this_scattering_type:
        procedure_output_dir_path = boot_outputs_dir_path / "forward"
        procedure_output_dir_path.mkdir(exist_ok=True)

        for front_ws_path in sorted(inputs_forward_path.iterdir(), key=sorting_order):
            sample_index = bootstrap_helpers.get_bootstrap_sample_sort_key(front_ws_path)[0]
            sample_output_directory = procedure_output_dir_path / (front_ws_path.stem.split("_")[0] + "_fwd_" + str(sample_index))
            bootstrap_helpers.update_sample_inputs_outputs(
                back_inputs=BackwardAnalysisInputs,
                front_inputs=ForwardAnalysisInputs,
                back_ws_path=None,
                front_ws_path=front_ws_path,
                output_path=sample_output_directory,
            )

            AnalysisDataService.clear()
            Load(Filename=str(front_ws_path), OutputWorkspace=front_ws_path.stem)
            _run_reduction_with_injected_workspaces(front_ws_to_fit=front_ws_path.stem)
            plt.close("all")


if __name__ == "__main__":
    run_bootstrap(BOOTSTRAP_INPUTS_DIRECTORY)

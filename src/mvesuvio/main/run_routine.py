from mvesuvio.analysis_fitting import FitInYSpace
from mvesuvio.util import handle_config
from mvesuvio.util.analysis_helpers import (
    calculate_resolution,
    fix_profile_parameters,
    isolate_lighest_mass_data,
    load_raw_and_empty_from_path,
    cropAndMaskWorkspace,
    calculate_h_ratio,
    ws_history_matches_inputs,
    save_ws_from_load_vesuvio,
    is_hydrogen_present,
    create_profiles_table,
    create_table_for_hydrogen_to_mass_ratios,
    print_table_workspace,
    convert_to_list_of_spectrum_numbers,
)
from mvesuvio.analysis_reduction import VesuvioAnalysisRoutine
from mvesuvio import globals

from mantid.api import AnalysisDataService
from mantid.simpleapi import mtd, RenameWorkspace, SaveAscii, Load
from mantid.kernel import logger
from mantid.api import AlgorithmFactory, AlgorithmManager

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import importlib
import sys
import dill  # To convert constraints to string
import re
import os
import traceback


class Runner:
    def __init__(
        self,
        override_back_workspace="",
        override_front_workspace="",
        bootstrap_inputs_directory="",
        minimal_output=False,
        output_directory="",
        running_tests=False,
    ) -> None:
        self.running_tests = running_tests
        self.inputs_path = Path(handle_config.read_config_var("caching.inputs"))
        self.override_back_workspace = override_back_workspace
        self.override_front_workspace = override_front_workspace
        self.bootstrap_inputs_directory = bootstrap_inputs_directory
        self.minimal_output = minimal_output
        self.mantid_log_file = "mantid.log"

        # I/O paths
        inputs_script_path = Path(handle_config.read_config_var("caching.inputs"))
        script_name = handle_config.get_script_name()
        self.experiment_path = inputs_script_path.parent / script_name
        self.input_ws_path = self.experiment_path / "input_workspaces"
        self.input_ws_path.mkdir(parents=True, exist_ok=True)
        if not output_directory:
            self.output_directory = Path(self.experiment_path) / "output_files"
        else:
            self.output_directory = Path(output_directory)
        self.reduction_directory = self.output_directory / "reduction"
        self.fitting_directory = self.output_directory / "fitting"

        self.analysis_result = None
        self.fitting_result = None

        self.setup()

    def setup(self):
        ai = self.import_from_inputs()

        self.bckwd_ai = ai.BackwardAnalysisInputs
        self.fwd_ai = ai.ForwardAnalysisInputs

        self.fwd_ai.minimal_output = self.minimal_output
        self.bckwd_ai.minimal_output = self.minimal_output

        self.fwd_ai.name = handle_config.get_script_name() + "_" + globals.FORWARD_TAG
        self.bckwd_ai.name = handle_config.get_script_name() + "_" + globals.BACKWARD_TAG

        self.fwd_ai.override_input_workspace = self.override_front_workspace
        self.bckwd_ai.override_input_workspace = self.override_back_workspace
        self.fwd_ai.minimal_output = self.minimal_output
        self.bckwd_ai.minimal_output = self.minimal_output

        self.update_ws_names_from_override_input_workspaces()

    def import_from_inputs(self):
        name = "analysis_inputs"
        spec = importlib.util.spec_from_file_location(name, self.inputs_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    def run_bootstrap(self):
        if self.h_ratio_is_zero_when_h_present():
            logger.error("Hydrogen ratio not set, run analysis on sample first before attempting bootstrap.")
            return

        input_dirs = self.get_bootstrap_input_directories()
        if not input_dirs:
            return
        inputs_parent_path, inputs_backward_path, inputs_forward_path = input_dirs

        # Sort files based on last character, assumed to be integer
        def sorting_order(p):
            return int(p.stem[-1])

        boot_outputs_dir_path = inputs_parent_path.parent / (inputs_parent_path.name + "_outputs")
        boot_outputs_dir_path.mkdir(exist_ok=True)

        if self.bckwd_ai.run_this_scattering_type and self.fwd_ai.run_this_scattering_type:
            procedure_output_dir_path = boot_outputs_dir_path / "joint"
            procedure_output_dir_path.mkdir(exist_ok=True)

            for back_ws_path, front_ws_path in zip(
                sorted(inputs_backward_path.iterdir(), key=sorting_order), sorted(inputs_forward_path.iterdir(), key=sorting_order)
            ):
                common_prefix = self.get_common_prefix_of_bootstrap_sample_names(back_ws_path.stem, front_ws_path.stem)
                if not common_prefix:
                    return
                sample_output_directory = procedure_output_dir_path / (common_prefix + "_joint_" + back_ws_path.stem[-1])

                self.update_sample_inputs_outputs(back_ws_path, front_ws_path, sample_output_directory)
                AnalysisDataService.clear()
                self.run()
                plt.close("all")  # Close any plots that might still be open
            return

        if self.bckwd_ai.run_this_scattering_type:
            procedure_output_dir_path = boot_outputs_dir_path / "backward"
            procedure_output_dir_path.mkdir(exist_ok=True)
            for back_ws_path in sorted(inputs_backward_path.iterdir(), key=sorting_order):
                sample_output_directory = procedure_output_dir_path / (back_ws_path.stem.split("_")[0] + "_bckwd_" + back_ws_path.stem[-1])

                self.update_sample_inputs_outputs(back_ws_path, None, sample_output_directory)
                AnalysisDataService.clear()
                self.run()
                plt.close("all")  # Close any plots that might still be open
            return

        if self.fwd_ai.run_this_scattering_type:
            procedure_output_dir_path = boot_outputs_dir_path / "forward"
            procedure_output_dir_path.mkdir(exist_ok=True)
            for front_ws_path in sorted(inputs_forward_path.iterdir(), key=sorting_order):
                sample_output_directory = procedure_output_dir_path / (front_ws_path.stem.split("_")[0] + "_fwd_" + front_ws_path.stem[-1])

                self.update_sample_inputs_outputs(None, front_ws_path, sample_output_directory)
                AnalysisDataService.clear()
                self.run()
                plt.close("all")  # Close any plots that might still be open
            return
        return

    def update_sample_inputs_outputs(self, back_ws_path, front_ws_path, output_path):
        output_path.mkdir(exist_ok=True)
        self.update_output_directory(output_path)

        self.bckwd_ai.override_input_workspace = str(back_ws_path.absolute()) if back_ws_path is not None else ""
        self.fwd_ai.override_input_workspace = str(front_ws_path.absolute()) if front_ws_path is not None else ""
        self.update_ws_names_from_override_input_workspaces()

        self.bckwd_ai.show_plots = False
        self.fwd_ai.show_plots = False
        return

    def get_common_prefix_of_bootstrap_sample_names(self, back_ws_name, front_ws_name):
        if back_ws_name == front_ws_name:
            logger.error(f"Bootstrap Error: backward and forward inputs should not have the same name: {front_ws_name}.")
            return ""
        if back_ws_name[-1] != front_ws_name[-1]:
            logger.error(f"Bootstrap Error: inputs {back_ws_name} and {front_ws_name} do not have the same last character.")
            return ""

        def longest_common_prefix(s1, s2):
            return s1[: next((i for i, (a, b) in enumerate(zip(s1, s2)) if a != b or a == "_" or b == "_"), min(len(s1), len(s2)))]

        common_prefix = longest_common_prefix(back_ws_name, front_ws_name)
        if not common_prefix:
            logger.error(f"Bootstrap Error: inputs {back_ws_name} and {front_ws_name} do not have a common prefix.")
            return ""
        return common_prefix

    def get_bootstrap_input_directories(self):
        inputs_dir_path = Path(self.bootstrap_inputs_directory)
        if not inputs_dir_path.is_dir():
            logger.error("The inputs directory path provided for bootstrap is not a directory.")
            return ()

        inputs_backward_path = inputs_dir_path / "backward"
        if not inputs_backward_path.exists():
            inputs_backward_path.mkdir(exist_ok=True)
            logger.error(f"Created backward directory. Please place your backward samples here: {str(inputs_backward_path)}")
            return ()

        inputs_forward_path = inputs_dir_path / "forward"
        if not inputs_forward_path.exists():
            inputs_forward_path.mkdir(exist_ok=True)
            logger.error(f"Created forward directory. Please place your forward samples here: {str(inputs_forward_path)}")
            return ()
        return (inputs_dir_path, inputs_backward_path, inputs_forward_path)

    def update_output_directory(self, path):
        self.output_directory = path
        self.reduction_directory = self.output_directory / "reduction"
        self.fitting_directory = self.output_directory / "fitting"

    def update_ws_names_from_override_input_workspaces(self):
        if self.fwd_ai.override_input_workspace:
            self.fwd_ai.name = Path(self.fwd_ai.override_input_workspace).stem

        if self.bckwd_ai.override_input_workspace:
            self.bckwd_ai.name = Path(self.bckwd_ai.override_input_workspace).stem

    def run(self):
        if not self.bckwd_ai.run_this_scattering_type and not self.fwd_ai.run_this_scattering_type:
            return

        # Erase previous log
        # Not working on Windows due to shared file locks
        if os.name == "posix":
            with open(self.mantid_log_file, "w") as file:
                file.write("")

        if self.runAnalysisFitting():
            self.make_summarised_log_file()
            return self.analysis_result, self.fitting_result

        if self.h_ratio_is_zero_when_h_present():
            self.run_estimate_h_ratio()
            self.make_summarised_log_file()
            return

        self.runAnalysisRoutine()
        self.runAnalysisFitting()

        self.make_summarised_log_file()
        # Return results used only in tests
        return self.analysis_result, self.fitting_result

    def h_ratio_is_zero_when_h_present(self):
        if self.bckwd_ai.run_this_scattering_type:
            if is_hydrogen_present(self.fwd_ai.masses):
                if self.bckwd_ai.intensity_ratio_of_hydrogen_to_chosen_mass == 0:
                    return True
            else:
                logger.warning("Ignoring Hydrogen ratio because not detected in masses.")
                self.bckwd_ai.intensity_ratio_of_hydrogen_to_chosen_mass = 0
        return False

    def make_summarised_log_file(self):
        pattern = re.compile(r"^\d{4}-\d{2}-\d{2}")

        log_file_save_path = self.make_log_file_name()

        try:
            with open(self.mantid_log_file, "r") as infile, open(log_file_save_path, "w") as outfile:
                for line in infile:
                    if "VesuvioAnalysisRoutine" in line:
                        outfile.write(line)

                    if "Notice Python" in line:  # For Fitting notices
                        outfile.write(line)

                    if not pattern.match(line):
                        outfile.write(line)
        except OSError:
            logger.error("Mantid log file not available. Unable to produce a summarized log file for this routine.")
        return

    def make_log_file_name(self):
        filename = ""
        if self.bckwd_ai.run_this_scattering_type:
            filename += "bckwd_" + self.bckwd_ai.fitting_model
        if self.fwd_ai.run_this_scattering_type:
            filename += "fwd_" + self.fwd_ai.fitting_model
        return self.experiment_path / (filename + ".log")

    def runAnalysisFitting(self):
        success = False
        if self.bckwd_ai.fit_in_y_space:
            try:
                success |= self.run_y_space_reduction_and_fit(self.bckwd_ai)
            except Exception:
                success |= True
                logger.error(traceback.format_exc())
        if self.fwd_ai.fit_in_y_space:
            try:
                success |= self.run_y_space_reduction_and_fit(self.fwd_ai)
            except Exception:
                success |= True
                logger.error(traceback.format_exc())
        return success

    def run_y_space_reduction_and_fit(self, ai):
        ws_to_fit_name = ai.name + "_" + str(ai.number_of_iterations_for_corrections)
        try:
            ws_to_fit = mtd[ws_to_fit_name]
            ws_to_fit_ncps = mtd[ws_to_fit_name + "_ncp_group"]
        except KeyError:
            logger.warning(f"Either {ws_to_fit_name} or {ws_to_fit_name}_ncp_group not found. Skipping fitting in Y-Space.")
            return False

        # TODO: Move resolution calculation to end of analysis, instead of beggining of fitting
        ws_resolution = calculate_resolution(min(ai.masses), ws_to_fit, ai.range_for_rebinning_in_y_space)
        # NOTE: Resolution workspace is useful for scientists outside mantid
        self.fitting_directory.mkdir(exist_ok=True)
        SaveAscii(ws_resolution.name(), str(self.fitting_directory / ws_resolution.name()))

        ws_lighest_data, ws_lighest_ncp = isolate_lighest_mass_data(ws_to_fit, ws_to_fit_ncps, ai.subtract_calculated_fse_from_data)
        self.fitting_result = FitInYSpace(ai, ws_lighest_data, ws_lighest_ncp, ws_resolution, outputs_dir=self.fitting_directory).run()
        return True

    def runAnalysisRoutine(self):
        if self.bckwd_ai.override_input_workspace:
            if not self.bckwd_ai.override_input_workspace.endswith(".nxs"):
                logger.error("Currently only nexus files are supported for overriding input workspaces. Try again with a .nxs file.")
                return
        if self.fwd_ai.override_input_workspace:
            if not self.fwd_ai.override_input_workspace.endswith(".nxs"):
                logger.error("Currently only nexus files are supported for overriding input workspaces. Try again with a .nxs file.")
                return

        if self.bckwd_ai.run_this_scattering_type and self.fwd_ai.run_this_scattering_type:
            # Either both overrides are empty or they're both set to a workspace path
            if (not self.bckwd_ai.override_input_workspace) != (not self.fwd_ai.override_input_workspace):
                logger.error(
                    "Running joint analysis requires that both backward and forward workspaces are explicitly passed as inputs to override default input workspaces."
                )
                return
            self.run_joint_analysis()
            return

        if self.bckwd_ai.run_this_scattering_type:
            if not self.bckwd_ai.override_input_workspace and self.fwd_ai.override_input_workspace:
                logger.error(
                    "Forward input workspace was explicitly set but analysis is running backward routine. Please provide input workspace for backward analysis."
                )
                return

            self.run_single_analysis(self.bckwd_ai)
            return

        if self.fwd_ai.run_this_scattering_type:
            if not self.fwd_ai.override_input_workspace and self.bckwd_ai.override_input_workspace:
                logger.error(
                    "Backward input workspace was explicitly set but analysis is running forward routine. Please provide input workspace for forward analysis."
                )
                return
            self.run_single_analysis(self.fwd_ai)
            return
        return

    def run_single_analysis(self, ai):
        AnalysisDataService.clear()
        alg = self._create_analysis_algorithm(ai)
        alg.execute()
        self.analysis_result = alg
        return

    def run_joint_analysis(self):
        AnalysisDataService.clear()
        back_alg = self._create_analysis_algorithm(self.bckwd_ai)
        front_alg = self._create_analysis_algorithm(self.fwd_ai)
        self.run_joint_algs(back_alg, front_alg)
        return

    @classmethod
    def run_joint_algs(cls, back_alg, front_alg):
        back_alg.execute()

        incoming_means_table = mtd[back_alg.getPropertyValue("OutputMeansTable")]
        h_ratio = back_alg.getProperty("HRatioToChosenMass").value

        assert incoming_means_table is not None, "Means table from backward routine not correctly accessed."
        assert h_ratio is not None, "H ratio from backward routine not correctly accesssed."

        receiving_profiles_table = mtd[front_alg.getPropertyValue("InputProfiles")]

        fixed_profiles_table = fix_profile_parameters(incoming_means_table, receiving_profiles_table, h_ratio)

        # Update original profiles table
        RenameWorkspace(fixed_profiles_table, receiving_profiles_table.name())
        print_table_workspace(mtd[receiving_profiles_table.name()])
        # Even if the name is the same, need to trigger update
        front_alg.setPropertyValue("InputProfiles", receiving_profiles_table.name())

        front_alg.execute()
        return

    def run_estimate_h_ratio(self):
        """
        Used when H is present and H to first mass ratio is not known.
        Preliminary forward scattering is run to get rough estimate of H to first mass ratio.
        Runs iterative procedure with alternating back and forward scattering.
        """

        if not self.running_tests:
            try:
                userInput = input("\nHydrogen intensity ratio to lowest mass is not set. Press Enter to start estimate procedure.")
                if not userInput == "":
                    raise EOFError

            except EOFError:
                logger.error("Estimation of Hydrogen intensity ratio interrupted.")
                return

        chosen_mass = self.bckwd_ai.masses[self.bckwd_ai.chosen_mass_index]

        table_h_ratios = create_table_for_hydrogen_to_mass_ratios()

        back_alg = self._create_analysis_algorithm(self.bckwd_ai)
        front_alg = self._create_analysis_algorithm(self.fwd_ai)

        front_alg.execute()

        means_table = mtd[front_alg.getPropertyValue("OutputMeansTable")]
        current_ratio = calculate_h_ratio(means_table, chosen_mass)

        table_h_ratios.addRow([current_ratio])
        previous_ratio = np.nan

        while not np.isclose(current_ratio, previous_ratio, rtol=0.01):
            back_alg.setProperty("HRatioToChosenMass", current_ratio)
            self.run_joint_algs(back_alg, front_alg)

            previous_ratio = current_ratio

            means_table = mtd[front_alg.getPropertyValue("OutputMeansTable")]
            current_ratio = calculate_h_ratio(means_table, chosen_mass)

            table_h_ratios.addRow([current_ratio])

            SaveAscii(table_h_ratios.name(), str(self.output_directory / table_h_ratios.name()))

        logger.notice("\nProcedute to estimate Hydrogen ratio finished.")
        print_table_workspace(table_h_ratios)
        return

    def _create_analysis_algorithm(self, ai):
        if not ai.override_input_workspace:
            raw_path, empty_path = self._save_ws_if_not_on_path(ai)
            ws = load_raw_and_empty_from_path(
                raw_path=raw_path,
                empty_path=empty_path,
                tof_binning=ai.time_of_flight_binning,
                name=ai.name,
                raw_scale_factor=ai.scale_raw_workspace,
                empty_scale_factor=ai.scale_empty_workspace,
                raw_minus_empty=ai.subtract_empty_workspace_from_raw,
            )
            first_detector, last_detector = [int(s) for s in ai.detectors.split("-")]
            input_ws = cropAndMaskWorkspace(
                ws,
                firstSpec=first_detector,
                lastSpec=last_detector,
                maskedDetectors=ai.mask_detectors,
                maskTOFRange=ai.mask_time_of_flight_range,
            )
        else:
            input_ws = Load(Filename=str(Path(ai.override_input_workspace).absolute()), OutputWorkspace=ai.name)

        profiles_table = create_profiles_table(input_ws.name() + "_initial_parameters", ai)
        print_table_workspace(profiles_table)
        ipFilesPath = Path(handle_config.read_config_var("caching.ipfolder"))
        kwargs = {
            "InputWorkspace": input_ws.name(),
            "InputProfiles": profiles_table.name(),
            "InstrumentParametersFile": str(ipFilesPath / ai.instrument_parameters_file),
            "ChosenMassIndex": ai.chosen_mass_index if hasattr(ai, "chosen_mass_index") else 0,
            "HRatioToChosenMass": ai.intensity_ratio_of_hydrogen_to_chosen_mass
            if hasattr(ai, "intensity_ratio_of_hydrogen_to_chosen_mass")
            else 0,
            "NumberOfIterations": int(ai.number_of_iterations_for_corrections),
            "InvalidDetectors": convert_to_list_of_spectrum_numbers(ai.mask_detectors),
            "MultipleScatteringCorrection": ai.do_multiple_scattering_correction,
            "SampleShapeXml": ai.sample_shape_xml,
            "GammaCorrection": ai.do_gamma_correction,
            "ModeRunning": "BACKWARD" if self.get_scattering_type(ai) == globals.BACKWARD_TAG else "FORWARD",
            "TransmissionGuess": ai.transmission_guess,
            "MultipleScatteringOrder": int(ai.multiple_scattering_order),
            "NumberOfEvents": int(ai.multiple_scattering_number_of_events),
            "Constraints": str(dill.dumps(ai.constraints)),
            "ResultsPath": str(self.reduction_directory.absolute()),
            "MinimalOutputFiles": ai.minimal_output,
            "OutputMeansTable": " Final_Means",
        }

        if self.running_tests:
            alg = VesuvioAnalysisRoutine()
        else:
            AlgorithmFactory.subscribe(VesuvioAnalysisRoutine)
            alg = AlgorithmManager.createUnmanaged("VesuvioAnalysisRoutine")

        alg.initialize()
        alg.setProperties(kwargs)
        return alg

    def _save_ws_if_not_on_path(self, ai):
        raw_name = handle_config.get_script_name() + "_" + "raw" + "_" + self.get_scattering_type(ai) + ".nxs"
        empty_name = handle_config.get_script_name() + "_" + "empty" + "_" + self.get_scattering_type(ai) + ".nxs"

        raw_path = self.input_ws_path / raw_name
        empty_path = self.input_ws_path / empty_name

        ip_files_path = Path(handle_config.read_config_var("caching.ipfolder"))

        if not ws_history_matches_inputs(ai.runs, ai.mode, ai.instrument_parameters_file, raw_path):
            save_ws_from_load_vesuvio(ai.runs, ai.mode, str(ip_files_path / ai.instrument_parameters_file), raw_path)

        if not ws_history_matches_inputs(ai.empty_runs, ai.mode, ai.instrument_parameters_file, empty_path):
            save_ws_from_load_vesuvio(ai.empty_runs, ai.mode, str(ip_files_path / ai.instrument_parameters_file), empty_path)
        return raw_path, empty_path

    def get_scattering_type(self, ai):
        if ai.__name__ in ["BackwardAnalysisInputs"]:
            return globals.BACKWARD_TAG
        elif ai.__name__ in ["ForwardAnalysisInputs"]:
            return globals.FORWARD_TAG
        else:
            raise ValueError(f"Input class for workspace not valid: {ai.__name__}")


if __name__ == "__main__":
    Runner().run()

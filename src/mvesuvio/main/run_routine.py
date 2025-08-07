from mvesuvio.analysis_fitting import FitInYSpace
from mvesuvio.util import handle_config
from mvesuvio.util.analysis_helpers import (
    calculate_resolution,
    fix_profile_parameters,
    isolate_lighest_mass_data,
    load_raw_and_empty_from_path,
    cropAndMaskWorkspace,
    calculate_h_ratio,
    name_for_starting_ws,
    scattering_type,
    ws_history_matches_inputs,
    save_ws_from_load_vesuvio,
    is_hydrogen_present,
    create_profiles_table,
    create_table_for_hydrogen_to_mass_ratios,
    print_table_workspace,
    convert_to_list_of_spectrum_numbers,
)
from mvesuvio.analysis_reduction import VesuvioAnalysisRoutine

from mantid.api import AnalysisDataService
from mantid.simpleapi import mtd, RenameWorkspace, SaveAscii
from mantid.kernel import logger
from mantid.api import AlgorithmFactory, AlgorithmManager

import numpy as np
from pathlib import Path
import importlib
import sys
import dill  # To convert constraints to string
import re
import os


class Runner:
    def __init__(self, running_tests=False) -> None:
        self.running_tests = running_tests
        self.inputs_path = Path(handle_config.read_config_var("caching.inputs"))
        self.setup()

    def setup(self):
        ai = self.import_from_inputs()

        self.bckwd_ai = ai.BackwardAnalysisInputs
        self.fwd_ai = ai.ForwardAnalysisInputs

        # Names of workspaces to check if they exist to skip analysis
        self.ws_to_fit_y_space = []
        self.classes_to_fit_y_space = []
        for ai_cls in [self.bckwd_ai, self.fwd_ai]:
            if ai_cls.fit_in_y_space:
                self.ws_to_fit_y_space.append(name_for_starting_ws(ai_cls) + "_" + str(ai_cls.number_of_iterations_for_corrections))
                self.classes_to_fit_y_space.append(ai_cls)

        self.analysis_result = None
        self.fitting_result = None

        # I/O paths
        inputs_script_path = Path(handle_config.read_config_var("caching.inputs"))
        script_name = handle_config.get_script_name()
        self.experiment_path = inputs_script_path.parent / script_name
        self.input_ws_path = self.experiment_path / "input_workspaces"
        self.input_ws_path.mkdir(parents=True, exist_ok=True)

        # TODO: remove this by fixing circular import
        self.fwd_ai.name = name_for_starting_ws(self.fwd_ai)
        self.bckwd_ai.name = name_for_starting_ws(self.bckwd_ai)

        self.mantid_log_file = "mantid.log"

    def import_from_inputs(self):
        name = "analysis_inputs"
        spec = importlib.util.spec_from_file_location(name, self.inputs_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module

    def run(self):
        if not self.bckwd_ai.run_this_scattering_type and not self.fwd_ai.run_this_scattering_type:
            return

        # Erase previous log
        # Not working on Windows due to shared file locks
        if os.name == "posix":
            with open(self.mantid_log_file, "w") as file:
                file.write("")

        # If any ws for y fit already loaded
        wsInMtd = [ws in mtd for ws in self.ws_to_fit_y_space]  # Bool list
        if (len(wsInMtd) > 0) and all(wsInMtd):
            self.runAnalysisFitting()
            self.make_summarised_log_file()
            return self.analysis_result, self.fitting_result

        if self.bckwd_ai.run_this_scattering_type:
            if is_hydrogen_present(self.fwd_ai.masses) & (self.bckwd_ai.intensity_ratio_of_hydrogen_to_lowest_mass == 0):
                self.run_estimate_h_ratio()
                return

            # TODO: make this automatic
            assert is_hydrogen_present(self.fwd_ai.masses) != (self.bckwd_ai.intensity_ratio_of_hydrogen_to_lowest_mass == 0), (
                "No Hydrogen detected, intensity_ratio_of_hydrogen_to_lowest_mass has to be set to 0"
            )

        self.runAnalysisRoutine()
        self.runAnalysisFitting()

        # Return results used only in tests
        self.make_summarised_log_file()
        return self.analysis_result, self.fitting_result

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
        for wsName, i_cls in zip(self.ws_to_fit_y_space, self.classes_to_fit_y_space):
            ws_lighest_data, ws_lighest_ncp = isolate_lighest_mass_data(
                mtd[wsName], mtd[wsName + "_ncp_group"], i_cls.subtract_calculated_fse_from_data
            )
            # TODO: Move resolution calculation to end of analysis, instead of beggining of fitting
            ws_resolution = calculate_resolution(min(i_cls.masses), mtd[wsName], i_cls.range_for_rebinning_in_y_space)
            # NOTE: Set saving path like this for now
            i_cls.save_path = self.experiment_path / "output_files" / "fitting"
            i_cls.save_path.mkdir(exist_ok=True, parents=True)
            # NOTE: Resolution workspace is useful for scientists outside mantid
            SaveAscii(ws_resolution.name(), str(i_cls.save_path / ws_resolution.name()))
            self.fitting_result = FitInYSpace(i_cls, ws_lighest_data, ws_lighest_ncp, ws_resolution).run()
        return

    def runAnalysisRoutine(self):
        if self.bckwd_ai.run_this_scattering_type and self.fwd_ai.run_this_scattering_type:
            self.run_joint_analysis()
            return
        if self.bckwd_ai.run_this_scattering_type:
            self.run_single_analysis(self.bckwd_ai)
            return
        if self.fwd_ai.run_this_scattering_type:
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
        h_ratio = back_alg.getProperty("HRatioToLowestMass").value

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

        table_h_ratios = create_table_for_hydrogen_to_mass_ratios()

        back_alg = self._create_analysis_algorithm(self.bckwd_ai)
        front_alg = self._create_analysis_algorithm(self.fwd_ai)

        front_alg.execute()

        means_table = mtd[front_alg.getPropertyValue("OutputMeansTable")]
        current_ratio = calculate_h_ratio(means_table)

        table_h_ratios.addRow([current_ratio])
        previous_ratio = np.nan

        while not np.isclose(current_ratio, previous_ratio, rtol=0.01):
            back_alg.setProperty("HRatioToLowestMass", current_ratio)
            self.run_joint_algs(back_alg, front_alg)

            previous_ratio = current_ratio

            means_table = mtd[front_alg.getPropertyValue("OutputMeansTable")]
            current_ratio = calculate_h_ratio(means_table)

            table_h_ratios.addRow([current_ratio])

            SaveAscii(table_h_ratios.name(), str(self.experiment_path / "output_files" / table_h_ratios.name()))

        logger.notice("\nProcedute to estimate Hydrogen ratio finished.")
        print_table_workspace(table_h_ratios)
        return

    def _create_analysis_algorithm(self, ai):
        raw_path, empty_path = self._save_ws_if_not_on_path(ai)

        ws = load_raw_and_empty_from_path(
            raw_path=raw_path,
            empty_path=empty_path,
            tof_binning=ai.time_of_flight_binning,
            name=name_for_starting_ws(ai),
            raw_scale_factor=ai.scale_raw_workspace,
            empty_scale_factor=ai.scale_empty_workspace,
            raw_minus_empty=ai.subtract_empty_workspace_from_raw,
        )
        first_detector, last_detector = [int(s) for s in ai.detectors.split("-")]
        cropedWs = cropAndMaskWorkspace(
            ws,
            firstSpec=first_detector,
            lastSpec=last_detector,
            maskedDetectors=ai.mask_detectors,
            maskTOFRange=ai.mask_time_of_flight_range,
        )
        profiles_table = create_profiles_table(cropedWs.name() + "_initial_parameters", ai)
        print_table_workspace(profiles_table)
        ipFilesPath = Path(handle_config.read_config_var("caching.ipfolder"))
        kwargs = {
            "InputWorkspace": cropedWs.name(),
            "InputProfiles": profiles_table.name(),
            "InstrumentParametersFile": str(ipFilesPath / ai.instrument_parameters_file),
            "HRatioToLowestMass": ai.intensity_ratio_of_hydrogen_to_lowest_mass
            if hasattr(ai, "intensity_ratio_of_hydrogen_to_lowest_mass")
            else 0,
            "NumberOfIterations": int(ai.number_of_iterations_for_corrections),
            "InvalidDetectors": convert_to_list_of_spectrum_numbers(ai.mask_detectors),
            "MultipleScatteringCorrection": ai.do_multiple_scattering_correction,
            "SampleShapeXml": ai.sample_shape_xml,
            "GammaCorrection": ai.do_gamma_correction,
            "ModeRunning": scattering_type(ai),
            "TransmissionGuess": ai.transmission_guess,
            "MultipleScatteringOrder": int(ai.multiple_scattering_order),
            "NumberOfEvents": int(ai.multiple_scattering_number_of_events),
            "Constraints": str(dill.dumps(ai.constraints)),
            "ResultsPath": str(self.experiment_path / "output_files" / "reduction"),
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

    def _save_ws_if_not_on_path(self, load_ai):
        scatteringType = scattering_type(load_ai).lower()
        scriptName = handle_config.get_script_name()

        rawWSName = scriptName + "_" + "raw" + "_" + scatteringType + ".nxs"
        emptyWSName = scriptName + "_" + "empty" + "_" + scatteringType + ".nxs"

        rawPath = self.input_ws_path / rawWSName
        emptyPath = self.input_ws_path / emptyWSName

        ipFilesPath = Path(handle_config.read_config_var("caching.ipfolder"))

        if not ws_history_matches_inputs(load_ai.runs, load_ai.mode, load_ai.instrument_parameters_file, rawPath):
            save_ws_from_load_vesuvio(load_ai.runs, load_ai.mode, str(ipFilesPath / load_ai.instrument_parameters_file), rawPath)

        if not ws_history_matches_inputs(load_ai.empty_runs, load_ai.mode, load_ai.instrument_parameters_file, emptyPath):
            save_ws_from_load_vesuvio(load_ai.empty_runs, load_ai.mode, str(ipFilesPath / load_ai.instrument_parameters_file), emptyPath)
        return rawPath, emptyPath


if __name__ == "__main__":
    Runner().run()

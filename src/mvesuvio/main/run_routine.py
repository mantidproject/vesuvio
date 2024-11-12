from mvesuvio.analysis_fitting import fitInYSpaceProcedure
from mvesuvio.util import handle_config
from mvesuvio.util.analysis_helpers import fix_profile_parameters,  \
                            loadRawAndEmptyWsFromUserPath, cropAndMaskWorkspace, \
                            calculate_h_ratio, name_for_starting_ws, \
                            scattering_type, ws_history_matches_inputs, save_ws_from_load_vesuvio, \
                            is_hydrogen_present, create_profiles_table, create_table_for_hydrogen_to_mass_ratios
from mvesuvio.analysis_reduction import VesuvioAnalysisRoutine

from mantid.api import mtd
from mantid.api import AnalysisDataService
from mantid.simpleapi import mtd, RenameWorkspace
from mantid.api import AlgorithmFactory, AlgorithmManager

import numpy as np
from pathlib import Path
import importlib
import sys
import dill         # To convert constraints to string


class Runner:
    def __init__(self, running_tests=False) -> None:
        self.running_tests = running_tests
        self.inputs_path = Path(handle_config.read_config_var("caching.inputs"))
        self.setup()


    def setup(self):
        
        ai = self.import_from_inputs()

        self.bckwd_ai = ai.BackwardAnalysisInputs
        self.fwd_ai = ai.ForwardAnalysisInputs
        self.yFitIC = ai.YSpaceFitInputs

        # Names of workspaces to check if they exist to skip analysis
        self.ws_to_fit_y_space = []
        self.classes_to_fit_y_space = []
        for ai_cls in [self.bckwd_ai, self.fwd_ai]:
            if ai_cls.fit_in_y_space:
                self.ws_to_fit_y_space.append(name_for_starting_ws(ai_cls) + '_' + str(ai_cls.noOfMSIterations))
                self.classes_to_fit_y_space.append(ai_cls)

        self.analysis_result = None
        self.fitting_result = None

        # I/O paths
        inputs_script_path = Path(handle_config.read_config_var("caching.inputs"))
        script_name = handle_config.get_script_name()
        self.experiment_path = inputs_script_path.parent / script_name
        self.input_ws_path =  self.experiment_path / "input_ws"
        self.input_ws_path.mkdir(parents=True, exist_ok=True)

        # TODO: Output paths should probably not be set like this 
        self._set_output_paths(self.bckwd_ai) 
        self._set_output_paths(self.fwd_ai) 

        # TODO: remove this by fixing circular import 
        self.fwd_ai.name = name_for_starting_ws(self.fwd_ai)
        self.bckwd_ai.name = name_for_starting_ws(self.bckwd_ai)

        # TODO: sort out yfit inputs
        figSavePath = self.experiment_path / "figures"
        figSavePath.mkdir(exist_ok=True)
        self.yFitIC.figSavePath = figSavePath


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
        # Default workflow for procedure + fit in y space

        # If any ws for y fit already loaded
        wsInMtd = [ws in mtd for ws in self.ws_to_fit_y_space]  # Bool list
        if (len(wsInMtd) > 0) and all(wsInMtd):
            self.runAnalysisFitting()
            return self.analysis_result, self.fitting_result  

        self.runAnalysisRoutine()
        self.runAnalysisFitting()

        # Return results used only in tests
        return self.analysis_result, self.fitting_result  


    def runAnalysisFitting(self):
        for wsName, i_cls in zip(self.ws_to_fit_y_space, self.classes_to_fit_y_space):
            self.fitting_result = fitInYSpaceProcedure(self.yFitIC, i_cls, wsName)
        return


    def runAnalysisRoutine(self):

        if self.bckwd_ai.run_this_scattering_type:

            if is_hydrogen_present(self.fwd_ai.masses) & (self.bckwd_ai.HToMassIdxRatio==0):
                self.run_estimate_h_ratio()
                return

            # TODO: make this automatic
            assert is_hydrogen_present(self.fwd_ai.masses) != (
                self.bckwd_ai.HToMassIdxRatio==0 
            ), "No Hydrogen detected, HToMassIdxRatio has to be set to 0"

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

        # assert (
        #     bckwdIC.runningSampleWS is False
        # ), "Preliminary procedure not suitable for Bootstrap."
        # fwdIC.runningPreliminary = True

        userInput = input(
            "\nHydrogen intensity ratio to lowest mass is not set. Run procedure to estimate it?"
        )
        if not ((userInput == "y") or (userInput == "Y")):
            raise KeyboardInterrupt("Procedure interrupted.")

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

        print("\nProcedute to estimate Hydrogen ratio finished.",
              "\nEstimates at each iteration converged:",
              f"\n{table_h_ratios.column(0)}")
        return


    def _create_analysis_algorithm(self, ai):

        raw_path, empty_path = self._save_ws_if_not_on_path(ai)

        ws = loadRawAndEmptyWsFromUserPath(
            userWsRawPath=raw_path,
            userWsEmptyPath=empty_path,
            tofBinning=ai.tofBinning,
            name=name_for_starting_ws(ai),
            scaleRaw=ai.scaleRaw,
            scaleEmpty=ai.scaleEmpty,
            subEmptyFromRaw=ai.subEmptyFromRaw
        )
        cropedWs = cropAndMaskWorkspace(
            ws, 
            firstSpec=ai.firstSpec,
            lastSpec=ai.lastSpec,
            maskedDetectors=ai.maskedSpecAllNo,
            maskTOFRange=ai.maskTOFRange
        )
        profiles_table = create_profiles_table(cropedWs.name()+"_initial_parameters", ai)
        ipFilesPath = Path(handle_config.read_config_var("caching.ipfolder"))
        kwargs = {
            "InputWorkspace": cropedWs.name(),
            "InputProfiles": profiles_table.name(),
            "InstrumentParametersFile": str(ipFilesPath / ai.ipfile),
            "HRatioToLowestMass": ai.HToMassIdxRatio if hasattr(ai, 'HRatioToLowestMass') else 0,
            "NumberOfIterations": int(ai.noOfMSIterations),
            "InvalidDetectors": [int(det) for det in ai.maskedSpecAllNo],
            "MultipleScatteringCorrection": ai.MSCorrectionFlag,
            "SampleVerticalWidth": ai.vertical_width, 
            "SampleHorizontalWidth": ai.horizontal_width, 
            "SampleThickness": ai.thickness,
            "GammaCorrection": ai.GammaCorrectionFlag,
            "ModeRunning": scattering_type(ai),
            "TransmissionGuess": ai.transmission_guess,
            "MultipleScatteringOrder": int(ai.multiple_scattering_order),
            "NumberOfEvents": int(ai.number_of_events),
            "Constraints": str(dill.dumps(ai.constraints)),
            "ResultsPath": str(self.results_save_path),
            "FiguresPath": str(self.fig_save_path),
            "OutputMeansTable":" Final_Means"
        }

        if self.running_tests:
            alg = VesuvioAnalysisRoutine()
        else:
            AlgorithmFactory.subscribe(VesuvioAnalysisRoutine)
            alg = AlgorithmManager.createUnmanaged("VesuvioAnalysisRoutine")

        alg.initialize()
        alg.setProperties(kwargs)
        return alg 


    def _make_neutron_profiles(cls, ai):
        profiles = []
        for mass, intensity, width, center, intensity_bound, width_bound, center_bound in zip(
            ai.masses, ai.initPars[::3], ai.initPars[1::3], ai.initPars[2::3],
            ai.bounds[::3], ai.bounds[1::3], ai.bounds[2::3]
        ):
            profiles.append(NeutronComptonProfile(
                label=str(float(mass)), mass=float(mass), intensity=float(intensity), width=float(width), center=float(center),
                intensity_bounds=list(intensity_bound), width_bounds=list(width_bound), center_bounds=list(center_bound)
            ))
        return profiles


    def _save_ws_if_not_on_path(self, load_ai):

        scatteringType = scattering_type(load_ai).lower()
        scriptName = handle_config.get_script_name()

        rawWSName = scriptName + "_" + "raw" + "_" + scatteringType + ".nxs"
        emptyWSName = scriptName + "_" + "empty" + "_" + scatteringType + ".nxs"

        rawPath = self.input_ws_path / rawWSName
        emptyPath = self.input_ws_path / emptyWSName

        ipFilesPath = Path(handle_config.read_config_var("caching.ipfolder"))

        if not ws_history_matches_inputs(load_ai.runs, load_ai.mode, load_ai.ipfile, rawPath):
            save_ws_from_load_vesuvio(load_ai.runs, load_ai.mode, str(ipFilesPath/load_ai.ipfile), rawPath)

        if not ws_history_matches_inputs(load_ai.empty_runs, load_ai.mode, load_ai.ipfile, emptyPath):
            save_ws_from_load_vesuvio(load_ai.empty_runs, load_ai.mode, str(ipFilesPath/load_ai.ipfile), emptyPath)
        return rawPath, emptyPath


    def _set_output_paths(self, ai):
        experimentPath = self.experiment_path
        outputPath = experimentPath / "output_files"
        outputPath.mkdir(parents=True, exist_ok=True)

        # Build Filename based on ic
        corr = ""
        if ai.GammaCorrectionFlag & (ai.noOfMSIterations > 0):
            corr += "_GC"
        if ai.MSCorrectionFlag & (ai.noOfMSIterations > 0):
            corr += "_MS"

        fileName = (
            f"spec_{ai.firstSpec}-{ai.lastSpec}_iter_{ai.noOfMSIterations}{corr}" + ".npz"
        )
        fileNameYSpace = fileName + "_ySpaceFit" + ".npz"

        self.results_save_path = outputPath / fileName
        ai.ySpaceFitSavePath = outputPath / fileNameYSpace

        # Set directories for figures
        figSavePath = experimentPath / "figures"
        figSavePath.mkdir(exist_ok=True)
        self.fig_save_path = figSavePath
        return


if __name__ == "__main__":
    Runner().run()

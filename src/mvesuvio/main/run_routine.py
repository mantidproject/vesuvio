from mvesuvio.util.process_inputs import (
    completeICFromInputs,
    completeYFitIC,
)
from mvesuvio.analysis_fitting import fitInYSpaceProcedure
from mvesuvio.util import handle_config
from mvesuvio.util.analysis_helpers import fix_profile_parameters,  \
                            loadRawAndEmptyWsFromUserPath, cropAndMaskWorkspace, \
                            NeutronComptonProfile, calculate_h_ratio
from mvesuvio.analysis_reduction import VesuvioAnalysisRoutine

from mantid.api import mtd
from mantid.api import AnalysisDataService
from mantid.simpleapi import CreateEmptyTableWorkspace, mtd, RenameWorkspace
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

        self.wsBackIC = ai.LoadVesuvioBackParameters
        self.wsFrontIC = ai.LoadVesuvioFrontParameters
        self.bckwdIC = ai.BackwardInitialConditions
        self.fwdIC = ai.ForwardInitialConditions
        self.yFitIC = ai.YSpaceFitInitialConditions
        self.userCtr = ai.UserScriptControls

        # Set extra attributes from user attributes
        completeICFromInputs(self.fwdIC, self.wsFrontIC)
        completeICFromInputs(self.bckwdIC, self.wsBackIC)
        completeYFitIC(self.yFitIC)
        checkInputs(self.userCtr)

        # Names of workspaces to check if they exist to do fitting
        self.ws_to_fit_y_space = []
        self.classes_to_fit_y_space = []
        for mode, i_cls in zip(["BACKWARD", "FORWARD"], [self.bckwdIC, self.fwdIC]):
            if (self.userCtr.fitInYSpace == mode) | (self.userCtr.fitInYSpace == "JOINT"):
                self.ws_to_fit_y_space.append(i_cls.name + '_' + str(i_cls.noOfMSIterations))
                self.classes_to_fit_y_space.append(i_cls)

        self.analysis_result = None
        self.fitting_result = None


    def import_from_inputs(self):
        name = "analysis_inputs"
        spec = importlib.util.spec_from_file_location(name, self.inputs_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        return module


    def run(self):
        if not self.userCtr.runRoutine:
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
        routine_type = self.userCtr.procedure

        if routine_type is None:
            return

        if (routine_type == "BACKWARD") | (routine_type== "JOINT"):

            if isHPresent(self.fwdIC.masses) & (self.bckwdIC.HToMassIdxRatio==0):
                self.run_estimate_h_ratio()
                return

            # TODO: make this automatic
            assert isHPresent(self.fwdIC.masses) != (
                self.bckwdIC.HToMassIdxRatio==0 
            ), "No Hydrogen detected, HToMassIdxRatio has to be set to 0"

        if routine_type == "BACKWARD":
            self.run_single_analysis(self.wsBackIC, self.bckwdIC)
        if routine_type == "FORWARD":
            self.run_single_analysis(self.wsFrontIC, self.fwdIC)
        if routine_type == "JOINT":
            self.run_joint_analysis()
        return 


    def run_single_analysis(self, load_ai, ai):
        AnalysisDataService.clear()
        alg = self._create_analysis_algorithm(load_ai, ai)
        alg.execute()
        self.analysis_result = alg
        return


    def run_joint_analysis(self):
        AnalysisDataService.clear()
        back_alg = self._create_analysis_algorithm(self.wsBackIC, self.bckwdIC)
        front_alg = self._create_analysis_algorithm(self.wsFrontIC, self.fwdIC)
        self.analysis_result = run_joint_algs(back_alg, front_alg)
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

        table_h_ratios = createTableWSHRatios()

        back_alg = self._create_analysis_algorithm(self.wsBackIC, self.bckwdIC)
        front_alg = self._create_analysis_algorithm(self.wsFrontIC, self.fwdIC)

        front_alg.execute()

        means_table = mtd[front_alg.getPropertyValue("OutputMeansTable")]
        current_ratio = calculate_h_ratio(means_table) 

        table_h_ratios.addRow([current_ratio])
        previous_ratio = np.nan 

        while not np.isclose(current_ratio, previous_ratio, rtol=0.01):

            back_alg.setProperty("HRatioToLowestMass", current_ratio)
            run_joint_algs(back_alg, front_alg)

            previous_ratio = current_ratio

            means_table = mtd[front_alg.getPropertyValue("OutputMeansTable")]
            current_ratio = calculate_h_ratio(means_table) 

            table_h_ratios.addRow([current_ratio])

        print("\nProcedute to estimate Hydrogen ratio finished.",
              "\nEstimates at each iteration converged:",
              f"\n{table_h_ratios.column(0)}")
        return


    def _create_analysis_algorithm(self, load_ai, ai):
        ws = loadRawAndEmptyWsFromUserPath(
            userWsRawPath=ai.userWsRawPath,
            userWsEmptyPath=ai.userWsEmptyPath,
            tofBinning=ai.tofBinning,
            name=ai.name,
            scaleRaw=load_ai.scaleRaw,
            scaleEmpty=load_ai.scaleEmpty,
            subEmptyFromRaw=load_ai.subEmptyFromRaw
        )
        cropedWs = cropAndMaskWorkspace(
            ws, 
            firstSpec=ai.firstSpec,
            lastSpec=ai.lastSpec,
            maskedDetectors=ai.maskedSpecAllNo,
            maskTOFRange=ai.maskTOFRange
        )

        profiles = []
        for mass, intensity, width, center, intensity_bound, width_bound, center_bound in zip(
            ai.masses, ai.initPars[::3], ai.initPars[1::3], ai.initPars[2::3],
            ai.bounds[::3], ai.bounds[1::3], ai.bounds[2::3]
        ):
            profiles.append(NeutronComptonProfile(
                label=str(mass), mass=mass, intensity=intensity, width=width, center=center,
                intensity_bounds=list(intensity_bound), width_bounds=list(width_bound), center_bounds=list(center_bound)
            ))

        profiles_table = create_profiles_table(cropedWs.name()+"_initial_parameters", profiles)

        ipFilesPath = Path(handle_config.read_config_var("caching.ipfolder"))

        kwargs = {
            "InputWorkspace": cropedWs.name(),
            "InputProfiles": profiles_table.name(),
            "InstrumentParametersFile": str(ipFilesPath / ai.instrParsFile),
            "HRatioToLowestMass": ai.HToMassIdxRatio if hasattr(ai, 'HRatioToLowestMass') else 0,
            "NumberOfIterations": int(ai.noOfMSIterations),
            "InvalidDetectors": ai.maskedSpecAllNo.astype(int).tolist(),
            "MultipleScatteringCorrection": ai.MSCorrectionFlag,
            "SampleVerticalWidth": ai.vertical_width, 
            "SampleHorizontalWidth": ai.horizontal_width, 
            "SampleThickness": ai.thickness,
            "GammaCorrection": ai.GammaCorrectionFlag,
            "ModeRunning": ai.modeRunning,
            "TransmissionGuess": ai.transmission_guess,
            "MultipleScatteringOrder": int(ai.multiple_scattering_order),
            "NumberOfEvents": int(ai.number_of_events),
            "Constraints": str(dill.dumps(ai.constraints)),
            "ResultsPath": str(ai.resultsSavePath),
            "FiguresPath": str(ai.figSavePath),
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


def checkInputs(crtIC):
    try:
        if ~crtIC.runRoutine:
            return
    except AttributeError:
        if ~crtIC.runBootstrap:
            return

    for flag in [crtIC.procedure, crtIC.fitInYSpace]:
        assert (
            (flag == "BACKWARD")
            | (flag == "FORWARD")
            | (flag == "JOINT")
            | (flag is None)
        ), "Option not recognized."

    if (crtIC.procedure != "JOINT") & (crtIC.fitInYSpace is not None):
        assert crtIC.procedure == crtIC.fitInYSpace


def isHPresent(masses) -> bool:
    Hmask = np.abs(masses - 1) / 1 < 0.1  # H mass whithin 10% of 1 au

    if np.any(Hmask):  # H present
        print("\nH mass detected.\n")
        assert (
            len(Hmask) > 1
        ), "When H is only mass present, run independent forward procedure, not joint."
        assert Hmask[0], "H mass needs to be the first mass in masses and initPars."
        assert sum(Hmask) == 1, "More than one mass very close to H were detected."
        return True
    else:
        return False


def create_profiles_table(name, profiles: list[NeutronComptonProfile]):
    table = CreateEmptyTableWorkspace(OutputWorkspace=name)
    table.addColumn(type="str", name="label")
    table.addColumn(type="float", name="mass")
    table.addColumn(type="float", name="intensity")
    table.addColumn(type="str", name="intensity_bounds")
    table.addColumn(type="float", name="width")
    table.addColumn(type="str", name="width_bounds")
    table.addColumn(type="float", name="center")
    table.addColumn(type="str", name="center_bounds")
    for p in profiles:
        table.addRow([str(getattr(p, attr)) 
            if "bounds" in attr 
            else getattr(p, attr) 
            for attr in table.getColumnNames()])
    return table


def run_joint_algs(back_alg, front_alg):

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


def createTableWSHRatios():
    table = CreateEmptyTableWorkspace(
        OutputWorkspace="hydrogen_intensity_ratios_estimates"
    )
    table.addColumn(type="float", name="Hydrogen intensity ratio to lowest mass at each iteration")
    return table


if __name__ == "__main__":
    Runner().run()

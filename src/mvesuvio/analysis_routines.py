# from .analysis_reduction import iterativeFitForDataReduction
from mantid.api import AnalysisDataService
from mantid.simpleapi import CreateEmptyTableWorkspace, mtd, RenameWorkspace
from mantid.api import AlgorithmFactory, AlgorithmManager
import numpy as np
import dill         # To convert constraints to string

from mvesuvio.util.analysis_helpers import fix_profile_parameters,  \
                            loadRawAndEmptyWsFromUserPath, cropAndMaskWorkspace, \
                            NeutronComptonProfile, calculate_h_ratio
from mvesuvio.analysis_reduction import VesuvioAnalysisRoutine
from mvesuvio.util import handle_config

from pathlib import Path

def _create_analysis_object_from_current_interface(IC, running_tests=False):
    ws = loadRawAndEmptyWsFromUserPath(
        userWsRawPath=IC.userWsRawPath,
        userWsEmptyPath=IC.userWsEmptyPath,
        tofBinning=IC.tofBinning,
        name=IC.name,
        scaleRaw=IC.scaleRaw,
        scaleEmpty=IC.scaleEmpty,
        subEmptyFromRaw=IC.subEmptyFromRaw
    )
    cropedWs = cropAndMaskWorkspace(
        ws, 
        firstSpec=IC.firstSpec,
        lastSpec=IC.lastSpec,
        maskedDetectors=IC.maskedSpecAllNo,
        maskTOFRange=IC.maskTOFRange
    )

    profiles = []
    for mass, intensity, width, center, intensity_bound, width_bound, center_bound in zip(
        IC.masses, IC.initPars[::3], IC.initPars[1::3], IC.initPars[2::3],
        IC.bounds[::3], IC.bounds[1::3], IC.bounds[2::3]
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
        "InstrumentParametersFile": str(ipFilesPath / IC.instrParsFile),
        "HRatioToLowestMass": IC.HToMassIdxRatio if hasattr(IC, 'HRatioToLowestMass') else 0,
        "NumberOfIterations": int(IC.noOfMSIterations),
        "InvalidDetectors": IC.maskedSpecAllNo.astype(int).tolist(),
        "MultipleScatteringCorrection": IC.MSCorrectionFlag,
        "SampleVerticalWidth": IC.vertical_width, 
        "SampleHorizontalWidth": IC.horizontal_width, 
        "SampleThickness": IC.thickness,
        "GammaCorrection": IC.GammaCorrectionFlag,
        "ModeRunning": IC.modeRunning,
        "TransmissionGuess": IC.transmission_guess,
        "MultipleScatteringOrder": int(IC.multiple_scattering_order),
        "NumberOfEvents": int(IC.number_of_events),
        "Constraints": str(dill.dumps(IC.constraints)),
        "ResultsPath": str(IC.resultsSavePath),
        "FiguresPath": str(IC.figSavePath),
        "OutputMeansTable":" Final_Means"
    }

    if running_tests:
        alg = VesuvioAnalysisRoutine()
    else:
        AlgorithmFactory.subscribe(VesuvioAnalysisRoutine)
        alg = AlgorithmManager.createUnmanaged("VesuvioAnalysisRoutine")

    alg.initialize()
    alg.setProperties(kwargs)
    return alg 


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


def runIndependentIterativeProcedure(IC, clearWS=True, running_tests=False):
    """
    Runs the iterative fitting of NCP, cleaning any previously stored workspaces.
    input: Backward or Forward scattering initial conditions object
    output: Final workspace that was fitted, object with results arrays
    """

    # Clear worksapces before running one of the procedures below
    if clearWS:
        AnalysisDataService.clear()

    alg = _create_analysis_object_from_current_interface(IC, running_tests=running_tests)
    alg.execute()
    return alg


def runJointBackAndForwardProcedure(bckwdIC, fwdIC, clearWS=True):
    assert (
        bckwdIC.modeRunning == "BACKWARD"
    ), "Missing backward IC, args usage: (bckwdIC, fwdIC)"
    assert (
        fwdIC.modeRunning == "FORWARD"
    ), "Missing forward IC, args usage: (bckwdIC, fwdIC)"

    # Clear worksapces before running one of the procedures below
    if clearWS:
        AnalysisDataService.clear()

    back_alg= _create_analysis_object_from_current_interface(bckwdIC)
    front_alg= _create_analysis_object_from_current_interface(fwdIC)

    return run_joint_algs(back_alg, front_alg)


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


def runPreProcToEstHRatio(bckwdIC, fwdIC):
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

    back_alg = _create_analysis_object_from_current_interface(bckwdIC)
    front_alg = _create_analysis_object_from_current_interface(fwdIC)

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


def createTableWSHRatios():
    table = CreateEmptyTableWorkspace(
        OutputWorkspace="hydrogen_intensity_ratios_estimates"
    )
    table.addColumn(type="float", name="Hydrogen intensity ratio to lowest mass at each iteration")
    return table



# from .analysis_reduction import iterativeFitForDataReduction
from mantid.api import AnalysisDataService
from mantid.simpleapi import CreateEmptyTableWorkspace
import numpy as np

from mvesuvio.util.analysis_helpers import loadRawAndEmptyWsFromUserPath, cropAndMaskWorkspace
from mvesuvio.analysis_reduction import AnalysisRoutine
from mvesuvio.analysis_reduction import NeutronComptonProfile


def _create_analysis_object_from_current_interface(IC):
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
    AR = AnalysisRoutine(
        cropedWs,
        ip_file=IC.InstrParsPath,
        h_ratio_to_lowest_mass=IC.HToMassIdxRatio,
        number_of_iterations=IC.noOfMSIterations,
        mask_spectra=IC.maskedSpecAllNo,
        multiple_scattering_correction=IC.MSCorrectionFlag,
        vertical_width=IC.vertical_width, 
        horizontal_width=IC.horizontal_width, 
        thickness=IC.thickness,
        gamma_correction=IC.GammaCorrectionFlag,
        mode_running=IC.modeRunning,
        transmission_guess=IC.transmission_guess,
        multiple_scattering_order=IC.multiple_scattering_order,
        number_of_events=IC.number_of_events,
        results_path=IC.resultsSavePath,
        figures_path=IC.figSavePath,
        constraints=IC.constraints
    )
    profiles = []
    for mass, intensity, width, center, intensity_bound, width_bound, center_bound in zip(
        IC.masses, IC.initPars[::3], IC.initPars[1::3], IC.initPars[2::3],
        IC.bounds[::3], IC.bounds[1::3], IC.bounds[2::3]
    ):
        profiles.append(NeutronComptonProfile(
            label=str(mass), mass=mass, intensity=intensity, width=width, center=center,
            intensity_bounds=intensity_bound, width_bounds=width_bound, center_bounds=center_bound
        ))
    AR.add_profiles(*profiles)
    return AR


def runIndependentIterativeProcedure(IC, clearWS=True):
    """
    Runs the iterative fitting of NCP, cleaning any previously stored workspaces.
    input: Backward or Forward scattering initial conditions object
    output: Final workspace that was fitted, object with results arrays
    """

    # Clear worksapces before running one of the procedures below
    if clearWS:
        AnalysisDataService.clear()

    AR = _create_analysis_object_from_current_interface(IC)
    return AR.run()


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

    return runJoint(bckwdIC, fwdIC)


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

    backRoutine = _create_analysis_object_from_current_interface(bckwdIC)
    frontRoutine = _create_analysis_object_from_current_interface(fwdIC)

    frontRoutine.run()
    current_ratio = frontRoutine.calculate_h_ratio()
    table_h_ratios.addRow([current_ratio])
    previous_ratio = np.nan 

    while not np.isclose(current_ratio, previous_ratio, rtol=0.01):

        backRoutine._h_ratio = current_ratio
        backRoutine.run()
        frontRoutine.set_initial_profiles_from(backRoutine)
        frontRoutine.run()

        previous_ratio = current_ratio
        current_ratio = frontRoutine.calculate_h_ratio()

        table_h_ratios.addRow([current_ratio])

    print("\nProcedute to estimate Hydrogen ratio finished.",
          "\nEstimates at each iteration converged:",
          f"\n{table_h_ratios.column(0)}")
    return


def createTableWSHRatios():
    table = CreateEmptyTableWorkspace(
        OutputWorkspace="H_Ratios_Estimates"
    )
    table.addColumn(type="float", name="H Ratio to lowest mass at each iteration")
    return table


def runJoint(bckwdIC, fwdIC):

    backRoutine = _create_analysis_object_from_current_interface(bckwdIC)
    frontRoutine = _create_analysis_object_from_current_interface(fwdIC)

    backRoutine.run()
    frontRoutine.set_initial_profiles_from(backRoutine)
    frontRoutine.run()
    return


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

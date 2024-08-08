# from .analysis_reduction import iterativeFitForDataReduction
from mantid.api import AnalysisDataService
from mantid.simpleapi import CreateEmptyTableWorkspace
from mvesuvio.oop.analysis_helpers import loadRawAndEmptyWsFromUserPath, cropAndMaskWorkspace
from mvesuvio.oop.AnalysisRoutine import AnalysisRoutine
from mvesuvio.oop.NeutronComptonProfile import NeutronComptonProfile
import numpy as np


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
        figures_path=IC.figSavePath
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

    assert (
        bckwdIC.runningSampleWS is False
    ), "Preliminary procedure not suitable for Bootstrap."
    fwdIC.runningPreliminary = True

    # Store original no of MS and set MS iterations to zero
    oriMS = []
    for IC in [bckwdIC, fwdIC]:
        oriMS.append(IC.noOfMSIterations)
        IC.noOfMSIterations = 0

    nIter = askUserNoOfIterations()

    HRatios = []  # List to store HRatios
    massIdxs = []
    # Run preliminary forward with a good guess for the widths of non-H masses
    wsFinal, fwdScatResults = iterativeFitForDataReduction(fwdIC)
    for i in range(int(nIter)):  # Loop until convergence is achieved
        AnalysisDataService.clear()  # Clears all Workspaces

        # Update H ratio
        massIdx, HRatio = calculateHToMassIdxRatio(fwdScatResults)
        bckwdIC.HToMassIdxRatio = HRatio
        bckwdIC.massIdx = massIdx
        HRatios.append(HRatio)
        massIdxs.append(massIdx)

        wsFinal, bckwdScatResults, fwdScatResults = runJoint(bckwdIC, fwdIC)

    print(f"\nIdxs of masses for H ratio for each iteration: \n{massIdxs}")
    print(f"\nCorresponding H ratios: \n{HRatios}")

    fwdIC.runningPreliminary = (
        False  # Change to default since end of preliminary procedure
    )

    # Set original number of MS iterations
    for IC, ori in zip([bckwdIC, fwdIC], oriMS):
        IC.noOfMSIterations = ori

    # Update the H ratio with the best estimate, chages bckwdIC outside function
    massIdx, HRatio = calculateHToMassIdxRatio(fwdScatResults)
    bckwdIC.HToMassIdxRatio = HRatio
    bckwdIC.massIdx = massIdx
    HRatios.append(HRatio)
    massIdxs.append(massIdx)

    return HRatios, massIdxs


def createTableWSHRatios(HRatios, massIdxs):
    tableWS = CreateEmptyTableWorkspace(
        OutputWorkspace="H_Ratios_From_Preliminary_Procedure"
    )
    tableWS.setTitle("H Ratios and Idxs at each iteration")
    tableWS.addColumn(type="int", name="iter")
    tableWS.addColumn(type="float", name="H Ratio")
    tableWS.addColumn(type="int", name="Mass Idx")
    for i, (hr, hi) in enumerate(zip(HRatios, massIdxs)):
        tableWS.addRow([i, hr, hi])
    return


def askUserNoOfIterations():
    print("\nH was detected but HToMassIdxRatio was not provided.")
    print(
        "\nSugested preliminary procedure:\n\nrun_forward\nfor n:\n    estimate_HToMassIdxRatio\n    run_backward\n"
        "    run_forward"
    )
    userInput = input(
        "\n\nDo you wish to run preliminary procedure to estimate HToMassIdxRatio? (y/n)"
    )
    if not ((userInput == "y") or (userInput == "Y")):
        raise KeyboardInterrupt("Preliminary procedure interrupted.")

    nIter = int(input("\nHow many iterations do you wish to run? n="))
    return nIter


def calculateHToMassIdxRatio(fwdScatResults):
    """
    Calculate H ratio to mass with highest peak.
    Returns idx of mass and corresponding H ratio.
    """
    fwdMeanIntensityRatios = fwdScatResults.all_mean_intensities[-1]

    # To find idx of mass in backward scattering, take out first mass H
    fwdIntensitiesNoH = fwdMeanIntensityRatios[1:]

    massIdx = np.argmax(
        fwdIntensitiesNoH
    )  # Idex of forward inensities, which include H
    assert (
        fwdIntensitiesNoH[massIdx] != 0
    ), "Cannot estimate H intensity since maximum peak from backscattering is zero."

    HRatio = fwdMeanIntensityRatios[0] / fwdIntensitiesNoH[massIdx]

    return massIdx, HRatio


def runJoint(bckwdIC, fwdIC):

    backRoutine = _create_analysis_object_from_current_interface(bckwdIC)
    frontRoutine = _create_analysis_object_from_current_interface(fwdIC)

    backRoutine.run()
    frontRoutine.set_initial_profiles_from(backRoutine)
    print("\nCHANGED STARTING POINT OF PROFILES\n")
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

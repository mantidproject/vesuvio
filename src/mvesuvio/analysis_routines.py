# from .analysis_reduction import iterativeFitForDataReduction
from mantid.api import AnalysisDataService
from mantid.simpleapi import CreateEmptyTableWorkspace
from mantid.api import AlgorithmFactory
import numpy as np

from mvesuvio.util.analysis_helpers import loadRawAndEmptyWsFromUserPath, cropAndMaskWorkspace
from mvesuvio.analysis_reduction import AnalysisRoutine
from mvesuvio.analysis_reduction import NeutronComptonProfile
from tests.testhelpers.calibration.algorithms import create_algorithm

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

    profiles = []
    for mass, intensity, width, center, intensity_bound, width_bound, center_bound in zip(
        IC.masses, IC.initPars[::3], IC.initPars[1::3], IC.initPars[2::3],
        IC.bounds[::3], IC.bounds[1::3], IC.bounds[2::3]
    ):
        profiles.append(NeutronComptonProfile(
            label=str(mass), mass=mass, intensity=intensity, width=width, center=center,
            intensity_bounds=list(intensity_bound), width_bounds=list(width_bound), center_bounds=list(center_bound)
        ))

    profiles_table = create_profiles_table(cropedWs.name()+"_Initial_Parameters", profiles)

    AlgorithmFactory.subscribe(AnalysisRoutine)
    alg = create_algorithm("AnalysisRoutine",
        InputWorkspace=cropedWs,
        InputProfiles=profiles_table.name(),
        InstrumentParametersFile=str(IC.InstrParsPath),
        HRatioToLowestMass=IC.HToMassIdxRatio,
        NumberOfIterations=int(IC.noOfMSIterations),
        InvalidDetectors=IC.maskedSpecAllNo.astype(int).tolist(),
        MultipleScatteringCorrection=IC.MSCorrectionFlag,
        SampleVerticalWidth=IC.vertical_width, 
        SampleHorizontalWidth=IC.horizontal_width, 
        SampleThickness=IC.thickness,
        GammaCorrection=IC.GammaCorrectionFlag,
        ModeRunning=IC.modeRunning,
        TransmissionGuess=IC.transmission_guess,
        MultipleScatteringOrder=int(IC.multiple_scattering_order),
        NumberOfEvents=int(IC.number_of_events),
        # Constraints=IC.constraints,
        ResultsPath=str(IC.resultsSavePath),
        FiguresPath=str(IC.figSavePath)
    )
    # alg.add_profiles(*profiles)

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
        table.addRow([str(getattr(p, attr)) if "bounds" in attr else getattr(p, attr) for attr in table.getColumnNames()])

    for p in profiles:
        print(str(getattr(p, "intensity_bounds")))
        print(str(getattr(p, "width_bounds")))
    return table

def set_initial_profiles_from(self, source: 'AnalysisRoutine'):
    
    # Set intensities
    for p in self._profiles.values():
        if np.isclose(p.mass, 1, atol=0.1):    # Hydrogen present
            p.intensity = source._h_ratio * source._get_lightest_profile().mean_intensity
            continue
        p.intensity = source.profiles[p.label].mean_intensity

    # Normalise intensities
    sum_intensities = sum([p.intensity for p in self._profiles.values()])
    for p in self._profiles.values():
        p.intensity /= sum_intensities
        
    # Set widths
    for p in self._profiles.values():
        try:
            p.width = source.profiles[p.label].mean_width
        except KeyError:
            continue

    # Fix all widths except lightest mass
    for p in self._profiles.values():
        if p == self._get_lightest_profile():
            continue
        p.width_bounds = [p.width, p.width]

    return

def _get_lightest_profile(self):
    profiles = [p for p in self._profiles.values()]
    masses = [p.mass for p in self._profiles.values()]
    return profiles[np.argmin(masses)]

def runIndependentIterativeProcedure(IC, clearWS=True):
    """
    Runs the iterative fitting of NCP, cleaning any previously stored workspaces.
    input: Backward or Forward scattering initial conditions object
    output: Final workspace that was fitted, object with results arrays
    """

    # Clear worksapces before running one of the procedures below
    if clearWS:
        AnalysisDataService.clear()

    alg = _create_analysis_object_from_current_interface(IC)
    return alg.execute()


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

    frontRoutine.execute()
    current_ratio = frontRoutine.calculate_h_ratio()
    table_h_ratios.addRow([current_ratio])
    previous_ratio = np.nan 

    while not np.isclose(current_ratio, previous_ratio, rtol=0.01):

        backRoutine._h_ratio = current_ratio
        backRoutine.execute()
        frontRoutine.set_initial_profiles_from(backRoutine)
        frontRoutine.execute()

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

    backRoutine.execute()
    frontRoutine.set_initial_profiles_from(backRoutine)
    frontRoutine.execute()
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

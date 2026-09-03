from dataclasses import dataclass
import mvesuvio
from mantid.api import AnalysisDataService
from mantid.simpleapi import Load, Rebin, Scale, Minus, SumSpectra
from mantid.kernel import logger
from pathlib import Path
from mvesuvio import ConfigArgInputs
from mvesuvio.util import reduction_helpers


@dataclass
class SampleParameters:
    # Sample slab parameters, expressed in meters
    slab_height = 0.1
    slab_width = 0.1
    slab_thickness = 0.001

    sample_shape_xml = f'''<cuboid id="sample-shape">
        <left-front-bottom-point x="{slab_width / 2}" y="{-slab_height / 2}" z="{slab_thickness / 2}" />
        <left-front-top-point x="{slab_width / 2}" y="{slab_height / 2}" z="{slab_thickness / 2}" />
        <left-back-bottom-point x="{slab_width / 2}" y="{-slab_height / 2}" z="{-slab_thickness / 2}" />
        <right-front-bottom-point x="{-slab_width / 2}" y="{-slab_height / 2}" z="{slab_thickness / 2}" />
        </cuboid>'''


@dataclass
class BackwardAnalysisInputs(SampleParameters):
    run_this_scattering_type = False
    fit_in_y_space = False
    name = "back"
    minimal_output = False

    runs = "43066-43076"  # Runs of your sample dataset
    empty_runs = "41876-41923"  # Empty CCR
    mode = "DoubleDifference"
    instrument_parameters_file = "ip2019.par"
    detectors = "3-134"
    mask_detectors = [18, 34, 42, 43, 59, 60, 62, 118, 119, 133]  # Can also be a string "18, 34, 42-43, 59-60, 62, 118-119, 133"
    time_of_flight_binning = "275.,1.,420"
    mask_time_of_flight_range = None  # Can be string eg. "110-120, 200-210"
    # Scaling factors, leave at default of 1 for most cases
    scale_empty_workspace = 1
    scale_raw_workspace = 1

    # Atomic mass in a.m.u. of each element/isotope present in sample + cell EXCEPT HYDROGEN
    masses = [12, 16, 27]

    initial_fitting_parameters = [  # NCP intensities, NCP widths, NCP centers
        1,
        12,
        0.0,
        1,
        12,
        0.0,
        1,
        12.5,
        0.0,
    ]
    fitting_bounds = [
        [0, None],
        [8, 16],
        [-3, 1],
        [0, None],
        [8, 16],
        [-3, 1],
        [0, None],
        [11, 14],
        [-3, 1],
    ]
    constraints = ()

    number_of_iterations_for_corrections = 0  # 4
    do_multiple_scattering_correction = True
    # Known stoichiometry of any mass in the sample to Hydrogen, to estimate intensity ratio as a guess
    chosen_mass_index = 0  # index in 'masses' list (index from 0 to n-1), ignored if H not present
    intensity_ratio_of_hydrogen_to_chosen_mass = (
        # 0
        19.0620008206  # Set to zero to estimate, with 1 iteration for corrections, ignored if H not present
    )
    transmission_guess = 0.8  # [1 - 2(1-T)] --> Twice the absorption, T: Experimental value from VesuvioTransmission
    multiple_scattering_order = 2
    multiple_scattering_number_of_events = 1.0e5  # 1.0e6 for smoother correction, at the cost of higher execution time
    do_gamma_correction = False


@dataclass
class ForwardAnalysisInputs(SampleParameters):
    run_this_scattering_type = True
    fit_in_y_space = True
    name = "front"
    minimal_output = False

    runs = "43066-43076"
    empty_runs = "43868-43911"  # Empty CCR
    mode = "SingleDifference"
    instrument_parameters_file = "ip2018_3.par"
    detectors = "144-182"
    mask_detectors = [173, 174, 179]  # Can also be a string "173-174, 179"
    time_of_flight_binning = "110,1,430"
    mask_time_of_flight_range = None  # Can be string Eg. "110-120, 200-210"
    # Scaling factors, leave at default of 1 for most cases
    scale_empty_workspace = 1
    scale_raw_workspace = 1

    masses = [1.0079, 12, 16, 27]  # Atomic mass in a.m.u. of each element/isotope present in sample + cell
    initial_fitting_parameters = [  # Intensities, NCP widths, NCP centers
        1,
        4.7,
        0.0,
        1,
        12.71,
        0.0,
        1,
        8.76,
        0.0,
        1,
        13.897,
        0.0,
    ]
    fitting_bounds = [
        [0, None],
        [3, 6],
        [-3, 1],
        [0, None],
        [12.71, 12.71],
        [-3, 1],
        [0, None],
        [8.76, 8.76],
        [-3, 1],
        [0, None],
        [13.897, 13.897],
        [-3, 1],
    ]
    constraints = ()

    number_of_iterations_for_corrections = 0  # 4
    do_multiple_scattering_correction = True
    transmission_guess = 0.9  # T : Experimental value from VesuvioTransmission
    multiple_scattering_order = 2
    multiple_scattering_number_of_events = 1.0e5  # 1.0e6 for smoother correction, at the cost of higher execution time
    do_gamma_correction = True


########################
### END OF USER EDIT ###
########################

mvesuvio.main(ConfigArgInputs(analysis_inputs=str(Path(__file__)), ip_folder=""))

# Optional workspace-name overrides for bootstrap script injection.
BACK_WS_TO_FIT = globals().get("BACK_WS_TO_FIT", "")
FRONT_WS_TO_FIT = globals().get("FRONT_WS_TO_FIT", "")

# Preserve standalone behavior when no bootstrap overrides are injected.
if not BACK_WS_TO_FIT and not FRONT_WS_TO_FIT:
    AnalysisDataService.clear()

if BackwardAnalysisInputs.run_this_scattering_type:
    if BACK_WS_TO_FIT:
        BackwardAnalysisInputs.name = str(BACK_WS_TO_FIT)
        if not AnalysisDataService.doesExist(BackwardAnalysisInputs.name):
            logger.error(f"Injected backward workspace does not exist in ADS: {BackwardAnalysisInputs.name}")
            BACK_WS_TO_FIT = ""
    else:
        raw_path, empty_path = reduction_helpers.load_and_save_input_ws_if_not_on_path(BackwardAnalysisInputs)

        raw_name = raw_path.stem
        empty_name = empty_path.stem

        Load(Filename=str(raw_path), OutputWorkspace=raw_name)
        Load(Filename=str(empty_path), OutputWorkspace=empty_name)

        Rebin(InputWorkspace=raw_name, Params=BackwardAnalysisInputs.time_of_flight_binning, OutputWorkspace=raw_name)
        Rebin(InputWorkspace=empty_name, Params=BackwardAnalysisInputs.time_of_flight_binning, OutputWorkspace=empty_name)

        Scale(InputWorkspace=raw_name, Factor=BackwardAnalysisInputs.scale_raw_workspace, OutputWorkspace=raw_name)
        Scale(InputWorkspace=empty_name, Factor=BackwardAnalysisInputs.scale_empty_workspace, OutputWorkspace=empty_name)

        Minus(LHSWorkspace=raw_name, RHSWorkspace=empty_name, OutputWorkspace=BackwardAnalysisInputs.name)

        # TODO: Take out sums from here
        SumSpectra(InputWorkspace=raw_name, OutputWorkspace=raw_name + "_sum")
        SumSpectra(InputWorkspace=empty_name, OutputWorkspace=empty_name + "_sum")
        BACK_WS_TO_FIT = BackwardAnalysisInputs.name


if ForwardAnalysisInputs.run_this_scattering_type:
    if FRONT_WS_TO_FIT:
        ForwardAnalysisInputs.name = str(FRONT_WS_TO_FIT)
        if not AnalysisDataService.doesExist(ForwardAnalysisInputs.name):
            logger.error(f"Injected forward workspace does not exist in ADS: {ForwardAnalysisInputs.name}")
            FRONT_WS_TO_FIT = ""
    else:
        raw_path, empty_path = reduction_helpers.load_and_save_input_ws_if_not_on_path(ForwardAnalysisInputs)

        raw_name = raw_path.stem
        empty_name = empty_path.stem

        Load(Filename=str(raw_path), OutputWorkspace=raw_name)
        Load(Filename=str(empty_path), OutputWorkspace=empty_name)

        Rebin(InputWorkspace=raw_name, Params=ForwardAnalysisInputs.time_of_flight_binning, OutputWorkspace=raw_name)
        Rebin(InputWorkspace=empty_name, Params=ForwardAnalysisInputs.time_of_flight_binning, OutputWorkspace=empty_name)

        Scale(InputWorkspace=raw_name, Factor=ForwardAnalysisInputs.scale_raw_workspace, OutputWorkspace=raw_name)
        Scale(InputWorkspace=empty_name, Factor=ForwardAnalysisInputs.scale_empty_workspace, OutputWorkspace=empty_name)

        Minus(LHSWorkspace=raw_name, RHSWorkspace=empty_name, OutputWorkspace=ForwardAnalysisInputs.name)

        # TODO: Take out sums from here
        SumSpectra(InputWorkspace=raw_name, OutputWorkspace=raw_name + "_sum")
        SumSpectra(InputWorkspace=empty_name, OutputWorkspace=empty_name + "_sum")
        FRONT_WS_TO_FIT = ForwardAnalysisInputs.name


if not BACK_WS_TO_FIT:
    BACK_WS_TO_FIT = BackwardAnalysisInputs.name
if not FRONT_WS_TO_FIT:
    FRONT_WS_TO_FIT = ForwardAnalysisInputs.name

reduction_helpers.crop_and_mask_workspace(BACK_WS_TO_FIT, BackwardAnalysisInputs)
reduction_helpers.crop_and_mask_workspace(FRONT_WS_TO_FIT, ForwardAnalysisInputs)
back_alg = reduction_helpers.init_analysis_algorithm(BACK_WS_TO_FIT, BackwardAnalysisInputs)
front_alg = reduction_helpers.init_analysis_algorithm(FRONT_WS_TO_FIT, ForwardAnalysisInputs)

if reduction_helpers.h_ratio_is_zero_when_h_present(BackwardAnalysisInputs, ForwardAnalysisInputs):
    reduction_helpers.run_estimate_h_ratio(
        back_alg=back_alg,
        front_alg=front_alg,
        back_masses=BackwardAnalysisInputs.masses,
        back_chosen_mass_index=BackwardAnalysisInputs.chosen_mass_index,
        rtol=0.01,
        max_iter=3,
    )

if (
    BackwardAnalysisInputs.run_this_scattering_type
    and ForwardAnalysisInputs.run_this_scattering_type
    and back_alg is not None
    and front_alg is not None
):
    reduction_helpers.execute_joint_algorithms(back_alg=back_alg, front_alg=front_alg)
elif BackwardAnalysisInputs.run_this_scattering_type and back_alg is not None:
    back_alg.execute()
elif ForwardAnalysisInputs.run_this_scattering_type and front_alg is not None:
    front_alg.execute()

reduction_helpers.make_summarised_log_file()

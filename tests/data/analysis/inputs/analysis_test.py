
class SampleParameters:
    transmission_guess = 0.8537  # Experimental value from VesuvioTransmission
    multiple_scattering_order = 2 
    multiple_scattering_number_of_events = 1.0e5
    vertical_width, horizontal_width, thickness = 0.1, 0.1, 0.001  # Expressed in meters


class BackwardAnalysisInputs(SampleParameters):
    run_this_scattering_type = False
    fit_in_y_space = False 
    instrument_parameters_file = "ip2019.par"
    runs = "43066-43076"  
    empty_runs = "41876-41923"  
    detectors = "3-134"
    mode = "DoubleDifference"
    time_of_flight_binning = "275.,1.,420"  # Binning of ToF spectra
    mask_time_of_flight_range = None
    mask_detectors = [18, 34, 42, 43, 59, 60, 62, 118, 119, 133]
    subtract_empty_workspace_from_raw = True 
    scale_empty_workspace = 1 
    scale_raw_workspace = 1

    masses = [12, 16, 27]
    initial_fitting_parameters = [1, 12, 0.0, 1, 12, 0.0, 1, 12.5, 0.0]
    fitting_bounds =[
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

    number_of_iterations_for_corrections = 3  # 4
    do_multiple_scattering_correction = True
    intensity_ratio_of_hydrogen_to_lowest_mass = 19.0620008206  # Set to zero or None when H is not present
    do_gamma_correction = False


    show_plots = False
    do_symmetrisation = True
    subtract_calculated_fse_from_data = False
    range_for_rebinning_in_y_space = "-20, 0.5, 20"  # Needs to be symetric
    fitting_model = "SINGLE_GAUSSIAN"
    run_minos = True
    do_global_fit = None
    number_of_global_fit_groups = 4
    mask_zeros_with = None


class ForwardAnalysisInputs(SampleParameters):
    run_this_scattering_type = True
    fit_in_y_space = False

    instrument_parameters_file = "ip2018_3.par"
    runs = "43066-43076"
    empty_runs = "43868-43911"
    spectra = "144-182"
    mode = "SingleDifference"
    time_of_flight_binning = "110,1,430"  # Binning of ToF spectra
    mask_time_of_flight_range = None
    mask_detectors = [173, 174, 179]
    detectors = '144-182'
    subtract_empty_workspace_from_raw= False 
    scale_empty_workspace= 1 
    scale_raw_workspace= 1


    masses = [1.0079, 12, 16, 27]
    initial_fitting_parameters= [1, 4.7, 0, 1, 12.71, 0.0, 1, 8.76, 0.0, 1, 13.897, 0.0]
    fitting_bounds =[
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
    number_of_iterations_for_corrections= 1  # 4
    do_multiple_scattering_correction= True
    do_gamma_correction= True


    show_plots = False
    do_symmetrisation = True
    subtract_calculated_fse_from_data = False
    range_for_rebinning_in_y_space = "-20, 0.5, 20"  # Needs to be symetric
    fitting_model = "SINGLE_GAUSSIAN"
    run_minos = True
    do_global_fit = None
    number_of_global_fit_groups = 4
    mask_zeros_with = None


if (__name__ == "__main__") or (__name__ == "mantidqt.widgets.codeeditor.execution"):
    import mvesuvio
    from pathlib import Path
    mvesuvio.set_config(inputs_file=Path(__file__))
    mvesuvio.run()

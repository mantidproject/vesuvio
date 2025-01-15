from dataclasses import dataclass


@dataclass
class SampleParameters:
    # Sample slab parameters, expressed in meters
    vertical_width = 0.1 
    horizontal_width = 0.1 
    thickness = 0.001


@dataclass
class BackwardAnalysisInputs(SampleParameters):
    run_this_scattering_type = True
    fit_in_y_space = True

    runs = "43066-43076"
    empty_runs = "41876-41923" 
    mode = "DoubleDifference"
    instrument_parameters_file = "ip2019.par"
    detectors = "3-134"
    mask_detectors = [18, 34, 42, 43, 59, 60, 62, 118, 119, 133]
    time_of_flight_binning = "275.,1.,420"
    mask_time_of_flight_range = None
    subtract_empty_workspace_from_raw = True
    scale_empty_workspace = 1  # None or scaling factor
    scale_raw_workspace = 1

    masses = [12, 16, 27]
    # Intensities, NCP widths, NCP centers
    initial_fitting_parameters = [1, 12, 0.0, 1, 12, 0.0, 1, 12.5, 0.0]
    fitting_bounds = [
        [0, None], [8, 16], [-3, 1],
        [0, None], [8, 16], [-3, 1],
        [0, None], [11, 14], [-3, 1],
        ]
    constraints = ()

    number_of_iterations_for_corrections = 0  # 4
    do_multiple_scattering_correction = True
    intensity_ratio_of_hydrogen_to_lowest_mass = 19.0620008206  # Set to zero to disable
    transmission_guess = 0.8537  # Experimental value from VesuvioTransmission
    multiple_scattering_order = 2
    multiple_scattering_number_of_events = 1.0e5
    do_gamma_correction = False


    show_plots = True
    do_symmetrisation = False
    subtract_calculated_fse_from_data = True
    range_for_rebinning_in_y_space = "-25, 0.5, 25"  # Needs to be symetric
    # Fitting model options
    # 'gauss': Single Gaussian
    # 'gcc4': Gram-Charlier with C4 parameter
    # 'gcc6': Gram-Charlier with C6 parameter
    # 'doublewell': Double Well function 
    # 'gauss3D': 3-Dimensional Gaussian 
    fitting_model = "gauss"
    run_minos = True
    do_global_fit = True   # Performs global fit with Minuit by default
    # Number of groups of detectors to perform global (simultaneous) fit on
    # Either an integer less than the number of detectors 
    # or option 'all', which does not form groups and fits all spectra simultaneously and individualy 
    number_of_global_fit_groups = 4
    # Type of masking 
    # 'nan': Zeros in workspace being fit are ignored
    # 'ncp': Zeros in workspace being fit are replaced by the fitted neutron compton profile 
    mask_zeros_with = "nan"   


@dataclass
class ForwardAnalysisInputs(SampleParameters):
    run_this_scattering_type = True
    fit_in_y_space = True

    runs = "43066-43076"
    empty_runs = "43868-43911"
    mode = "SingleDifference"
    instrument_parameters_file = "ip2018_3.par"
    detectors = '144-182'
    mask_detectors = [173, 174, 179]
    time_of_flight_binning = "110,1,430"
    mask_time_of_flight_range = None
    subtract_empty_workspace_from_raw = False
    scale_empty_workspace = 1  # None or scaling factor
    scale_raw_workspace = 1

    masses = 1.0079, 12, 16, 27
    # Intensities, NCP widths, NCP centers
    initial_fitting_parameters = [1, 4.7, 0, 1, 12.71, 0.0, 1, 8.76, 0.0, 1, 13.897, 0.0]
    fitting_bounds = [
        [0, None], [3, 6], [-3, 1],
        [0, None], [12.71, 12.71], [-3, 1],
        [0, None], [8.76, 8.76], [-3, 1],
        [0, None], [13.897, 13.897], [-3, 1],
        ]
    constraints = ()

    number_of_iterations_for_corrections = 0  # 4
    do_multiple_scattering_correction = True
    transmission_guess = 0.8537  # Experimental value from VesuvioTransmission
    multiple_scattering_order = 2
    multiple_scattering_number_of_events = 1.0e5
    do_gamma_correction = True


    show_plots = True
    do_symmetrisation = False
    subtract_calculated_fse_from_data = True
    range_for_rebinning_in_y_space = "-25, 0.5, 25"  # Needs to be symetric
    # Fitting model options
    # 'gauss': Single Gaussian
    # 'gcc4': Gram-Charlier with C4 parameter
    # 'gcc6': Gram-Charlier with C6 parameter
    # 'doublewell': Double Well function 
    # 'gauss3D': 3-Dimensional Gaussian 
    fitting_model = "gauss"
    run_minos = True
    do_global_fit = True   # Performs global fit with Minuit by default
    # Number of groups of detectors to perform global (simultaneous) fit on
    # Either an integer less than the number of detectors 
    # or option 'all', which does not form groups and fits all spectra simultaneously and individualy 
    number_of_global_fit_groups = 4
    # Type of masking 
    # 'nan': Zeros in workspace being fit are ignored
    # 'ncp': Zeros in workspace being fit are replaced by the fitted neutron compton profile 
    mask_zeros_with = "nan"   


########################
### END OF USER EDIT ###
########################


if (__name__ == "__main__") or (__name__ == "mantidqt.widgets.codeeditor.execution"):
    import mvesuvio
    from pathlib import Path
    mvesuvio.set_config(inputs_file=Path(__file__))
    mvesuvio.run()

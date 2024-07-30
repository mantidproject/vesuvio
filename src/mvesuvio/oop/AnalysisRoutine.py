from mvesuvio.oop.NeutronComptonProfile import NeutronComptonProfile 

class AnalysisRoutine:

    def __init__(self, number_of_iterations, spectrum_range, mask_spectra,
                 multiple_scattering_correction, gamma_correction,
                 transmission_guess, multiple_scattering_order, number_of_events):

        self._number_of_iterations = number_of_iterations
        self._spectrum_range = spectrum_range
        self._mask_spectra = mask_spectra
        self._transmission_guess = transmission_guess
        self._multiple_scattering_order = multiple_scattering_order
        self._number_of_events = number_of_events

        self._multiple_scattering_correction = multiple_scattering_correction
        self._gamma_correction = gamma_correction

        self._h_ratio = None
        self._constraints = [] 
        self._profiles = {} 

        # Only used for system tests, remove once tests are updated
        self._run_hist_data = True
        self._run_norm_voigt = False

        # Links to another AnalysisRoutine object:
        self._profiles_destination = None
        self._h_ratio_destination = None


    def add_profiles(self, **args: NeutronComptonProfile):
        for profile in args:
            self._profiles[profile.label] = profile


    def add_constraint(self, constraint_string: str):
        self._constraints.append(constraint_string)


    def add_h_ratio_to_next_lowest_mass(self, ratio: float):
        self._h_ratio_to_next_lowest_mass = ratio


    def send_ncp_fit_parameters(self):
       self._profiles_destination.profiles = self.profiles


    def send_h_ratio(self):
        self._h_ratio_destination.h_ratio_to_next_lowest_mass = self._h_ratio

    @property
    def h_ratio_to_next_lowest_mass(self):
        return self._h_ratio

    @h_ratio_to_next_lowest_mass.setter
    def h_ratio_to_next_lowest_mass(self, value):
        self.h_ratio_to_next_lowest_mass = value

    @property
    def profiles(self):
        return self._profiles

    @profiles.setter
    def profiles(self, incoming_profiles):
        assert(isinstance(incoming_profiles, dict))
        common_keys = self._profiles.keys() & incoming_profiles.keys() 
        common_keys_profiles = {k: incoming_profiles[k] for k in common_keys}
        self._profiles = {**self._profiles, **common_keys_profiles}



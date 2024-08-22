import numpy as np 
import matplotlib.pyplot as plt
from scipy import optimize
from mantid.simpleapi import mtd, CreateEmptyTableWorkspace, SumSpectra, \
                            CloneWorkspace, DeleteWorkspace, VesuvioCalculateGammaBackground, \
                            VesuvioCalculateMS, Scale, RenameWorkspace, Minus, CreateSampleShape, \
                            VesuvioThickness, Integration, Divide, Multiply, DeleteWorkspaces, \
                            CreateWorkspace

from mvesuvio.util.analysis_helpers import histToPointData, loadConstants, \
                                        gaussian, lorentzian, numericalThirdDerivative

from dataclasses import dataclass


np.set_printoptions(suppress=True, precision=4, linewidth=200)


@dataclass(frozen=False)
class NeutronComptonProfile:
    label: str
    mass: float

    intensity: float
    width: float
    center: float

    intensity_bounds: tuple[float, float]
    width_bounds: tuple[float, float]
    center_bounds: tuple[float, float]

    mean_intensity: float = None
    mean_width: float = None
    mean_center: float = None


class AnalysisRoutine:

    def __init__(self, workspace, ip_file, h_ratio_to_lowest_mass, number_of_iterations, mask_spectra,
                 multiple_scattering_correction, vertical_width, horizontal_width, thickness,
                 gamma_correction, mode_running, transmission_guess=None, multiple_scattering_order=None, 
                 number_of_events=None, results_path=None, figures_path=None, constraints=()):

        self._name = workspace.name()
        self._ip_file = ip_file
        self._number_of_iterations = number_of_iterations
        self._mask_spectra = mask_spectra
        self._transmission_guess = transmission_guess
        self._multiple_scattering_order = multiple_scattering_order
        self._number_of_events = number_of_events
        self._vertical_width = vertical_width
        self._horizontal_width = horizontal_width
        self._thickness = thickness
        self._mode_running = mode_running
        self._multiple_scattering_correction = multiple_scattering_correction
        self._gamma_correction = gamma_correction
        self._save_results_path = results_path
        self._save_figures_path = figures_path
        self._h_ratio = h_ratio_to_lowest_mass 
        self._constraints = constraints 

        self._profiles = {} 

        # Variables changing during fit
        self._workspace_for_corrections = workspace
        self._workspace_being_fit = workspace
        self._row_being_fit = 0 
        self._zero_columns_boolean_mask = None
        self._table_fit_results = None
        self._fit_profiles_workspaces = {}

        # Only used for system tests, remove once tests are updated
        self._run_hist_data = True 
        self._run_norm_voigt = False


    def add_profiles(self, *args: NeutronComptonProfile):
        for profile in args:
            self._profiles[profile.label] = profile


    @property
    def profiles(self):
        return self._profiles


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


    def calculate_h_ratio(self):

        if not np.isclose(self._get_lightest_profile().mass, 1, atol=0.1):    # Hydrogen present
            return None
        
        # Hydrogen is present 
        intensities = np.array([p.mean_intensity for p in self._profiles.values()])
        masses = np.array([p.mass for p in self._profiles.values()])

        sorted_intensities = intensities[np.argsort(masses)]
        return sorted_intensities[0] / sorted_intensities[1] 
        

    @property
    def profiles(self):
        return self._profiles


    @profiles.setter
    def profiles(self, incoming_profiles):
        assert(isinstance(incoming_profiles, dict))
        common_keys = self._profiles.keys() & incoming_profiles.keys() 
        common_keys_profiles = {k: incoming_profiles[k] for k in common_keys}
        self._profiles = {**self._profiles, **common_keys_profiles}


    def _update_workspace_data(self):

        self._dataX = self._workspace_being_fit.extractX()
        self._dataY = self._workspace_being_fit.extractY()
        self._dataE = self._workspace_being_fit.extractE()

        if self._run_hist_data:  # Converts point data from workspaces to histogram data
            self._dataY, self._dataX, self._dataE = histToPointData(self._dataY, self._dataX, self._dataE)

        self._set_up_kinematic_arrays()

        self._fit_parameters = np.zeros((len(self._dataY), 3 * len(self._profiles) + 3))
        self._row_being_fit = 0 
        self._table_fit_results = self._initialize_table_fit_parameters()

        # Initialise workspaces for fitted ncp 
        self._fit_profiles_workspaces = {}
        for element, p in self._profiles.items():
            self._fit_profiles_workspaces[element] = self._create_emtpy_ncp_workspace(f'_{element}_ncp')
        self._fit_profiles_workspaces['total'] = self._create_emtpy_ncp_workspace(f'_total_ncp')

        # Initialise empty means
        self._mean_widths = None
        self._std_widths = None
        self._mean_intensity_ratios = None
        self._std_intensity_ratios = None


    def _initialize_table_fit_parameters(self):
        table = CreateEmptyTableWorkspace(
            OutputWorkspace=self._workspace_being_fit.name()+ "_fit_results"
        )
        table.setTitle("SciPy Fit Parameters")
        table.addColumn(type="float", name="Spectrum")
        for key in self._profiles.keys():
            table.addColumn(type="float", name=f"{key} Intensity")
            table.addColumn(type="float", name=f"{key} Width")
            table.addColumn(type="float", name=f"{key} Center ")
        table.addColumn(type="float", name="Normalised Chi2")
        table.addColumn(type="float", name="Number of Iterations")
        return table


    @property
    def mean_widths(self):
        return self._mean_widths


    @mean_widths.setter
    def mean_widths(self, value):
        self._mean_widths = value
        for i, p in enumerate(self._profiles.values()):
            p.mean_width = self._mean_widths[i]
        return


    @property
    def mean_intensity_ratios(self):
        return self._mean_intensity_ratios

    @mean_intensity_ratios.setter
    def mean_intensity_ratios(self, value):
        self._mean_intensity_ratios = value
        for i, p in enumerate(self.profiles.values()):
            p.mean_intensity = self._mean_intensity_ratios[i]
        return


    def _create_emtpy_ncp_workspace(self, suffix):
        return CreateWorkspace(
            DataX=self._dataX,
            DataY=np.zeros(self._dataY.size),
            DataE=np.zeros(self._dataE.size),
            Nspec=self._workspace_being_fit.getNumberHistograms(),
            OutputWorkspace=self._workspace_being_fit.name()+suffix,
            ParentWorkspace=self._workspace_being_fit
    )


    def _set_up_kinematic_arrays(self):
        resolutionPars, instrPars, kinematicArrays, ySpacesForEachMass = self.prepareFitArgs()
        self._resolution_params = resolutionPars
        self._instrument_params = instrPars
        self._kinematic_arrays = kinematicArrays
        self._y_space_arrays = ySpacesForEachMass


    def run(self):

        assert len(self.profiles) > 0, "Add profiles before attempting to run the routine!"

        self._create_table_initial_parameters()

        # Legacy code from Bootstrap
        # if self.runningSampleWS:
        #     initialWs = RenameWorkspace(
        #         InputWorkspace=ic.sampleWS, OutputWorkspace=initialWs.name()
        #     )

        CloneWorkspace(
            InputWorkspace=self._workspace_being_fit.name(), 
            OutputWorkspace=self._name + '0' 
        )

        for iteration in range(self._number_of_iterations + 1):

            self._workspace_being_fit = mtd[self._name + str(iteration)]
            self._update_workspace_data()

            self._fit_neutron_compton_profiles()

            self._create_summed_workspaces()
            self._save_plots()
            self._set_means_and_std()

            # When last iteration, skip MS and GC
            if iteration == self._number_of_iterations:
                break

            # Do this because MS and Gamma corrections do not accept zero columns 
            if iteration==0:
                self._replace_zero_columns_with_ncp_fit()

            CloneWorkspace(
                InputWorkspace=self._workspace_for_corrections.name(), 
                OutputWorkspace="next_iteration"
            )
            self._correct_for_gamma_and_multiple_scattering("next_iteration")

            # Need to remask columns of output of corrections 
            self._remask_columns_with_zeros("next_iteration")

            RenameWorkspace(
                InputWorkspace="next_iteration", 
                OutputWorkspace=self._name + str(iteration + 1)
            )

        self._set_results()
        self._save_results()
        return self 


    def _create_table_initial_parameters(self):
        meansTableWS = CreateEmptyTableWorkspace(
            OutputWorkspace=self._name + "_Initial_Parameters"
        )
        meansTableWS.addColumn(type="float", name="Mass")
        meansTableWS.addColumn(type="float", name="Initial Widths")
        meansTableWS.addColumn(type="str", name="Bounds Widths")
        meansTableWS.addColumn(type="float", name="Initial Intensities")
        meansTableWS.addColumn(type="str", name="Bounds Intensities")
        meansTableWS.addColumn(type="float", name="Initial Centers")
        meansTableWS.addColumn(type="str", name="Bounds Centers")

        print("\nCreated Table with Initial Parameters:")
        for p in self._profiles.values():
            meansTableWS.addRow([p.mass, p.width, str(p.width_bounds), 
                                 p.intensity, str(p.intensity_bounds), 
                                 p.center, str(p.center_bounds)])
            print("\nMass: ", p.mass)
            print(f"{'Initial Intensity:':>20s} {p.intensity:<8.3f} Bounds: {p.intensity_bounds}")
            print(f"{'Initial Width:':>20s} {p.width:<8.3f} Bounds: {p.width_bounds}")
            print(f"{'Initial Center:':>20s} {p.center:<8.3f} Bounds: {p.center_bounds}")
        print("\n")
        return


    def _fit_neutron_compton_profiles(self):
        """
        Performs the fit of neutron compton profiles to the workspace being fit.
        The profiles are fit on a spectrum by spectrum basis.
        """
        print("\nFitting Neutron Compton Prolfiles:\n")

        self._row_being_fit = 0
        while self._row_being_fit != len(self._dataY):
            self._fit_neutron_compton_profiles_to_row()
            self._row_being_fit += 1

        assert np.any(self._fit_parameters), "Fitting parameters cannot be zero for all spectra!"
        return


    def prepareFitArgs(self):
        instrPars = self.loadInstrParsFileIntoArray()
        resolutionPars = self.loadResolutionPars(instrPars)

        v0, E0, delta_E, delta_Q = self.calculateKinematicsArrays(instrPars)
        kinematicArrays = np.array([v0, E0, delta_E, delta_Q])
        ySpacesForEachMass = self.convertDataXToYSpacesForEachMass(
            self._dataX, delta_Q, delta_E
        )
        kinematicArrays = np.swapaxes(kinematicArrays, 0, 1)
        ySpacesForEachMass = np.swapaxes(ySpacesForEachMass, 0, 1)
        return resolutionPars, instrPars, kinematicArrays, ySpacesForEachMass


    def loadInstrParsFileIntoArray(self):
        """Loads instrument parameters into array, from the file in the specified path"""

        data = np.loadtxt(self._ip_file, dtype=str)[1:].astype(float)
        spectra = data[:, 0]

        workspace_spectrum_list = self._workspace_being_fit.getSpectrumNumbers()
        first_spec = min(workspace_spectrum_list)
        last_spec = max(workspace_spectrum_list)

        select_rows = np.where((spectra >= first_spec) & (spectra <= last_spec))
        instrPars = data[select_rows]
        return instrPars


    @staticmethod
    def loadResolutionPars(instrPars):
        """Resolution of parameters to propagate into TOF resolution
        Output: matrix with each parameter in each column"""
        spectrums = instrPars[:, 0]
        L = len(spectrums)
        # For spec no below 135, back scattering detectors, mode is double difference
        # For spec no 135 or above, front scattering detectors, mode is single difference
        dE1 = np.where(spectrums < 135, 88.7, 73)  # meV, STD
        dE1_lorz = np.where(spectrums < 135, 40.3, 24)  # meV, HFHM
        dTOF = np.repeat(0.37, L)  # us
        dTheta = np.repeat(0.016, L)  # rad
        dL0 = np.repeat(0.021, L)  # meters
        dL1 = np.repeat(0.023, L)  # meters

        resolutionPars = np.vstack((dE1, dTOF, dTheta, dL0, dL1, dE1_lorz)).transpose()
        return resolutionPars


    def calculateKinematicsArrays(self, instrPars):
        """Kinematics quantities calculated from TOF data"""

        dataX = self._dataX

        mN, Ef, en_to_vel, vf, hbar = loadConstants()
        det, plick, angle, T0, L0, L1 = np.hsplit(instrPars, 6)  # each is of len(dataX)
        t_us = dataX - T0  # T0 is electronic delay due to instruments
        v0 = vf * L0 / (vf * t_us - L1)
        E0 = np.square(
            v0 / en_to_vel
        )  # en_to_vel is a factor used to easily change velocity to energy and vice-versa

        delta_E = E0 - Ef
        delta_Q2 = (
            2.0
            * mN
            / hbar**2
            * (E0 + Ef - 2.0 * np.sqrt(E0 * Ef) * np.cos(angle / 180.0 * np.pi))
        )
        delta_Q = np.sqrt(delta_Q2)
        return v0, E0, delta_E, delta_Q  # shape(no of spectrums, no of bins)


    def convertDataXToYSpacesForEachMass(self, dataX, delta_Q, delta_E):
        """"Calculates y spaces from TOF data, each row corresponds to one mass"""

        # Prepare arrays to broadcast
        dataX = dataX[np.newaxis, :, :]
        delta_Q = delta_Q[np.newaxis, :, :]
        delta_E = delta_E[np.newaxis, :, :]

        mN, Ef, en_to_vel, vf, hbar = loadConstants()
        masses = np.array([p.mass for p in self._profiles.values()]).reshape(-1, 1, 1)

        energyRecoil = np.square(hbar * delta_Q) / 2.0 / masses
        ySpacesForEachMass = (
            masses / hbar**2 / delta_Q * (delta_E - energyRecoil)
        )  # y-scaling
        return ySpacesForEachMass


    def _save_plots(self):
        # if IC.runningSampleWS:  # Skip saving figure if running bootstrap
        #     return

        if not self._save_figures_path:
            return

        lw = 2

        fig, ax = plt.subplots(subplot_kw={"projection": "mantid"})

        ws_data_sum = mtd[self._workspace_being_fit.name()+"_Sum"]
        ax.errorbar(ws_data_sum, fmt="k.", label="Sum of spectra")

        for key, ws in self._fit_profiles_workspaces.items():
            ws_sum = mtd[ws.name()+"_Sum"] 
            ax.plot(ws_sum, label=f'Sum of {key} profile', linewidth=lw)

        ax.set_xlabel("TOF")
        ax.set_ylabel("Counts")
        ax.set_title("Sum of NCP fits")
        ax.legend()

        fileName = self._workspace_being_fit.name() + "_profiles_sum.pdf"
        savePath = self._save_figures_path / fileName
        plt.savefig(savePath, bbox_inches="tight")
        plt.close(fig)
        return


    def _create_summed_workspaces(self):

        SumSpectra(
            InputWorkspace=self._workspace_being_fit.name(), 
            OutputWorkspace=self._workspace_being_fit.name() + "_Sum")

        for ws in self._fit_profiles_workspaces.values():
            SumSpectra(
                InputWorkspace=ws.name(),
                OutputWorkspace=ws.name() + "_Sum"
            )

    def _set_means_and_std(self):
        """Calculate mean widths and intensities from tableWorkspace"""

        fitParsTable = self._table_fit_results
        widths = np.zeros((len(self._profiles), fitParsTable.rowCount()))
        intensities = np.zeros(widths.shape)
        for i, p in enumerate(self._profiles.values()):
            widths[i] = fitParsTable.column(f"{p.label} Width")
            intensities[i] = fitParsTable.column(f"{p.label} Intensity")
        (
            meanWidths,
            stdWidths,
            meanIntensityRatios,
            stdIntensityRatios,
        ) = self.calculateMeansAndStds(widths, intensities)

        assert (
            len(meanWidths) == len(self._profiles) 
        ), "Number of mean widths must match number of profiles!"

        self.mean_widths = meanWidths     # Use setter
        self._std_widths = stdWidths
        self.mean_intensity_ratios = meanIntensityRatios   # Use setter
        self._std_intensity_ratios = stdIntensityRatios

        self.createMeansAndStdTableWS()
        return


    def createMeansAndStdTableWS(self):
        meansTableWS = CreateEmptyTableWorkspace(
            OutputWorkspace=self._workspace_being_fit.name() + "_MeanWidthsAndIntensities"
        )
        meansTableWS.addColumn(type="str", name="Mass")
        meansTableWS.addColumn(type="float", name="Mean Widths")
        meansTableWS.addColumn(type="float", name="Std Widths")
        meansTableWS.addColumn(type="float", name="Mean Intensities")
        meansTableWS.addColumn(type="float", name="Std Intensities")

        print("\nCreated Table with means and std:")
        print("\nMass    Mean \u00B1 Std Widths    Mean \u00B1 Std Intensities\n")
        for p, mean_width, std_width, mean_intensity, std_intensity in zip(
            self._profiles.values(),
            self._mean_widths,
            self._std_widths,
            self._mean_intensity_ratios,
            self._std_intensity_ratios,
        ):
            meansTableWS.addRow([p.label, mean_width, std_width, mean_intensity, std_intensity])
            print(f"{p.label:5s}  {mean_width:10.5f} \u00B1 {std_width:7.5f}  \
                    {mean_intensity:10.5f} \u00B1 {std_intensity:7.5f}\n")
        return


    def calculateMeansAndStds(self, widthsIn, intensitiesIn):
        betterWidths, betterIntensities = self.filterWidthsAndIntensities(widthsIn, intensitiesIn)

        meanWidths = np.nanmean(betterWidths, axis=1)
        stdWidths = np.nanstd(betterWidths, axis=1)

        meanIntensityRatios = np.nanmean(betterIntensities, axis=1)
        stdIntensityRatios = np.nanstd(betterIntensities, axis=1)

        return meanWidths, stdWidths, meanIntensityRatios, stdIntensityRatios


    def filterWidthsAndIntensities(self, widthsIn, intensitiesIn):
        """Puts nans in places to be ignored"""

        widths = widthsIn.copy()  # Copy to avoid accidental changes in arrays
        intensities = intensitiesIn.copy()

        zeroSpecs = np.all(
            widths == 0, axis=0
        )  # Catches all failed fits, not just masked spectra
        widths[:, zeroSpecs] = np.nan
        intensities[:, zeroSpecs] = np.nan

        meanWidths = np.nanmean(widths, axis=1)[:, np.newaxis]

        widthDeviation = np.abs(widths - meanWidths)
        stdWidths = np.nanstd(widths, axis=1)[:, np.newaxis]

        # Put nan in places where width deviation is bigger than std
        filterMask = widthDeviation > stdWidths
        betterWidths = np.where(filterMask, np.nan, widths)

        maskedIntensities = np.where(filterMask, np.nan, intensities)
        betterIntensities = maskedIntensities / np.sum(
            maskedIntensities, axis=0
        )  # Not nansum()

        # Keep this around in case it is needed again
        # When trying to estimate HToMassIdxRatio and normalization fails, skip normalization
        # if np.all(np.isnan(betterIntensities)) & IC.runningPreliminary:
        #     assert IC.noOfMSIterations == 0, (
        #         "Calculation of mean intensities failed, cannot proceed with MS correction."
        #         "Try to run again with noOfMSIterations=0."
        #     )
        #     betterIntensities = maskedIntensities
        # else:
        #     pass

        assert np.all(meanWidths != np.nan), "At least one mean of widths is nan!"
        assert np.sum(filterMask) >= 1, "No widths survive filtering condition"
        assert not (np.all(np.isnan(betterWidths))), "All filtered widths are nan"
        assert not (np.all(np.isnan(betterIntensities))), "All filtered intensities are nan"
        assert np.nanmax(betterWidths) != np.nanmin(
            betterWidths
        ), f"All filtered widths have the same value: {np.nanmin(betterWidths)}"
        assert np.nanmax(betterIntensities) != np.nanmin(
            betterIntensities
        ), f"All filtered intensities have the same value: {np.nanmin(betterIntensities)}"

        return betterWidths, betterIntensities


    def _fit_neutron_compton_profiles_to_row(self):

        if np.all(self._dataY[self._row_being_fit] == 0):
            self._table_fit_results.addRow(np.zeros(3*len(self._profiles)+3))
            return

        # Need to transform profiles into parameter array for minimize
        initial_parameters = []
        bounds = []
        for p in self._profiles.values():
            for attr in ['intensity', 'width', 'center']:
                initial_parameters.append(getattr(p, attr))
            for attr in ['intensity_bounds', 'width_bounds', 'center_bounds']:
                bounds.append(getattr(p, attr))

        result = optimize.minimize(
            self.errorFunction,
            initial_parameters,
            method="SLSQP",
            bounds=bounds,
            constraints=self._constraints,
        )
        fitPars = result["x"]

        # Pass fit parameters to results table
        noDegreesOfFreedom = len(self._dataY[self._row_being_fit]) - len(fitPars)
        normalised_chi2 = result["fun"] / noDegreesOfFreedom
        number_iterations = result["nit"]
        spectrum_number = self._instrument_params[self._row_being_fit, 0]
        tableRow = np.hstack((spectrum_number, fitPars, normalised_chi2, number_iterations))
        self._table_fit_results.addRow(tableRow)

        # Store results for easier access when calculating means
        self._fit_parameters[self._row_being_fit] = tableRow 

        print(tableRow)

        # Pass fit profiles into workspaces
        individual_ncps = self._neutron_compton_profiles(fitPars)
        for ncp, element in zip(individual_ncps, self._profiles.keys()):
            self._fit_profiles_workspaces[element].dataY(self._row_being_fit)[:] = ncp

        self._fit_profiles_workspaces['total'].dataY(self._row_being_fit)[:] = np.sum(individual_ncps, axis=0)
        return


    def errorFunction(self, pars):
        """Error function to be minimized, in TOF space"""

        ncpForEachMass = self._neutron_compton_profiles(pars)
        ncpTotal = np.sum(ncpForEachMass, axis=0)

        # Ignore any masked values from Jackknife or masked tof range
        zerosMask = self._dataY[self._row_being_fit] == 0
        ncpTotal = ncpTotal[~zerosMask]
        dataY = self._dataY[self._row_being_fit, ~zerosMask]
        dataE = self._dataE[self._row_being_fit, ~zerosMask]

        if np.all(self._dataE[self._row_being_fit] == 0):  # When errors not present
            return np.sum((ncpTotal - dataY) ** 2)

        return np.sum((ncpTotal - dataY) ** 2 / dataE**2)


    def _neutron_compton_profiles(self, pars):
        """
        Neutron Compton Profile distribution on TOF space for a single spectrum. 
        Calculated from kinematics, J(y) and resolution functions.
        """

        intensities = pars[::3].reshape(-1, 1)
        widths = pars[1::3].reshape(-1, 1)
        centers = pars[2::3].reshape(-1, 1)
        masses = np.array([p.mass for p in self._profiles.values()]).reshape(-1, 1)

        v0, E0, deltaE, deltaQ = self._kinematic_arrays[self._row_being_fit]

        gaussRes, lorzRes = self.caculateResolutionForEachMass(centers)
        totalGaussWidth = np.sqrt(widths**2 + gaussRes**2)

        JOfY = self.pseudoVoigt(self._y_space_arrays[self._row_being_fit] - centers, totalGaussWidth, lorzRes)

        FSE = (
            -numericalThirdDerivative(self._y_space_arrays[self._row_being_fit], JOfY)
            * widths**4
            / deltaQ
            * 0.72
        )
        return intensities * (JOfY + FSE) * E0 * E0 ** (-0.92) * masses / deltaQ


    def caculateResolutionForEachMass(self, centers):
        """Calculates the gaussian and lorentzian resolution
        output: two column vectors, each row corresponds to each mass"""

        gaussianResWidth = self.calcGaussianResolution(centers)
        lorentzianResWidth = self.calcLorentzianResolution(centers)
        return gaussianResWidth, lorentzianResWidth


    def kinematicsAtYCenters(self, centers):
        """v0, E0, deltaE, deltaQ at the peak of the ncpTotal for each mass"""

        shapeOfArrays = centers.shape
        proximityToYCenters = np.abs(self._y_space_arrays[self._row_being_fit] - centers)
        yClosestToCenters = proximityToYCenters.min(axis=1).reshape(shapeOfArrays)
        yCentersMask = proximityToYCenters == yClosestToCenters

        v0, E0, deltaE, deltaQ = self._kinematic_arrays[self._row_being_fit]

        # Expand arrays to match shape of yCentersMask
        v0 = v0 * np.ones(shapeOfArrays)
        E0 = E0 * np.ones(shapeOfArrays)
        deltaE = deltaE * np.ones(shapeOfArrays)
        deltaQ = deltaQ * np.ones(shapeOfArrays)

        v0 = v0[yCentersMask].reshape(shapeOfArrays)
        E0 = E0[yCentersMask].reshape(shapeOfArrays)
        deltaE = deltaE[yCentersMask].reshape(shapeOfArrays)
        deltaQ = deltaQ[yCentersMask].reshape(shapeOfArrays)
        return v0, E0, deltaE, deltaQ


    def calcGaussianResolution(self, centers):
        masses = np.array([p.mass for p in self._profiles.values()]).reshape(-1, 1)
        v0, E0, delta_E, delta_Q = self.kinematicsAtYCenters(centers)
        det, plick, angle, T0, L0, L1 = self._instrument_params[self._row_being_fit]
        dE1, dTOF, dTheta, dL0, dL1, dE1_lorz = self._resolution_params[self._row_being_fit]
        mN, Ef, en_to_vel, vf, hbar = loadConstants()

        angle = angle * np.pi / 180

        dWdE1 = 1.0 + (E0 / Ef) ** 1.5 * (L1 / L0)
        dWdTOF = 2.0 * E0 * v0 / L0
        dWdL1 = 2.0 * E0**1.5 / Ef**0.5 / L0
        dWdL0 = 2.0 * E0 / L0

        dW2 = (
            dWdE1**2 * dE1**2
            + dWdTOF**2 * dTOF**2
            + dWdL1**2 * dL1**2
            + dWdL0**2 * dL0**2
        )
        # conversion from meV^2 to A^-2, dydW = (M/q)^2
        dW2 *= (masses / hbar**2 / delta_Q) ** 2

        dQdE1 = (
            1.0
            - (E0 / Ef) ** 1.5 * L1 / L0
            - np.cos(angle) * ((E0 / Ef) ** 0.5 - L1 / L0 * E0 / Ef)
        )
        dQdTOF = 2.0 * E0 * v0 / L0
        dQdL1 = 2.0 * E0**1.5 / L0 / Ef**0.5
        dQdL0 = 2.0 * E0 / L0
        dQdTheta = 2.0 * np.sqrt(E0 * Ef) * np.sin(angle)

        dQ2 = (
            dQdE1**2 * dE1**2
            + (dQdTOF**2 * dTOF**2 + dQdL1**2 * dL1**2 + dQdL0**2 * dL0**2)
            * np.abs(Ef / E0 * np.cos(angle) - 1)
            + dQdTheta**2 * dTheta**2
        )
        dQ2 *= (mN / hbar**2 / delta_Q) ** 2

        # in A-1    #same as dy^2 = (dy/dw)^2*dw^2 + (dy/dq)^2*dq^2
        gaussianResWidth = np.sqrt(dW2 + dQ2)
        return gaussianResWidth


    def calcLorentzianResolution(self, centers):
        masses = np.array([p.mass for p in self._profiles.values()]).reshape(-1, 1)
        v0, E0, delta_E, delta_Q = self.kinematicsAtYCenters(centers)
        det, plick, angle, T0, L0, L1 = self._instrument_params[self._row_being_fit]
        dE1, dTOF, dTheta, dL0, dL1, dE1_lorz = self._resolution_params[self._row_being_fit]
        mN, Ef, en_to_vel, vf, hbar = loadConstants()

        angle = angle * np.pi / 180

        dWdE1_lor = (1.0 + (E0 / Ef) ** 1.5 * (L1 / L0)) ** 2
        # conversion from meV^2 to A^-2
        dWdE1_lor *= (masses / hbar**2 / delta_Q) ** 2

        dQdE1_lor = (
            1.0
            - (E0 / Ef) ** 1.5 * L1 / L0
            - np.cos(angle) * ((E0 / Ef) ** 0.5 + L1 / L0 * E0 / Ef)
        ) ** 2
        dQdE1_lor *= (mN / hbar**2 / delta_Q) ** 2

        lorentzianResWidth = np.sqrt(dWdE1_lor + dQdE1_lor) * dE1_lorz  # in A-1
        return lorentzianResWidth


    def pseudoVoigt(self, x, sigma, gamma):
        """Convolution between Gaussian with std sigma and Lorentzian with HWHM gamma"""
        fg, fl = 2.0 * sigma * np.sqrt(2.0 * np.log(2.0)), 2.0 * gamma
        f = 0.5346 * fl + np.sqrt(0.2166 * fl**2 + fg**2)
        eta = 1.36603 * fl / f - 0.47719 * (fl / f) ** 2 + 0.11116 * (fl / f) ** 3
        sigma_v, gamma_v = f / (2.0 * np.sqrt(2.0 * np.log(2.0))), f / 2.0
        pseudo_voigt = eta * lorentzian(x, gamma_v) + (1.0 - eta) * gaussian(x, sigma_v)

        norm = (
            np.abs(np.trapz(pseudo_voigt, x, axis=1))[:, np.newaxis] if self._run_norm_voigt else 1
        )
        return pseudo_voigt / norm


    def _get_parsed_constraints(self):

        parsed_constraints = []

        for constraint in  self._constraints:
            constraint['fun'] = self._get_parsed_constraint_function(constraint['fun']) 

            parsed_constraints.append(constraint)

        return parsed_constraints


    def _get_parsed_constraint_function(self, function_string: str):

        profile_order = [key for key in self._profiles.keys()]
        attribute_order = ['intensity', 'width', 'center']

        words = function_string.split(' ')
        for i, word in enumerate(words):
            if '.' in word:

                try:    # Skip floats 
                    float(word) 
                except ValueError: 
                    continue

                profile, attribute = word
                words[i] = f"pars[{profile_order.index(profile) + attribute_order.index(attribute)}]" 

        return eval(f"lambda pars: {' '.join(words)}")
        

    def _replace_zero_columns_with_ncp_fit(self):
        """
        If the initial input contains columns with zeros 
        (to mask resonance peaks) then these sections must be approximated 
        by the total fitted function because multiple scattering and 
        gamma correction algorithms do not accept columns with zeros.
        If no masked columns are present then the input workspace 
        for corrections is left unchanged.
        """
        dataY = self._workspace_for_corrections.extractY()
        ncp = self._fit_profiles_workspaces['total'].extractY()

        self._zero_columns_boolean_mask = np.all(dataY == 0, axis=0)  # Masked Cols

        self._workspace_for_corrections = CloneWorkspace(
            InputWorkspace=self._workspace_for_corrections.name(), 
            OutputWorkspace=self._workspace_for_corrections.name() + "_CorrectionsInput"
        )
        for row in range(self._workspace_for_corrections.getNumberHistograms()):
            # TODO: Once the option to change point to hist is removed, remove [:len(ncp)]
            self._workspace_for_corrections.dataY(row)[self._zero_columns_boolean_mask] = ncp[row, self._zero_columns_boolean_mask[:len(ncp[row])]]

        SumSpectra(
            InputWorkspace=self._workspace_for_corrections.name(), 
            OutputWorkspace=self._workspace_for_corrections.name() + "_Sum"
        )
        return


    def _remask_columns_with_zeros(self, ws_to_remask_name):
        """
        Uses previously stored information on masked columns in the
        initial workspace to set these columns again to zero on the
        workspace resulting from the multiple scattering or gamma correction.
        """
        ws_to_remask = mtd[ws_to_remask_name]
        for row in range(ws_to_remask.getNumberHistograms()):
            ws_to_remask.dataY(row)[self._zero_columns_boolean_mask] = 0
            ws_to_remask.dataE(row)[self._zero_columns_boolean_mask] = 0
        return


    def _correct_for_gamma_and_multiple_scattering(self, ws_name):

        if self._gamma_correction:
            gamma_correction_ws = self.create_gamma_workspaces()
            Minus(
                LHSWorkspace=ws_name,
                RHSWorkspace=gamma_correction_ws.name(),
                OutputWorkspace=ws_name
            )

        if self._multiple_scattering_correction:
            multiple_scattering_ws = self.create_multiple_scattering_workspaces()
            Minus(
                LHSWorkspace=ws_name,
                RHSWorkspace=multiple_scattering_ws.name(), 
                OutputWorkspace=ws_name
            )
        return


    def create_multiple_scattering_workspaces(self):
        """Creates _MulScattering and _TotScattering workspaces used for the MS correction"""

        self.createSlabGeometry(self._workspace_for_corrections)  # Sample properties for MS correction

        sampleProperties = self.calcMSCorrectionSampleProperties(self._mean_widths, self._mean_intensity_ratios)
        print(
            "\nThe sample properties for Multiple Scattering correction are:\n\n",
            sampleProperties,
            "\n",
        )

        return self.createMulScatWorkspaces(self._workspace_for_corrections, sampleProperties)


    def createSlabGeometry(self, wsNCPM):
        half_height, half_width, half_thick = (
            0.5 * self._vertical_width,
            0.5 * self._horizontal_width,
            0.5 * self._thickness,
        )
        xml_str = (
            ' <cuboid id="sample-shape"> '
            + '<left-front-bottom-point x="%f" y="%f" z="%f" /> '
            % (half_width, -half_height, half_thick)
            + '<left-front-top-point x="%f" y="%f" z="%f" /> '
            % (half_width, half_height, half_thick)
            + '<left-back-bottom-point x="%f" y="%f" z="%f" /> '
            % (half_width, -half_height, -half_thick)
            + '<right-front-bottom-point x="%f" y="%f" z="%f" /> '
            % (-half_width, -half_height, half_thick)
            + "</cuboid>"
        )

        CreateSampleShape(self._workspace_for_corrections, xml_str)


    def calcMSCorrectionSampleProperties(self, meanWidths, meanIntensityRatios):
        masses = [p.mass for p in self._profiles.values()]

        # If Backscattering mode and H is present in the sample, add H to MS properties
        if self._mode_running == "BACKWARD":
            if self._h_ratio is not None:  # If H is present, ratio is a number
                masses = np.append(masses, 1.0079)
                meanWidths = np.append(meanWidths, 5.0)

                HIntensity = self._h_ratio * meanIntensityRatios[np.argmin(masses)]
                meanIntensityRatios = np.append(meanIntensityRatios, HIntensity)
                meanIntensityRatios /= np.sum(meanIntensityRatios)

        MSProperties = np.zeros(3 * len(masses))
        MSProperties[::3] = masses
        MSProperties[1::3] = meanIntensityRatios
        MSProperties[2::3] = meanWidths
        sampleProperties = list(MSProperties)

        return sampleProperties


    def createMulScatWorkspaces(self, ws, sampleProperties):
        """Uses the Mantid algorithm for the MS correction to create two Workspaces _TotScattering and _MulScattering"""

        print("\nEvaluating the Multiple Scattering Correction...\n")
        # selects only the masses, every 3 numbers
        MS_masses = sampleProperties[::3]
        # same as above, but starts at first intensities
        MS_amplitudes = sampleProperties[1::3]

        dens, trans = VesuvioThickness(
            Masses=MS_masses,
            Amplitudes=MS_amplitudes,
            TransmissionGuess=self._transmission_guess,
            Thickness=0.1,
        )

        _TotScattering, _MulScattering = VesuvioCalculateMS(
            ws,
            NoOfMasses=len(MS_masses),
            SampleDensity=dens.cell(9, 1),
            AtomicProperties=sampleProperties,
            BeamRadius=2.5,
            NumScatters=self._multiple_scattering_order,
            NumEventsPerRun=int(self._number_of_events),
        )

        data_normalisation = Integration(ws)
        simulation_normalisation = Integration("_TotScattering")
        for workspace in ("_MulScattering", "_TotScattering"):
            Divide(
                LHSWorkspace=workspace,
                RHSWorkspace=simulation_normalisation,
                OutputWorkspace=workspace,
            )
            Multiply(
                LHSWorkspace=workspace,
                RHSWorkspace=data_normalisation,
                OutputWorkspace=workspace,
            )
            RenameWorkspace(InputWorkspace=workspace, OutputWorkspace=ws.name() + workspace)
            SumSpectra(
                ws.name() + workspace, OutputWorkspace=ws.name() + workspace + "_Sum"
            )

        DeleteWorkspaces([data_normalisation, simulation_normalisation, trans, dens])
        # The only remaining workspaces are the _MulScattering and _TotScattering
        return mtd[ws.name() + "_MulScattering"]


    def create_gamma_workspaces(self):
        """Creates _gamma_background correction workspace to be subtracted from the main workspace"""

        inputWS = self._workspace_for_corrections.name()

        profiles = self.calcGammaCorrectionProfiles(self._mean_widths, self._mean_intensity_ratios)

        # Approach below not currently suitable for current versions of Mantid, but will be in the future
        # background, corrected = VesuvioCalculateGammaBackground(InputWorkspace=inputWS, ComptonFunction=profiles)
        # DeleteWorkspace(corrected)
        # RenameWorkspace(InputWorkspace= background, OutputWorkspace = inputWS+"_Gamma_Background")

        ws = CloneWorkspace(InputWorkspace=inputWS, OutputWorkspace="tmpGC")
        for spec in range(ws.getNumberHistograms()):
            background, corrected = VesuvioCalculateGammaBackground(
                InputWorkspace=inputWS, ComptonFunction=profiles, WorkspaceIndexList=spec
            )
            ws.dataY(spec)[:], ws.dataE(spec)[:] = (
                background.dataY(0)[:],
                background.dataE(0)[:],
            )
        DeleteWorkspace(background)
        DeleteWorkspace(corrected)
        RenameWorkspace(
            InputWorkspace="tmpGC", OutputWorkspace=inputWS + "_Gamma_Background"
        )

        Scale(
            InputWorkspace=inputWS + "_Gamma_Background",
            OutputWorkspace=inputWS + "_Gamma_Background",
            Factor=0.9,
            Operation="Multiply",
        )
        return mtd[inputWS + "_Gamma_Background"]


    def calcGammaCorrectionProfiles(self, meanWidths, meanIntensityRatios):
        masses = [p.mass for p in self._profiles.values()]
        profiles = ""
        for mass, width, intensity in zip(masses, meanWidths, meanIntensityRatios):
            profiles += (
                "name=GaussianComptonProfile,Mass="
                + str(mass)
                + ",Width="
                + str(width)
                + ",Intensity="
                + str(intensity)
                + ";"
            )
        print("\n The sample properties for Gamma Correction are:\n", profiles)
        return profiles


    def _set_results(self):
        """Used to collect results from workspaces and store them in .npz files for testing."""

        self.wsFinal = mtd[self._name + str(self._number_of_iterations)]

        allIterNcp = []
        allFitWs = []
        allTotNcp = []
        allBestPar = []
        allMeanWidhts = []
        allMeanIntensities = []
        allStdWidths = []
        allStdIntensities = []
        j = 0
        while True:
            try:
                wsIterName = self._name + str(j)

                # Extract ws that were fitted
                ws = mtd[wsIterName]
                allFitWs.append(ws.extractY())

                # Extract total ncp
                totNcpWs = mtd[wsIterName + "_total_ncp"]
                allTotNcp.append(totNcpWs.extractY())

                # Extract best fit parameters
                fitParTable = mtd[wsIterName + "_fit_results"]
                bestFitPars = []
                for key in fitParTable.keys():
                    bestFitPars.append(fitParTable.column(key))
                allBestPar.append(np.array(bestFitPars).T)

                # Extract individual ncp
                allNCP = []
                for p in self._profiles.values():
                    ncpWsToAppend = mtd[
                        wsIterName + f"_{p.label}_ncp"
                    ]
                    allNCP.append(ncpWsToAppend.extractY())
                allNCP = np.swapaxes(np.array(allNCP), 0, 1)
                allIterNcp.append(allNCP)

                # Extract Mean and Std Widths, Intensities
                meansTable = mtd[wsIterName + "_MeanWidthsAndIntensities"]
                allMeanWidhts.append(meansTable.column("Mean Widths"))
                allStdWidths.append(meansTable.column("Std Widths"))
                allMeanIntensities.append(meansTable.column("Mean Intensities"))
                allStdIntensities.append(meansTable.column("Std Intensities"))

                j += 1
            except KeyError:
                break

        self.all_fit_workspaces = np.array(allFitWs)
        self.all_spec_best_par_chi_nit = np.array(allBestPar)
        self.all_tot_ncp = np.array(allTotNcp)
        self.all_ncp_for_each_mass = np.array(allIterNcp)

        self.all_mean_widths = np.array(allMeanWidhts)
        self.all_mean_intensities = np.array(allMeanIntensities)
        self.all_std_widths = np.array(allStdWidths)
        self.all_std_intensities = np.array(allStdIntensities)

    def _save_results(self):
        """Saves all of the arrays stored in this object"""

        maskedDetectorIdx = np.array(self._mask_spectra) - min(self._workspace_being_fit.getSpectrumNumbers())

        # TODO: Take out nans next time when running original results
        # Because original results were recently saved with nans, mask spectra with nans
        self.all_spec_best_par_chi_nit[:, maskedDetectorIdx, :] = np.nan
        self.all_ncp_for_each_mass[:, maskedDetectorIdx, :, :] = np.nan
        self.all_tot_ncp[:, maskedDetectorIdx, :] = np.nan

        if not self._save_results_path:
            return 

        np.savez(
            self._save_results_path,
            all_fit_workspaces=self.all_fit_workspaces,
            all_spec_best_par_chi_nit=self.all_spec_best_par_chi_nit,
            all_mean_widths=self.all_mean_widths,
            all_mean_intensities=self.all_mean_intensities,
            all_std_widths=self.all_std_widths,
            all_std_intensities=self.all_std_intensities,
            all_tot_ncp=self.all_tot_ncp,
            all_ncp_for_each_mass=self.all_ncp_for_each_mass,
        )


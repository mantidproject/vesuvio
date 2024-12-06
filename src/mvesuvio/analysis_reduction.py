import numpy as np 
import matplotlib.pyplot as plt
import scipy
import dill      # Only for converting constraints from string
from mantid.kernel import StringListValidator, Direction, IntArrayBoundedValidator, IntArrayProperty,\
     IntBoundedValidator, FloatBoundedValidator
from mantid.api import FileProperty, FileAction, PythonAlgorithm, MatrixWorkspaceProperty
from mantid.dataobjects import TableWorkspaceProperty
from mantid.simpleapi import mtd, CreateEmptyTableWorkspace, SumSpectra, \
                            CloneWorkspace, DeleteWorkspace, VesuvioCalculateGammaBackground, \
                            VesuvioCalculateMS, Scale, RenameWorkspace, Minus, CreateSampleShape, \
                            VesuvioThickness, Integration, Divide, Multiply, DeleteWorkspaces, \
                            CreateWorkspace

from mvesuvio.util.analysis_helpers import numerical_third_derivative, load_resolution, load_instrument_params, \
                                            extend_range_of_array

np.set_printoptions(suppress=True, precision=4, linewidth=200)

NEUTRON_MASS = 1.008  # a.m.u.
ENERGY_FINAL = 4906.0  # meV
ENERGY_TO_VELOCITY = 4.3737 * 1.0e-4
VELOCITY_FINAL = np.sqrt(ENERGY_FINAL) * ENERGY_TO_VELOCITY  # m/us
H_BAR = 2.0445


class VesuvioAnalysisRoutine(PythonAlgorithm):

    def summary(self):
        return "Runs the analysis reduction routine for VESUVIO."

    def category(self):
        return "VesuvioAnalysis"

    def PyInit(self):
        self.declareProperty(MatrixWorkspaceProperty(
            name="InputWorkspace",
            defaultValue="",
            direction=Direction.Input),
            doc="Workspace to fit Neutron Compton Profiles."
        )
        self.declareProperty(TableWorkspaceProperty(
            name="InputProfiles",
            defaultValue="",
            direction=Direction.Input),
            doc="Table workspace containing starting parameters for profiles."
        )
        self.declareProperty(FileProperty(
            name='InstrumentParametersFile', 
            defaultValue='', 
            action=FileAction.Load, 
            extensions=["par", "dat"]),
            doc="Filename of the instrument parameter file."
        )
        self.declareProperty(
            name="HRatioToLowestMass", 
            defaultValue=0.0,
            validator=FloatBoundedValidator(lower=0), 
            doc="Intensity ratio between H peak and lowest mass peak."
        )
        self.declareProperty(
            name="NumberOfIterations", 
            defaultValue=2,
            validator=IntBoundedValidator(lower=0)
        )
        self.declareProperty(IntArrayProperty(
            name="InvalidDetectors",
            validator=IntArrayBoundedValidator(lower=3, upper=198),
            direction=Direction.Input),
            doc="List of invalid detectors whithin range 3-198."
        )
        self.declareProperty(
            name="MultipleScatteringCorrection", 
            defaultValue=False, 
            doc="Whether to run multiple scattering correction."
        )
        self.declareProperty(
            name="GammaCorrection", 
            defaultValue=False, 
            doc="Whether to run gamma correction."
        )
        self.declareProperty(
            name="SampleVerticalWidth",
            defaultValue=-1.0,
            validator=FloatBoundedValidator(lower=0)
        )
        self.declareProperty(
            name="SampleHorizontalWidth",
            defaultValue=-1.0,
            validator=FloatBoundedValidator(lower=0)
        )
        self.declareProperty(
            name="SampleThickness",
            defaultValue=-1.0,
            validator=FloatBoundedValidator(lower=0)
        )
        self.declareProperty(
            name="ModeRunning",
            defaultValue="BACKWARD",
            validator=StringListValidator(["BACKWARD", "FORWARD"]),
            doc="Whether running backward or forward scattering.")

        self.declareProperty(
            name="OutputDirectory",
            defaultValue="",
            doc="Directory where to save analysis results."
        )
        self.declareProperty(
            name="Constraints",
            defaultValue="",
            doc="Constraints to use during fitting profiles."
        )
        self.declareProperty(
            name="TransmissionGuess",
            defaultValue=-1.0,
            validator=FloatBoundedValidator(lower=0, upper=1)
        )
        self.declareProperty(
            name="MultipleScatteringOrder",
            defaultValue=-1,
            validator=IntBoundedValidator(lower=0)
        )
        self.declareProperty(
            name="NumberOfEvents",
            defaultValue=-1,
            validator=IntBoundedValidator(lower=0)
        )
        self.declareProperty(
            name="ResultsPath",
            defaultValue="",
            doc="Directory to store results, to be deleted later."
        )
        self.declareProperty(
            name="FiguresPath",
            defaultValue="",
            doc="Directory to store figures, to be deleted later."
        )
        # Outputs
        self.declareProperty(TableWorkspaceProperty(
            name="OutputMeansTable",
            defaultValue="",
            direction=Direction.Output),
            doc="TableWorkspace containing final means of intensity and widths.")

                                    
    def PyExec(self):
        self._setup()
        self.run()

    def _setup(self):
        self._name = self.getPropertyValue("InputWorkspace")
        self._ip_file = self.getProperty("InstrumentParametersFile").value
        self._number_of_iterations = self.getProperty("NumberOfIterations").value
        self._mask_spectra = self.getProperty("InvalidDetectors").value 
        self._transmission_guess = self.getProperty("TransmissionGuess").value 
        self._multiple_scattering_order = self.getProperty("MultipleScatteringOrder").value 
        self._number_of_events = self.getProperty("NumberOfEvents").value 
        self._vertical_width = self.getProperty("SampleVerticalWidth").value 
        self._horizontal_width = self.getProperty("SampleHorizontalWidth").value 
        self._thickness = self.getProperty("SampleThickness").value 
        self._mode_running = self.getProperty("ModeRunning").value 
        self._multiple_scattering_correction = self.getProperty("MultipleScatteringCorrection").value 
        self._gamma_correction = self.getProperty("GammaCorrection").value 
        self._save_results_path = self.getProperty("ResultsPath").value
        self._save_figures_path = self.getProperty("FiguresPath").value 
        self._h_ratio = self.getProperty("HRatioToLowestMass").value 
        self._constraints = dill.loads(eval(self.getProperty("Constraints").value))
        self._profiles_table = self.getProperty("InputProfiles").value

        self._instrument_params = load_instrument_params(self._ip_file, self.getProperty("InputWorkspace").value.getSpectrumNumbers())
        self._resolution_params = load_resolution(self._instrument_params)

        # Need to transform profiles table into parameter array for optimize.minimize()
        self._initial_fit_parameters = []
        for intensity, width, center in zip(
            self._profiles_table.column("intensity"),
            self._profiles_table.column("width"),
            self._profiles_table.column("center")
        ):
            self._initial_fit_parameters.extend([intensity, width, center])

        self._initial_fit_bounds = []
        for intensity_bounds, width_bounds, center_bounds in zip(
            self._profiles_table.column("intensity_bounds"),
            self._profiles_table.column("width_bounds"),
            self._profiles_table.column("center_bounds")
        ):
            self._initial_fit_bounds.extend([eval(intensity_bounds), eval(width_bounds), eval(center_bounds)])

        # Masses need to be defined in the same order
        self._masses = np.array(self._profiles_table.column("mass"))

        # Variables changing during fit
        self._workspace_for_corrections = self.getProperty("InputWorkspace").value 
        self._workspace_being_fit = self.getProperty("InputWorkspace").value
        self._row_being_fit = 0 
        self._zero_columns_boolean_mask = None
        self._table_fit_results = None
        self._fit_profiles_workspaces = {}


    def _update_workspace_data(self):

        self._dataX = self._workspace_being_fit.extractX()
        self._dataY = self._workspace_being_fit.extractY()
        self._dataE = self._workspace_being_fit.extractE()

        self._set_kinematic_arrays(self._dataX)
        self._set_gaussian_resolution()
        self._set_lorentzian_resolution()
        self._set_y_space_arrays()

        self._fit_parameters = np.zeros((len(self._dataY), 3 * self._profiles_table.rowCount() + 3))
        self._row_being_fit = 0 
        self._table_fit_results = self._initialize_table_fit_parameters()

        # Initialise workspaces for fitted ncp 
        self._fit_profiles_workspaces = {}
        for element in self._profiles_table.column("label"):
            self._fit_profiles_workspaces[element] = self._create_emtpy_ncp_workspace(f'_{element}_ncp')
        self._fit_profiles_workspaces['total'] = self._create_emtpy_ncp_workspace(f'_total_ncp')

        # Initialise workspaces for fitted ncp 
        self._fit_fse_workspaces = {}
        for element in self._profiles_table.column("label"):
            self._fit_fse_workspaces[element] = self._create_emtpy_ncp_workspace(f'_{element}_fse')
        self._fit_fse_workspaces['total'] = self._create_emtpy_ncp_workspace(f'_total_fse')

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
        for label in self._profiles_table.column("label"):
            table.addColumn(type="float", name=f"{label} intensity")
            table.addColumn(type="float", name=f"{label} width")
            table.addColumn(type="float", name=f"{label} center ")
        table.addColumn(type="float", name="normalised chi2")
        table.addColumn(type="float", name="no of iterations")
        return table


    def _create_emtpy_ncp_workspace(self, suffix):
        return CreateWorkspace(
            DataX=self._dataX,
            DataY=np.zeros(self._dataY.size),
            DataE=np.zeros(self._dataE.size),
            Nspec=self._workspace_being_fit.getNumberHistograms(),
            UnitX="TOF",    # I had hoped for a method like .XUnit() but alas
            OutputWorkspace=self._workspace_being_fit.name()+suffix,
            ParentWorkspace=self._workspace_being_fit,
            Distribution=True
    )


    def run(self):

        assert self._profiles_table.rowCount() > 0, "Need at least one profile to run the routine!"

        # Legacy code from Bootstrap
        # if self.runningSampleWS:
        #     initialWs = RenameWorkspace(
        #         InputWorkspace=ic.sampleWS, OutputWorkspace=initialWs.name()
        #     )

        CloneWorkspace(
            InputWorkspace=self._workspace_being_fit.name(), 
            OutputWorkspace=self._name + '_0' 
        )

        for iteration in range(self._number_of_iterations + 1):

            self._workspace_being_fit = mtd[self._name + '_' + str(iteration)]
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
                OutputWorkspace=self._name + '_' + str(iteration + 1)
            )

        self._set_results()
        self._save_results()
        return self 


    def _fit_neutron_compton_profiles(self):
        """
        Performs the fit of neutron compton profiles to the workspace being fit.
        The profiles are fit on a spectrum by spectrum basis.
        """
        self.log().notice("\nFitting neutron compton profiles ...\n")

        self._row_being_fit = 0
        while self._row_being_fit != len(self._dataY):
            self._fit_neutron_compton_profiles_to_row()
            self._row_being_fit += 1

        assert np.any(self._fit_parameters), "Fitting parameters cannot be zero for all spectra!"
        return


    def _set_kinematic_arrays(self, dataX):

        # Extend range due to third derivative cutting edges
        dataX = extend_range_of_array(dataX, 6)

        det, plick, angle, T0, L0, L1 = np.hsplit(self._instrument_params, 6)  # each is of len(dataX)

        # T0 is electronic delay due to instruments
        t_us = dataX - T0  
        self._v0 = VELOCITY_FINAL * L0 / (VELOCITY_FINAL * t_us - L1)
        # en_to_vel is a factor used to easily change velocity to energy and vice-versa
        self._E0 = np.square(self._v0 / ENERGY_TO_VELOCITY)  
        self._deltaE = self._E0 - ENERGY_FINAL
        delta_Q2 = (
            2.0
            * NEUTRON_MASS 
            / H_BAR**2
            * (self._E0 + ENERGY_FINAL - 2.0 * np.sqrt(self._E0 * ENERGY_FINAL) * np.cos(angle / 180.0 * np.pi))
        )
        self._deltaQ = np.sqrt(delta_Q2)
        return


    def _set_y_space_arrays(self):

        delta_Q = self._deltaQ[np.newaxis, :, :]
        delta_E = self._deltaE[np.newaxis, :, :]
        masses = self._masses.reshape(-1, 1, 1)

        energy_recoil = np.square(H_BAR * delta_Q) / 2.0 / masses
        y_spaces = masses / H_BAR**2 / delta_Q * (delta_E - energy_recoil)

        # Swap axis so that first axis selects spectra
        self._y_space_arrays = np.swapaxes(y_spaces, 0, 1) 
        return


    def _save_plots(self):
        # if IC.runningSampleWS:  # Skip saving figure if running bootstrap
        #     return

        if not self._save_figures_path:
            return

        lw = 2

        fig, ax = plt.subplots(subplot_kw={"projection": "mantid"})

        ws_data_sum = mtd[self._workspace_being_fit.name()+"_sum"]
        ax.errorbar(ws_data_sum, fmt="k.", label="Sum of spectra")

        for key, ws in self._fit_profiles_workspaces.items():
            ws_sum = mtd[ws.name()+"_sum"] 
            ax.plot(ws_sum, label=f'Sum of {key} profile', linewidth=lw)

        ax.set_xlabel("TOF")
        ax.set_ylabel("Counts")
        ax.set_title("Sum of NCP fits")
        ax.legend()

        fileName = self._workspace_being_fit.name() + "_profiles_sum.pdf"
        savePath = self._save_figures_path + '/' + fileName
        plt.savefig(savePath, bbox_inches="tight")
        plt.close(fig)
        return


    def _create_summed_workspaces(self):

        SumSpectra(
            InputWorkspace=self._workspace_being_fit.name(), 
            OutputWorkspace=self._workspace_being_fit.name() + "_sum")

        for ws in self._fit_profiles_workspaces.values():
            SumSpectra(
                InputWorkspace=ws.name(),
                OutputWorkspace=ws.name() + "_sum"
            )

        for ws in self._fit_fse_workspaces.values():
            SumSpectra(
                InputWorkspace=ws.name(),
                OutputWorkspace=ws.name() + "_sum"
            )

    def _set_means_and_std(self):
        widths = np.zeros((self._profiles_table.rowCount(), self._table_fit_results.rowCount()))
        intensities = np.zeros(widths.shape)

        for i, label in enumerate(self._profiles_table.column("label")):
            widths[i] = self._table_fit_results.column(f"{label} width")
            intensities[i] = self._table_fit_results.column(f"{label} intensity")
            self._set_means_and_std_arrays(widths, intensities)

        self._create_means_table()
        return


    def _set_means_and_std_arrays(self, widths, intensities):
        # Remove failed fits and masked spectra
        non_zero_columns = np.any(widths!=0, axis=0)
        widths = widths[:, non_zero_columns]
        intensities = intensities[:, non_zero_columns]

        widths_mean = np.mean(widths, axis=1).reshape(-1, 1)
        widths_std = np.std(widths, axis=1).reshape(-1, 1)

        widths_deviations = np.abs(widths - widths_mean)

        # Remove width outliers
        widths[widths_deviations > widths_std] = np.nan
        intensities[widths_deviations > widths_std] = np.nan

        # Use sum instead of nansum to propagate nans
        intensities = intensities / intensities.sum(axis=0)

        self._mean_widths = np.nanmean(widths, axis=1) 
        self._std_widths = np.nanstd(widths, axis=1) 
        self._mean_intensity_ratios = np.nanmean(intensities, axis=1) 
        self._std_intensity_ratios = np.nanstd(intensities, axis=1) 
        return


    def _create_means_table(self):
        table = CreateEmptyTableWorkspace(
            OutputWorkspace=self._workspace_being_fit.name() + "_means"
        )
        table.addColumn(type="str", name="label")
        table.addColumn(type="float", name="mass")
        table.addColumn(type="float", name="mean_width")
        table.addColumn(type="float", name="std_width")
        table.addColumn(type="float", name="mean_intensity")
        table.addColumn(type="float", name="std_intensity")

        self.log().notice("\nmass    mean widths    mean intensities\n")
        for label, mass, mean_width, std_width, mean_intensity, std_intensity in zip(
            self._profiles_table.column("label"),
            self._masses,
            self._mean_widths,
            self._std_widths,
            self._mean_intensity_ratios,
            self._std_intensity_ratios,
        ):
            # Explicit conversion to float required to match profiles table
            table.addRow([label, float(mass), float(mean_width), float(std_width), float(mean_intensity), float(std_intensity)])
            self.log().notice(f"{label:6s}  {mean_width:10.5f} \u00B1 {std_width:7.5f}" + \
                f"{mean_intensity:10.5f} \u00B1 {std_intensity:7.5f}\n")

        self.setPropertyValue("OutputMeansTable", table.name())
        return table


    def _fit_neutron_compton_profiles_to_row(self):

        if np.all(self._dataY[self._row_being_fit] == 0):
            self._table_fit_results.addRow(np.zeros(3*self._profiles_table.rowCount()+3))
            return

        result = scipy.optimize.minimize(
            self._error_function,
            self._initial_fit_parameters,
            method="SLSQP",
            bounds=self._initial_fit_bounds,
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

        self.log().notice(' '.join(str(tableRow).split(",")).replace('[', '').replace(']', ''))

        # Pass fit profiles into workspaces
        ncp_for_each_mass, fse_for_each_mass = self._neutron_compton_profiles(fitPars)
        for ncp, fse, element in zip(ncp_for_each_mass, fse_for_each_mass, self._profiles_table.column("label")):
            self._fit_profiles_workspaces[element].dataY(self._row_being_fit)[:] = ncp
            self._fit_fse_workspaces[element].dataY(self._row_being_fit)[:] = fse

        self._fit_profiles_workspaces['total'].dataY(self._row_being_fit)[:] = np.sum(ncp_for_each_mass, axis=0)
        self._fit_fse_workspaces['total'].dataY(self._row_being_fit)[:] = np.sum(fse_for_each_mass, axis=0)
        return


    def _error_function(self, pars):
        """Error function to be minimized, in TOF space"""

        ncp_for_each_mass, fse_for_each_mass = self._neutron_compton_profiles(pars)

        ncp_total = np.sum(ncp_for_each_mass, axis=0)
        data_y = self._dataY[self._row_being_fit]
        data_e = self._dataE[self._row_being_fit]

        # Ignore any masked values on tof range
        ncp_total = ncp_total[np.nonzero(data_y)]
        data_y = data_y[np.nonzero(data_y)]
        data_e = data_e[np.nonzero(data_y)]

        if np.all(data_e == 0):  # When errors not present
            return np.sum((ncp_total - data_y) ** 2)

        return np.sum((ncp_total - data_y) ** 2 / data_e**2)


    def _neutron_compton_profiles(self, pars):
        """
        Neutron Compton Profile distribution on TOF space for a single spectrum. 
        Calculated from kinematics, J(y) and resolution functions.
        """
        intensities = pars[::3].reshape(-1, 1)
        widths = pars[1::3].reshape(-1, 1)
        centers = pars[2::3].reshape(-1, 1)
        masses = self._masses.reshape(-1, 1)

        gaussian_width = self._get_gaussian_resolution(centers)
        lorentzian_width = self._get_lorentzian_resolution(centers)
        total_gaussian_width = np.sqrt(widths**2 + gaussian_width**2)

        JOfY = scipy.special.voigt_profile(self._y_space_arrays[self._row_being_fit] - centers, total_gaussian_width, lorentzian_width)

        # Third derivative cuts edges of array by 6 indices
        JOfY_third_derivative = numerical_third_derivative(self._y_space_arrays[self._row_being_fit], JOfY)

        deltaQ = self._deltaQ[self._row_being_fit, 6: -6]
        E0 = self._E0[self._row_being_fit, 6: -6]
        JOfY = JOfY[:, 6:-6]

        FSE = - JOfY_third_derivative * widths**4 / deltaQ * 0.72

        NCP = intensities * (JOfY+FSE) * E0 * E0 ** (-0.92) * masses / deltaQ
        FSE = intensities * FSE * E0 * E0 ** (-0.92) * masses / deltaQ
        return NCP, FSE


    def _get_gaussian_resolution(self, centers):
        proximity_to_y_centers = np.abs(self._y_space_arrays[self._row_being_fit] - centers)
        gauss_resolution = self._gaussian_resolution[self._row_being_fit]
        assert proximity_to_y_centers.shape==gauss_resolution.shape
        return np.take_along_axis(gauss_resolution, proximity_to_y_centers.argmin(axis=1, keepdims=True), axis=1)


    def _set_gaussian_resolution(self):
        masses = self._masses.reshape(-1, 1, 1)
        v0 = self._v0
        E0 = self._E0
        delta_Q = self._deltaQ
        det, plick, angle, T0, L0, L1 = np.hsplit(self._instrument_params, 6)
        dE1, dTOF, dTheta, dL0, dL1, dE1_lorz = np.hsplit(self._resolution_params, 6)

        angle = angle * np.pi / 180

        dWdE1 = 1.0 + (E0 / ENERGY_FINAL) ** 1.5 * (L1 / L0)
        dWdTOF = 2.0 * E0 * v0 / L0
        dWdL1 = 2.0 * E0**1.5 / ENERGY_FINAL**0.5 / L0
        dWdL0 = 2.0 * E0 / L0

        dW2 = (
            dWdE1**2 * dE1**2
            + dWdTOF**2 * dTOF**2
            + dWdL1**2 * dL1**2
            + dWdL0**2 * dL0**2
        ) * np.ones((masses.size, 1, 1))
        # conversion from meV^2 to A^-2, dydW = (M/q)^2
        dW2 *= (masses / H_BAR**2 / delta_Q) ** 2

        dQdE1 = (
            1.0
            - (E0 / ENERGY_FINAL) ** 1.5 * L1 / L0
            - np.cos(angle) * ((E0 / ENERGY_FINAL) ** 0.5 - L1 / L0 * E0 / ENERGY_FINAL)
        )
        dQdTOF = 2.0 * E0 * v0 / L0
        dQdL1 = 2.0 * E0**1.5 / L0 / ENERGY_FINAL**0.5
        dQdL0 = 2.0 * E0 / L0
        dQdTheta = 2.0 * np.sqrt(E0 * ENERGY_FINAL) * np.sin(angle)

        dQ2 = (
            dQdE1**2 * dE1**2
            + (dQdTOF**2 * dTOF**2 + dQdL1**2 * dL1**2 + dQdL0**2 * dL0**2)
            * np.abs(ENERGY_FINAL / E0 * np.cos(angle) - 1)
            + dQdTheta**2 * dTheta**2
        )
        dQ2 *= (NEUTRON_MASS / H_BAR**2 / delta_Q) ** 2

        # in A-1    #same as dy^2 = (dy/dw)^2*dw^2 + (dy/dq)^2*dq^2
        gaussianResWidth = np.sqrt(dW2 + dQ2)
        self._gaussian_resolution = np.swapaxes(gaussianResWidth, 0, 1)
        return


    def _get_lorentzian_resolution(self, centers):
        proximity_to_y_centers = np.abs(self._y_space_arrays[self._row_being_fit] - centers)
        lorentzian_resolution = self._lorentzian_resolution[self._row_being_fit]
        assert proximity_to_y_centers.shape==lorentzian_resolution.shape
        return np.take_along_axis(lorentzian_resolution, proximity_to_y_centers.argmin(axis=1, keepdims=True), axis=1)


    def _set_lorentzian_resolution(self):
        masses = self._masses.reshape(-1, 1, 1)
        E0 = self._E0
        delta_Q = self._deltaQ
        det, plick, angle, T0, L0, L1 = np.hsplit(self._instrument_params, 6)
        dE1, dTOF, dTheta, dL0, dL1, dE1_lorz = np.hsplit(self._resolution_params, 6)

        angle = angle * np.pi / 180

        dWdE1_lor = (1.0 + (E0 / ENERGY_FINAL) ** 1.5 * (L1 / L0)) ** 2 * np.ones((masses.size, 1, 1))
        # conversion from meV^2 to A^-2
        dWdE1_lor *= (masses / H_BAR**2 / delta_Q) ** 2 

        dQdE1_lor = (
            1.0
            - (E0 / ENERGY_FINAL) ** 1.5 * L1 / L0
            - np.cos(angle) * ((E0 / ENERGY_FINAL) ** 0.5 + L1 / L0 * E0 / ENERGY_FINAL)
        ) ** 2
        dQdE1_lor *= (NEUTRON_MASS / H_BAR**2 / delta_Q) ** 2

        lorentzianResWidth = np.sqrt(dWdE1_lor + dQdE1_lor) * dE1_lorz  # in A-1
        self._lorentzian_resolution = np.swapaxes(lorentzianResWidth, 0, 1)
        return


    def _get_parsed_constraints(self):

        parsed_constraints = []

        for constraint in  self._constraints:
            constraint['fun'] = self._get_parsed_constraint_function(constraint['fun']) 

            parsed_constraints.append(constraint)

        return parsed_constraints


    def _get_parsed_constraint_function(self, function_string: str):

        profile_order = [label for label in self._profiles_table.column("label")]
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

        for row in range(self._workspace_for_corrections.getNumberHistograms()):
            self._workspace_for_corrections.dataY(row)[self._zero_columns_boolean_mask] = ncp[row, self._zero_columns_boolean_mask]

        SumSpectra(
            InputWorkspace=self._workspace_for_corrections.name(), 
            OutputWorkspace=self._workspace_for_corrections.name() + "_sum"
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
        self.log().notice(
            "\nSample properties for multiple scattering correction:\n\n" + \
            "mass   intensity   width\n" + \
            str(np.array(sampleProperties).reshape(-1, 3)).replace('[', '').replace(']', '') + "\n"
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
        masses = self._masses

        # If Backscattering mode and H is present in the sample, add H to MS properties
        if self._mode_running == "BACKWARD":
            if self._h_ratio > 0:  # If H is present, ratio is a number
                HIntensity = self._h_ratio * meanIntensityRatios[np.argmin(masses)]
                meanIntensityRatios = np.append(meanIntensityRatios, HIntensity)
                meanIntensityRatios /= np.sum(meanIntensityRatios)

                masses = np.append(masses, 1.0079)
                meanWidths = np.append(meanWidths, 5.0)

        MSProperties = np.zeros(3 * len(masses))
        MSProperties[::3] = masses
        MSProperties[1::3] = meanIntensityRatios
        MSProperties[2::3] = meanWidths
        sampleProperties = list(MSProperties)

        return sampleProperties


    def createMulScatWorkspaces(self, ws, sampleProperties):
        """Uses the Mantid algorithm for the MS correction to create two Workspaces _tot_sctr and _mltp_sctr"""

        self.log().notice("\nEvaluating multiple scattering correction ...\n")
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

        _tot_sctr, _mltp_sctr = VesuvioCalculateMS(
            ws,
            NoOfMasses=len(MS_masses),
            SampleDensity=dens.cell(9, 1),
            AtomicProperties=sampleProperties,
            BeamRadius=2.5,
            NumScatters=self._multiple_scattering_order,
            NumEventsPerRun=int(self._number_of_events),
        )

        data_normalisation = Integration(ws)
        simulation_normalisation = Integration("_tot_sctr")
        for workspace in ("_mltp_sctr", "_tot_sctr"):
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
                ws.name() + workspace, OutputWorkspace=ws.name() + workspace + "_sum"
            )

        DeleteWorkspaces([data_normalisation, simulation_normalisation, trans, dens])
        # The only remaining workspaces are the _mltp_sctr and _tot_sctr
        return mtd[ws.name() + "_mltp_sctr"]


    def create_gamma_workspaces(self):
        """Creates _gamma_background correction workspace to be subtracted from the main workspace"""

        inputWS = self._workspace_for_corrections.name()

        profiles = self.calcGammaCorrectionProfiles(self._mean_widths, self._mean_intensity_ratios)

        background, corrected = VesuvioCalculateGammaBackground(InputWorkspace=inputWS, ComptonFunction=profiles)
        DeleteWorkspace(corrected)
        RenameWorkspace(InputWorkspace= background, OutputWorkspace = inputWS + "_gamma_backgr")

        Scale(
            InputWorkspace=inputWS + "_gamma_backgr",
            OutputWorkspace=inputWS + "_gamma_backgr",
            Factor=0.9,
            Operation="Multiply",
        )
        return mtd[inputWS + "_gamma_backgr"]


    def calcGammaCorrectionProfiles(self, meanWidths, meanIntensityRatios):
        profiles = ""
        for mass, width, intensity in zip(self._masses, meanWidths, meanIntensityRatios):
            profiles += (
                "name=GaussianComptonProfile,Mass="
                + str(mass)
                + ",Width="
                + str(width)
                + ",Intensity="
                + str(intensity)
                + ";"
            )
        self.log().notice("\nThe sample properties for Gamma Correction are:\n\n" + \
                str(profiles).replace(';', '\n\n').replace(',', '\n'))
        return profiles


    def _set_results(self):
        """Used to collect results from workspaces and store them in .npz files for testing."""

        self.wsFinal = mtd[self._name + '_' + str(self._number_of_iterations)]

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
                wsIterName = self._name + '_' + str(j)

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
                for label in self._profiles_table.column("label"):
                    ncpWsToAppend = mtd[
                        wsIterName + f"_{label}_ncp"
                    ]
                    allNCP.append(ncpWsToAppend.extractY())
                allNCP = np.swapaxes(np.array(allNCP), 0, 1)
                allIterNcp.append(allNCP)

                # Extract Mean and Std Widths, Intensities
                meansTable = mtd[wsIterName + "_means"]
                allMeanWidhts.append(meansTable.column("mean_width"))
                allStdWidths.append(meansTable.column("std_width"))
                allMeanIntensities.append(meansTable.column("mean_intensity"))
                allStdIntensities.append(meansTable.column("std_intensity"))

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


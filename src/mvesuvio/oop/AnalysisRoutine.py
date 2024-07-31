from mvesuvio.oop.NeutronComptonProfile import NeutronComptonProfile 
from mvesuvio.oop.analysis_helpers import extractWS, histToPointData, loadConstants, \
                                            gaussian, lorentizian, numericalThirdDerivative, \
                                            switchFirstTwoAxis, createWS
from mantid.simpleapi import mtd, CreateEmptyTableWorkspace, MaskDetectors, SumSpectra, \
                            CloneWorkspace
from mvesuvio.analysis_fitting import passDataIntoWS, replaceZerosWithNCP
import numpy as np 
from scipy import optimize
import sys

class AnalysisRoutine:

    def __init__(self, workspace, ip_file, number_of_iterations, mask_spectra,
                 multiple_scattering_correction, gamma_correction,
                 transmission_guess=None, multiple_scattering_order=None, number_of_events=None):

        self._workspace_to_fit = workspace
        self._name = workspace.name()
        self._ip_file = ip_file
        self._number_of_iterations = number_of_iterations
        spectrum_list = workspace.getSpectrumNumbers()
        self._firstSpec = min(spectrum_list)
        self._lastSpec = max(spectrum_list)
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
        self._run_hist_data = False #True
        self._run_norm_voigt = False

        # Links to another AnalysisRoutine object:
        self._profiles_destination = None
        self._h_ratio_destination = None


    def add_profiles(self, *args: NeutronComptonProfile):
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


    def _preprocess(self):
        # Set up variables used during fitting
        self._masses = np.array([p.mass for p in self._profiles.values()])
        self._dataX, self._dataY, self._dataE = extractWS(self._workspace_to_fit)
        resolutionPars, instrPars, kinematicArrays, ySpacesForEachMass = self.prepareFitArgs()
        self._resolution_params = resolutionPars
        self._instrument_params = instrPars
        self._kinematic_arrays = kinematicArrays
        self._y_space_arrays = ySpacesForEachMass

        self._initial_fit_parameters = []
        self._bounds = []
        for p in self._profiles.values():
            self._initial_fit_parameters.append(p.intensity)
            self._initial_fit_parameters.append(p.width)
            self._initial_fit_parameters.append(p.center)
            self._bounds.append(p._intensity_bounds)
            self._bounds.append(p._width_bounds)
            self._bounds.append(p._center_bounds)

        # self._intensities = np.array([p.intensity for p in self._profiles.values()])[:, np.newaxis]
        # self._widths = np.array([p.width for p in self._profiles.values()])[:, np.newaxis]
        # self._centers = np.array([p.center for p in self._profiles.values()])[:, np.newaxis]
        self._fit_parameters = np.zeros((len(self._dataY), 3 * len(self._profiles) + 3))

        self._fig_save_path = None
        

    def run(self):

        assert len(self.profiles) > 0, "Add profiles before atempting to run the routine!"

        self._preprocess()

        self.createTableInitialParameters()

        # Legacy code from Bootstrap
        # if self.runningSampleWS:
        #     initialWs = RenameWorkspace(
        #         InputWorkspace=ic.sampleWS, OutputWorkspace=initialWs.name()
        #     )

        self._workspace_to_fit = CloneWorkspace(
            InputWorkspace=self._workspace_to_fit, 
            OutputWorkspace=self._name + "0"
        )

        for iteration in range(self._number_of_iterations + 1):
            # Workspace from previous iteration
            wsToBeFitted = mtd[self._name + str(iteration)]

            ncpTotal = self.fitNcpToWorkspace(wsToBeFitted)

            mWidths, stdWidths, mIntRatios, stdIntRatios = self.extractMeans(wsToBeFitted.name())

            self.createMeansAndStdTableWS(
                wsToBeFitted.name(), mWidths, stdWidths, mIntRatios, stdIntRatios
            )

            # When last iteration, skip MS and GC
            if iteration == self._number_of_iterations:
                break

            # TODO: Refactored until here -------

            # Replace zero columns (bins) with ncp total fit
            # If ws has no zero column, then remains unchanged
            if iteration == 0:
                wsNCPM = replaceZerosWithNCP(mtd[ic.name], ncpTotal)

            CloneWorkspace(InputWorkspace=ic.name, OutputWorkspace="tmpNameWs")

            if ic.GammaCorrectionFlag:
                wsGC = createWorkspacesForGammaCorrection(ic, mWidths, mIntRatios, wsNCPM)
                Minus(
                    LHSWorkspace="tmpNameWs", RHSWorkspace=wsGC, OutputWorkspace="tmpNameWs"
                )

            if ic.MSCorrectionFlag:
                wsMS = createWorkspacesForMSCorrection(ic, mWidths, mIntRatios, wsNCPM)
                Minus(
                    LHSWorkspace="tmpNameWs", RHSWorkspace=wsMS, OutputWorkspace="tmpNameWs"
                )

            remaskValues(ic.name, "tmpNameWS")  # Masks cols in the same place as in ic.name
            RenameWorkspace(
                InputWorkspace="tmpNameWs", OutputWorkspace=ic.name + str(iteration + 1)
            )

        wsFinal = mtd[ic.name + str(ic.noOfMSIterations)]
        fittingResults = resultsObject(ic)
        fittingResults.save()
        return wsFinal, fittingResults


    def remaskValues(wsName, wsToMaskName):
        """
        Uses the ws before the MS correction to look for masked columns or dataE
        and implement the same masked values after the correction.
        """
        ws = mtd[wsName]
        dataX, dataY, dataE = extractWS(ws)
        mask = np.all(dataY == 0, axis=0)

        wsM = mtd[wsToMaskName]
        dataXM, dataYM, dataEM = extractWS(wsM)
        dataYM[:, mask] = 0
        if np.all(dataE == 0):
            dataEM = np.zeros(dataEM.shape)

        passDataIntoWS(dataXM, dataYM, dataEM, wsM)
        return


    def createTableInitialParameters(self):
        # print("\nRUNNING ", self.modeRunning, " SCATTERING.\n")
        # if self.modeRunning == "BACKWARD":
        #     print(f"\nH ratio to next lowest mass = {self._h_ratio}\n")

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


    def fitNcpToWorkspace(self, ws):
        """
        Performs the fit of ncp to the workspace.
        Firtly the arrays required for the fit are prepared and then the fit is performed iteratively
        on a spectrum by spectrum basis.
        """

        if self._run_hist_data:  # Converts point data from workspaces to histogram data
            self._dataY, self._dataX, self._dataE = histToPointData(self._dataY, self._dataX, self._dataE)

        print("\nFitting NCP:\n")

        arrFitPars = self.fitNcpToArray()

        self.createTableWSForFitPars(ws.name(), len(self._profiles), arrFitPars)

        arrBestFitPars = arrFitPars[:, 1:-2]

        ncpForEachMass, ncpTotal = self.calculateNcpArr(arrBestFitPars)
        ncpSumWSs = self.createNcpWorkspaces(ncpForEachMass, ncpTotal, ws)

        wsDataSum = SumSpectra(InputWorkspace=ws, OutputWorkspace=ws.name() + "_Sum")
        self.plotSumNCPFits(wsDataSum, *ncpSumWSs)
        return ncpTotal


    def prepareFitArgs(self):
        instrPars = self.loadInstrParsFileIntoArray()
        resolutionPars = self.loadResolutionPars(instrPars)

        v0, E0, delta_E, delta_Q = self.calculateKinematicsArrays(instrPars)
        kinematicArrays = np.array([v0, E0, delta_E, delta_Q])
        ySpacesForEachMass = self.convertDataXToYSpacesForEachMass(
            self._dataX, delta_Q, delta_E
        )
        kinematicArrays = self.reshapeArrayPerSpectrum(kinematicArrays)
        ySpacesForEachMass = self.reshapeArrayPerSpectrum(ySpacesForEachMass)
        return resolutionPars, instrPars, kinematicArrays, ySpacesForEachMass


    def loadInstrParsFileIntoArray(self):
        """Loads instrument parameters into array, from the file in the specified path"""

        data = np.loadtxt(self._ip_file, dtype=str)[1:].astype(float)

        spectra = data[:, 0]
        select_rows = np.where((spectra >= self._firstSpec) & (spectra <= self._lastSpec))
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


    @staticmethod
    def reshapeArrayPerSpectrum(A):
        """
        Exchanges the first two axes of an array A.
        Rearranges array to match iteration per spectrum
        """
        return np.stack(np.split(A, len(A), axis=0), axis=2)[0]


    def convertDataXToYSpacesForEachMass(self, dataX, delta_Q, delta_E):
        "Calculates y spaces from TOF data, each row corresponds to one mass"

        # Prepare arrays to broadcast
        dataX = dataX[np.newaxis, :, :]
        delta_Q = delta_Q[np.newaxis, :, :]
        delta_E = delta_E[np.newaxis, :, :]

        mN, Ef, en_to_vel, vf, hbar = loadConstants()
        masses = self._masses.reshape(self._masses.size, 1, 1)

        energyRecoil = np.square(hbar * delta_Q) / 2.0 / masses
        ySpacesForEachMass = (
            masses / hbar**2 / delta_Q * (delta_E - energyRecoil)
        )  # y-scaling
        return ySpacesForEachMass


    def fitNcpToArray(self):
        """Takes dataY as a 2D array and returns the 2D array best fit parameters."""

        for row in range(len(self._dataY)):

            specFitPars = self.fitNcpToSingleSpec(row)

            self._fit_parameters[row] = specFitPars

            if np.all(specFitPars == 0):
                print("Skipped spectra.")
            else:
                with np.printoptions(
                    suppress=True, precision=4, linewidth=200, threshold=sys.maxsize
                ):
                    print(specFitPars)

        assert ~np.all(
            self._fit_parameters == 0
        ), "Either Fits are all zero or assignment of fitting not working"
        return self._fit_parameters


    def createTableWSForFitPars(self, wsName, noOfMasses, arrFitPars):
        tableWS = CreateEmptyTableWorkspace(
            OutputWorkspace=wsName + "_Best_Fit_NCP_Parameters"
        )
        tableWS.setTitle("SCIPY Fit")
        tableWS.addColumn(type="float", name="Spec Idx")
        for i in range(int(noOfMasses)):
            tableWS.addColumn(type="float", name=f"Intensity {i}")
            tableWS.addColumn(type="float", name=f"Width {i}")
            tableWS.addColumn(type="float", name=f"Center {i}")
        tableWS.addColumn(type="float", name="Norm Chi2")
        tableWS.addColumn(type="float", name="No Iter")

        for row in arrFitPars:  # Pass array onto table ws
            tableWS.addRow(row)
        return


    def calculateNcpArr(self, arrBestFitPars):
        """Calculates the matrix of NCP from matrix of best fit parameters"""

        allNcpForEachMass = []
        for row in range(len(arrBestFitPars)):
            ncpForEachMass = self.calculateNcpRow(arrBestFitPars[row], row)

            allNcpForEachMass.append(ncpForEachMass)

        allNcpForEachMass = np.array(allNcpForEachMass)
        allNcpTotal = np.sum(allNcpForEachMass, axis=1)
        return allNcpForEachMass, allNcpTotal


    def calculateNcpRow(self, initPars, row):
        """input: all row shape
        output: row shape with the ncpTotal for each mass"""

        if np.all(initPars == 0):
            # return np.zeros(self._y_space_arrays.shape)
            return np.zeros_like(self._y_space_arrays[row])

        ncpForEachMass, ncpTotal = self.calculateNcpSpec(initPars, row) 
        return ncpForEachMass


    def createNcpWorkspaces(self, ncpForEachMass, ncpTotal, ws):
        """Creates workspaces from ncp array data"""

        # Need to rearrage array of yspaces into seperate arrays for each mass
        ncpForEachMass = switchFirstTwoAxis(ncpForEachMass)

        # Use ws dataX to match with histogram data
        dataX = ws.extractX()[
            :, : ncpTotal.shape[1]
        ]  # Make dataX match ncp shape automatically
        assert (
            ncpTotal.shape == dataX.shape
        ), "DataX and DataY in ws need to be the same shape."

        ncpTotWS = CloneWorkspace(InputWorkspace=ws.name(), OutputWorkspace=ws.name() + "_TOF_Fitted_Profiles")
        passDataIntoWS(dataX, ncpTotal, np.zeros_like(dataX), ncpTotWS)
        # ncpTotWS = createWS(
        #     dataX, ncpTotal, np.zeros(dataX.shape), ws.name() + "_TOF_Fitted_Profiles"
        # )
        # MaskDetectors(Workspace=ncpTotWS, WorkspaceIndexList=ic.maskedDetectorIdx)
        MaskDetectors(Workspace=ncpTotWS, SpectraList=self._mask_spectra)
        wsTotNCPSum = SumSpectra(
            InputWorkspace=ncpTotWS, OutputWorkspace=ncpTotWS.name() + "_Sum"
        )

        # Individual ncp workspaces
        wsMNCPSum = []
        for i, ncp_m in enumerate(ncpForEachMass):
            ncpMWS =  CloneWorkspace(InputWorkspace=ws.name(), OutputWorkspace=ws.name()+"_TOF_Fitted_Profile_" + str(i))
            passDataIntoWS(dataX, ncp_m, np.zeros_like(dataX), ncpMWS)
            # ncpMWS = createWS(
            #     dataX,
            #     ncp_m,
            #     np.zeros(dataX.shape),
            #     ws.name() + "_TOF_Fitted_Profile_" + str(i),
            # )
            MaskDetectors(Workspace=ncpMWS, SpectraList=self._mask_spectra)
            wsNCPSum = SumSpectra(
                InputWorkspace=ncpMWS, OutputWorkspace=ncpMWS.name() + "_Sum"
            )
            wsMNCPSum.append(wsNCPSum)

        return wsTotNCPSum, wsMNCPSum


    def plotSumNCPFits(self, wsDataSum, wsTotNCPSum, wsMNCPSum):
        # if IC.runningSampleWS:  # Skip saving figure if running bootstrap
        #     return

        if not self._fig_save_path:
            return
        lw = 2

        fig, ax = plt.subplots(subplot_kw={"projection": "mantid"})
        ax.errorbar(wsDataSum, "k.", label="Spectra")

        ax.plot(wsTotNCPSum, "r-", label="Total NCP", linewidth=lw)
        for m, wsNcp in zip(self._masses, wsMNCPSum):
            ax.plot(wsNcp, label=f"NCP m={m}", linewidth=lw)

        ax.set_xlabel("TOF")
        ax.set_ylabel("Counts")
        ax.set_title("Sum of NCP fits")
        ax.legend()

        fileName = wsDataSum.name() + "_NCP_Fits.pdf"
        savePath = self._fig_save_path / fileName
        plt.savefig(savePath, bbox_inches="tight")
        plt.close(fig)
        return


    def extractMeans(self, wsName):
        """Extract widths and intensities from tableWorkspace"""

        fitParsTable = mtd[wsName + "_Best_Fit_NCP_Parameters"]
        widths = np.zeros((self._masses.size, fitParsTable.rowCount()))
        intensities = np.zeros(widths.shape)
        for i in range(self._masses.size):
            widths[i] = fitParsTable.column(f"Width {i}")
            intensities[i] = fitParsTable.column(f"Intensity {i}")

        (
            meanWidths,
            stdWidths,
            meanIntensityRatios,
            stdIntensityRatios,
        ) = self.calculateMeansAndStds(widths, intensities)

        assert (
            len(widths) == self._masses.size 
        ), "Widths and intensities must be in shape (noOfMasses, noOfSpec)"
        return meanWidths, stdWidths, meanIntensityRatios, stdIntensityRatios


    def createMeansAndStdTableWS(
        self, wsName, meanWidths, stdWidths, meanIntensityRatios, stdIntensityRatios
    ):
        meansTableWS = CreateEmptyTableWorkspace(
            OutputWorkspace=wsName + "_Mean_Widths_And_Intensities"
        )
        meansTableWS.addColumn(type="float", name="Mass")
        meansTableWS.addColumn(type="float", name="Mean Widths")
        meansTableWS.addColumn(type="float", name="Std Widths")
        meansTableWS.addColumn(type="float", name="Mean Intensities")
        meansTableWS.addColumn(type="float", name="Std Intensities")

        print("\nCreated Table with means and std:")
        print("\nMass    Mean \u00B1 Std Widths    Mean \u00B1 Std Intensities\n")
        for m, mw, stdw, mi, stdi in zip(
            self._masses.astype(float),
            meanWidths,
            stdWidths,
            meanIntensityRatios,
            stdIntensityRatios,
        ):
            meansTableWS.addRow([m, mw, stdw, mi, stdi])
            print(f"{m:5.2f}  {mw:10.5f} \u00B1 {stdw:7.5f}  {mi:10.5f} \u00B1 {stdi:7.5f}")
        print("\n")
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

        # TODO: sort this out
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
        ), f"All fitered widths have the same value: {np.nanmin(betterWidths)}"
        assert np.nanmax(betterIntensities) != np.nanmin(
            betterIntensities
        ), f"All fitered widths have the same value: {np.nanmin(betterIntensities)}"

        return betterWidths, betterIntensities


    def fitNcpToSingleSpec(self, row):
        """Fits the NCP and returns the best fit parameters for one spectrum"""

        if np.all(self._dataY[row] == 0):
            return np.zeros(len(self._initial_fit_parameters) + 3)

        result = optimize.minimize(
            self.errorFunction,
            self._initial_fit_parameters,
            args=(row),
            method="SLSQP",
            bounds=self._bounds,
            constraints=self._constraints,
        )
        fitPars = result["x"]

        noDegreesOfFreedom = len(self._dataY) - len(fitPars)
        specFitPars = np.append(self._instrument_params[row, 0], fitPars)
        return np.append(specFitPars, [result["fun"] / noDegreesOfFreedom, result["nit"]])


    def errorFunction(self, pars, row):
        """Error function to be minimized, operates in TOF space"""

        ncpForEachMass, ncpTotal = self.calculateNcpSpec(pars, row)

        # Ignore any masked values from Jackknife or masked tof range
        zerosMask = self._dataY[row] == 0
        ncpTotal = ncpTotal[~zerosMask]
        dataYf = self._dataY[row, ~zerosMask]
        dataEf = self._dataE[row, ~zerosMask]

        if np.all(self._dataE[row] == 0):  # When errors not present
            return np.sum((ncpTotal - dataYf) ** 2)

        return np.sum((ncpTotal - dataYf) ** 2 / dataEf**2)


    def calculateNcpSpec(self, pars, row):
        """Creates a synthetic C(t) to be fitted to TOF values of a single spectrum, from J(y) and resolution functions
        Shapes: datax (1, n), ySpacesForEachMass (4, n), res (4, 2), deltaQ (1, n), E0 (1,n),
        where n is no of bins"""

        masses, intensities, widths, centers = self.prepareArraysFromPars(pars)
        v0, E0, deltaE, deltaQ = self._kinematic_arrays[row]

        gaussRes, lorzRes = self.caculateResolutionForEachMass(centers, row)
        totalGaussWidth = np.sqrt(widths**2 + gaussRes**2)

        JOfY = self.pseudoVoigt(self._y_space_arrays[row] - centers, totalGaussWidth, lorzRes)

        FSE = (
            -numericalThirdDerivative(self._y_space_arrays[row], JOfY)
            * widths**4
            / deltaQ
            * 0.72
        )
        ncpForEachMass = intensities * (JOfY + FSE) * E0 * E0 ** (-0.92) * masses / deltaQ
        ncpTotal = np.sum(ncpForEachMass, axis=0)
        return ncpForEachMass, ncpTotal


    def prepareArraysFromPars(self, initPars):
        """Extracts the intensities, widths and centers from the fitting parameters
        Reshapes all of the arrays to collumns, for the calculation of the ncp,"""

        masses = self._masses[:, np.newaxis]
        intensities = initPars[::3].reshape(masses.shape)
        widths = initPars[1::3].reshape(masses.shape)
        centers = initPars[2::3].reshape(masses.shape)
        return masses, intensities, widths, centers


    def caculateResolutionForEachMass(self, centers, row):
        """Calculates the gaussian and lorentzian resolution
        output: two column vectors, each row corresponds to each mass"""

        gaussianResWidth = self.calcGaussianResolution(centers, row)
        lorentzianResWidth = self.calcLorentzianResolution(centers, row)
        return gaussianResWidth, lorentzianResWidth


    def kinematicsAtYCenters(self, centers, row):
        """v0, E0, deltaE, deltaQ at the peak of the ncpTotal for each mass"""

        shapeOfArrays = centers.shape
        proximityToYCenters = np.abs(self._y_space_arrays[row] - centers)
        yClosestToCenters = proximityToYCenters.min(axis=1).reshape(shapeOfArrays)
        yCentersMask = proximityToYCenters == yClosestToCenters

        v0, E0, deltaE, deltaQ = self._kinematic_arrays[row]

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


    def calcGaussianResolution(self, centers, row):
        masses = self._masses.reshape((self._masses.size, 1))
        v0, E0, delta_E, delta_Q = self.kinematicsAtYCenters(centers, row)
        det, plick, angle, T0, L0, L1 = self._instrument_params[row]
        dE1, dTOF, dTheta, dL0, dL1, dE1_lorz = self._resolution_params[row]
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


    def calcLorentzianResolution(self, centers, row):
        masses = self._masses.reshape((self._masses.size, 1))
        v0, E0, delta_E, delta_Q = self.kinematicsAtYCenters(centers, row)
        det, plick, angle, T0, L0, L1 = self._instrument_params[row]
        dE1, dTOF, dTheta, dL0, dL1, dE1_lorz = self._resolution_params[row]
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
        pseudo_voigt = eta * lorentizian(x, gamma_v) + (1.0 - eta) * gaussian(x, sigma_v)

        norm = (
            np.abs(np.trapz(pseudo_voigt, x, axis=1))[:, np.newaxis] if self._run_norm_voigt else 1
        )
        return pseudo_voigt / norm


    def createWorkspacesForMSCorrection(ic, meanWidths, meanIntensityRatios, wsNCPM):
        """Creates _MulScattering and _TotScattering workspaces used for the MS correction"""

        createSlabGeometry(ic, wsNCPM)  # Sample properties for MS correction

        sampleProperties = calcMSCorrectionSampleProperties(
            ic, meanWidths, meanIntensityRatios
        )
        print(
            "\nThe sample properties for Multiple Scattering correction are:\n\n",
            sampleProperties,
            "\n",
        )

        return createMulScatWorkspaces(ic, wsNCPM, sampleProperties)


    def createSlabGeometry(ic, wsNCPM):
        half_height, half_width, half_thick = (
            0.5 * ic.vertical_width,
            0.5 * ic.horizontal_width,
            0.5 * ic.thickness,
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

        CreateSampleShape(wsNCPM, xml_str)


    def calcMSCorrectionSampleProperties(ic, meanWidths, meanIntensityRatios):
        masses = ic.masses.flatten()

        # If Backsscattering mode and H is present in the sample, add H to MS properties
        if ic.modeRunning == "BACKWARD":
            if ic.HToMassIdxRatio is not None:  # If H is present, ratio is a number
                masses = np.append(masses, 1.0079)
                meanWidths = np.append(meanWidths, 5.0)

                HIntensity = ic.HToMassIdxRatio * meanIntensityRatios[ic.massIdx]
                meanIntensityRatios = np.append(meanIntensityRatios, HIntensity)
                meanIntensityRatios /= np.sum(meanIntensityRatios)

        MSProperties = np.zeros(3 * len(masses))
        MSProperties[::3] = masses
        MSProperties[1::3] = meanIntensityRatios
        MSProperties[2::3] = meanWidths
        sampleProperties = list(MSProperties)

        return sampleProperties


    def createMulScatWorkspaces(ic, ws, sampleProperties):
        """Uses the Mantid algorithm for the MS correction to create two Workspaces _TotScattering and _MulScattering"""

        print("\nEvaluating the Multiple Scattering Correction...\n")
        # selects only the masses, every 3 numbers
        MS_masses = sampleProperties[::3]
        # same as above, but starts at first intensities
        MS_amplitudes = sampleProperties[1::3]

        dens, trans = VesuvioThickness(
            Masses=MS_masses,
            Amplitudes=MS_amplitudes,
            TransmissionGuess=ic.transmission_guess,
            Thickness=0.1,
        )

        _TotScattering, _MulScattering = VesuvioCalculateMS(
            ws,
            NoOfMasses=len(MS_masses),
            SampleDensity=dens.cell(9, 1),
            AtomicProperties=sampleProperties,
            BeamRadius=2.5,
            NumScatters=ic.multiple_scattering_order,
            NumEventsPerRun=int(ic.number_of_events),
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


    def createWorkspacesForGammaCorrection(ic, meanWidths, meanIntensityRatios, wsNCPM):
        """Creates _gamma_background correction workspace to be subtracted from the main workspace"""

        inputWS = wsNCPM.name()

        profiles = calcGammaCorrectionProfiles(ic.masses, meanWidths, meanIntensityRatios)

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


    def calcGammaCorrectionProfiles(masses, meanWidths, meanIntensityRatios):
        masses = masses.flatten()
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


    class resultsObject:
        """Used to collect results from workspaces and store them in .npz files for testing."""

        def __init__(self, ic):
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
                    wsIterName = ic.name + str(j)

                    # Extract ws that were fitted
                    ws = mtd[wsIterName]
                    allFitWs.append(ws.extractY())

                    # Extract total ncp
                    totNcpWs = mtd[wsIterName + "_TOF_Fitted_Profiles"]
                    allTotNcp.append(totNcpWs.extractY())

                    # Extract best fit parameters
                    fitParTable = mtd[wsIterName + "_Best_Fit_NCP_Parameters"]
                    bestFitPars = []
                    for key in fitParTable.keys():
                        bestFitPars.append(fitParTable.column(key))
                    allBestPar.append(np.array(bestFitPars).T)

                    # Extract individual ncp
                    allNCP = []
                    i = 0
                    while True:  # By default, looks for all ncp ws until it breaks
                        try:
                            ncpWsToAppend = mtd[
                                wsIterName + "_TOF_Fitted_Profile_" + str(i)
                            ]
                            allNCP.append(ncpWsToAppend.extractY())
                            i += 1
                        except KeyError:
                            break
                    allNCP = switchFirstTwoAxis(np.array(allNCP))
                    allIterNcp.append(allNCP)

                    # Extract Mean and Std Widths, Intensities
                    meansTable = mtd[wsIterName + "_Mean_Widths_And_Intensities"]
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

            # Pass all attributes of ic into attributes to be used whithin this object
            self.maskedDetectorIdx = ic.maskedDetectorIdx
            self.masses = ic.masses
            self.noOfMasses = ic.noOfMasses
            self.resultsSavePath = ic.resultsSavePath

        def save(self):
            """Saves all of the arrays stored in this object"""

            # TODO: Take out nans next time when running original results
            # Because original results were recently saved with nans, mask spectra with nans
            self.all_spec_best_par_chi_nit[:, self.maskedDetectorIdx, :] = np.nan
            self.all_ncp_for_each_mass[:, self.maskedDetectorIdx, :, :] = np.nan
            self.all_tot_ncp[:, self.maskedDetectorIdx, :] = np.nan

            savePath = self.resultsSavePath
            np.savez(
                savePath,
                all_fit_workspaces=self.all_fit_workspaces,
                all_spec_best_par_chi_nit=self.all_spec_best_par_chi_nit,
                all_mean_widths=self.all_mean_widths,
                all_mean_intensities=self.all_mean_intensities,
                all_std_widths=self.all_std_widths,
                all_std_intensities=self.all_std_intensities,
                all_tot_ncp=self.all_tot_ncp,
                all_ncp_for_each_mass=self.all_ncp_for_each_mass,
            )


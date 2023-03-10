from __future__ import (absolute_import, division, print_function, unicode_literals)

from mantid.kernel import *
from mantid.api import *
from mantid.simpleapi import *
from mantid.kernel import  StringListValidator, IntListValidator, FloatBoundedValidator, Direction
from Inelastic.vesuvio.analysisHelpers import *


class VesuvioAnalysis(PythonAlgorithm):   # Will run only either Forward or Backward scattering
    def category(self):
        return "Inelastic\\Indirect\\Vesuvio"

    def seeAlso(self):
        return ["VesuvioCalculateGammaBackground", "ConvertToYSpace","VesuvioThickness",
                "VesuvioCalculateMS","Integration","VesuvioResolution","Fit",]

    def PyInit(self):
        self.declareProperty(
            "ModeRunning", "FORWARD",
            doc="Either forward or backward scattering",
            validator=StringListValidator(["FORWARD", "BACKWARD"])
            )
        self.declareProperty(
            FileProperty(
                "IPFile",
                "ip2018.par",
                action=FileAction.Load,
                direction=Direction.Input,
                extensions=["par"]),
            doc="The instrument parameter file"
        )
    # Input of masses? How is that done?




# Example of original

class VesuvioAnalysis(PythonAlgorithm):
    def category(self):
        return "Inelastic\\Indirect\\Vesuvio"

    def seeAlso(self):
        return ["VesuvioCalculateGammaBackground", "ConvertToYSpace","VesuvioThickness",
                "VesuvioCalculateMS","Integration","VesuvioResolution","Fit",]

    def PyInit(self):
        self.declareProperty(
            "AnalysisMode",
            "LoadReduceAnalyse",
            doc="In the first case, all the algorithm is run. In the second case, the data are not re-loaded, and only"
            " the TOF and y-scaling bits are run. In the third case, only the y-scaling final analysis is run. In the"
            " fourth case, the data is re-loaded and the TOF bits are run. In the fifth case, only the TOF bits are run.",
            validator=StringListValidator(
                [
                    "LoadReduceAnalyse",
                    "ReduceAnalyse",
                    "Analyse",
                    "LoadReduce",
                    "Reduce"]))
        self.declareProperty(
            FileProperty(
                "IPFile",
                "ip2018.par",
                action=FileAction.Load,
                direction=Direction.Input,
                extensions=["par"]),
            doc="The instrument parameter file")
        self.declareProperty("NumberOfIterations",2, doc="Number of time the reduction is reiterated.",
                             validator=IntListValidator([0,1,2,3,4]))
        self.declareProperty("OutputName", "polyethylene",doc="The base name for the outputs." )
        self.declareProperty("Runs", "38898-38906", doc="List of Vesuvio run numbers (e.g. 20934-20937, 30924)")
        self.declareProperty(IntArrayProperty("Spectra",[135,182]), doc="Range of spectra to be analysed (first, last). Please note that "
                                                                         "spectra with a number lower than 135 are treated as back "
                                                                         "scattering spectra and are therefore not considered valid input "
                                                                         "as this algorithm is only for forward scattering data.")
        self.declareProperty(FloatArrayProperty("TOFRangeVector", [110.,1.5,460.]),
                             doc="In micro seconds (lower bound, binning, upper bound).")
        self.declareProperty(
            "TransmissionGuess",
            0.9174,
            doc="A number from 0 to 1 to represent the experimental transmission value of the sample for epithermal"
            " neutrons. This value is used for the multiple scattering corrections. If 1, the multiple scattering correction is not run.",
            validator=FloatBoundedValidator(
                0,
                1) )
        self.declareProperty("MultipleScatteringOrder", 2, doc="Order of multiple scattering events in MC simultation.",
                             validator=IntListValidator([1,2,3,4]))
        self.declareProperty("MonteCarloEvents", 1.e6, doc="Number of events for MC multiple scattering simulation.")
        self.declareProperty(ITableWorkspaceProperty("ComptonProfile",
                                                     "",
                                                     direction=Direction.Input),doc="Table for Compton profiles")
        self.declareProperty(ITableWorkspaceProperty("ConstraintsProfile",
                                                     "",
                                                     Direction.Input, PropertyMode.Optional),
                                                     doc="Table with LHS and RHS element of constraint on "
                                              "intensities of element peaks. A constraint can only be set when there are at least two "
                                              "elements in the ComptonProfile. For each constraint the ratio of the first to second "
                                              "intensities, each equal to atom stoichiometry times bound scattering "
                                              "cross section is defined in the column ScatteringCrossSection. Simple arithmetic can be "
                                              "included but the result may be rounded. The column State allows the values 'eq' and 'ineq'.")
        self.declareProperty(IntArrayProperty("SpectraToBeMasked", []))#173,174,181
        self.declareProperty("SubtractResonancesFunction", "", doc="Function for resonance subtraction. Empty means no subtraction.")
        self.declareProperty("YSpaceFitFunctionTies", "",doc="The TOF spectra are subtracted by all the fitted profiles"
                             " about the first element specified in the elements string. Then such spectra are converted to the Y space"
                             " of the first element (using the ConvertToYSPace algorithm). The spectra are summed together and"
                             " symmetrised. A fit on the resulting spectrum is performed using a Gauss Hermite function up to the sixth"
                             " order.")


    def PyExec(self):
        IPFile = self.getProperty("IPFile").value
        g_log = logger(self.log())
        analysisMode = self.getProperty("AnalysisMode").value
        # This is the number of iterations for the reduction analysis in time-of-flight.
        number_of_iterations =  self.getProperty("NumberOfIterations").value
        # Parameters of the measurement
        ws_name=self.getProperty("OutputName").value
        runs = self.getProperty("Runs").value
        first_spectrum, last_spectrum = self.getProperty("Spectra").value
        tof_range = self.getProperty("TOFRangeVector").value
        # Parameters for the multiple-scattering correction, including the shape of the sample.
        transmission_guess = self.getProperty("TransmissionGuess").value
        multiple_scattering_order = self.getProperty("MultipleScatteringOrder").value
        number_of_events = self.getProperty("MonteCarloEvents").value

        # parameters of the neutron Compton profiles to be fitted.
        elements = generate_elements(self.getProperty("ComptonProfile").value)

        # constraint on the intensities of element peaks
        constraints = generate_constraints(self.getProperty("ConstraintsProfile").value)

        # spectra to be masked
        spectra_to_be_masked = self.getProperty("SpectraToBeMasked").value

        subtract_resonances = True
        resonance_function = self.getProperty("SubtractResonancesFunction").value
        if not resonance_function:
            subtract_resonances = False

        fit_hydrogen_in_Y_space = True# If True, corrected time-of-flight spectra containing H only are transformed to Y-space and fitted.

        y_fit_ties = self.getProperty("YSpaceFitFunctionTies").value
        if not y_fit_ties:
            fit_hydrogen_in_Y_space = False
        ####### END OF USER INPUT

        #
        # Start of the reduction and analysis procedure
        #
        if "Load" in analysisMode:
            spectrum_list=str(first_spectrum)+'-'+str(last_spectrum)
            LoadVesuvio(
                Filename=runs,
                SpectrumList=spectrum_list,
                Mode="SingleDifference",
                SumSpectra=False,
                InstrumentParFile=IPFile,
                OutputWorkspace=ws_name)
            # chose limits such that there is at list one non-nan bin among all the spectra between -30 and 30 \AA-1
            Rebin(InputWorkspace=ws_name,Params=tof_range,OutputWorkspace=ws_name)

        if "Reduce" in analysisMode:
            vertical_width, horizontal_width, thickness = 0.1, 0.1, 0.001 # expressed in meters
            create_slab_geometry(ws_name,vertical_width, horizontal_width, thickness)

            masses, par, bounds, constraints = prepare_fit_arguments(elements, constraints)
            fit_arguments = [bounds, constraints]

            # Iterative analysis and correction of time-of-flight spectra.
            for iteration in range(number_of_iterations):
                if iteration == 0:
                    ws_to_be_fitted = CloneWorkspace(InputWorkspace = ws_name, OutputWorkspace = ws_name+"_cor")
                ws_to_be_fitted = mtd[ws_name+"_cor"]
                MaskDetectors(Workspace=ws_to_be_fitted,SpectraList=spectra_to_be_masked)

                # Fit and plot where the spectra for the current iteration
                spectra, widths, intensities, centres = block_fit_ncp(
                    par, first_spectrum,last_spectrum, masses, ws_to_be_fitted, fit_arguments, g_log,IPFile, g_log)

                # Calculate mean widths and intensities
                mean_widths, mean_intensity_ratios = calculate_mean_widths_and_intensities(
                    masses, widths, intensities, spectra, g_log) # at present is not multiplying for 0,9

                if (number_of_iterations - iteration -1 > 0):
                    # evaluate gamma background correction ---------- This creates a
                    # background workspace with name :  str(ws_name)+"_gamma_background"
                    sample_properties = calculate_sample_properties(masses, mean_widths, mean_intensity_ratios, "GammaBackground", g_log)
                    correct_for_gamma_background(ws_name, first_spectrum,last_spectrum, sample_properties, g_log)#

                    Scale(InputWorkspace = str(ws_name)+"_gamma_background", OutputWorkspace = str(ws_name)+"_gamma_background",
                          Factor=0.9, Operation = "Multiply")
                    # evaluate multiple scattering correction --------- This creates a
                    # background workspace with name :  str(ws_name)+"_MulScattering"
                    if transmission_guess < 1. :
                        sample_properties = calculate_sample_properties(
                            masses, mean_widths, mean_intensity_ratios, "MultipleScattering", g_log)
                        correct_for_multiple_scattering(ws_name, first_spectrum,last_spectrum, sample_properties, transmission_guess,
                                                        multiple_scattering_order, number_of_events,g_log, masses,mean_intensity_ratios)
                    # Create corrected workspace
                    Minus(LHSWorkspace= ws_name, RHSWorkspace = str(ws_name)+"_gamma_background",
                          OutputWorkspace = ws_name+"_cor")
                    if transmission_guess < 1. :
                        Minus(LHSWorkspace= ws_name+"_cor", RHSWorkspace = str(ws_name)+"_MulScattering",
                              OutputWorkspace = ws_name+"_cor")
                    if subtract_resonances :
                        Minus(LHSWorkspace=ws_name+"_cor",RHSWorkspace=ws_name+"_cor_fit",
                              OutputWorkspace=ws_name+"_cor_residuals")
                        ws = CloneWorkspace(ws_name+"_cor_residuals")
                        for index in range( ws.getNumberHistograms() ):
                            Fit(Function=resonance_function, InputWorkspace=ws_name+"_cor_residuals", WorkspaceIndex=index,
                                MaxIterations=10000,
                                Output=ws_name+"_cor_residuals", OutputCompositeMembers=True, StartX=110., EndX=460.)
                            fit_ws=mtd[ws_name+"_cor_residuals_Workspace"]
                            for bin in range( ws.blocksize() ) :
                                ws.dataY(index)[bin]=fit_ws.readY(1)[bin]
                                ws.dataE(index)[bin]=0.
                        RenameWorkspace(InputWorkspace="ws", OutputWorkspace=ws_name+"_fitted_resonances")
                        Minus(LHSWorkspace=ws_name+"_cor",RHSWorkspace=ws_name+"_fitted_resonances",
                              OutputWorkspace=ws_name+"_cor" )
                        Minus(LHSWorkspace=ws_name+"_cor_residuals",RHSWorkspace=ws_name+"_fitted_resonances",
                              OutputWorkspace=ws_name+"_cor_residuals" )
                else:
                    if fit_hydrogen_in_Y_space and "Analyse" in analysisMode:
                        hydrogen_ws = subtract_other_masses(ws_name+"_cor", widths, intensities, centres, spectra, masses,IPFile,g_log)
                        RenameWorkspace(hydrogen_ws, ws_name+'_H')
                        SumSpectra(InputWorkspace=ws_name+'_H', OutputWorkspace=ws_name+'_H_sum')
                        calculate_mantid_resolutions(ws_name, masses[0])

            # # Fit of the summed and symmetrised hydrogen neutron Compton profile in its Y space using MANTID.
            # if "Analyse" in analysisMode:
            #     if fit_hydrogen_in_Y_space:
            #         #calculate_mantid_resolutions(ws_name, masses[0])
            #         #max_Y
            #         _ = convert_to_y_space_and_symmetrise(ws_name+"_H",masses[0])
            #         # IT WOULD BE GOOD TO HAVE THE TIES DEFINED IN THE USER SECTION!!!
            #         constraints = "   sigma1=3.0,c4=0.0, c6=0.0,A=0.08, B0=0.00, ties = {}".format(y_fit_ties)
            #         correct_for_offsets = True
            #         y_range = (-20., 20.)
            #         final_fit(ws_name+'_H_JoY_sym', constraints, y_range,correct_for_offsets, masses, g_log)
            #     else:
            #         g_log.notice("Did not compute analysis. The YSpaceFitFunctionTies must be stated.")

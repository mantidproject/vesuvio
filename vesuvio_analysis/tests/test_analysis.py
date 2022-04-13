
from ..core_functions.procedures import runIndependentIterativeProcedure
from ..ICHelpers import completeICFromInputs
import unittest
import numpy as np
import numpy.testing as nptest
from pathlib import Path
import matplotlib.pyplot as plt


ipFilesPath = Path(__file__).absolute().parent.parent / "ip_files"
ipFilePath = ipFilesPath / "ip2018_3.par"  

testPath = Path(__file__).absolute().parent 


class LoadVesuvioFrontParameters:
    runs='43066-43076'         # 100K        # The numbers of the runs to be analysed
    empty_runs='43868-43911'   # 100K        # The numbers of the empty runs to be subtracted
    spectra='144-182'                        # Spectra to be analysed
    mode='SingleDifference'
    ipfile=str(ipFilesPath / "ip2018_3.par") 


class GeneralInitialConditions:
    """Used to define initial conditions shared by both Back and Forward scattering"""
    
    transmission_guess =  0.8537        # Experimental value from VesuvioTransmission
    multiple_scattering_order, number_of_events = 2, 1.e5
    # Sample slab parameters
    vertical_width, horizontal_width, thickness = 0.1, 0.1, 0.001  # Expressed in meters


class ForwardInitialConditions(GeneralInitialConditions):

    InstrParsPath = ipFilePath

    masses = np.array([1.0079, 12, 16, 27]) 
    # noOfMasses = len(masses)

    initPars = np.array([ 
    # Intensities, NCP widths, NCP centers  
        1, 4.7, 0, 
        1, 12.71, 0.,    
        1, 8.76, 0.,   
        1, 13.897, 0.    
    ])
    bounds = np.array([
        [0, np.nan], [3, 6], [-3, 1],
        [0, np.nan], [12.71, 12.71], [-3, 1],
        [0, np.nan], [8.76, 8.76], [-3, 1],
        [0, np.nan], [13.897, 13.897], [-3, 1]
    ])
    constraints = ()

    noOfMSIterations = 2   #4
    firstSpec = 164   #144
    lastSpec = 175    #182

    # Boolean Flags to control script
    MSCorrectionFlag = True
    GammaCorrectionFlag = True

    maskedSpecAllNo = np.array([173, 174, 179])

    tof_binning="110,1.,430"                 # Binning of ToF spectra


class bootstrapInitialConditions:
    speedQuick = None
    nSamples = None


icWSFront = LoadVesuvioFrontParameters
fwdIC = ForwardInitialConditions
wsBootIC = bootstrapInitialConditions

completeICFromInputs(fwdIC, "tests", icWSFront, wsBootIC)


wsFinal, forwardScatteringResults = runIndependentIterativeProcedure(fwdIC)

# Test the results
np.set_printoptions(suppress=True, precision=8, linewidth=150)

oriPath = testPath / "stored_analysis.npz"   # Original data
storedResults = np.load(oriPath)
currentResults = forwardScatteringResults


def displayMask(mask, rtol, string):
    noDiff = np.sum(mask)
    maskSize = mask.size
    print("\nNo of different "+string+f", rtol={rtol}:\n",
        noDiff, " out of ", maskSize,
        f"ie {100*noDiff/maskSize:.1f} %")    


class TestFitParameters(unittest.TestCase):
    def setUp(self):
        oriPars = storedResults["all_spec_best_par_chi_nit"]
        self.orispec = oriPars[:, :, 0]
        self.orichi2 = oriPars[:, :, -2]
        self.orinit = oriPars[:, :, -1]
        self.orimainPars = oriPars[:, :, 1:-2]
        self.oriintensities = self.orimainPars[:, :, 0::3]
        self.oriwidths = self.orimainPars[:, :, 1::3]
        self.oricenters = self.orimainPars[:, :, 2::3]

        optPars = currentResults.all_spec_best_par_chi_nit 
        self.optspec = optPars[:, :, 0]
        self.optchi2 = optPars[:, :, -2]
        self.optnit = optPars[:, :, -1]
        self.optmainPars = optPars[:, :, 1:-2]
        self.optintensities = self.optmainPars[:, :, 0::3]
        self.optwidths = self.optmainPars[:, :, 1::3]
        self.optcenters = self.optmainPars[:, :, 2::3]

        self.rtol = 1e-7
        self.equal_nan = True

    def test_mainPars(self):
        for orip, optp in zip(self.orimainPars, self.optmainPars):
            mask = ~np.isclose(orip, optp, rtol=self.rtol, equal_nan=True)
            displayMask(mask, self.rtol, "Main Pars")
        nptest.assert_array_equal(self.orimainPars, self.optmainPars)

    def test_chi2(self):
        nptest.assert_array_equal(self.orichi2, self.optchi2)

    def test_nit(self):
        nptest.assert_array_equal(self.orinit, self.optnit)

    def test_intensities(self):
        nptest.assert_array_equal(self.oriintensities, self.optintensities)


class TestNcp(unittest.TestCase):
    def setUp(self):
        self.orincp = storedResults["all_tot_ncp"]
        
        self.optncp = currentResults.all_tot_ncp

        self.rtol = 1e-7
        self.equal_nan = True

    def test_ncp(self):
        for orincp, optncp in zip(self.orincp, self.optncp):
            mask = ~np.isclose(orincp, optncp, rtol=self.rtol, equal_nan=True)
            displayMask(mask, self.rtol, "NCP")
        nptest.assert_array_equal(self.orincp, self.optncp)


class TestMeanWidths(unittest.TestCase):
    def setUp(self):
        self.orimeanwidths = storedResults["all_mean_widths"]

        self.optmeanwidths = currentResults.all_mean_widths
    
    def test_widths(self):
        # nptest.assert_allclose(self.orimeanwidths, self.optmeanwidths)
        nptest.assert_array_equal(self.orimeanwidths, self.optmeanwidths)


class TestMeanIntensities(unittest.TestCase):
    def setUp(self):
        self.orimeanintensities = storedResults["all_mean_intensities"]

        self.optmeanintensities = currentResults.all_mean_intensities

    def test_intensities(self):
        # nptest.assert_allclose(self.orimeanintensities, self.optmeanintensities)
        nptest.assert_array_equal(self.orimeanintensities, self.optmeanintensities)


class TestFitWorkspaces(unittest.TestCase):
    def setUp(self):
        self.oriws = storedResults["all_fit_workspaces"]
        
        self.optws = currentResults.all_fit_workspaces

        self.decimal = 8
        self.rtol = 1e-7
        self.equal_nan = True

    def test_FinalWS(self):
        for oriws, optws in zip(self.oriws, self.optws):
            mask = ~np.isclose(oriws, optws, rtol=self.rtol, equal_nan=True)
            displayMask(mask, self.rtol, "wsFinal")
        nptest.assert_array_equal(self.optws, self.oriws)


from ..core_functions.fit_in_yspace import fitInYSpaceProcedure
from mantid.simpleapi import Load
from mantid.api import AnalysisDataService
from pathlib import Path
import numpy as np
import unittest
import numpy.testing as nptest

from ..ICHelpers import completeICFromInputs


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

class YSpaceFitInitialConditions(ForwardInitialConditions):
    symmetrisationFlag = True
    rebinParametersForYSpaceFit = "-20, 0.5, 20"    # Needs to be symetric
    singleGaussFitToHProfile = True      # When False, use Hermite expansion
    globalFitFlag = True
    forceManualMinos = False
    nGlobalFitGroups = 4

yfitIC = YSpaceFitInitialConditions

completeICFromInputs(fwdIC, "tests", icWSFront, wsBootIC)


AnalysisDataService.clear()

wsPath = testPath / "ws_test_yspace_fit.nxs"
wsFinal = Load(str(wsPath))

oriPath = testPath / "stored_analysis.npz"
AllNCP = np.load(oriPath)["all_ncp_for_each_mass"][-1]

ySpaceFitResults = fitInYSpaceProcedure(yfitIC, wsFinal, AllNCP)

# Test yspace
np.set_printoptions(suppress=True, precision=8, linewidth=150)

oriPath = testPath / "stored_yspace_fit.npz"
storedResults = np.load(oriPath)
currentResults = ySpaceFitResults


class TestSymSumYSpace(unittest.TestCase):
    def setUp(self):
        self.oridataY = storedResults["YSpaceSymSumDataY"]
        self.oridataE = storedResults["YSpaceSymSumDataE"]

        self.optdataY = currentResults.YSpaceSymSumDataY
        self.optdataE = currentResults.YSpaceSymSumDataE
        self.rtol = 0.000001
        self.equal_nan = True
        self.decimal = 6

    def test_YSpaceDataY(self):
        nptest.assert_allclose(self.oridataY, self.optdataY)

 
    def test_YSpaceDataE(self):
        nptest.assert_allclose(self.oridataE, self.optdataE)


class TestResolution(unittest.TestCase):
    def setUp(self): 
        self.orires = storedResults["resolution"]

        self.optres = currentResults.resolution

        self.rtol = 0.0001
        self.equal_nan = True
        self.decimal = 8

    def test_resolution(self):
        nptest.assert_array_equal(self.orires, self.optres)


class TestHdataY(unittest.TestCase):
    def setUp(self):
        self.oriHdataY = storedResults["HdataY"]

        self.optHdataY = currentResults.HdataY

        self.rtol = 0.0001
        self.equal_nan = True
        self.decimal = 4

    def test_HdataY(self):
        # mask = np.isclose(self.oriHdataY, self.optHdataY, rtol=1e-9)
        # plt.imshow(mask, aspect="auto", cmap=plt.cm.RdYlGn, 
        #                 interpolation="nearest", norm=None)
        # plt.show()
        nptest.assert_array_equal(self.oriHdataY, self.optHdataY)


class TestFinalRawDataY(unittest.TestCase):
    def setUp(self):
        self.oriFinalDataY = storedResults["finalRawDataY"]

        self.optFinalDataY = currentResults.finalRawDataY

        self.rtol = 1e-6
        self.equal_nan = True
        self.decimal = 10

    def test_FinalDataY(self):
        nptest.assert_array_equal(self.oriFinalDataY, self.optFinalDataY)


class TestFinalRawDataE(unittest.TestCase):
    def setUp(self):
        self.oriFinalDataE = storedResults["finalRawDataE"]

        self.optFinalDataE = currentResults.finalRawDataE

        self.rtol = 1e-6
        self.equal_nan = True
        self.decimal = 10

    def test_HdataE(self):
        nptest.assert_array_equal(self.oriFinalDataE, self.optFinalDataE)


class Testpopt(unittest.TestCase):
    def setUp(self):
        self.oripopt = storedResults["popt"]

        # Select only Fit results due to Mantid Fit
        self.optpopt = currentResults.popt
    
    def test_opt(self):
        print("\nori:\n", self.oripopt, "\nopt:\n", self.optpopt)
        nptest.assert_array_equal(self.oripopt, self.optpopt)


class Testperr(unittest.TestCase):
    def setUp(self):
        self.oriperr = storedResults["perr"]

        self.optperr = currentResults.perr
    
    def test_perr(self):
        # print("\norierr:\n", self.oriperr, "\nopterr:\n", self.optperr)
        nptest.assert_array_equal( self.oriperr, self.optperr)



if __name__ == "__main__":
    unittest.main()
import unittest
import numpy as np
from numpy.core.fromnumeric import size
import numpy.testing as nptest
from pathlib import Path
currentPath = Path(__file__).absolute().parent  # Path to the repository


pathToStarter = currentPath / "current_yspacefit.npz"
pathToOriginal = currentPath / "starter_yspacefit.npz" 


class TestSymSumYSpace(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        self.oridataY = originalResults["YSpaceSymSumDataY"]
        self.oridataE = originalResults["YSpaceSymSumDataE"]

        optimizedResults = np.load(pathToStarter)
        self.optdataY = optimizedResults["YSpaceSymSumDataY"]
        self.optdataE = optimizedResults["YSpaceSymSumDataE"]
        self.rtol = 0.000001
        self.equal_nan = True
        self.decimal = 6

    def test_YSpaceDataY(self):
        nptest.assert_allclose(self.oridataY, self.optdataY)

 
    def test_YSpaceDataE(self):
        nptest.assert_allclose(self.oridataE, self.optdataE)


class TestResolution(unittest.TestCase):
    def setUp(self): 
        originalResults = np.load(pathToOriginal)
        self.orires = originalResults["resolution"]

        optimizedResults = np.load(pathToStarter)
        self.optres = optimizedResults["resolution"]

        self.rtol = 0.0001
        self.equal_nan = True
        self.decimal = 8

    def test_resolution(self):
        nptest.assert_array_equal(self.orires, self.optres)


class TestHdataY(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        self.oriHdataY = originalResults["HdataY"]

        optimizedResults = np.load(pathToStarter)
        self.optHdataY = optimizedResults["HdataY"]

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
        originalResults = np.load(pathToOriginal)
        self.oriFinalDataY = originalResults["finalRawDataY"]

        optimizedResults = np.load(pathToStarter)
        self.optFinalDataY = optimizedResults["finalRawDataY"]

        self.rtol = 1e-6
        self.equal_nan = True
        self.decimal = 10

    def test_FinalDataY(self):
        nptest.assert_array_equal(self.oriFinalDataY, self.optFinalDataY)


class TestFinalRawDataE(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        self.oriFinalDataE = originalResults["finalRawDataE"]

        optimizedResults = np.load(pathToStarter)
        self.optFinalDataE = optimizedResults["finalRawDataE"]

        self.rtol = 1e-6
        self.equal_nan = True
        self.decimal = 10

    def test_HdataE(self):
        nptest.assert_array_equal(self.oriFinalDataE, self.optFinalDataE)


class Testpopt(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        self.oripopt = originalResults["popt"]

        optimizedResults = np.load(pathToStarter)
        # Select only Fit results due to Mantid Fit
        self.optpopt = optimizedResults["popt"]
    
    def test_opt(self):
        nptest.assert_array_equal(self.oripopt, self.optpopt)


class Testperr(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        self.oriperr = originalResults["perr"]

        optimizedResults = np.load(pathToStarter)
        self.optperr = optimizedResults["perr"]
    
    def test_perr(self):
        nptest.assert_array_equal( self.oriperr, self.optperr)



if __name__ == "__main__":
    unittest.main()
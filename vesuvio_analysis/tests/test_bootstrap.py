from ..core_functions.bootstrap import runBootstrap
from ..ICHelpers import completeICFromInputs
import unittest
import numpy as np
import numpy.testing as nptest
from pathlib import Path
from .tests_IC import fwdIC, bckwdIC, bootIC, yfitIC
testPath = Path(__file__).absolute().parent 

np.random.seed(1)
bootBackSamples, bootFrontSamples, bootYFitSamples = runBootstrap(bckwdIC, fwdIC, bootIC, yfitIC)

oriBootBack = testPath / "stored_boot_back.npz"
oriBootFront = testPath / "stored_boot_front.npz"
oriBootYFit = testPath / "stored_boot_yfit.npz"

class TestBootstrap(unittest.TestCase):

    def setUp(self):
        self.oriBack = np.load(oriBootBack)["boot_samples"]
        self.oriFront = np.load(oriBootFront)["boot_samples"]
        self.oriYFit = np.load(oriBootYFit)["boot_vals"]

    def testBack(self):
        nptest.assert_array_almost_equal(bootBackSamples, self.oriBack)

    def testFront(self):
        nptest.assert_array_almost_equal(bootFrontSamples, self.oriFront)

    def testYFit(self):
        nptest.assert_array_almost_equal(bootYFitSamples, self.oriYFit)
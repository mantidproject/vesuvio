from ..core_functions.bootstrap import runJointBootstrap
from ..ICHelpers import completeICFromInputs
import unittest
import numpy as np
import numpy.testing as nptest
from pathlib import Path
from .tests_IC import fwdIC, bckwdIC, nSamples, yfitIC
testPath = Path(__file__).absolute().parent 

np.random.seed(3)   # Set seed so that tests match everytime

class BootstrapInitialConditions:
    runningJackknife = True
    nSamples = 3   # Overwritten by running Jackknife
    skipMSIterations = False
    runningTest = True
    userConfirmation = False

bootIC = BootstrapInitialConditions

jackJointResults = runJointBootstrap(bckwdIC, fwdIC, bootIC, yfitIC)
# jackJointResults = runJointJackknife(bckwdIC, fwdIC, yfitIC, fastBootstrap=False, runningTest=True)


jackSamples = []
for jackRes in jackJointResults:
    jackSamples.append(jackRes.bootSamples)

jackBackSamples, jackFrontSamples = jackSamples


oriJackBack = testPath / "stored_joint_jack_back.npz"
oriJackFront = testPath / "stored_joint_jack_front.npz"

class TestJointBootstrap(unittest.TestCase):

    def setUp(self):
        self.oriJointBack = np.load(oriJackBack)["boot_samples"]
        self.oriJointFront = np.load(oriJackFront)["boot_samples"]

    def testBack(self):
        nptest.assert_array_almost_equal(jackBackSamples, self.oriJointBack)

    def testFront(self):
        nptest.assert_array_almost_equal(jackFrontSamples, self.oriJointFront)

 
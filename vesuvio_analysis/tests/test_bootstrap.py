from vesuvio_analysis.core_functions.run_script import runScript
from ..core_functions.bootstrap import runJointBootstrap, runIndependentBootstrap
from ..ICHelpers import completeICFromInputs
import unittest
import numpy as np
import numpy.testing as nptest
from pathlib import Path
from .tests_IC import  scriptName, wsBackIC, wsFrontIC, bckwdIC, fwdIC, yFitIC
testPath = Path(__file__).absolute().parent 

np.random.seed(1)   # Set seed so that tests match everytime


class BootstrapInitialConditions:
    runningJackknife = False
    nSamples = 3
    skipMSIterations = False
    runningTest = False
    userConfirmation = False

class UserScriptControls:
    procedure = "FORWARD"   
    fitInYSpace = None    
    bootstrap = "JOINT"   

bootIC = BootstrapInitialConditions
userCtr = UserScriptControls

bootRes, noneRes = runScript(userCtr, scriptName, wsBackIC, wsFrontIC, bckwdIC, fwdIC, yFitIC, bootIC)

#TODO: Figure out why doing the two tests simultaneously fails the testing
# Probably because running bootstrap alters the initial conditions of forward scattering
# Test Joint procedure

bootJointResults = bootRes


# completeICFromInputs(fwdIC, "tests", wsFrontIC)
# completeICFromInputs(bckwdIC, "tests", wsBackIC)
# bootJointResults = runJointBootstrap(bckwdIC, fwdIC, bootIC, yFitIC)

bootSamples = []
for bootRes in bootJointResults:
    bootSamples.append(bootRes.bootSamples)

bootBackSamples, bootFrontSamples, bootYFitSamples = bootSamples

oriBootBack = testPath / "stored_boot_back.npz"
oriBootFront = testPath / "stored_boot_front.npz"
oriBootYFit = testPath / "stored_boot_yfit.npz"

class TestJointBootstrap(unittest.TestCase):

    def setUp(self):
        self.oriJointBack = np.load(oriBootBack)["boot_samples"]
        self.oriJointFront = np.load(oriBootFront)["boot_samples"]
        self.oriJointYFit = np.load(oriBootYFit)["boot_vals"]

    def testBack(self):
        nptest.assert_array_almost_equal(bootBackSamples, self.oriJointBack)

    def testFront(self):
        nptest.assert_array_almost_equal(bootFrontSamples, self.oriJointFront)

    def testYFit(self):
        nptest.assert_array_almost_equal(bootYFitSamples, self.oriJointYFit)


# # Test Single procedure
# bootSingleResults = runIndependentBootstrap(bckwdIC, bootIC, yfitIC)

# bootSingleBackSamples = bootSingleResults[0].bootSamples
# bootSingleYFitSamples = bootSingleResults[1].bootSamples

# oriSingleBootBack = testPath / "stored_single_boot_back.npz"
# oriSingleBootYFit = testPath / "stored_single_boot_back_yfit.npz"

# class TestIndependentBootstrap(unittest.TestCase):
#     def setUp(self):
#         self.oriBack = np.load(oriSingleBootBack)["boot_samples"]
#         self.oriYFit = np.load(oriSingleBootYFit)["boot_vals"]

#     def testBack(self):
#         nptest.assert_array_almost_equal(bootSingleBackSamples, self.oriBack)

#     def testYFit(self):
#         nptest.assert_array_almost_equal(bootSingleYFitSamples, self.oriYFit)



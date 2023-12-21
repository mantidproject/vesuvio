from mvesuvio.vesuvio_analysis.run_script import runScript
import unittest
import numpy as np
import numpy.testing as nptest
from pathlib import Path
from mvesuvio.system_tests.tests_IC import (
    scriptName,
    wsBackIC,
    wsFrontIC,
    bckwdIC,
    fwdIC,
    yFitIC,
)


class BootstrapInitialConditions:
    runBootstrap = True

    procedure = "JOINT"
    fitInYSpace = "FORWARD"

    bootstrapType = "BOOT_RESIDUALS"
    nSamples = 3
    skipMSIterations = False
    runningTest = True
    userConfirmation = False


class UserScriptControls:
    runRoutine = False
    procedure = "FORWARD"
    fitInYSpace = None
    # bootstrap = "JOINT"


class TestJointBootstrap(unittest.TestCase):
    @classmethod
    def _run_analysis(cls):
        np.random.seed(1)  # Set seed so that tests match everytime
        bootIC = BootstrapInitialConditions
        userCtr = UserScriptControls

        # Change yFItIC to default settings, running tests for yfit before hand changes this
        yFitIC.fitModel = "SINGLE_GAUSSIAN"
        yFitIC.symmetrisationFlag = True

        bootRes, noneRes = runScript(
            userCtr, scriptName, wsBackIC, wsFrontIC, bckwdIC, fwdIC, yFitIC, bootIC, True
        )

        # TODO: Figure out why doing the two tests simultaneously fails the testing
        # Probably because running bootstrap alters the initial conditions of forward scattering
        # Test Joint procedure

        cls._bootBackSamples = bootRes["bckwdScat"].bootSamples
        cls._bootFrontSamples = bootRes["fwdScat"].bootSamples
        cls._bootYFitSamples = bootRes["fwdYFit"].bootSamples

        # bootJointResults = bootRes

        # bootSamples = []
        # for bootRes in bootJointResults:
        #     bootSamples.append(bootRes.bootSamples)

        # bootBackSamples, bootFrontSamples, bootYFitSamples = bootSamples

    @classmethod
    def _load_benchmark_results(cls):
        testPath = Path(__file__).absolute().parent
        cls._oriJointBack = np.load(str(testPath / "stored_boot_back.npz"))[
            "boot_samples"
        ]
        cls._oriJointFront = np.load(str(testPath / "stored_boot_front.npz"))[
            "boot_samples"
        ]
        cls._oriJointYFit = np.load(str(testPath / "stored_boot_yfit.npz"))[
            "boot_samples"
        ]

    @classmethod
    def setUpClass(cls):
        cls._run_analysis()
        cls._load_benchmark_results()

    def xtestBack(self):
        nptest.assert_array_almost_equal(self._bootBackSamples, self._oriJointBack)

    def xtestFront(self):
        nptest.assert_array_almost_equal(self._bootFrontSamples, self._oriJointFront)

    def xtestYFit(self):
        nptest.assert_array_almost_equal(self._bootYFitSamples, self._oriJointYFit)


# # Test Single procedure
# class TestIndependentBootstrap(unittest.TestCase):
#     def _run_analysis(self):
#         np.random.seed(1)  # Set seed so that tests match everytime
#         bootIC = BootstrapInitialConditions
#         userCtr = UserScriptControls
#
#         # Change yFItIC to default settings, running tests for yfit before hand changes this
#         yFitIC.fitModel = "SINGLE_GAUSSIAN"
#         yFitIC.symmetrisationFlag = True
#
#         bootSingleResults = runIndependentBootstrap(bckwdIC, bootIC, yFitIC)
#
#         self._bootSingleBackSamples = bootSingleResults[0].bootSamples
#         self._bootSingleYFitSamples = bootSingleResults[1].bootSamples
#
#     def _load_benchmark_results(self):
#         testPath = Path(__file__).absolute().parent
#         self._oriBack = np.load(testPath / "stored_single_boot_back.npz")["boot_samples"]
#         self._oriYFit = np.load(testPath / "stored_single_boot_back_yfit.npz")["boot_vals"]
#
#     def setUp(self):
#         self._run_analysis()
#         self._load_benchmark_results()
#
#     def testBack(self):
#         nptest.assert_array_almost_equal(self._bootSingleBackSamples, self._oriBack)
#
#     def testYFit(self):
#         nptest.assert_array_almost_equal(self._bootSingleYFitSamples, self._oriYFit)
#

if __name__ == "__main__":
    unittest.main()

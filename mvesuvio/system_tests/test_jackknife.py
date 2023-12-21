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
    fitInYSpace = None

    bootstrapType = "JACKKNIFE"
    nSamples = 3  # Overwritten by running Jackknife
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
        np.random.seed(3)  # Set seed so that tests match everytime

        bootIC = BootstrapInitialConditions
        userCtr = UserScriptControls

        bootRes, noneRes = runScript(
            userCtr, scriptName, wsBackIC, wsFrontIC, bckwdIC, fwdIC, yFitIC, bootIC, True
        )

        cls._jackBackSamples = bootRes["bckwdScat"].bootSamples
        cls._jackFrontSamples = bootRes["fwdScat"].bootSamples

        # jackJointResults = bootRes

        # jackSamples = []
        # for jackRes in jackJointResults:
        #     jackSamples.append(jackRes.bootSamples)

        # jackBackSamples, jackFrontSamples = jackSamples

    @classmethod
    def _load_benchmark_results(cls):
        testPath = Path(__file__).absolute().parent
        cls._oriJointBack = np.load(str(testPath / "stored_joint_jack_back.npz"))[
            "boot_samples"
        ]
        cls._oriJointFront = np.load(str(testPath / "stored_joint_jack_front.npz"))[
            "boot_samples"
        ]

    @classmethod
    def setUpClass(cls):
        cls._run_analysis()
        cls._load_benchmark_results()

    def xtestBack(self):
        nptest.assert_array_almost_equal(self._jackBackSamples, self._oriJointBack)

    def xtestFront(self):
        nptest.assert_array_almost_equal(self._jackFrontSamples, self._oriJointFront)


if __name__ == "__main__":
    unittest.main()

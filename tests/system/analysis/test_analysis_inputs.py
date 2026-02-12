import unittest
from mvesuvio.util import handle_config
import runpy
from pathlib import Path


class TestAnalysisInputsScript(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_analysis_inputs_script_runs(self):
        runpy.run_path(str(Path(handle_config.VESUVIO_PACKAGE_PATH, "config", "analysis_inputs.py")), run_name="__main__")

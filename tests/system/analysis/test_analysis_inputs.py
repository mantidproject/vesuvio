import unittest
import runpy
from pathlib import Path


class TestAnalysisInputsScript(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_analysis_inputs_script_runs(self):
        runpy.run_path(str(Path(__file__).absolute().parent.parent.parent.parent / "src" / "mvesuvio" / "config" / "analysis_inputs.py"), run_name="__main__")

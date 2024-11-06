import unittest
import numpy as np
import numpy.testing as nptest
from mock import MagicMock
from mvesuvio.analysis_reduction import VesuvioAnalysisRoutine 
from mantid.simpleapi import CreateWorkspace, DeleteWorkspace
import inspect


class TestAnalysisFunctions(unittest.TestCase):
    def setUp(self):
        pass

if __name__ == "__main__":
    unittest.main()

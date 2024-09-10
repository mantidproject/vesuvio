import unittest
import numpy as np
import numpy.testing as nptest
from mock import MagicMock
from mvesuvio.analysis_reduction import AnalysisRoutine 
from mantid.simpleapi import CreateWorkspace, DeleteWorkspace


class TestAnalysisFunctions(unittest.TestCase):
    def setUp(self):
        pass

    def test_calculate_h_ratio_masses_ordered(self):
        alg = AnalysisRoutine()
        alg._mean_intensity_ratios = np.array([0.91175, 0.06286, 0.00732, 0.01806])
        alg._masses = np.array([1.0079, 12.0, 16.0, 27.0])
        h_ratio = alg.calculate_h_ratio()
        self.assertAlmostEqual(14.504454343, h_ratio)

    def test_calculate_h_ratio_masses_unordered(self):
        alg = AnalysisRoutine()
        alg._mean_intensity_ratios = np.array([0.00732, 0.06286, 0.01806, 0.91175])
        alg._masses = np.array([16.0, 12.0, 27.0, 1.0079])
        h_ratio = alg.calculate_h_ratio()
        self.assertAlmostEqual(14.504454343, h_ratio)
    
    def test_calculate_h_ratio_hydrogen_missing(self):
        alg = AnalysisRoutine()
        alg._mean_intensity_ratios = np.array([0.00732, 0.06286, 0.01806])
        alg._masses = np.array([16.0, 12.0, 27.0])
        h_ratio = alg.calculate_h_ratio()
        self.assertAlmostEqual(None, h_ratio)


if __name__ == "__main__":
    unittest.main()

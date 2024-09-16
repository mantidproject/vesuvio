import unittest
import numpy as np
import numpy.testing as nptest
from mock import MagicMock
from mvesuvio.analysis_reduction import AnalysisRoutine 
from mantid.simpleapi import CreateWorkspace, DeleteWorkspace
import inspect


class TestAnalysisFunctions(unittest.TestCase):
    def setUp(self):
        pass

    def test_constraints_are_passed_correctly(self):
        alg = AnalysisRoutine()
        alg.initialize()
        constraints = (
            {'type': 'eq', 'fun': lambda par:  par[0] - 2.7527*par[3] }, {'type': 'eq', 'fun': lambda par:  par[3] - 0.7234*par[6] })
        for c in constraints:
            print(inspect.getsourcelines(c['fun'])[0])
        # alg.setProperty("Constraints", str(constraints))
        # print(str(constraints))
        # print(alg.getPropertyValue("Constraints"))
        # alg_constraints = eval(alg.getPropertyValue("Constraints"))
        # self.assertEqual(constraints, alg_constraints)

if __name__ == "__main__":
    unittest.main()

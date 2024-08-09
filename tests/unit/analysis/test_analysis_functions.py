import unittest
import numpy as np
import numpy.testing as nptest
from mock import MagicMock
from mvesuvio.util.analysis_helpers import extractWS
from mantid.simpleapi import CreateWorkspace, DeleteWorkspace


class TestAnalysisFunctions(unittest.TestCase):
    def setUp(self):
        pass

    def test_extract_ws(self):
        data = [1, 2, 3]
        ws = CreateWorkspace(DataX=data, DataY=data, DataE=data, NSpec=1, UnitX="some_unit")

        dataX, dataY, dataE = extractWS(ws)
        nptest.assert_array_equal([data], dataX)
        nptest.assert_array_equal([data], dataY)
        nptest.assert_array_equal([data], dataE)

        DeleteWorkspace(ws)


if __name__ == "__main__":
    unittest.main()

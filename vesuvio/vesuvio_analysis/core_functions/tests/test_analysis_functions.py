import unittest
from mock import MagicMock
from vesuvio.vesuvio_analysis.core_functions.analysis_functions import extractWS


class TestAnalysisFunctions(unittest.TestCase):
    def setUp(self):
        self.ws = MagicMock()
        self.ws.extractX = MagicMock()
        self.ws.extractY = MagicMock()
        self.ws.extractE = MagicMock()
        self.ws.extractX.return_value = 'x'
        self.ws.extractY.return_value = 'y'
        self.ws.extractE.return_value = 'e'

    def test_extract_ws_calls_extract_X_Y_and_E(self):
        _ = extractWS(self.ws)
        self.ws.extractX.assert_called_once()
        self.ws.extractY.assert_called_once()
        self.ws.extractE.assert_called_once()

    def test_extract_ws_returns_xye(self):
        returned_values = extractWS(self.ws)
        self.assertEqual(('x', 'y', 'e'), returned_values)


if __name__ == '__main__':
    unittest.main()

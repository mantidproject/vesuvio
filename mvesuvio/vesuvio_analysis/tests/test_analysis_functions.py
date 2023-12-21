import unittest
from mock import MagicMock
from mvesuvio.vesuvio_analysis.analysis_functions import extractWS


class TestAnalysisFunctions(unittest.TestCase):
    def setUp(self):
        self.ws = MagicMock()
        self.ws.extractX = MagicMock()
        self.ws.extractY = MagicMock()
        self.ws.extractE = MagicMock()

    def test_extract_ws_returns_3_values(self):
        returned_values = extractWS(self.ws)
        self.assertEqual(3, len(returned_values))

    def test_extract_ws_calls_extract_X_Y_and_E(self):
        _ = extractWS(self.ws)
        self.ws.extractX.assert_called_once()
        self.ws.extractY.assert_called_once()
        self.ws.extractE.assert_called_once()


if __name__ == "__main__":
    unittest.main()

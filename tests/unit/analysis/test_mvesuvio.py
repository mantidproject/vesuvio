import unittest
from mock import patch
import mvesuvio


class TestPackageAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @patch("mvesuvio.main")
    def test_set_config(self, mock_main):

        mvesuvio.config(analysis_inputs="mock_file", ip_folder="mock_ipfolder")

        args, _kwargs = mock_main.call_args
        self.assertEqual(args[0].analysis_inputs, "mock_file")
        self.assertEqual(args[0].ip_folder, "mock_ipfolder")

    @patch("mvesuvio.main")
    def test_run(self, mock_main):

        mvesuvio.run()

        args, _kwargs = mock_main.call_args
        self.assertEqual(args[0].command, "run")

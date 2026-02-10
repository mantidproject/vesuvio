import unittest
from mock import patch
import mvesuvio


class TestPackageAPI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @patch("mvesuvio.main")
    def test_set_config(self, mock_main):

        mvesuvio.set_config(analysis_inputs="mock_file", ip_folder="mock_ipfolder")

        args, _kwargs = mock_main.call_args
        self.assertEqual(args[0].analysis_inputs, "mock_file")
        self.assertEqual(args[0].ip_folder, "mock_ipfolder")

    @patch("mvesuvio.main")
    def test_run(self, mock_main):

        mvesuvio.run(back_workspace="bws", front_workspace="fws", minimal_output=True, outputs_dir="out")

        args, _kwargs = mock_main.call_args
        self.assertEqual(args[0].back_workspace, "bws")
        self.assertEqual(args[0].front_workspace, "fws")
        self.assertEqual(args[0].minimal_output, True)
        self.assertEqual(args[0].outputs_dir, "out")

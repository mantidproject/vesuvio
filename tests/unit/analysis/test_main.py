import unittest
from mock import patch
from mvesuvio.main import set_up_parser


class TestMain(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_set_up_parser_config(self):

        parser = set_up_parser()
        args = parser.parse_args(["config", "--analysis-inputs", "analysis_inputs.py", "--ip-folder", "mock_ip_folder"])

        self.assertEqual(args.analysis_inputs, "analysis_inputs.py")
        self.assertEqual(args.ip_folder, "mock_ip_folder")

    def test_set_up_parser_config_defaults(self):

        parser = set_up_parser()
        args = parser.parse_args(["config"])

        self.assertEqual(args.analysis_inputs, "")
        self.assertEqual(args.ip_folder, "")

    def test_set_up_parser_run(self):

        parser = set_up_parser()
        args = parser.parse_args(["run", "--front-workspace", "fws", "--back-workspace", "bws", "--minimal-output", "--outputs-dir", "out"])

        self.assertEqual(args.front_workspace, "fws")
        self.assertEqual(args.back_workspace, "bws")
        self.assertEqual(args.minimal_output, True)
        self.assertEqual(args.outputs_dir, "out")

    def test_set_up_parser_run_defaults(self):

        parser = set_up_parser()
        args = parser.parse_args(["run"])

        self.assertEqual(args.front_workspace, "")
        self.assertEqual(args.back_workspace, "")
        self.assertEqual(args.minimal_output, False)
        self.assertEqual(args.outputs_dir, "")


    @patch("mvesuvio.run.handle_config")
    def test_setup_config(self, mock_handle_config):
        return



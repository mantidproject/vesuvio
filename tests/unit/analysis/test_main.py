import unittest
from unittest.mock import patch, MagicMock
from mvesuvio.main import _set_up_parser, _setup_config


class TestParser(unittest.TestCase):
    """Test cases for command-line parser setup."""

    @classmethod
    def setUpClass(cls):
        pass

    def test_set_up_parser_config(self):
        parser = _set_up_parser()
        args = parser.parse_args(["config", "--analysis-inputs", "analysis_inputs.py", "--ip-folder", "mock_ip_folder"])

        self.assertEqual(args.analysis_inputs, "analysis_inputs.py")
        self.assertEqual(args.ip_folder, "mock_ip_folder")

    def test_set_up_parser_config_defaults(self):
        parser = _set_up_parser()
        args = parser.parse_args(["config"])

        self.assertEqual(args.analysis_inputs, "")
        self.assertEqual(args.ip_folder, "")

    def test_set_up_parser_run(self):
        parser = _set_up_parser()
        args = parser.parse_args(["run", "--front-workspace", "fws", "--back-workspace", "bws", "--minimal-output", "--outputs-dir", "out"])

        self.assertEqual(args.front_workspace, "fws")
        self.assertEqual(args.back_workspace, "bws")
        self.assertEqual(args.minimal_output, True)
        self.assertEqual(args.outputs_dir, "out")

    def test_set_up_parser_run_defaults(self):
        parser = _set_up_parser()
        args = parser.parse_args(["run"])

        self.assertEqual(args.front_workspace, "")
        self.assertEqual(args.back_workspace, "")
        self.assertEqual(args.minimal_output, False)
        self.assertEqual(args.outputs_dir, "")

    def test_set_up_parser_bootstrap(self):
        parser = _set_up_parser()
        args = parser.parse_args(["bootstrap", "--inputs-dir", "/path/to/inputs"])

        self.assertEqual(args.inputs_dir, "/path/to/inputs")

    def test_set_up_parser_bootstrap_defaults(self):
        parser = _set_up_parser()
        args = parser.parse_args(["bootstrap"])

        self.assertEqual(args.inputs_dir, "")

    def test_set_up_parser_config_short_flags(self):
        parser = _set_up_parser()
        args = parser.parse_args(["config", "-i", "inputs.py", "-p", "/ip/folder"])

        self.assertEqual(args.analysis_inputs, "inputs.py")
        self.assertEqual(args.ip_folder, "/ip/folder")

    def test_set_up_parser_run_short_flags(self):
        parser = _set_up_parser()
        args = parser.parse_args(["run", "-b", "back_ws", "-f", "front_ws", "-o", "/output"])

        self.assertEqual(args.back_workspace, "back_ws")
        self.assertEqual(args.front_workspace, "front_ws")
        self.assertEqual(args.outputs_dir, "/output")

    def test_set_up_parser_requires_command(self):
        parser = _set_up_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])


class TestSetupConfig(unittest.TestCase):
    """Test cases for _setup_config function."""

    @patch("mvesuvio.main.handle_config")
    @patch("mantid.kernel.ConfigService")
    def test_setup_config_with_args_but_no_custom_paths(self, mock_config_service, mock_handle_config):
        mock_handle_config.config_set.return_value = True
        mock_handle_config.read_config_var.side_effect = ["/default/inputs.py", "/default/ip_folder"]

        mock_args = MagicMock()
        mock_args.set_inputs = None
        mock_args.set_ipfolder = None

        _setup_config(mock_args)

        mock_handle_config.set_config_vars.assert_called_once()
        call_args = mock_handle_config.set_config_vars.call_args[0][0]
        self.assertEqual(call_args["caching.inputs"], "/default/inputs.py")
        self.assertEqual(call_args["caching.ipfolder"], "/default/ip_folder")

    @patch("mvesuvio.main.Path")
    @patch("mvesuvio.main.handle_config")
    @patch("mantid.kernel.ConfigService")
    def test_setup_config_with_custom_inputs_path(self, mock_config_service, mock_handle_config, mock_path):
        mock_handle_config.config_set.return_value = True
        mock_handle_config.read_config_var.side_effect = ["/default/inputs.py", "/default/ip_folder"]

        mock_path_obj = MagicMock()
        mock_path_obj.absolute.return_value = "/absolute/custom/inputs.py"
        mock_path.return_value = mock_path_obj

        mock_args = MagicMock()
        mock_args.set_inputs = "/custom/inputs.py"
        mock_args.set_ipfolder = None

        _setup_config(mock_args)

        mock_path.assert_called_with("/custom/inputs.py")
        mock_path_obj.absolute.assert_called_once()

        mock_handle_config.set_config_vars.assert_called_once()
        call_args = mock_handle_config.set_config_vars.call_args[0][0]
        self.assertEqual(call_args["caching.inputs"], "/absolute/custom/inputs.py")
        self.assertEqual(call_args["caching.ipfolder"], "/default/ip_folder")

    @patch("mvesuvio.main.Path")
    @patch("mvesuvio.main.handle_config")
    @patch("mantid.kernel.ConfigService")

    def test_setup_config_with_custom_ip_folder(self, mock_config_service, mock_handle_config, mock_path):
        mock_handle_config.config_set.return_value = True
        mock_handle_config.read_config_var.side_effect = ["/default/inputs.py", "/default/ip_folder"]

        mock_path_obj = MagicMock()
        mock_path_obj.absolute.return_value = "/absolute/custom/ip_folder"
        mock_path.return_value = mock_path_obj

        mock_args = MagicMock()
        mock_args.set_inputs = None
        mock_args.set_ipfolder = "/custom/ip_folder"

        _setup_config(mock_args)

        mock_path.assert_called_with("/custom/ip_folder")
        mock_path_obj.absolute.assert_called_once()

        mock_handle_config.set_config_vars.assert_called_once()
        call_args = mock_handle_config.set_config_vars.call_args[0][0]
        self.assertEqual(call_args["caching.inputs"], "/default/inputs.py")
        self.assertEqual(call_args["caching.ipfolder"], "/absolute/custom/ip_folder")

    @patch("mvesuvio.main.handle_config")
    @patch("mantid.kernel.ConfigService")
    def test_setup_config_calls_mantid_logging_setup(self, mock_config_service, mock_handle_config):
        mock_handle_config.config_set.return_value = True
        mock_handle_config.read_config_var.side_effect = ["/default/inputs.py", "/default/ip_folder"]

        _setup_config(None)

        # ConfigService.setString should be called multiple times
        self.assertGreater(mock_config_service.setString.call_count, 0)


class TestMainFunction(unittest.TestCase):

    @patch("mvesuvio.main._run_bootstrap")
    @patch("mvesuvio.main._run_analysis")
    @patch("mvesuvio.main._setup_config")
    @patch("mvesuvio.main.handle_config")
    def test_main_with_config_command(self, mock_config, mock_setup_config, mock_run_analysis, mock_run_bootstrap):
        from mvesuvio.main import main

        mock_args = MagicMock()
        mock_args.command = "config"

        main(manual_args=mock_args)

        mock_setup_config.assert_called_once_with(mock_args)

    @patch("mvesuvio.main._run_bootstrap")
    @patch("mvesuvio.main._run_analysis")
    @patch("mvesuvio.main._setup_config")
    @patch("mvesuvio.main.handle_config")
    def test_main_with_run_command(self, mock_config, mock_setup_config, mock_run_analysis, mock_run_bootstrap):
        from mvesuvio.main import main

        mock_args = MagicMock()
        mock_args.command = "run"
        mock_config.config_set.return_value = True

        main(manual_args=mock_args)

        mock_run_analysis.assert_called_once_with(mock_args)

    @patch("mvesuvio.main._run_bootstrap")
    @patch("mvesuvio.main._run_analysis")
    @patch("mvesuvio.main._setup_config")
    @patch("mvesuvio.main.handle_config")
    def test_main_setup_config_if_not_configured(self, mock_config, mock_setup_config, mock_run_analysis, mock_run_bootstrap):
        from mvesuvio.main import main

        mock_args = MagicMock()
        mock_args.command = "run"
        mock_config.config_set.return_value = False  # Config is not set

        main(manual_args=mock_args)

        # Config setup should be called before run analysis
        mock_setup_config.assert_called_once_with(None)
        mock_run_analysis.assert_called_once()


if __name__ == "__main__":
    unittest.main()



import unittest
from mock import MagicMock, patch, call
from mvesuvio.util import handle_config
import tempfile
from textwrap import dedent
import os

class TestHandleConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def test_parse_config_env_var_with_env_variable(self):
        with patch("mvesuvio.util.handle_config.os.getenv") as mock_getenv:
            mock_getenv.return_value = "/some/random/path/to/vesuvio.user.properties"
            # getattr because Python mangles name
            config_path, config_file = getattr(handle_config, "__parse_config_env_var")()
            self.assertEqual(str(config_path), "/some/random/path/to")
            self.assertEqual(str(config_file), "vesuvio.user.properties")


    def test_parse_config_env_var_default(self):
        # getattr because Python mangles name
        config_path, config_file = getattr(handle_config, "__parse_config_env_var")()
        self.assertEqual(str(config_path), os.path.join(os.path.expanduser('~'), '.mvesuvio'))
        self.assertEqual(str(config_file), "vesuvio.user.properties")

    def test_read_config(self):
        file = tempfile.NamedTemporaryFile()
        file.write(dedent("""
            caching.inputs=/inputs.py
            caching.ipfolder=/ip_files
            """).encode())
        file.seek(0)
        lines = getattr(handle_config, "__read_config")(file.name)
        self.assertEqual(lines, ['\n', "caching.inputs=/inputs.py\n", "caching.ipfolder=/ip_files\n"])


    def test_read_config_throws(self):
        with self.assertRaises(RuntimeError):
            lines = getattr(handle_config, "__read_config")("/not.there")


    def test_set_config_vars(self):
        file = tempfile.NamedTemporaryFile() 
        mock_dir, mock_file = os.path.split(file.name)
        with (
            patch("mvesuvio.util.handle_config.__read_config") as mock_read_config,
            patch.object(handle_config, "VESUVIO_CONFIG_PATH", mock_dir),
            patch.object(handle_config, "VESUVIO_CONFIG_FILE", mock_file)

        ):
            mock_read_config.return_value = ['\n', 'caching.inputs=\n', 'caching.ipfolder=\n']
            handle_config.set_config_vars({'caching.inputs': '/inputs.py', 'caching.ipfolder': '/ipfiles'})

            file = open(os.path.join(mock_dir, mock_file), "r")
            self.assertEqual(file.read(), "\ncaching.inputs=/inputs.py\ncaching.ipfolder=/ipfiles\n")
        file.close()


    def test_read_config_vars(self):
        file = tempfile.NamedTemporaryFile() 
        file.write("\ncaching.inputs=/inputs.py\ncaching.ipfolder=/ipfiles\n".encode())
        file.seek(0)
        mock_dir, mock_file = os.path.split(file.name)

        with (
            patch.object(handle_config, "VESUVIO_CONFIG_PATH", mock_dir),
            patch.object(handle_config, "VESUVIO_CONFIG_FILE", mock_file)
        ):
            self.assertEqual(handle_config.read_config_var('caching.inputs'), '/inputs.py')
            self.assertEqual(handle_config.read_config_var('caching.ipfolder'), '/ipfiles')
        file.close()


    def test_read_config_vars_throws(self):
        file = tempfile.NamedTemporaryFile() 
        file.write("\ncaching.inputs=/inputs.py\ncaching.ipfolder=/ipfiles\n".encode())
        file.seek(0)
        mock_dir, mock_file = os.path.split(file.name)

        with (
            patch.object(handle_config, "VESUVIO_CONFIG_PATH", mock_dir),
            patch.object(handle_config, "VESUVIO_CONFIG_FILE", mock_file),
            self.assertRaises(ValueError)
        ):
            handle_config.read_config_var('non.existent') 
        file.close()


    def test_get_script_name(self):
        with patch("mvesuvio.util.handle_config.read_config_var") as mock_read_config:
            mock_read_config.return_value = "path/to/inputs.py"
            self.assertEqual(handle_config.get_script_name(), "inputs")


    def test_setup_default_inputs(self):
        tempdir = tempfile.gettempdir()
        mock_path = os.path.join(tempdir, 'mock_inputs.py')
        # Make sure file does not exist from previous tests
        try:
            os.remove(mock_path)
        except FileNotFoundError:
            pass

        with patch.object(handle_config, "VESUVIO_INPUTS_PATH", mock_path):

            handle_config.setup_default_inputs()

            file = open(mock_path, 'r')
            original_content = file.read()
            file.close()

            file = open(mock_path, 'w+')
            file.write('Overwrite file!')
            file.seek(0)
            # Check that file was overwritten
            self.assertEqual("Overwrite file!", file.read())
            file.close()

            handle_config.setup_default_inputs()

            file = open(mock_path, 'r')
            self.assertEqual(original_content, file.read())
            file.close()
            os.remove(mock_path)

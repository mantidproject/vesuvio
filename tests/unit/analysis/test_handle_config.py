import unittest
from mock import patch
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
            file = tempfile.NamedTemporaryFile()
            mock_getenv.return_value = file.name
            # getattr because Python mangles name
            config_path, config_file = getattr(handle_config, "__parse_config_env_var")()
            expected_path, expected_file = os.path.split(file.name)
            self.assertEqual(str(config_path), expected_path)
            self.assertEqual(str(config_file), expected_file)


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
            getattr(handle_config, "__read_config")("/not.there")


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
        with (
            patch("mvesuvio.util.handle_config.os.path.basename") as mock_basename,
            patch("mvesuvio.util.handle_config.read_config_var") as mock_read_config_var # noqa : F841
        ):
            mock_basename.return_value = "inputs.py"
            self.assertEqual(handle_config.get_script_name(), "inputs")


    def test_setup_config_dir(self):
        # Can't use TemporaryDirectory here because that creates directory
        tempdir = os.path.join(tempfile.gettempdir(), ".mvesuvio")
        with patch.object(handle_config, "VESUVIO_CONFIG_PATH", tempdir):
            handle_config.setup_config_dir()

        vesuvio_file = open(os.path.join(tempdir, "vesuvio.user.properties"), "r")
        self.assertEqual(vesuvio_file.read(), "caching.inputs=\ncaching.ipfolder=\n")
        vesuvio_file.close()
        mantid_file = open(os.path.join(tempdir, "Mantid.user.properties"), "r")
        self.assertEqual(mantid_file.read(), "default.facility=ISIS\ndefault.instrument=Vesuvio\ndatasearch.searcharchive=On\n")
        mantid_file.close()
        os.remove(vesuvio_file.name)
        os.remove(mantid_file.name)
        os.rmdir(tempdir)


    def test_setup_config_dir_dir_already_exists(self):
        tempdir = tempfile.TemporaryDirectory()
        with (
            patch.object(handle_config, "VESUVIO_CONFIG_PATH", tempdir.name),
            patch("mvesuvio.util.handle_config.os.makedirs") as mock_mkdirs,
            patch("mvesuvio.util.handle_config.copyfile") as mock_copyfile
        ):
            handle_config.setup_config_dir()
            mock_mkdirs.assert_not_called()
            mock_copyfile.assert_not_called()


    def test_setup_default_inputs(self):
        tempdir = tempfile.TemporaryDirectory()
        mock_path = os.path.join(tempdir.name, 'mock_inputs.py')
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
            tempdir.cleanup()


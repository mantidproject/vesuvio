import unittest
from mock import patch
from mvesuvio.util import handle_config
import tempfile
from textwrap import dedent
import os
from pathlib import Path
import shutil

class TestHandleConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass


    def test_read_config(self):
        file = tempfile.NamedTemporaryFile(delete=False)
        file.write(dedent("""
            caching.inputs=/inputs.py
            caching.ipfolder=/ip_files
            """).encode())
        file.seek(0)
        file.flush()
        file.close()
        lines = getattr(handle_config, "__read_config")(file.name)
        self.assertEqual(lines, ['\n', "caching.inputs=/inputs.py\n", "caching.ipfolder=/ip_files\n"])
        file.close()
        os.unlink(file.name)


    def test_read_config_throws(self):
        with self.assertRaises(RuntimeError):
            getattr(handle_config, "__read_config")("/not.there")


    def test_set_config_vars(self):
        mock_dir = tempfile.TemporaryDirectory()
        mock_file = os.path.join(mock_dir.name, "config", "mock.vesuvio.properties")
        mock_file = Path(mock_file)
        mock_file.parent.mkdir(parents=True, exist_ok=True)
        mock_file.write_text("")

        with (
            patch("mvesuvio.util.handle_config.__read_config") as mock_read_config,
            patch.object(handle_config, "VESUVIO_PACKAGE_PATH", mock_dir.name),
            patch.object(handle_config, "VESUVIO_PROPERTIES_FILE", mock_file.name)

        ):
            mock_read_config.return_value = ['\n', 'caching.inputs=\n', 'caching.ipfolder=\n']
            handle_config.set_config_vars({'caching.inputs': '/inputs.py', 'caching.ipfolder': '/ipfiles'})

            file = open(mock_file, "r")
            self.assertEqual(file.read(), "\ncaching.inputs=/inputs.py\ncaching.ipfolder=/ipfiles\n")

            file.close()
            mock_dir.cleanup()


    def test_set_default_config_vars(self):
        mock_dir = tempfile.TemporaryDirectory()
        mock_file = os.path.join(mock_dir.name, "config", "mock.vesuvio.properties")
        mock_file = Path(mock_file)
        mock_file.parent.mkdir(parents=True, exist_ok=True)
        mock_file.write_text("")

        with (
            patch("mvesuvio.util.handle_config.__read_config") as mock_read_config,
            patch.object(handle_config, "VESUVIO_PACKAGE_PATH", mock_dir.name),
            patch.object(handle_config, "VESUVIO_PROPERTIES_FILE", mock_file.name),
            patch.object(handle_config, "VESUVIO_CONFIG_PATH", str(Path("path", "to", ".mvesuvio"))),
            patch.object(handle_config, "ANALYSIS_INPUTS_FILE", "analysis_inputs.py"),
            patch.object(handle_config, "IP_FOLDER", "ip_files"),

        ):
            mock_read_config.return_value = ['\n', 'caching.inputs=\n', 'caching.ipfolder=\n']
            handle_config.set_default_config_vars()

            file = open(mock_file, "r")
            self.assertEqual(file.read(), f"\ncaching.inputs={str(Path('path', 'to', '.mvesuvio', 'analysis_inputs.py'))}\ncaching.ipfolder={str(Path('path', 'to', '.mvesuvio', 'ip_files'))}\n")

            file.close()
            mock_dir.cleanup()


    def test_read_config_vars(self):
        mock_dir = tempfile.TemporaryDirectory()
        mock_file = os.path.join(mock_dir.name, "config", "mock.vesuvio.properties")
        mock_file = Path(mock_file)
        mock_file.parent.mkdir(parents=True, exist_ok=True)
        mock_file.write_text("\ncaching.inputs=/inputs.py\ncaching.ipfolder=/ipfiles\n")

        with (
            patch.object(handle_config, "VESUVIO_PACKAGE_PATH", mock_dir.name),
            patch.object(handle_config, "VESUVIO_PROPERTIES_FILE", mock_file.name)
        ):
            self.assertEqual(handle_config.read_cached_var('caching.inputs'), '/inputs.py')
            self.assertEqual(handle_config.read_cached_var('caching.ipfolder'), '/ipfiles')
            mock_dir.cleanup()


    def test_read_config_vars_throws(self):

        mock_dir = tempfile.TemporaryDirectory()
        mock_file = os.path.join(mock_dir.name, "config", "mock.vesuvio.properties")
        mock_file = Path(mock_file)
        mock_file.parent.mkdir(parents=True, exist_ok=True)
        mock_file.write_text("\ncaching.inputs=/inputs.py\ncaching.ipfolder=/ipfiles\n")

        with (
            patch.object(handle_config, "VESUVIO_PACKAGE_PATH", mock_dir.name),
            patch.object(handle_config, "VESUVIO_PROPERTIES_FILE", mock_file.name),
            self.assertRaises(ValueError)
        ):
            handle_config.read_cached_var('non.existent')
            mock_dir.cleanup()


    def test_get_script_name(self):
        with (
            patch("mvesuvio.util.handle_config.os.path.basename") as mock_basename,
            patch("mvesuvio.util.handle_config.read_cached_var") as mock_read_cached_var # noqa : F841
        ):
            mock_basename.return_value = "inputs.py"
            self.assertEqual(handle_config.get_script_name(), "inputs")


    def test_setup_config_dir(self):
        # Use string because want to test when directory does not exist
        tempdir = os.path.join(tempfile.gettempdir(), ".mvesuvio")
        # Clean up any mess from previous tests
        shutil.rmtree(tempdir, ignore_errors=True)

        with patch.object(handle_config, "VESUVIO_CONFIG_PATH", tempdir):
            handle_config.setup_config_dir()

            self.assertTrue(Path(tempdir).exists())

        shutil.rmtree(tempdir)


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
            tempdir.cleanup()


    def test_setup_default_ipfile_dir(self):
        tempdir = tempfile.TemporaryDirectory()

        mock_path = os.path.join(tempdir.name, 'ip_folder')
        with (
            patch.object(handle_config, "VESUVIO_CONFIG_PATH", tempdir.name),
            patch.object(handle_config, "IP_FOLDER", "ip_folder"),
        ):
            handle_config.setup_default_ipfile_dir()

            # Check all par files inside config/ip_files were copied to destination
            for ip_file in Path(handle_config.VESUVIO_PACKAGE_PATH, "config", "ip_files").iterdir():
                if not ip_file.name.endswith("par"):
                    continue

                self.assertTrue(Path(mock_path, ip_file.name).exists())

            tempdir.cleanup()


    def test_setup_default_ipfile_dir_dir_already_exists(self):
        tempdir = tempfile.TemporaryDirectory()
        # Create ip folder
        Path(tempdir.name, "ip_folder").mkdir()
        self.assertTrue(Path(tempdir.name, "ip_folder").exists())

        with (
            patch.object(handle_config, "VESUVIO_CONFIG_PATH", tempdir.name),
            patch.object(handle_config, "IP_FOLDER", "ip_folder"),
            patch("mvesuvio.util.handle_config.copytree") as mock_mkdirs,
        ):
            handle_config.setup_default_ipfile_dir()
            mock_mkdirs.assert_not_called()
            tempdir.cleanup()


    def test_setup_default_inputs(self):
        tempdir = tempfile.TemporaryDirectory()
        mock_path = os.path.join(tempdir.name, 'analysis_inputs.py')
        with (
            patch.object(handle_config, "VESUVIO_CONFIG_PATH", tempdir.name),
            patch.object(handle_config, "ANALYSIS_INPUTS_FILE", "analysis_inputs.py"),
        ):

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


    def test_refresh_config_dir_and_contents_dir_doesnt_exist(self):
        # Use string because want to test when directory does not exist
        tempdir = os.path.join(tempfile.gettempdir(), ".mvesuvio")
        # Clean up any mess from previous tests
        shutil.rmtree(tempdir, ignore_errors=True)

        with patch.object(handle_config, "VESUVIO_CONFIG_PATH", tempdir):
            handle_config.refresh_config_dir_and_contents()

        self.assertTrue(Path(tempdir, "script_to_create_figures.py").exists())
        self.assertTrue(Path(tempdir, "vesuvio.plots.mplstyle").exists())
        self.assertTrue(Path(tempdir, "analysis_inputs.py").exists())
        self.assertTrue(Path(tempdir, "ip_files").exists())

        shutil.rmtree(tempdir)

    def test_refresh_config_dir_and_contents_dir_exists(self):
        tempdir = tempfile.TemporaryDirectory()

        with patch.object(handle_config, "VESUVIO_CONFIG_PATH", tempdir.name):
            handle_config.refresh_config_dir_and_contents()

            self.assertTrue(Path(tempdir.name, "script_to_create_figures.py").exists())
            self.assertTrue(Path(tempdir.name, "vesuvio.plots.mplstyle").exists())
            self.assertTrue(Path(tempdir.name, "analysis_inputs.py").exists())
            self.assertTrue(Path(tempdir.name, "ip_files").exists())
        tempdir.cleanup()

    def test_refresh_config_dir_and_contents_dont_overwrite_ip_files(self):
        tempdir = tempfile.TemporaryDirectory()
        Path(tempdir.name, "ip_files").mkdir()
        Path(tempdir.name, "ip_files", "ip.par").touch()

        with patch.object(handle_config, "VESUVIO_CONFIG_PATH", tempdir.name):
            handle_config.refresh_config_dir_and_contents()
            # Did not overwrite ip files
            self.assertTrue(Path(tempdir.name, "ip_files", "ip.par").exists())
        tempdir.cleanup()


    def test_refresh_config_dir_and_contents_dont_overwrite_plots_config(self):
        tempdir = tempfile.TemporaryDirectory()
        Path(tempdir.name, "vesuvio.plots.mplstyle").write_text("mock config")

        with patch.object(handle_config, "VESUVIO_CONFIG_PATH", tempdir.name):
            handle_config.refresh_config_dir_and_contents()
            # Check did not overwrite plots config
            self.assertEqual(Path(tempdir.name, "vesuvio.plots.mplstyle").read_text(), "mock config")
        tempdir.cleanup()


    def test_refresh_config_dir_and_contents_dont_overwrite_script_figures(self):
        tempdir = tempfile.TemporaryDirectory()
        Path(tempdir.name, "script_to_create_figures.py").write_text("mock script")

        with patch.object(handle_config, "VESUVIO_CONFIG_PATH", tempdir.name):
            handle_config.refresh_config_dir_and_contents()
            # Check did not overwrite plots config
            self.assertEqual(Path(tempdir.name, "script_to_create_figures.py").read_text(), "mock script")
        tempdir.cleanup()


    def test_refresh_config_dir_and_contents_dont_overwrites_analysis_inputs(self):
        tempdir = tempfile.TemporaryDirectory()
        Path(tempdir.name, "analysis_inputs.py").write_text("mock script")

        with patch.object(handle_config, "VESUVIO_CONFIG_PATH", tempdir.name):
            handle_config.refresh_config_dir_and_contents()

            # Analysis inputs overwritten
            self.assertEqual(
                Path(tempdir.name, "analysis_inputs.py").read_text(),
                Path(handle_config.VESUVIO_PACKAGE_PATH, "config", "analysis_inputs.py").read_text()
            )
        tempdir.cleanup()

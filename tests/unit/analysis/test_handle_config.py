import unittest
from mock import patch
from mvesuvio.util import handle_config
import tempfile
from textwrap import dedent
import os
from pathlib import Path

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
        mock_file = Path(mock_dir.name, "mock.vesuvio.properties")
        mock_file.write_text("")

        with (
            patch("mvesuvio.util.handle_config.__read_config") as mock_read_config,
            patch.object(handle_config, "VESUVIO_PROPERTIES_PATH", mock_file)
        ):
            mock_read_config.return_value = ['\n', 'caching.inputs=\n', 'caching.ipfolder=\n']
            handle_config.set_config_vars({'caching.inputs': '/inputs.py', 'caching.ipfolder': '/ipfiles'})

            file = open(mock_file, "r")
            self.assertEqual(file.read(), "\ncaching.inputs=/inputs.py\ncaching.ipfolder=/ipfiles\n")

            file.close()
            mock_dir.cleanup()


    def test_set_default_config_vars(self):
        mock_dir = tempfile.TemporaryDirectory()
        mock_file = Path(mock_dir.name, "mock.vesuvio.properties")
        mock_file.write_text("")

        with (
            patch("mvesuvio.util.handle_config.__read_config") as mock_read_config,
            patch.object(handle_config, "VESUVIO_PROPERTIES_PATH", mock_file),
            patch.object(handle_config, "USER_CONFIG_PATH", Path("path", "to", ".mvesuvio")),
        ):
            mock_read_config.return_value = ['\n', 'caching.inputs=\n', 'caching.ipfolder=\n']
            handle_config.set_default_config_vars()

            file = open(mock_file, "r")
            self.assertEqual(file.read(), f"\ncaching.inputs={str(Path('path', 'to', '.mvesuvio', 'experiment_template'))}\ncaching.ipfolder={str(Path('path', 'to', '.mvesuvio', 'ip_files'))}\n")

            file.close()
            mock_dir.cleanup()


    def test_read_config_vars(self):
        mock_dir = tempfile.TemporaryDirectory()
        mock_file = Path(mock_dir.name, "mock.vesuvio.properties")
        mock_file.write_text("\ncaching.inputs=/inputs.py\ncaching.ipfolder=/ipfiles\n")

        with (
            patch.object(handle_config, "VESUVIO_PROPERTIES_PATH", mock_file),
        ):
            self.assertEqual(handle_config.read_cached_var('caching.inputs'), '/inputs.py')
            self.assertEqual(handle_config.read_cached_var('caching.ipfolder'), '/ipfiles')
            mock_dir.cleanup()


    def test_read_config_vars_throws(self):

        mock_dir = tempfile.TemporaryDirectory()
        mock_file = Path(mock_dir.name, "mock.vesuvio.properties")
        mock_file.write_text("\ncaching.inputs=/inputs.py\ncaching.ipfolder=/ipfiles\n")

        with (
            patch.object(handle_config, "VESUVIO_PROPERTIES_PATH", mock_file),
            self.assertRaises(ValueError)
        ):
            handle_config.read_cached_var('non.existent')
            mock_dir.cleanup()


    def test_read_cached_var_uses_default_package_properties_path(self):
        mock_dir = tempfile.TemporaryDirectory()
        mock_file = Path(mock_dir.name, "mock.vesuvio.properties")
        with (
            patch("mvesuvio.util.handle_config.__read_config") as mock_read_config,
            patch.object(handle_config, "VESUVIO_PROPERTIES_PATH", mock_file),
        ):
            mock_read_config.return_value = ["caching.inputs=/inputs.py\n"]
            handle_config.read_cached_var("caching.inputs")
            mock_read_config.assert_called_once_with(mock_file, True)


    def test_set_config_vars_uses_default_package_properties_path(self):
        mock_dir = tempfile.TemporaryDirectory()
        mock_file = Path(mock_dir.name, "mock.vesuvio.properties")
        with (
            patch("mvesuvio.util.handle_config.__read_config") as mock_read_config,
            patch.object(handle_config, "VESUVIO_PROPERTIES_PATH", mock_file),
            patch("mvesuvio.util.handle_config.open", create=True) as mock_open,
        ):
            mock_read_config.return_value = ["caching.inputs=\n"]
            handle_config.set_config_vars({"caching.inputs": "/inputs.py"})

            mock_read_config.assert_called_once_with(mock_file)
            mock_open.assert_called_once_with(mock_file, "w")


    def test_get_experiment_name(self):
        with patch("mvesuvio.util.handle_config.read_cached_var") as mock_read_cached_var:
            mock_read_cached_var.return_value = str(Path("path", "to", "experiment"))
            self.assertEqual(handle_config.get_experiment_name(), "experiment")


    def test_refresh_config_dir_and_contents_creates_user_config_dir_when_missing(self):
        tempdir = tempfile.TemporaryDirectory()
        missing_dir = Path(tempdir.name, "missing_config_dir")

        self.assertFalse(missing_dir.exists())

        with patch.object(handle_config, "USER_CONFIG_PATH", missing_dir):
            handle_config.refresh_config_dir_and_contents()

        self.assertTrue(missing_dir.exists())
        tempdir.cleanup()


    def test_refresh_config_dir_and_contents_when_dir_exists_does_not_reset_dir(self):
        tempdir = tempfile.TemporaryDirectory()
        preserved_file = Path(tempdir.name, "custom_local_file.txt")
        preserved_file.write_text("keep me")

        with patch.object(handle_config, "USER_CONFIG_PATH", Path(tempdir.name)):
            handle_config.refresh_config_dir_and_contents()

        self.assertTrue(preserved_file.exists())
        self.assertEqual(preserved_file.read_text(), "keep me")
        tempdir.cleanup()


    def test_refresh_config_dir_and_contents_copies_dict_config_when_dir_missing(self):
        tempdir = tempfile.TemporaryDirectory()

        with patch.object(handle_config, "USER_CONFIG_PATH", Path(tempdir.name)):
            handle_config.refresh_config_dir_and_contents()

        self.assertTrue(Path(tempdir.name, "ip_files").exists())
        self.assertTrue(Path(tempdir.name, "vesuvio.plots.mplstyle").exists())
        self.assertTrue(Path(tempdir.name, "experiment_template", "script_to_create_figures.py").exists())
        self.assertTrue(Path(tempdir.name, "experiment_template", "run_reduction.py").exists())
        self.assertTrue(Path(tempdir.name, "experiment_template", "run_fitting.py").exists())

        tempdir.cleanup()


    def test_refresh_config_dir_and_contents_never_copies_dunder_files(self):
        tempdir = tempfile.TemporaryDirectory()

        with patch.object(handle_config, "USER_CONFIG_PATH", Path(tempdir.name)):
            handle_config.refresh_config_dir_and_contents()

        self.assertFalse(Path(tempdir.name, "__init__.py").exists())
        self.assertFalse(Path(tempdir.name, "__pycache__").exists())
        self.assertFalse(any(Path(tempdir.name).rglob("__*.py")))
        tempdir.cleanup()


    def test_refresh_config_dir_and_contents_never_copies_vesuvio_user_properties(self):
        tempdir = tempfile.TemporaryDirectory()

        with patch.object(handle_config, "USER_CONFIG_PATH", Path(tempdir.name)):
            handle_config.refresh_config_dir_and_contents()

        self.assertFalse(Path(tempdir.name, "vesuvio.user.properties").exists())
        tempdir.cleanup()


    def test_refresh_config_dir_and_contents_dont_overwrite_copy_if_present_file(self):
        tempdir = tempfile.TemporaryDirectory()
        template_dir = Path(tempdir.name, "experiment_template")
        template_dir.mkdir(parents=True, exist_ok=True)
        script_path = template_dir / "script_to_create_figures.py"
        script_path.write_text("mock script")

        with patch.object(handle_config, "USER_CONFIG_PATH", Path(tempdir.name)):
            handle_config.refresh_config_dir_and_contents()

        self.assertEqual(script_path.read_text(), "mock script")
        tempdir.cleanup()


    def test_refresh_config_dir_and_contents_dont_overwrite_copy_if_present_dir(self):
        tempdir = tempfile.TemporaryDirectory()
        ip_dir = Path(tempdir.name, "ip_files")
        ip_dir.mkdir(parents=True, exist_ok=True)
        marker = ip_dir / "ip.par"
        marker.write_text("keep")

        with patch.object(handle_config, "USER_CONFIG_PATH", Path(tempdir.name)):
            handle_config.refresh_config_dir_and_contents()

        self.assertEqual(marker.read_text(), "keep")
        tempdir.cleanup()


    def test_refresh_config_dir_and_contents_overwrites_always_copy_file(self):
        tempdir = tempfile.TemporaryDirectory()
        target = Path(tempdir.name, "experiment_template", "run_reduction.py")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("old content")

        with patch.object(handle_config, "USER_CONFIG_PATH", Path(tempdir.name)):
            handle_config.refresh_config_dir_and_contents()

        self.assertEqual(
            target.read_text(),
            Path(
                handle_config.PACKAGE_CONFIG_PATH,
                "experiment_template",
                "run_reduction.py",
            ).read_text(),
        )
        tempdir.cleanup()


    def test_refresh_config_dir_and_contents_ignores_dunder_files_in_ip_dir_copy(self):
        tempdir = tempfile.TemporaryDirectory()

        with patch.object(handle_config, "USER_CONFIG_PATH", Path(tempdir.name)):
            handle_config.refresh_config_dir_and_contents()

        self.assertFalse(Path(tempdir.name, "ip_files", "__init__.py").exists())
        tempdir.cleanup()

import unittest
from mock import MagicMock, patch, call
from mvesuvio.util import handle_config
import tempfile
import os

class TestHandleConfig(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

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

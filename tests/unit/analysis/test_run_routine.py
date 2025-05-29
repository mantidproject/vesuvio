
import unittest
import numpy as np
import numpy.testing as nptest
from mock import MagicMock, patch, call
from mvesuvio.main.run_routine import Runner
from mvesuvio.util import handle_config
from pathlib import Path
import mvesuvio
import tempfile
from textwrap import dedent
import os

class TestRunRoutine(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # TODO: Avoid doing this in the future, can probably replace it with mock
        mvesuvio.set_config(
            ip_folder=str(Path(handle_config.VESUVIO_PACKAGE_PATH).joinpath("config", "ip_files")),
            inputs_file=str(Path(__file__).absolute().parent.parent.parent / "data" / "analysis" / "inputs" / "system_test_inputs.py")
        )
        pass

    def test_make_summarised_log_file(self):
        runner = Runner()
        mock_log_file = tempfile.NamedTemporaryFile(delete=False)
        mock_mantid_log_file = tempfile.NamedTemporaryFile(delete=False)
        mock_mantid_log_file.write(dedent("""
            2025-01-08 10:48:44,832 [0] Notice CreateWorkspace - CreateWorkspace started (child)
            2025-01-08 10:48:44,844 [0] Notice CreateWorkspace - CreateWorkspace successful, Duration 0.01 seconds
            2025-01-08 10:48:44,860 [0] Notice VesuvioAnalysisRoutine -
            Fitting neutron compton profiles ...
            2025-01-08 10:48:45,319 [0] Notice VesuvioAnalysisRoutine - Fit spectrum 148: ✓
            2025-01-08 10:48:45,517 [0] Warning Python - Values in x were outside bounds during a minimize step, clipping to bounds
            2025-01-08 10:48:45,623 [0] Notice VesuvioAnalysisRoutine - Fit spectrum 151: ✓
            2025-01-08 10:48:48,568 [0] Notice CreateEmptyTableWorkspace - CreateEmptyTableWorkspace started (child)
            2025-01-08 10:48:48,570 [0] Notice CreateEmptyTableWorkspace - CreateEmptyTableWorkspace successful, Duration 0.00 seconds
            2025-01-08 10:48:48,573 [0] Notice Python - Table analysis_inputs_fwd_0_means:
            2025-01-08 10:48:48,574 [0] Notice Python -  ----------------------------------------------------------------
            2025-01-08 10:48:48,576 [0] Notice Python - |label |mass   |mean_width|std_width|mean_intensity|std_intensity|
            2025-01-08 10:48:48,578 [0] Notice Python - |1.0079|1.00790|5.29627   |0.19464  |0.91410       |0.00862      |
            2025-01-08 10:48:48,584 [0] Notice Python -  ----------------------------------------------------------------
            2025-01-08 10:48:48,588 [0] Notice VesuvioAnalysisRoutine - VesuvioAnalysisRoutine successful, Duration 3.89 seconds
            2025-01-08 10:48:49,390 [0] Notice Python -
            Shared Parameters: ['sigma']
            2025-01-08 10:48:49,391 [0] Notice Python -
            Unshared Parameters: ['A', 'x0']
        """).encode())
        mock_mantid_log_file.close()
        runner.mantid_log_file = mock_mantid_log_file.name
        runner.make_log_file_name = MagicMock(return_value=mock_log_file.name)
        runner.make_summarised_log_file()
        current_log_file_content = mock_log_file.read()
        self.assertEqual(dedent("""
            2025-01-08 10:48:44,860 [0] Notice VesuvioAnalysisRoutine -
            Fitting neutron compton profiles ...
            2025-01-08 10:48:45,319 [0] Notice VesuvioAnalysisRoutine - Fit spectrum 148: ✓
            2025-01-08 10:48:45,623 [0] Notice VesuvioAnalysisRoutine - Fit spectrum 151: ✓
            2025-01-08 10:48:48,573 [0] Notice Python - Table analysis_inputs_fwd_0_means:
            2025-01-08 10:48:48,574 [0] Notice Python -  ----------------------------------------------------------------
            2025-01-08 10:48:48,576 [0] Notice Python - |label |mass   |mean_width|std_width|mean_intensity|std_intensity|
            2025-01-08 10:48:48,578 [0] Notice Python - |1.0079|1.00790|5.29627   |0.19464  |0.91410       |0.00862      |
            2025-01-08 10:48:48,584 [0] Notice Python -  ----------------------------------------------------------------
            2025-01-08 10:48:48,588 [0] Notice VesuvioAnalysisRoutine - VesuvioAnalysisRoutine successful, Duration 3.89 seconds
            2025-01-08 10:48:49,390 [0] Notice Python -
            Shared Parameters: ['sigma']
            2025-01-08 10:48:49,391 [0] Notice Python -
            Unshared Parameters: ['A', 'x0']
                         """).encode(), current_log_file_content)
        os.remove(mock_mantid_log_file.name)
        os.remove(mock_log_file.name)



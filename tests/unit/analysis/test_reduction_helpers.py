
import os
import tempfile
import unittest
from textwrap import dedent
from unittest.mock import patch

from mvesuvio.util import reduction_helpers


class TestReductionHelpers(unittest.TestCase):
    def test_make_summarised_log_file(self):

        with tempfile.NamedTemporaryFile(delete=False) as mock_mantid_log_file, tempfile.NamedTemporaryFile(delete=False) as mock_summary_file:
            mock_mantid_log_file.write(
                dedent(
                    """
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
                    """
                ).encode()
            )

            mock_mantid_log_file_path = mock_mantid_log_file.name
            mock_summary_file_path = mock_summary_file.name

        with patch("mvesuvio.util.reduction_helpers.FilesManager.get_mantid_log_file", return_value=mock_mantid_log_file_path):
            with patch("mvesuvio.util.reduction_helpers.FilesManager.get_summarised_log_file", return_value=mock_summary_file_path):
                reduction_helpers.make_summarised_log_file()

        with open(mock_summary_file_path, "rb") as summary_file:
            current_log_file_content = summary_file.read()

        self.assertEqual(
            dedent(
                """
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
                             """
            ).encode(),
            current_log_file_content,
        )

        os.remove(mock_mantid_log_file_path)
        os.remove(mock_summary_file_path)



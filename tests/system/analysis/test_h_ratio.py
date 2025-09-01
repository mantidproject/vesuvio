import unittest
from pathlib import Path
from mvesuvio.main.run_routine import Runner
from mvesuvio.util import handle_config
import mvesuvio
from mantid.simpleapi import mtd, LoadAscii, AnalysisDataService, CompareWorkspaces
import os


class TestReduction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        pass

    def test_reduction(self):
        benchmark_path = Path(__file__).absolute().parent.parent.parent / "data" / "analysis" / "benchmark" / "h_ratio" / "hydrogen_intensity_ratios_estimates"
        results_path = Path(__file__).absolute().parent.parent.parent / "data" / "analysis" / "inputs" / "system_test_inputs" / "output_files" / "hydrogen_intensity_ratios_estimates"

        mvesuvio.set_config(
            ip_folder=str(Path(handle_config.VESUVIO_PACKAGE_PATH).joinpath("config", "ip_files")),
            inputs_file=str(Path(__file__).absolute().parent.parent.parent / "data" / "analysis" / "inputs" / "system_test_inputs.py")
        )
        # Delete outputs from previous runs
        if results_path.is_file():
            os.remove(str(results_path))

        runner = Runner(running_tests=True)
        runner.bckwd_ai.intensity_ratio_of_hydrogen_to_chosen_mass = 0
        runner.run()

        AnalysisDataService.clear()
        LoadAscii(str(benchmark_path), Separator="CSV", OutputWorkspace="bench_"+benchmark_path.name)
        LoadAscii(str(results_path), Separator="CSV", OutputWorkspace=results_path.name)

        for ws_name in mtd.getObjectNames():
            if ws_name.startswith('bench'):
                tol = 1e-3
                (result, messages) = CompareWorkspaces(ws_name, ws_name.replace("bench_", ""), Tolerance=tol)
                self.assertTrue(result)


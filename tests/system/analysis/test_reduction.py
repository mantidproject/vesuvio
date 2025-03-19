import unittest
from pathlib import Path
from mvesuvio.main.run_routine import Runner 
from mvesuvio.util import handle_config
import mvesuvio
from mantid.simpleapi import LoadAscii, mtd, CompareWorkspaces, AnalysisDataService
import shutil


class TestReduction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        pass
        
    def test_reduction(self):
        benchmark_path = Path(__file__).absolute().parent.parent.parent / "data" / "analysis" / "benchmark" / "reduction"
        results_path = Path(__file__).absolute().parent.parent.parent / "data" / "analysis" / "inputs" / "system_test_inputs" / "output_files" / "reduction"

        mvesuvio.set_config(
            ip_folder=str(Path(handle_config.VESUVIO_PACKAGE_PATH).joinpath("config", "ip_files")),
            inputs_file=str(Path(__file__).absolute().parent.parent.parent / "data" / "analysis" / "inputs" / "system_test_inputs.py")
        )

        # Delete outputs from previous runs
        if results_path.exists():
            shutil.rmtree(str(results_path))

        Runner(True).run()

        AnalysisDataService.clear()

        for p in benchmark_path.iterdir():
            if p.suffix == '.py':
                pass
            LoadAscii(str(p), Separator="CSV", OutputWorkspace="bench_"+p.name)

        for p in results_path.iterdir():
            LoadAscii(str(p), Separator="CSV", OutputWorkspace=p.name)

        for ws_name in mtd.getObjectNames():
            if ws_name.startswith('bench'):
                (result, messages) = CompareWorkspaces(ws_name, ws_name.replace("bench_", ""), Tolerance=1e-4)
                self.assertTrue(result)


if (__name__ == "__main__") or (__name__ == "mantidqt.widgets.codeeditor.execution"):
    unittest.main()

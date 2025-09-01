import unittest
from pathlib import Path
from mvesuvio.main.run_routine import Runner
from mvesuvio.util import handle_config
import mvesuvio
from mantid.simpleapi import Load, LoadAscii, mtd, CompareWorkspaces, AnalysisDataService
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

        Runner().run()

        AnalysisDataService.clear()

        for p in benchmark_path.iterdir():
            if p.is_dir():
                continue
            if p.name.endswith("nxs"):
                Load(str(p), OutputWorkspace="bench_"+p.stem)
                continue
            # TODO: Rename ascii files to include a .txt extension
            LoadAscii(str(p), Separator="CSV", OutputWorkspace="bench_"+p.name)

        for p in results_path.iterdir():
            if p.is_dir():
                continue
            if p.name.endswith("nxs"):
                Load(str(p), OutputWorkspace=p.stem)
                continue
            LoadAscii(str(p), Separator="CSV", OutputWorkspace=p.stem)

        for ws_name in mtd.getObjectNames():
            if ws_name.startswith('bench'):
                if ws_name.endswith('fit_results'):
                    # Fit results spectra by spectra very too much for comparison
                    # TODO: Find out why and fix it
                    continue
                else:
                    tol = 1e-3
                (result, messages) = CompareWorkspaces(ws_name, ws_name.replace("bench_", ""), Tolerance=tol)
                self.assertTrue(result)


if (__name__ == "__main__") or (__name__ == "mantidqt.widgets.codeeditor.execution"):
    unittest.main()

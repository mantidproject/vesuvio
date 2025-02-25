from dataclasses import dataclass
import unittest
from pathlib import Path
from mvesuvio.main.run_routine import Runner 
from mvesuvio.util import handle_config
from mvesuvio.analysis_fitting import FitInYSpace
import mvesuvio
from mantid.simpleapi import Load, LoadAscii, mtd, CompareWorkspaces, AnalysisDataService
import shutil


class TestFitting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    def setUp(self):
        pass
        
    def test_reduction(self):

        @dataclass
        class FitInputs:
            show_plots = False
            do_symmetrisation = True
            subtract_calculated_fse_from_data = True
            range_for_rebinning_in_y_space = "-25, 0.5, 25"  # Needs to be symetric
            fitting_model = "gauss"
            run_minos = True
            do_global_fit = True   # Performs global fit with Minuit by default
            number_of_global_fit_groups = 4
            mask_zeros_with = "nan"   

            # NOTE: Need to implement the save path where the results are to be saved
            save_path = Path(__file__).absolute().parent.parent.parent / "data" / "analysis" / "benchmark"
            # NOTE: Take out the masses argument in the future
            masses = [1.0079, 12, 16, 27]


        fi = FitInputs()

        inputs_path = Path(__file__).absolute().parent.parent.parent / "data" / "analysis" / "inputs" / "fitting_inputs"
        benchmark_path = Path(__file__).absolute().parent.parent.parent / "data" / "analysis" / "benchmark" / "gauss_fit"

        ws_to_fit = Load(str(inputs_path / "system_test_inputs_fwd_1_m0.nxs"), OutputWorkspace="ws")
        ws_resolution = Load(str(inputs_path / "system_test_inputs_fwd_1_resolution.nxs"), OutputWorkspace="resolution")

        alg = FitInYSpace(fi, ws_to_fit, ws_resolution)

        alg.run()

        AnalysisDataService.clear()

        for p in benchmark_path.iterdir():
            if p.suffix == '.py':
                pass
            LoadAscii(str(p), Separator="CSV", OutputWorkspace="bench_"+p.name)

        for p in fi.save_path.iterdir():
            LoadAscii(str(p), Separator="CSV", OutputWorkspace=p.name)

        for ws_name in mtd.getObjectNames():
            if ws_name.startswith('bench'):
                self.assertTrue(CompareWorkspaces(ws_name, ws_name.replace("bench_", "")))



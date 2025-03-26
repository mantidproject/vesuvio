from dataclasses import dataclass
from typing_extensions import override
import unittest
from pathlib import Path
from mvesuvio.main.run_routine import Runner 
from mvesuvio.util import handle_config
from mvesuvio.analysis_fitting import FitInYSpace
from mantid.simpleapi import Load, LoadAscii, mtd, CompareWorkspaces, AnalysisDataService

def ascii_workspaces_match(benchmark_dir, target_dir):
    for p in benchmark_dir.iterdir():
        if p.suffix == '.py':
            pass
        LoadAscii(str(p), Separator="CSV", OutputWorkspace="bench_"+p.name)

    for p in target_dir.iterdir():
        LoadAscii(str(p), Separator="CSV", OutputWorkspace=p.name)

    match = False
    for ws_name in mtd.getObjectNames():
        if ws_name.startswith('bench'):
            match = CompareWorkspaces(ws_name, ws_name.replace("bench_", ""), Tolerance=1e-4)
            if not match:
                return False
    return match


class TestFitting(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.inputs_path = Path(__file__).absolute().parent.parent.parent / "data" / "analysis" / "inputs" / "fitting_inputs"
        cls.benchmark_dir = Path(__file__).absolute().parent.parent.parent / "data" / "analysis" / "benchmark" / "fitting"
        cls.target_dir = Path(__file__).absolute().parent.parent.parent / "data" / "analysis" / "inputs" / "system_test_inputs" / "output_files" / "fitting"

        cls.ws_to_fit_path = cls.inputs_path / "system_test_inputs_fwd_1_m0_-fse.nxs"
        cls.ws_to_fit_ncp_path = cls.inputs_path / "system_test_inputs_fwd_1_1.0079_ncp_-fse.nxs"
        cls.ws_resolution_path = cls.inputs_path / "system_test_inputs_fwd_1_resolution.nxs"

        ipfile_path = Path(handle_config.VESUVIO_IPFOLDER_PATH) / "ip2018_3.par"

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

            save_path = cls.target_dir
            masses = [1.0079, 12, 16, 27]
            instrument_parameters_file = ipfile_path 
            detectors = '144-182'

        cls.fi = FitInputs()
        return

    def setUp(self):
        self.ws_to_fit = Load(str(self.ws_to_fit_path), OutputWorkspace="ws")
        self.ws_to_fit_ncp = Load(str(self.ws_to_fit_ncp_path), OutputWorkspace="ws_ncp")
        self.ws_resolution = Load(str(self.ws_resolution_path), OutputWorkspace="resolution")
        return
        
    def tearDown(self) -> None:
        AnalysisDataService.clear()
        return

    def test_gauss_with_symmetrisation_and_fse(self):
        fi = self.fi
        fi.fitting_model = "gauss"
        fi.do_symmetrisation = True 
        fi.subtract_calculated_fse_from_data = True
        alg = FitInYSpace(self.fi, self.ws_to_fit, self.ws_to_fit_ncp, self.ws_resolution)
        alg.run()
        self.assertTrue(ascii_workspaces_match(self.benchmark_dir / "gauss_fit", self.target_dir / "gauss_fit"))

    def test_gcc4c6_no_symmetrisation_and_fse(self):
        fi = self.fi
        fi.fitting_model = "gcc4c6"
        fi.do_symmetrisation = False
        fi.subtract_calculated_fse_from_data = True
        alg = FitInYSpace(fi, self.ws_to_fit, self.ws_to_fit_ncp, self.ws_resolution)
        alg.run()
        self.assertTrue(ascii_workspaces_match(self.benchmark_dir / "gcc4c6_fit", self.target_dir / "gcc4c6_fit"))

    def test_ansiogauss_with_symmetrisation_no_fse(self):
        fi = self.fi
        fi.fitting_model = "ansiogauss"
        fi.do_symmetrisation = True
        fi.subtract_calculated_fse_from_data = False
        alg = FitInYSpace(fi, self.ws_to_fit, self.ws_to_fit_ncp, self.ws_resolution)
        alg.run()
        self.assertTrue(ascii_workspaces_match(self.benchmark_dir / "ansiogauss_fit", self.target_dir / "ansiogauss_fit"))

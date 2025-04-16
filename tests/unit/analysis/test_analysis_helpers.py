import unittest
import numpy as np
from numpy.testing._private.utils import assert_allclose
import scipy
import dill
from pathlib import Path
import numpy.testing as nptest
from mock import MagicMock, Mock, patch, call
from mvesuvio.util.analysis_helpers import calculate_resolution, create_profiles_table, extractWS, _convert_dict_to_table,  \
    fix_profile_parameters, calculate_h_ratio, extend_range_of_array, is_hydrogen_present, isolate_lighest_mass_data, load_instrument_params, load_raw_and_empty_from_path, load_resolution, numerical_third_derivative,  \
    mask_time_of_flight_bins_with_zeros, make_gamma_correction_input_string, make_multiple_scattering_input_string, print_table_workspace, save_ws_from_load_vesuvio, ws_history_matches_inputs
from mantid.simpleapi import CreateWorkspace, DeleteWorkspace, GroupWorkspaces, RenameWorkspace, Load, SaveNexus, CompareWorkspaces, Rebin, AnalysisDataService
import tempfile
from textwrap import dedent

class TestAnalysisHelpers(unittest.TestCase):
    def setUp(self):
        pass

    def tearDown(self):
        AnalysisDataService.clear()

    def test_extract_ws(self):
        data = [1, 2, 3]
        ws = CreateWorkspace(DataX=data, DataY=data, DataE=data, NSpec=1, UnitX="some_unit")

        dataX, dataY, dataE = extractWS(ws)
        nptest.assert_array_equal([data], dataX)
        nptest.assert_array_equal([data], dataY)
        nptest.assert_array_equal([data], dataE)

        DeleteWorkspace(ws)


    def test_convert_dict_to_table(self):
        d = {'H': {'label': 'H', 'mass': 1, 'intensity': 1}}
        table = _convert_dict_to_table(d)
        self.assertEqual(['label', 'mass', 'intensity'], table.getColumnNames())
        self.assertEqual({'label': 'H', 'mass': 1, 'intensity': 1}, table.row(0))


    def test_fix_profile_parameters_with_H(self):
        means_table_mock = MagicMock()
        means_table_mock.rowCount.return_value = 3 
        means_table_mock.row.side_effect = [
            {'label': '16.0', 'mass': 16.0, 'mean_width': 8.974, 'std_width': 1.401, 'mean_intensity': 0.176, 'std_intensity': 0.08722},
            {'label': '27.0', 'mass': 27.0, 'mean_width': 15.397, 'std_width': 1.131, 'mean_intensity': 0.305, 'std_intensity': 0.04895},
            {'label': '12.0', 'mass': 12.0, 'mean_width': 13.932, 'std_width': 0.314, 'mean_intensity': 0.517, 'std_intensity': 0.09531}
        ]
        profiles_table_mock = MagicMock()
        profiles_table_mock.rowCount.return_value = 4
        profiles_table_mock.row.side_effect = [
            {'label': '1.0079', 'mass': 1.0078, 'intensity': 1.0, 'intensity_lb': 0 , 'intensity_ub': np.inf, 'width': 4.699, 'width_lb': 3, 'width_ub': 6, 'center': 0.0, 'center_lb': -3, 'center_ub' : 1},
            {'label': '16.0', 'mass': 16.0, 'intensity': 1.0, 'intensity_lb': 0 , 'intensity_ub': np.inf, 'width': 12, 'width_lb': 0, 'width_ub': np.inf, 'center': 0.0, 'center_lb': -3, 'center_ub': 1},
            {'label': '12.0', 'mass': 12.0, 'intensity': 1.0, 'intensity_lb': 0 , 'intensity_ub': np.inf, 'width': 8, 'width_lb': 0, 'width_ub': np.inf, 'center': 0.0, 'center_lb': -3, 'center_ub': 1},
            {'label': '27.0', 'mass': 27.0, 'intensity': 1.0, 'intensity_lb': 0 , 'intensity_ub': np.inf, 'width': 13, 'width_lb': 0, 'width_ub': np.inf, 'center': 0.0, 'center_lb': -3, 'center_ub': 1}
        ]

        result_table = fix_profile_parameters(means_table_mock, profiles_table_mock, h_ratio=14.7)

        # For some reason, reading floats from table introduces small variations in stored values
        # TODO: Fix floating positions eg. 8.973999977111816 -> 8.974
        self.assertEqual(
            result_table.row(0),
            {'label': '1.0079', 'mass': 1.0077999830245972, 'intensity': 0.8839251399040222, 'intensity_lb': 0.0, 'intensity_ub': np.inf, 'width': 4.698999881744385, 'width_lb': 3.0, 'width_ub': 6.0, 'center': 0.0, 'center_lb': -3.0, 'center_ub': 1.0}
        )
        self.assertEqual(
            result_table.row(1),
            {'label': '16.0', 'mass': 16.0, 'intensity': 0.020470114424824715, 'intensity_lb': 0.0, 'intensity_ub': np.inf, 'width': 8.973999977111816, 'width_lb': 8.973999977111816, 'width_ub': 8.973999977111816, 'center': 0.0, 'center_lb': -3.0 , 'center_ub': 1.0}
        )
        self.assertEqual(
            result_table.row(2),
            {'label': '12.0', 'mass': 12.0, 'intensity': 0.06013096123933792, 'intensity_lb': 0.0 , 'intensity_ub': np.inf, 'width': 13.932000160217285, 'width_lb': 13.932000160217285, 'width_ub': 13.932000160217285, 'center': 0.0, 'center_lb': -3.0, 'center_ub': 1.0}
        )
        self.assertEqual(
            result_table.row(3),
            {'label': '27.0', 'mass': 27.0, 'intensity': 0.0354737788438797, 'intensity_lb': 0.0 , 'intensity_ub': np.inf, 'width': 15.397000312805176, 'width_lb': 15.397000312805176, 'width_ub': 15.397000312805176, 'center': 0.0, 'center_lb': -3.0 , 'center_ub': 1.0}
        )
        

    def test_fix_profile_parameters_without_H(self):
        # TODO: Use a more physical example containing Deuterium
        # Same profiles as before, except when H is not present,
        # the first mass will be made free to vary and the intensities don't change

        means_table_mock = MagicMock()
        means_table_mock.rowCount.return_value = 3 
        means_table_mock.row.side_effect = [
            {'label': '16.0', 'mass': 16.0, 'mean_width': 8.974, 'std_width': 1.401, 'mean_intensity': 0.176, 'std_intensity': 0.08722},
            {'label': '27.0', 'mass': 27.0, 'mean_width': 15.397, 'std_width': 1.131, 'mean_intensity': 0.305, 'std_intensity': 0.04895},
            {'label': '12.0', 'mass': 12.0, 'mean_width': 13.932, 'std_width': 0.314, 'mean_intensity': 0.517, 'std_intensity': 0.09531}
        ]
        profiles_table_mock = MagicMock()
        profiles_table_mock.rowCount.return_value = 3
        profiles_table_mock.row.side_effect = [
            {'label': '16.0', 'mass': 16.0, 'intensity': 1.0, 'intensity_lb': 0 , 'intensity_ub': np.inf, 'width': 12, 'width_lb': 0 , 'width_ub': np.inf, 'center': 0.0, 'center_lb': -3 , 'center_ub': 1},
            {'label': '12.0', 'mass': 12.0, 'intensity': 1.0, 'intensity_lb': 0 , 'intensity_ub': np.inf, 'width': 8, 'width_lb': 0 , 'width_ub': np.inf, 'center': 0.0, 'center_lb': -3 , 'center_ub': 1},
            {'label': '27.0', 'mass': 27.0, 'intensity': 1.0, 'intensity_lb': 0 , 'intensity_ub': np.inf, 'width': 13, 'width_lb': 0 , 'width_ub': np.inf, 'center': 0.0, 'center_lb': -3 , 'center_ub': 1}
        ]

        result_table = fix_profile_parameters(means_table_mock, profiles_table_mock, h_ratio=14.7)

        # For some reason, reading floats from table introduces small variations in stored values
        # TODO: Fix floating positions eg. 8.973999977111816 -> 8.974
        self.assertEqual(
            result_table.row(0),
            {'label': '16.0', 'mass': 16.0, 'intensity': 0.17635270953178406, 'intensity_lb': 0.0, 'intensity_ub': np.inf, 'width': 8.973999977111816, 'width_lb': 8.973999977111816, 'width_ub': 8.973999977111816, 'center': 0.0, 'center_lb': -3.0 , 'center_ub': 1.0}
        )
        self.assertEqual(
            result_table.row(1),
            {'label': '12.0', 'mass': 12.0, 'intensity': 0.5180360674858093, 'intensity_lb': 0.0, 'intensity_ub': np.inf, 'width': 13.932000160217285, 'width_lb': 0.0, 'width_ub': np.inf, 'center': 0.0, 'center_lb': -3.0, 'center_ub': 1.0}
        )
        self.assertEqual(
            result_table.row(2),
            {'label': '27.0', 'mass': 27.0, 'intensity': 0.3056112229824066, 'intensity_lb': 0.0, 'intensity_ub': np.inf, 'width': 15.397000312805176, 'width_lb': 15.397000312805176, 'width_ub': 15.397000312805176, 'center': 0.0, 'center_lb': -3.0, 'center_ub': 1.0}
        )


    def test_calculate_h_ratio(self):
        means_table_mock = MagicMock()
        means_table_mock.column.side_effect = lambda x: [16, 1, 12] if x=="mass" else [0.1, 0.85, 0.05]
        h_ratio = calculate_h_ratio(means_table_mock)
        self.assertEqual(h_ratio, 0.85 / 0.05)


    def test_conversion_of_constraints(self):
        constraints = ({'type': 'eq', 'fun': lambda par:  par[0] - 2.7527*par[3] },{'type': 'eq', 'fun': lambda par:  par[3] - 0.7234*par[6] })
        # Used before passing constraints into Mantid algorithm
        string_constraints = str(dill.dumps(constraints))
        self.assertIsInstance(string_constraints, str)
        # Used inside Mantid algorithm to convert back to SciPy constraints
        converted_constraints = dill.loads(eval(string_constraints))
        self.assertEqual(converted_constraints[0]['fun']([3, 0, 0, 1]), 3-2.7527)
        self.assertEqual(converted_constraints[1]['fun']([0, 0, 0, 2, 0, 0, 1]), 2-0.7234)


    def test_extend_range_of_array_for_increasing_range(self):
        x = np.arange(10)
        x = np.vstack([x, 2*x])
        x_extended = extend_range_of_array(x, 5)
        np.testing.assert_array_equal(x_extended, np.vstack([np.arange(-5, 15, 1), np.arange(-10, 30, 2)]))


    def test_extend_range_of_array_for_decreasing_range(self):
        x = np.linspace(-5, 5, 21)
        x = np.vstack([x, 2*x])
        x_extended = extend_range_of_array(x, 5)
        np.testing.assert_array_equal(x_extended, np.vstack([np.linspace(-7.5, 7.5, 31), np.linspace(-15, 15, 31)]))


    def test_numerical_third_derivative(self):
        x= np.linspace(-20, 20, 300)    # Workspaces are about 300 points of range
        x = np.vstack([x, 2*x])
        y = scipy.special.voigt_profile(x, 5, 5)
        numerical_derivative = numerical_third_derivative(x, y)
        expected_derivative = np.array([np.gradient(np.gradient(np.gradient(y_i, x_i), x_i), x_i)[6: -6] for y_i, x_i in zip(y, x) ])
        np.testing.assert_allclose(numerical_derivative, expected_derivative, atol=1e-6)


    def test_mask_time_of_flight_bins_with_zeros(self):
        data_x = np.arange(10).reshape(1, -1) * np.ones((3, 1))
        data_y = np.ones((3, 10))
        data_e = np.ones((3, 10))
        workspace_mock = MagicMock()
        workspace_mock.extractX.return_value = data_x
        workspace_mock.extractY.return_value = data_y
        workspace_mock.extractE.return_value = data_e

        actual_data_x = np.zeros((3, 10))
        actual_data_y = np.zeros((3, 10))
        actual_data_e = np.zeros((3, 10))

        workspace_mock.dataY.side_effect = lambda i: actual_data_y[i]
        workspace_mock.dataX.side_effect = lambda i: actual_data_x[i]
        workspace_mock.dataE.side_effect = lambda i: actual_data_e[i]

        workspace_mock.getNumberHistograms.return_value = 3
        mask_time_of_flight_bins_with_zeros(workspace_mock, '4.5-7.3')

        np.testing.assert_allclose(actual_data_x, data_x)
        np.testing.assert_allclose(actual_data_e, data_e)
        expected_data_y = np.ones((3, 10)) 
        expected_data_y[(data_x >= 4.5) & (data_x <= 7.3)] = 0
        np.testing.assert_allclose(actual_data_y, expected_data_y)


    def test_isolate_lighest_mass_data_no_fse_subtraction(self):
        np.random.seed(0)

        dataX = np.linspace(0, 10, 100).reshape(1, -1)
        dataE = np.full_like(dataX, 0.1)
        ws_ncp1 = CreateWorkspace(DataX=dataX, DataY=np.exp(-(dataX-3)**2), DataE=dataE, NSpec=1, UnitX="some_unit", OutputWorkspace="_1_ncp")
        ws_ncp2 = CreateWorkspace(DataX=dataX, DataY=np.exp(-(dataX-7)**2), DataE=dataE, NSpec=1, UnitX="some_unit", OutputWorkspace="_2_ncp")
        ws_ncp3 = CreateWorkspace(DataX=dataX, DataY=np.exp(-(dataX-8)**2), DataE=dataE, NSpec=1, UnitX="some_unit", OutputWorkspace="_3_ncp")
        ws_total = ws_ncp1 + ws_ncp2 + ws_ncp3
        RenameWorkspace(ws_total, "_total_ncp")
        data = ws_total.extractY() + (np.random.random(100)-0.5) * 0.5
        data[0, :10] = 0
        ws_data = CreateWorkspace(DataX=dataX, DataY=data, DataE=1.5*dataE, NSpec=1, UnitX="some_unit")
        ncp_group = GroupWorkspaces([ws_ncp3, ws_ncp2, ws_ncp1, ws_total])

        ws_res, ws_res_ncp = isolate_lighest_mass_data(ws_data, ncp_group, False)

        ws_expected = ws_data - ws_ncp2 - ws_ncp3
        # Expected workspace should have masked values with zeros
        ws_expected.dataY(0)[:10] = 0

        np.testing.assert_allclose(ws_res.extractY(), ws_expected.extractY())
        np.testing.assert_allclose(ws_res.extractE(), ws_data.extractE())
        np.testing.assert_allclose(ws_res.extractX(), ws_data.extractX())
        np.testing.assert_allclose(ws_res_ncp.extractY(), ws_ncp1.extractY())
        np.testing.assert_allclose(ws_res_ncp.extractE(), ws_ncp1.extractE())
        np.testing.assert_allclose(ws_res_ncp.extractX(), ws_ncp1.extractX())


    def test_isolate_lighest_mass_data_with_fse_subtraction(self):
        np.random.seed(0)

        dataX = np.linspace(0, 10, 100).reshape(1, -1)
        dataE = np.full_like(dataX, 0.1)
        ws_ncp1 = CreateWorkspace(DataX=dataX, DataY=np.exp(-(dataX-3)**2), DataE=dataE, NSpec=1, UnitX="some_unit", OutputWorkspace="_1_ncp")
        ws_ncp2 = CreateWorkspace(DataX=dataX, DataY=np.exp(-(dataX-7)**2), DataE=dataE, NSpec=1, UnitX="some_unit", OutputWorkspace="_2_ncp")
        ws_ncp3 = CreateWorkspace(DataX=dataX, DataY=np.exp(-(dataX-8)**2), DataE=dataE, NSpec=1, UnitX="some_unit", OutputWorkspace="_3_ncp")
        ws_total = ws_ncp1 + ws_ncp2 + ws_ncp3
        RenameWorkspace(ws_total, "_total_ncp")
        data = ws_total.extractY() + (np.random.random(100)-0.5) * 0.5
        ws_data = CreateWorkspace(DataX=dataX, DataY=data, DataE=1.5*dataE, NSpec=1, UnitX="some_unit")
        ncp_group = GroupWorkspaces([ws_ncp3, ws_ncp2, ws_ncp1, ws_total])
        ws_fse1 = CreateWorkspace(DataX=dataX, DataY=0.4*np.sin(dataX)*np.exp(-(dataX-3)**2), DataE=0.5*dataE, NSpec=1, UnitX="some_unit", OutputWorkspace="_1_fse")

        ws_res, ws_res_ncp = isolate_lighest_mass_data(ws_data, ncp_group, True)

        ws_expected = ws_data - ws_ncp2 - ws_ncp3 - ws_fse1
        np.testing.assert_allclose(ws_res.extractY(), ws_expected.extractY())
        np.testing.assert_allclose(ws_res.extractE(), ws_data.extractE())
        np.testing.assert_allclose(ws_res.extractX(), ws_data.extractX())
        np.testing.assert_allclose(ws_res_ncp.extractY(), ws_ncp1.extractY() - ws_fse1.extractY())
        np.testing.assert_allclose(ws_res_ncp.extractE(), ws_ncp1.extractE())
        np.testing.assert_allclose(ws_res_ncp.extractX(), ws_ncp1.extractX())

    
    def test_vesuvio_resolution(self):
        import matplotlib.pyplot as plt
        ws_data = Load(str(Path(__file__).parent.parent.parent/"data/analysis/unit/analysis_fwd_0.nxs"))

        ws_res = calculate_resolution(1, ws_data, '-25, 5, 25')

        np.set_printoptions(precision=3)
        np.testing.assert_allclose(ws_res.dataY(0)[:], np.array([6.140e-05, 1.028e-04, 2.073e-04, 6.221e-04, 9.877e-02, 9.876e-02, 6.218e-04, 2.074e-04, 8.361e-05, 0.000e+00]), rtol=1e-3)
        np.testing.assert_allclose(ws_res.dataY(15)[:], np.array([8.188e-05, 1.555e-04, 3.137e-04, 9.517e-04, 9.811e-02, 9.812e-02, 9.517e-04, 3.137e-04, 1.555e-04, 9.288e-05]), rtol=1e-3)
        np.testing.assert_allclose(ws_res.dataY(30)[:], np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]), rtol=1e-3)


    def test_make_gamma_correction_input_string(self):
        masses = [1, 12]
        mean_widths = [5, 10]
        mean_intensity_ratios = [0.6, 0.4]

        profiles_string = make_gamma_correction_input_string(masses, mean_widths, mean_intensity_ratios)

        self.assertEqual(profiles_string, "name=GaussianComptonProfile,Mass=1,Width=5,Intensity=0.6;name=GaussianComptonProfile,Mass=12,Width=10,Intensity=0.4;")

    def test_make_multiple_scattering_input_string(self):

        masses = [1, 12]
        mean_intensity_ratios = [0.6, 0.4]
        mean_widths = [5, 10]

        profiles_list = make_multiple_scattering_input_string(masses, mean_widths, mean_intensity_ratios)

        self.assertEqual(profiles_list, [1.0, 0.6, 5.0, 12.0, 0.4, 10.0])


    def test_print_table_workspace(self):

        mock_table = Mock()
        mock_table.toDict.return_value = {
            "col1": ["a", "b", "c"],
            "col2": [1, 2, 3],
            "col3": [1.0, 2.0, 3.0]
        }
        mock_table.rowCount.return_value = 3
        mock_table.name.side_effect = lambda: "Mock Table Name"

        with patch('mvesuvio.util.analysis_helpers.logger') as mock_logger:

            print_table_workspace(mock_table)

            mock_logger.notice.assert_has_calls([
                call('Table Mock Table Name:'),
                call(' -------------- '),
                call('|col1|col2|col3|'),
                call('|a   |1   |1   |'),
                call('|b   |2   |2   |'),
                call('|c   |3   |3   |'),
                call(' -------------- ')
            ])

    def test_create_profiles_table(self):

        mock_ai = Mock()
        mock_ai.masses = [1, 12, 16]
        mock_ai.initial_fitting_parameters = [1, 5, 0, 1, 10, 0, 1, 13, 0]
        mock_ai.fitting_bounds = [[0, None], [2, 6], [-1, 3], [0, None], [8, 12], [-1, 3], [0, np.inf], [11, 15], [-1, 3]]

        with patch('mvesuvio.util.analysis_helpers.CreateEmptyTableWorkspace') as mock_create_table_ws:
            table_mock = MagicMock()
            mock_create_table_ws.return_value = table_mock

            create_profiles_table(table_mock, mock_ai)

            table_mock.addColumn.assert_has_calls([
                call(type='str', name='label'),
                call(type='float', name='mass'),
                call(type='float', name='intensity'),
                call(type='float', name='intensity_lb'),
                call(type='float', name='intensity_ub'),
                call(type='float', name='width'),
                call(type='float', name='width_lb'),
                call(type='float', name='width_ub'),
                call(type='float', name='center'),
                call(type='float', name='center_lb'),
                call(type='float', name='center_ub')
            ])
            table_mock.addRow.assert_has_calls([
                call(['1.0', 1.0, 1.0, 0.0, np.inf, 5.0, 2.0, 6.0, 0.0, -1.0, 3.0]),
                call(['12.0', 12.0, 1.0, 0.0, np.inf, 10.0, 8.0, 12.0, 0.0, -1.0, 3.0]),
                call(['16.0', 16.0, 1.0, 0.0, np.inf, 13.0, 11.0, 15.0, 0.0, -1.0, 3.0])
            ])

    def test_is_hydrogen_present_with_hydrogen(self):
        masses = np.array([1.0078, 12.0, 16.0])
        is_present = is_hydrogen_present(masses)
        self.assertTrue(is_present)


    def test_is_hydrogen_present_without_hydrogen(self):
        masses = np.array([2.0, 12.0, 16.0])
        is_present = is_hydrogen_present(masses)
        self.assertFalse(is_present)


    def test_is_hydrogen_present_bad_inputs(self):
        # Function not supposed to be used when only forward scattering should be run
        masses = np.array([1.01])
        with self.assertRaises(AssertionError):
            is_hydrogen_present(masses)

        # Hydrogen not first mass
        masses = np.array([2.0, 1.0, 12.0])
        with self.assertRaises(AssertionError):
            is_hydrogen_present(masses)

        # More than one hydrogen
        masses = np.array([1.0, 1.0078, 12.0])
        with self.assertRaises(AssertionError):
            is_hydrogen_present(masses)


    def test_is_hydrogen_present_one_mass_no_hydrogen(self):
        masses = np.array([2.0])
        is_present = is_hydrogen_present(masses)
        self.assertFalse(is_present)

    def test_ws_history_matches_inputs_invalid_path(self):
        path = Path("notthere.nxs")
        with patch('mvesuvio.util.analysis_helpers.logger') as mock_logger:
            match = ws_history_matches_inputs(0, 0, 0, path)
            mock_logger.notice.assert_has_calls([call('Cached workspace not found at notthere.nxs')])
            self.assertFalse(match)


    @patch('mvesuvio.util.analysis_helpers.Load')
    def test_ws_history_matches_inputs_bad_runs(self, mock_load):
        path = Mock()
        path.is_file.return_value = True
        props = {
            "Filename": "1234-1235",
            "Mode": "SingleDifference",
            "InstrumentParFile": "ip_par.txt"
        }
        mock_metadata = Mock()
        mock_metadata.getPropertyValue.side_effect = lambda key: props[key]
        mock_history = Mock()
        mock_history.getAlgorithmHistory.return_value = mock_metadata
        mock_ws = Mock()
        mock_ws.getHistory.return_value = mock_history
        mock_load.return_value = mock_ws

        with patch('mvesuvio.util.analysis_helpers.logger') as mock_logger:
            match = ws_history_matches_inputs("0000", "SingleDifference", "ip_par.txt", path)
            mock_logger.notice.assert_has_calls([call('Filename in saved workspace did not match: 1234-1235 and 0000')])
            self.assertFalse(match)


    @patch('mvesuvio.util.analysis_helpers.Load')
    def test_ws_history_matches_inputs_bad_mode(self, mock_load):
        path = Mock()
        path.is_file.return_value = True
        props = {
            "Filename": "1234-1235",
            "Mode": "SingleDifference",
            "InstrumentParFile": "ip_par.txt"
        }
        mock_metadata = Mock()
        mock_metadata.getPropertyValue.side_effect = lambda key: props[key]
        mock_history = Mock()
        mock_history.getAlgorithmHistory.return_value = mock_metadata
        mock_ws = Mock()
        mock_ws.getHistory.return_value = mock_history
        mock_load.return_value = mock_ws

        with patch('mvesuvio.util.analysis_helpers.logger') as mock_logger:
            match = ws_history_matches_inputs("1234-1235", "DoubleDifference", "ip_par.txt", path)
            mock_logger.notice.assert_has_calls([call('Mode in saved workspace did not match: SingleDifference and DoubleDifference')])
            self.assertFalse(match)

    @patch('mvesuvio.util.analysis_helpers.Load')
    def test_ws_history_matches_inputs_bad_mode(self, mock_load):
        path = Mock()
        path.is_file.return_value = True
        props = {
            "Filename": "1234-1235",
            "Mode": "SingleDifference",
            "InstrumentParFile": "ip_par.txt"
        }
        mock_metadata = Mock()
        mock_metadata.getPropertyValue.side_effect = lambda key: props[key]
        mock_history = Mock()
        mock_history.getAlgorithmHistory.return_value = mock_metadata
        mock_ws = Mock()
        mock_ws.getHistory.return_value = mock_history
        mock_load.return_value = mock_ws

        with patch('mvesuvio.util.analysis_helpers.logger') as mock_logger:
            match = ws_history_matches_inputs("1234-1235", "SingleDifference", "new_par.txt", path)
            mock_logger.notice.assert_has_calls([call('IP files in saved workspace did not match: ip_par.txt and new_par.txt')])
            self.assertFalse(match)

    @patch('mvesuvio.util.analysis_helpers.DeleteWorkspace')
    @patch('mvesuvio.util.analysis_helpers.Load')
    def test_ws_history_matches_good_inputs(self, mock_load, mock_delete):
        path = Mock()
        path.is_file.return_value = True
        props = {
            "Filename": "1234-1235",
            "Mode": "SingleDifference",
            "InstrumentParFile": "ip_par.txt"
        }
        mock_metadata = Mock()
        mock_metadata.getPropertyValue.side_effect = lambda key: props[key]
        mock_history = Mock()
        mock_history.getAlgorithmHistory.return_value = mock_metadata
        mock_ws = Mock()
        mock_ws.getHistory.return_value = mock_history
        mock_load.return_value = mock_ws

        with patch('mvesuvio.util.analysis_helpers.logger') as mock_logger:
            match = ws_history_matches_inputs("1234-1235", "SingleDifference", "ip_par.txt", path)
            mock_logger.notice.assert_has_calls([call('\nLocally saved workspace metadata matched with analysis inputs.\n')])
            self.assertTrue(match)


    @patch('mvesuvio.util.analysis_helpers.SaveNexus')
    @patch('mvesuvio.util.analysis_helpers.LoadVesuvio')
    def test_save_ws_from_load_vesuvio_backward(self, mock_load_vesuvio, mock_save_nexus):
        path = Path('notthere/raw_backward.nxs')
        save_ws_from_load_vesuvio("1234", "SingleDifference", "ipfile.txt", path)
        mock_load_vesuvio.assert_has_calls([
            call(Filename='1234', SpectrumList='3-134', Mode='SingleDifference', InstrumentParFile='ipfile.txt', OutputWorkspace='raw_backward.nxs', LoadLogFiles=False)
        ])
        args, kwargs = mock_save_nexus.call_args
        self.assertEqual(kwargs["Filename"], str(path.absolute()))


    @patch('mvesuvio.util.analysis_helpers.SaveNexus')
    @patch('mvesuvio.util.analysis_helpers.LoadVesuvio')
    def test_save_ws_from_load_vesuvio_forward(self, mock_load_vesuvio, mock_save_nexus):
        path = Path('notthere/raw_forward.nxs')
        save_ws_from_load_vesuvio("1234", "SingleDifference", "ipfile.txt", path)
        mock_load_vesuvio.assert_has_calls([
            call(Filename='1234', SpectrumList="135-198", Mode='SingleDifference', InstrumentParFile='ipfile.txt', OutputWorkspace='raw_forward.nxs', LoadLogFiles=False)
        ])
        args, kwargs = mock_save_nexus.call_args
        self.assertEqual(kwargs["Filename"], str(path.absolute()))


    def test_load_resolution(self):

        instrument_parameters = np.vstack([np.arange(130, 140), np.zeros(10), np.zeros(10)]).T
        res_pars = load_resolution(instrument_parameters)

        expected_res_pars = np.array([
            [8.87e+01, 3.70e-01, 1.60e-02, 2.10e-02, 2.30e-02, 4.03e+01],
            [8.87e+01, 3.70e-01, 1.60e-02, 2.10e-02, 2.30e-02, 4.03e+01],
            [8.87e+01, 3.70e-01, 1.60e-02, 2.10e-02, 2.30e-02, 4.03e+01],
            [8.87e+01, 3.70e-01, 1.60e-02, 2.10e-02, 2.30e-02, 4.03e+01],
            [8.87e+01, 3.70e-01, 1.60e-02, 2.10e-02, 2.30e-02, 4.03e+01],
            [7.30e+01, 3.70e-01, 1.60e-02, 2.10e-02, 2.30e-02, 2.40e+01],
            [7.30e+01, 3.70e-01, 1.60e-02, 2.10e-02, 2.30e-02, 2.40e+01],
            [7.30e+01, 3.70e-01, 1.60e-02, 2.10e-02, 2.30e-02, 2.40e+01],
            [7.30e+01, 3.70e-01, 1.60e-02, 2.10e-02, 2.30e-02, 2.40e+01],
            [7.30e+01, 3.70e-01, 1.60e-02, 2.10e-02, 2.30e-02, 2.40e+01]])

        np.testing.assert_allclose(res_pars, expected_res_pars)


    def test_load_instrument_params(self):

        ip_file_path = Path(__file__).parent.parent.parent / "data/analysis/unit/ip_example.par"
        ip = load_instrument_params(ip_file_path, np.array([5, 6, 7, 8]))

        print(str(ip).replace('\n', ',\n'))
        expected_ip = np.array([
            [ 5., 5., 133.892, -0.2, 11.005, 0.587558],
            [  6., 6., 133.753, -0.2, 11.005, 0.59536 ],
            [  7., 7., 133.246, -0.2, 11.005, 0.59228 ],
            [  8., 8., 131.671, -0.2, 11.005, 0.619911]
        ])
        np.testing.assert_allclose(ip, expected_ip)
        

    def test_load_raw_and_empty_from_path_with_subtraction(self):

        empty_path = Path(__file__).parent.parent.parent / "data/analysis/unit/system_test_inputs_empty_backward.nxs"
        raw_path = Path(__file__).parent.parent.parent / "data/analysis/unit/system_test_inputs_raw_backward.nxs"

        ws_result = load_raw_and_empty_from_path(raw_path, empty_path, "110, 5, 400", "test", 1.1, 0.9, True)

        raw_ws = Load(Filename=str(raw_path))
        raw_ws = Rebin(raw_ws, "110, 5, 400")
        empty_ws = Load(Filename=str(empty_path))
        empty_ws = Rebin(empty_ws, "110, 5, 400")
        expected_ws = 1.1 * raw_ws - 0.9 * empty_ws

        (match, messages) = CompareWorkspaces(ws_result.name(), expected_ws.name())
        error = ""
        if not match:
            error = messages.cell(0,0)
        self.assertTrue(match, error)


    def test_load_raw_and_empty_from_path_without_subtraction(self):

        empty_path = Path(__file__).parent.parent.parent / "data/analysis/unit/system_test_inputs_empty_backward.nxs"
        raw_path = Path(__file__).parent.parent.parent / "data/analysis/unit/system_test_inputs_raw_backward.nxs"

        ws_result = load_raw_and_empty_from_path(raw_path, empty_path, "110, 5, 400", "test", 0.5, 0.8, False)

        raw_ws = Load(Filename=str(raw_path))
        raw_ws = Rebin(raw_ws, "110, 5, 400")
        expected_ws = 0.5 * raw_ws

        (match, messages) = CompareWorkspaces(ws_result.name(), expected_ws.name())
        error = ""
        if not match:
            error = messages.cell(0,0)
        self.assertTrue(match, error)


if __name__ == "__main__":
    unittest.main()

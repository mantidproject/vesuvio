import unittest
import numpy as np
from numpy.testing._private.utils import assert_allclose
import scipy
import dill
from pathlib import Path
import numpy.testing as nptest
from mock import MagicMock, patch
from mvesuvio.util.analysis_helpers import calculate_resolution, extractWS, _convert_dict_to_table,  \
    fix_profile_parameters, calculate_h_ratio, extend_range_of_array, isolate_lighest_mass_data, numerical_third_derivative,  \
    mask_time_of_flight_bins_with_zeros
from mantid.simpleapi import CreateWorkspace, DeleteWorkspace, GroupWorkspaces, RenameWorkspace, Load


class TestAnalysisHelpers(unittest.TestCase):
    def setUp(self):
        pass

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


if __name__ == "__main__":
    unittest.main()

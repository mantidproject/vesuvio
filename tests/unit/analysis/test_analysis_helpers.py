import unittest
import numpy as np
import scipy
import dill
import numpy.testing as nptest
from mock import MagicMock, patch, call
from mvesuvio.util.analysis_helpers import extractWS, _convert_dict_to_table,  \
    fix_profile_parameters, calculate_h_ratio, extend_range_of_array, numerical_third_derivative,  \
    mask_time_of_flight_bins_with_zeros, pass_data_into_ws, print_table_workspace
from mantid.simpleapi import CreateWorkspace, DeleteWorkspace


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
            {'label': '1.0079', 'mass': 1.0078, 'intensity': 1.0, 'intensity_bounds': '[0, None]', 'width': 4.699, 'width_bounds': '[3, 6]', 'center': 0.0, 'center_bounds': '[-3, 1]'},
            {'label': '16.0', 'mass': 16.0, 'intensity': 1.0, 'intensity_bounds': '[0, None]', 'width': 12, 'width_bounds': '[0, None]', 'center': 0.0, 'center_bounds': '[-3, 1]'},
            {'label': '12.0', 'mass': 12.0, 'intensity': 1.0, 'intensity_bounds': '[0, None]', 'width': 8, 'width_bounds': '[0, None]', 'center': 0.0, 'center_bounds': '[-3, 1]'},
            {'label': '27.0', 'mass': 27.0, 'intensity': 1.0, 'intensity_bounds': '[0, None]', 'width': 13, 'width_bounds': '[0, None]', 'center': 0.0, 'center_bounds': '[-3, 1]'}
        ]

        result_table = fix_profile_parameters(means_table_mock, profiles_table_mock, h_ratio=14.7)

        # For some reason, reading floats from table introduces small variations in stored values
        # TODO: Fix floating positions eg. 8.973999977111816 -> 8.974
        self.assertEqual(
            result_table.row(0),
            {'label': '1.0079', 'mass': 1.0077999830245972, 'intensity': 0.8839251399040222, 'intensity_bounds': '[0, None]', 'width': 4.698999881744385, 'width_bounds': '[3, 6]', 'center': 0.0, 'center_bounds': '[-3, 1]'}
        )
        self.assertEqual(
            result_table.row(1),
            {'label': '16.0', 'mass': 16.0, 'intensity': 0.020470114424824715, 'intensity_bounds': '[0, None]', 'width': 8.973999977111816, 'width_bounds': '[8.974, 8.974]', 'center': 0.0, 'center_bounds': '[-3, 1]'}
        )
        self.assertEqual(
            result_table.row(2),
            {'label': '12.0', 'mass': 12.0, 'intensity': 0.06013096123933792, 'intensity_bounds': '[0, None]', 'width': 13.932000160217285, 'width_bounds': '[13.932, 13.932]', 'center': 0.0, 'center_bounds': '[-3, 1]'}
        )
        self.assertEqual(
            result_table.row(3),
            {'label': '27.0', 'mass': 27.0, 'intensity': 0.0354737788438797, 'intensity_bounds': '[0, None]', 'width': 15.397000312805176, 'width_bounds': '[15.397, 15.397]', 'center': 0.0, 'center_bounds': '[-3, 1]'}
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
            {'label': '16.0', 'mass': 16.0, 'intensity': 1.0, 'intensity_bounds': '[0, None]', 'width': 12, 'width_bounds': '[0, None]', 'center': 0.0, 'center_bounds': '[-3, 1]'},
            {'label': '12.0', 'mass': 12.0, 'intensity': 1.0, 'intensity_bounds': '[0, None]', 'width': 8, 'width_bounds': '[0, None]', 'center': 0.0, 'center_bounds': '[-3, 1]'},
            {'label': '27.0', 'mass': 27.0, 'intensity': 1.0, 'intensity_bounds': '[0, None]', 'width': 13, 'width_bounds': '[0, None]', 'center': 0.0, 'center_bounds': '[-3, 1]'}
        ]

        result_table = fix_profile_parameters(means_table_mock, profiles_table_mock, h_ratio=14.7)

        # For some reason, reading floats from table introduces small variations in stored values
        # TODO: Fix floating positions eg. 8.973999977111816 -> 8.974
        self.assertEqual(
            result_table.row(0),
            {'label': '16.0', 'mass': 16.0, 'intensity': 0.17635270953178406, 'intensity_bounds': '[0, None]', 'width': 8.973999977111816, 'width_bounds': '[8.974, 8.974]', 'center': 0.0, 'center_bounds': '[-3, 1]'}
        )
        self.assertEqual(
            result_table.row(1),
            {'label': '12.0', 'mass': 12.0, 'intensity': 0.5180360674858093, 'intensity_bounds': '[0, None]', 'width': 13.932000160217285, 'width_bounds': '[0, None]', 'center': 0.0, 'center_bounds': '[-3, 1]'}
        )
        self.assertEqual(
            result_table.row(2),
            {'label': '27.0', 'mass': 27.0, 'intensity': 0.3056112229824066, 'intensity_bounds': '[0, None]', 'width': 15.397000312805176, 'width_bounds': '[15.397, 15.397]', 'center': 0.0, 'center_bounds': '[-3, 1]'}
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


    def test_pass_data_into_ws(self):

        dataX = np.arange(20).reshape(4, 5)
        dataY = np.arange(20, 40).reshape(4, 5)
        dataE = np.arange(40, 60).reshape(4, 5)

        dataX_mock = np.zeros_like(dataX)
        dataY_mock = np.zeros_like(dataY)
        dataE_mock = np.zeros_like(dataE)

        ws_mock = MagicMock(
            dataY=lambda row: dataY_mock[row],
            dataX=lambda row: dataX_mock[row],
            dataE=lambda row: dataE_mock[row],
            getNumberHistograms=MagicMock(return_value=4)
        )

        pass_data_into_ws(dataX, dataY, dataE, ws_mock)

        np.testing.assert_allclose(dataX_mock, dataX)
        np.testing.assert_allclose(dataY_mock, dataY)
        np.testing.assert_allclose(dataE_mock, dataE)


    @patch('mantid.kernel.logger.notice')
    def test_print_table_workspace(self, mock_notice):
        mock_table = MagicMock()
        mock_table.name.return_value = "my_table"
        mock_table.rowCount.return_value = 3
        mock_table.toDict.return_value = {
            "names": ["1.0", "12.0", "16.0"],
            "mass": [1, 12.0, 16.00000],
            "width": [5, 10.3456, 15.23], 
            "bounds": ["[3, 6]", "[8, 13]", "[9, 17]"]
        } 

        print_table_workspace(mock_table, precision=2)

        mock_notice.assert_has_calls(
            [call('Table my_table:'),
             call(' ------------------------ '),
             call('|names|mass|width|bounds |'),
             call('|1.0  |1   |5    |[3, 6] |'),
             call('|12.0 |12  |10.35|[8, 13]|'),
             call('|16.0 |16  |15.23|[9, 17]|'),
             call(' ------------------------ ')]
        )


if __name__ == "__main__":
    unittest.main()

import unittest
import numpy as np
import numpy.testing as nptest
from mock import MagicMock
from mvesuvio.util.analysis_helpers import extractWS, _convert_dict_to_table,  \
    fix_profile_parameters, calculate_h_ratio
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
        means_table_mock.column.side_effect = lambda x: [16, 1, 12] if x is "mass" else [0.1, 0.85, 0.05]
        h_ratio = calculate_h_ratio(means_table_mock)
        self.assertEqual(h_ratio, 0.85 / 0.05)


if __name__ == "__main__":
    unittest.main()

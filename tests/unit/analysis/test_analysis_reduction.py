import unittest
import numpy as np
import numpy.testing as nptest
from mock import MagicMock, patch, call
from mvesuvio.analysis_reduction import VesuvioAnalysisRoutine 
from mvesuvio.util.analysis_helpers import load_resolution
from mantid.simpleapi import CreateWorkspace, DeleteWorkspace
import inspect
import scipy
import re


np.set_printoptions(suppress=True, precision=6, linewidth=200)

class TestAnalysisReduction(unittest.TestCase):
    def setUp(self):
        pass

    def test_calculate_kinematics(self):
        alg = VesuvioAnalysisRoutine()
        alg._instrument_params = np.array(
            [[3, 3, 131.12, -0.2, 11.005, 0.6039],
            [4, 4, 132.77, -0.2, 11.005, 0.5789],
            [5, 5, 133.69, -0.2, 11.005, 0.5696],
            ])
        dataX = np.array([
            [110.5, 111.5, 112.5, 113.5, 114.5],
            [110.5, 111.5, 112.5, 113.5, 114.5],
            [110.5, 111.5, 112.5, 113.5, 114.5],
        ])
        alg._set_kinematic_arrays(dataX)
        np.testing.assert_allclose(alg._v0, np.array([[0.12095, 0.11964, 0.11835, 0.11709, 0.11586],
                                       [0.11988, 0.11858, 0.11732, 0.11608, 0.11487],
                                       [0.11948, 0.1182 , 0.11694, 0.11571, 0.11451],
                                       ]), atol=1e-4)
        np.testing.assert_allclose(alg._E0, np.array([[76475.65823, 74821.94729, 73221.30191, 71671.47572, 70170.33998],
                                       [75122.06378, 73511.83023, 71952.81999, 70442.88326, 68979.98182],
                                       [74627.68443, 73033.23536, 71489.34475, 69993.89741, 68544.88764],
                                       ]), atol=1e-4)
        np.testing.assert_allclose(alg._deltaE, np.array([[71569.65823, 69915.94729, 68315.30191, 66765.47572, 65264.33998],
                                       [70216.06378, 68605.83023, 67046.81999, 65536.88326, 64073.98182],
                                       [69721.68443, 68127.23536, 66583.34475, 65087.89741, 63638.88764],
                                       ]), atol=1e-4)
        np.testing.assert_allclose(alg._deltaQ, np.array([[227.01905, 224.95887, 222.94348, 220.97148, 219.04148],
                                       [226.21278, 224.18766, 222.20618, 220.26696, 218.36867],
                                       [226.07138, 224.05877, 222.08939, 220.16185, 218.27485],
                                       ]), atol=1e-4)

    def test_set_y_space_arrays(self):
        alg = VesuvioAnalysisRoutine()
        alg._masses = np.array([1, 12, 16])
        dataX = np.array([
            [110.5, 111.5, 112.5, 113.5, 114.5],
            [110.5, 111.5, 112.5, 113.5, 114.5],
            [110.5, 111.5, 112.5, 113.5, 114.5],
        ])
        alg._deltaQ = np.array([[227.01905, 224.95887, 222.94348, 220.97148, 219.04148],
           [226.21278, 224.18766, 222.20618, 220.26696, 218.36867],
           [226.07138, 224.05877, 222.08939, 220.16185, 218.27485],
        ])
        alg._deltaE = np.array([[71569.65823, 69915.94729, 68315.30191, 66765.47572, 65264.33998],
           [70216.06378, 68605.83023, 67046.81999, 65536.88326, 64073.98182],
           [69721.68443, 68127.23536, 66583.34475, 65087.89741, 63638.88764],
        ])
        alg._set_y_space_arrays(dataX)
        y_spaces_expected = np.array(
            [[[-38.0885,-38.12637,-38.16414,-38.20185,-38.23948],
            [791.54274,779.75738,768.21943,756.92089,745.85436],
            [1093.22682,1077.16966,1061.44982,1046.05644,1030.97939]],

            [[-38.84807,-38.88304,-38.91795,-38.95279,-38.98756],
            [777.99344,766.43564,755.11863,744.0348,733.17695],
            [1075.02671,1059.2788,1043.8592,1028.75757,1013.96405]],

            [[-39.25409,-39.28749,-39.32085,-39.35414,-39.38735],
            [772.34348,760.87331,749.64145,738.64055,727.86342],
            [1067.46987,1051.84087,1036.53683,1021.54771,1006.8637,]]])
        np.testing.assert_allclose(alg._y_space_arrays, y_spaces_expected, atol=1e-4)


    def test_fit_neutron_compton_profiles_number_of_calls(self):
        alg = VesuvioAnalysisRoutine()
        alg._dataY = np.array([[1, 1], [2, 2], [3, 3]])
        alg._fit_parameters = np.ones(3)   # To avoid assertion error
        alg._fit_neutron_compton_profiles_to_row = MagicMock(return_value=None)
        alg._fit_neutron_compton_profiles()
        self.assertEqual(alg._fit_neutron_compton_profiles_to_row.call_count, 3)
        

    def test_neutron_compton_profile_fit_function(self):
        alg = VesuvioAnalysisRoutine()
        alg._dataX = np.arange(113, 430, 6).reshape(1, -1)
        alg._instrument_params = np.array([
            [144, 144, 54.1686, -0.41, 11.005, 0.720184]
            ])
        alg._resolution_params = load_resolution(alg._instrument_params)
        alg._masses = np.array([1, 12, 16, 27])
        alg._set_kinematic_arrays(alg._dataX)
        alg._set_y_space_arrays(alg._dataX)
        example_fit_parameters = np.array([7.1, 5.05, 0.02, 0.22, 12.71, 1.0, 0.0, 8.76, -1.1, 0.3, 13.897, 0.64])
        alg._row_being_fit = 0
        NCP, FSE = alg._neutron_compton_profiles(example_fit_parameters)
        expected_NCP = np.array([
            [0.00004, 0.000059, 0.00009, 0.000138, 0.000216, 0.000336, 0.000648, 0.000944, 0.001349, 0.001891, 0.002596, 0.003491, 0.004595, 0.005921, 0.007464, 0.009194, 0.011051, 0.012936, 0.014711, 0.016212, 0.017269, 0.017734, 0.017518, 0.016611, 0.01509, 0.013112, 0.010879, 0.008601, 0.006465, 0.004605, 0.003092, 0.001939, 0.001119, 0.000574, 0.00024, 0.000055, -0.000033, -0.000064, -0.000064, -0.00005, -0.000033, -0.000017, -0.000004, 0.000005, 0.000012, 0.000016, 0.000018, 0.000027, 0.000025, 0.000023, 0.000022, 0.000021, 0.00002], 
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000002, 0.000002, 0.000003, 0.000004, 0.000005, 0.000006, 0.000008, 0.000011, 0.000043, 0.000149, 0.00034, 0.000926, 0.002246, 0.003149, 0.00241, 0.000926, 0.00007, 0.000067, 0.00003, 0.000019, 0.000014, 0.000011, 0.000009], 
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.], 
            [ 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000001, 0.000002, 0.000002, 0.000002, 0.000003, 0.000004, 0.000005, 0.000006, 0.000008, 0.000011, 0.000017, 0.000021, 0.000052, 0.000152, 0.000455, 0.003186, 0.006654, 0.003274, 0.000527, 0.000117, 0.000058, 0.000037, 0.000026, 0.000019, 0.000015]
        ])
        np.testing.assert_allclose(NCP, expected_NCP, atol=1e-6)


    def test_fit_neutron_compton_profiles_to_row(self):
        alg = VesuvioAnalysisRoutine()
        alg._workspace_being_fit = MagicMock()
        alg._workspace_being_fit.name.return_value = "test_workspace"
        alg._workspace_being_fit.getNumberHistograms.return_value = 3 
        alg._workspace_being_fit.extractY.return_value = np.array(
            [[-0.0001, 0.0016, 0.0018, -0.0004, 0.0008, 0.002, 0.0025, 0.0033, 0.0012, 0.0012, 0.0024, 0.0035, 0.0019, 0.0069, 0.008, 0.0097, 0.0104, 0.0124, 0.0147, 0.0165, 0.0163, 0.0195, 0.0185, 0.0149, 0.0143, 0.0145, 0.0109, 0.0085, 0.0065, 0.0043, 0.0029, 0.0023, 0.001, -0.0001, 0.0009, 0.0004, -0., 0.0001, 0.0008, 0.0001, -0.0007, 0.0011, 0.0032, 0.0057, 0.0094, 0.0036, 0.0012, -0.0023, -0.0015, -0.0006, 0.0006, 0.0011, 0.0004, 0.0009], 
             [0.0008, 0., 0., 0., 0., 0., 0.0007, 0.0033, 0.0045, 0.0029, 0.0008, 0.0026, 0.0019, 0.0004, 0.0044, 0.0057, 0.0083, 0.0115, 0.012, 0.013, 0.0168, 0.0191, 0.0167, 0.0165, 0.0165, 0.018, 0.0131, 0.0131, 0.0111, 0.0069, 0.0045, 0.0049, 0.0008, 0.0022, 0.0017, -0., 0.0003, -0.0007, 0.0001, -0., 0.0009, 0.0017, 0.0033, 0.0061, 0.0097, 0.0044, 0.0016, 0.0003, -0.0002, 0.0008, -0.0009, 0.0004, 0.0001, 0.0025],
             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]
        )
        alg._workspace_being_fit.extractX.return_value = np.array(
            [[113., 119., 125., 131., 137., 143., 149., 155., 161., 167., 173., 179., 185., 191., 197., 203., 209., 215., 221., 227., 233., 239., 245., 251., 257., 263., 269., 275., 281., 287., 293., 299., 305., 311., 317., 323., 329., 335., 341., 347., 353., 359., 365., 371., 377., 383., 389., 395., 401., 407., 413., 419., 425., 429.], 
             [113., 119., 125., 131., 137., 143., 149., 155., 161., 167., 173., 179., 185., 191., 197., 203., 209., 215., 221., 227., 233., 239., 245., 251., 257., 263., 269., 275., 281., 287., 293., 299., 305., 311., 317., 323., 329., 335., 341., 347., 353., 359., 365., 371., 377., 383., 389., 395., 401., 407., 413., 419., 425., 429.]]
        )
        alg._workspace_being_fit.extractE.return_value = np.array(
            [[0.0015, 0.0014, 0.0014, 0.0014, 0.0014, 0.0014, 0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.0014, 0.0014, 0.0014, 0.0014, 0.0014, 0.0014, 0.0014, 0.0013, 0.0013, 0.0013, 0.0013, 0.0012, 0.0012, 0.0012, 0.0012, 0.0011, 0.0011, 0.0011, 0.0011, 0.0011, 0.0011, 0.0011, 0.0011, 0.0011, 0.0011, 0.0011, 0.0011, 0.0011, 0.0011, 0.001, 0.001, 0.001, 0.001, 0.001, 0.001, 0.0009, 0.0009, 0.0016], 
             [0.0014, 0.0014, 0.0014, 0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.0012, 0.0012, 0.0012, 0.0012, 0.0012, 0.0012, 0.0012, 0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.0013, 0.0012, 0.0012, 0.0012, 0.0011, 0.0011, 0.0011, 0.0011, 0.0011, 0.0011, 0.001, 0.001, 0.001, 0.001, 0.001, 0.0011, 0.0011, 0.0011, 0.001, 0.001, 0.001, 0.001, 0.0009, 0.0009, 0.0009, 0.0009, 0.0009, 0.0016]]
        )
        alg._instrument_params = np.array([
            [144, 144, 54.1686, -0.41, 11.005, 0.720184],
            [145, 145, 52.3407, -0.53, 11.005, 0.717311]
            ])
        alg._resolution_params = load_resolution(alg._instrument_params)
        alg._masses = np.array([1, 12, 16, 27])
        alg._initial_fit_parameters = np.array([1, 4.7, 0, 1, 12.71, 0.0, 1, 8.76, 0.0, 1, 13.897, 0.0])
        alg._initial_fit_bounds = np.array([
            [0, None], [3, 6], [-3, 1],
            [0, None], [12.71, 12.71], [-3, 1],
            [0, None], [8.76, 8.76], [-3, 1],
            [0, None], [13.897, 13.897], [-3, 1],
        ])
        alg._constraints = ()

        # Set up several fit arguments
        alg._profiles_table = MagicMock()
        alg._profiles_table.column.return_value = ['1', '12', '16', '27']
        alg._profiles_table.rowCount.return_value = 4 
        alg._create_emtpy_ncp_workspace = MagicMock(return_value = None)
        alg._update_workspace_data()

        # Create arrays for storing ncp and fse total result
        ncp_total_array = np.zeros_like(alg._dataY)
        fse_total_array = np.zeros_like(alg._dataY)

        alg._fit_profiles_workspaces = {
            "total": MagicMock(dataY=lambda row: ncp_total_array[row]),
            "1": MagicMock(), "12": MagicMock(), "16": MagicMock(), "27": MagicMock()
        }
        alg._fit_fse_workspaces = {
            "total": MagicMock(dataY=lambda row: fse_total_array[row]),
            "1": MagicMock(), "12": MagicMock(), "16": MagicMock(), "27": MagicMock()
        }
        # Fit ncp
        alg._row_being_fit = 0
        alg._fit_neutron_compton_profiles_to_row()
        alg._row_being_fit = 1
        alg._fit_neutron_compton_profiles_to_row()
        alg._row_being_fit = 2
        alg._fit_neutron_compton_profiles_to_row()
        # Compare results
        expected_total_ncp_fits = np.array([
            [0.00004, 0.000059, 0.000089, 0.000138, 0.000215, 0.000335, 0.000647, 0.000943, 0.001347, 0.001888, 0.002593, 0.003486, 0.004589, 0.005913, 0.007453, 0.009182, 0.011037, 0.01292, 0.014694, 0.016196, 0.017254, 0.017722, 0.017509, 0.016606, 0.01509, 0.013115, 0.010885, 0.00861, 0.006475, 0.004615, 0.003101, 0.001949, 0.001128, 0.000583, 0.000251, 0.000068, -0.000016, -0.000041, -0.000005, 0.000119, 0.000355, 0.00105, 0.002667, 0.006238, 0.008899, 0.004131, 0.000602, -0.000026, 0.000111, 0.000078, 0.000061, 0.00005, 0.000043, 0.00004],
            [0.000021, 0.000029, 0.000042, 0.000062, 0.000095, 0.000148, 0.000324, 0.000486, 0.00072, 0.001046, 0.001492, 0.002084, 0.002851, 0.003818, 0.005006, 0.006424, 0.008066, 0.009895, 0.011838, 0.013776, 0.015551, 0.016978, 0.017872, 0.018092, 0.017566, 0.016325, 0.014492, 0.012265, 0.009878, 0.007552, 0.005464, 0.003723, 0.002369, 0.001389, 0.00073, 0.000322, 0.000095, -0.000012, -0.000016, 0.000117, 0.000383, 0.001178, 0.003156, 0.006523, 0.009388, 0.00489, 0.00077, -0.000037, 0.000125, 0.000087, 0.000067, 0.000055, 0.000047, 0.000043], 
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        ]) 
        np.testing.assert_allclose(ncp_total_array, expected_total_ncp_fits, atol=1e-6)

        expected_fit_parameters = np.array([
            [144., 7.095403, 5.051053, 0.015996, 0.218192, 12.71, 1., 0., 8.76, -1.091821, 0.29291, 13.897, 0.639245, 0.848425, 21.], 
            [145., 6.825532, 5.073835, -0.087401, 0.29456, 12.71, 1., 0., 8.76, -0.14256, 0.26776, 13.897, -2.563441, 1.246451, 20.],
            [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
        ])
        np.testing.assert_allclose(alg._fit_parameters, expected_fit_parameters, atol=1e-6)


    def test_fit_neutron_compton_profiles_to_row_with_masked_tof(self):
        alg = VesuvioAnalysisRoutine()
        alg._workspace_being_fit = MagicMock()
        alg._workspace_being_fit.name.return_value = "test_workspace"
        alg._workspace_being_fit.getNumberHistograms.return_value = 1
        dataX = np.arange(113, 430).reshape(1, -1)
        dataE = np.full_like(dataX, 0.0015)
        dataY = scipy.special.voigt_profile(dataX - 235, 30, 0) + 0.005*(np.random.random_sample(dataX.shape)-0.5)

        # Mask TOF range
        cut_off_idx = 100
        dataY[:, :cut_off_idx] = 0

        alg._workspace_being_fit.extractY.return_value = dataY
        alg._workspace_being_fit.extractX.return_value = dataX
        alg._workspace_being_fit.extractE.return_value = dataE
        alg._instrument_params = np.array([[144, 144, 54.1686, -0.41, 11.005, 0.720184]])
        alg._resolution_params = load_resolution(alg._instrument_params)
        alg._masses = np.array([1])
        alg._initial_fit_parameters = np.array([1, 5.2, 0])
        alg._initial_fit_bounds = np.array([[0, None], [3, 6], [None, None]])
        alg._constraints = ()

        # Set up several fit arguments
        alg._profiles_table = MagicMock()
        alg._profiles_table.column.return_value = ['1'] 
        alg._profiles_table.rowCount.return_value = 1 
        alg._create_emtpy_ncp_workspace = MagicMock()
        alg._update_workspace_data()

        # Create mock for storing ncp total result
        ncp_array_masked = np.zeros_like(dataY)
        alg._fit_profiles_workspaces = {"total": MagicMock(dataY=lambda arg: ncp_array_masked), "1": MagicMock()}

        # Fit ncp
        alg._row_being_fit = 0
        alg._fit_neutron_compton_profiles_to_row()
        fit_parameters_masked = alg._fit_parameters.copy()

        # Now cut range so that zeros are not part of dataY
        # (Still need to leave a padding with 6 zeros due to numerical derivative in ncp)
        alg._workspace_being_fit.extractY.return_value = dataY[:, cut_off_idx - 6:].reshape(1, -1)
        alg._workspace_being_fit.extractX.return_value = dataX[:, cut_off_idx - 6:].reshape(1, -1)
        alg._workspace_being_fit.extractE.return_value = dataE[:, cut_off_idx - 6:].reshape(1, -1)
        alg._update_workspace_data()

        ncp_array_cut = ncp_array_masked[:, cut_off_idx - 6:].reshape(1, -1)
        alg._fit_profiles_workspaces = {"total": MagicMock(dataY=lambda arg: ncp_array_cut), "1": MagicMock()}

        alg._fit_neutron_compton_profiles_to_row()
        fit_parameters_cut = alg._fit_parameters.copy()

        np.testing.assert_allclose(fit_parameters_cut, fit_parameters_masked, atol=1e-6)
        np.testing.assert_allclose(ncp_array_masked[:, cut_off_idx-6:], ncp_array_cut, atol=1e-6)


if __name__ == "__main__":
    unittest.main()

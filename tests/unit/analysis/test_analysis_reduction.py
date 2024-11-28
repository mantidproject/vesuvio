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
np.random.seed(4)


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
        alg._set_gaussian_resolution()
        alg._set_lorentzian_resolution()
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


    def test_error_function(self):
        alg = VesuvioAnalysisRoutine()
        alg._dataX = np.arange(113, 430, 6).reshape(1, -1)
        alg._instrument_params = np.array([
            [144, 144, 54.1686, -0.41, 11.005, 0.720184]
            ])
        alg._resolution_params = load_resolution(alg._instrument_params)
        alg._masses = np.array([1, 12, 16, 27])
        alg._set_kinematic_arrays(alg._dataX)
        alg._set_gaussian_resolution()
        alg._set_lorentzian_resolution()
        alg._set_y_space_arrays(alg._dataX)
        example_fit_parameters = np.array([7.1, 5.05, 0.02, 0.22, 12.71, 1.0, 0.0, 8.76, -1.1, 0.3, 13.897, 0.64])
        alg._row_being_fit = 0

        NCP, FSE = alg._neutron_compton_profiles(example_fit_parameters)
        alg._dataY = NCP.sum(axis=0) + 0.002*(np.random.random_sample(alg._dataX.shape)-0.5)

        alg._dataE = np.zeros_like(alg._dataX)
        chi2_without_errors = alg._error_function(example_fit_parameters)

        alg._dataE = np.full_like(alg._dataX, 0.0015, dtype=np.double)
        chi2_with_errors = alg._error_function(example_fit_parameters)

        alg._dataY[:, 5:15] = 0
        chi2_with_errors_with_tof_masked = alg._error_function(example_fit_parameters)

        self.assertEqual(chi2_without_errors, chi2_with_errors * 0.0015**2)
        self.assertEqual(chi2_without_errors, 1.762715850011478e-05)
        self.assertEqual(chi2_with_errors, 7.834292666717679)
        self.assertEqual(chi2_with_errors_with_tof_masked, 5.5864483595799195)


    def test_fit_neutron_compton_profiles_to_row(self):
        alg = VesuvioAnalysisRoutine()
        alg._instrument_params = np.array([
            [144, 144, 54.1686, -0.41, 11.005, 0.720184],
            [145, 145, 52.3407, -0.53, 11.005, 0.717311],
            [146, 146, 50.7811, -1.14, 11.005, 0.742482]
            ])
        alg._resolution_params = load_resolution(alg._instrument_params)
        alg._masses = np.array([1, 12, 16, 27])

        alg._dataX = np.array([
            [113., 119., 125., 131., 137., 143., 149., 155., 161., 167., 173., 179., 185., 191., 197., 203., 209., 215., 221., 227., 233., 239., 245., 251., 257., 263., 269., 275., 281., 287., 293., 299., 305., 311., 317., 323., 329., 335., 341., 347., 353., 359., 365., 371., 377., 383., 389., 395., 401., 407., 413., 419., 425., 429.], 
            [113., 119., 125., 131., 137., 143., 149., 155., 161., 167., 173., 179., 185., 191., 197., 203., 209., 215., 221., 227., 233., 239., 245., 251., 257., 263., 269., 275., 281., 287., 293., 299., 305., 311., 317., 323., 329., 335., 341., 347., 353., 359., 365., 371., 377., 383., 389., 395., 401., 407., 413., 419., 425., 429.], 
            [113., 119., 125., 131., 137., 143., 149., 155., 161., 167., 173., 179., 185., 191., 197., 203., 209., 215., 221., 227., 233., 239., 245., 251., 257., 263., 269., 275., 281., 287., 293., 299., 305., 311., 317., 323., 329., 335., 341., 347., 353., 359., 365., 371., 377., 383., 389., 395., 401., 407., 413., 419., 425., 429.], 
        ])
        alg._set_kinematic_arrays(alg._dataX)
        alg._set_gaussian_resolution()
        alg._set_lorentzian_resolution()
        alg._set_y_space_arrays(alg._dataX)

        synthetic_ncp = np.zeros_like(alg._dataX)
        example_fit_parameters = np.array([
            [7.1, 5.05, 0.02, 0.22, 12.71, 1.0, 0.0, 8.76, -1.1, 0.3, 13.897, 0.64],
            [5, 4, -1, 0.5, 12, -2, 0.8, 8., -1.1, 0.6, 14., 0.5]
        ])
        for i, pars in enumerate(example_fit_parameters):
            alg._row_being_fit = i
            NCP, FSE = alg._neutron_compton_profiles(pars)
            synthetic_ncp[i] = NCP.sum(axis=0)

        alg._dataY = synthetic_ncp.copy()
        # Add noise (except for third row that is masked)
        alg._dataY[:2, :] += 0.002*(np.random.random_sample((2, synthetic_ncp.shape[-1]))-0.5)
        # Mask tof range on second row
        alg._dataY[1, 5:15] = np.zeros(10)
        alg._dataE = np.full_like(alg._dataX, 0.0015, dtype=np.double)

        alg._initial_fit_parameters = np.array([1, 5.0, 0.1, 1, 12.0, 0.1, 1, 8.0, 0.1, 1, 13.0, 0.1])
        alg._initial_fit_bounds = np.array([
            [0, None], [3, 6], [-3, 1],
            [0, None], [9, 13], [-3, 1],
            [0, None], [6, 11], [-3, 1],
            [0, None], [11, 15], [-3, 1],
        ])
        alg._constraints = ()

        # Create arrays for storing results 
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
        # Mock methods that are not used
        alg._table_fit_results = MagicMock()
        alg._profiles_table = MagicMock()
        alg._fit_parameters = MagicMock()

        # Fit
        alg._row_being_fit = 0
        alg._fit_neutron_compton_profiles_to_row()
        alg._row_being_fit = 1
        alg._fit_neutron_compton_profiles_to_row()
        alg._row_being_fit = 2
        alg._fit_neutron_compton_profiles_to_row()

        np.testing.assert_allclose(synthetic_ncp, ncp_total_array, atol=1e-3)
        # Check masked row is correctly ignored
        np.testing.assert_allclose(synthetic_ncp[-1], np.zeros(synthetic_ncp.shape[-1]))


    def test_fit_neutron_compton_profiles_to_row_with_masked_tof(self):
        alg = VesuvioAnalysisRoutine()
        alg._masses = np.array([1])
        alg._instrument_params = np.array([[144, 144, 54.1686, -0.41, 11.005, 0.720184]])
        alg._resolution_params = load_resolution(alg._instrument_params)
        alg._initial_fit_parameters = np.array([1, 5.2, 0])
        alg._initial_fit_bounds = np.array([[0, None], [3, 6], [None, None]])
        alg._constraints = ()
        alg._table_fit_results = MagicMock()
        alg._profiles_table = MagicMock()
        alg._fit_parameters = MagicMock()

        alg._dataX = np.arange(113, 430).reshape(1, -1)
        alg._dataE = np.full_like(alg._dataX, 0.0015, dtype=np.double)
        alg._dataY = scipy.special.voigt_profile(alg._dataX - 235, 30, 0) + 0.005*(np.random.random_sample(alg._dataX.shape)-0.5)
        # Mask TOF range
        cut_off_idx = 100
        alg._dataY[:, :cut_off_idx] = 0
        alg._set_kinematic_arrays(alg._dataX)
        alg._set_gaussian_resolution()
        alg._set_lorentzian_resolution()
        alg._set_y_space_arrays(alg._dataX)

        # Create mock for storing ncp total result
        ncp_array_masked = np.zeros_like(alg._dataY)
        alg._fit_profiles_workspaces = {"total": MagicMock(dataY=lambda arg: ncp_array_masked), "1": MagicMock()}
        alg._fit_fse_workspaces = {"total": MagicMock(), "1": MagicMock()}

        # Fit ncp
        alg._row_being_fit = 0
        alg._fit_neutron_compton_profiles_to_row()
        fit_parameters_masked = alg._fit_parameters.copy()

        # Now cut range so that zeros are not part of dataY
        # (Still need to leave a padding with 6 zeros due to numerical derivative in ncp)
        alg._dataY = alg._dataY[:, cut_off_idx - 6:].reshape(1, -1)
        alg._dataX = alg._dataX[:, cut_off_idx - 6:].reshape(1, -1)
        alg._dataE = alg._dataE[:, cut_off_idx - 6:].reshape(1, -1)
        
        # Run methods that depend on dataX
        alg._set_kinematic_arrays(alg._dataX)
        alg._set_gaussian_resolution()
        alg._set_lorentzian_resolution()
        alg._set_y_space_arrays(alg._dataX)

        ncp_array_cut = ncp_array_masked[:, cut_off_idx - 6:].reshape(1, -1)
        alg._fit_profiles_workspaces = {"total": MagicMock(dataY=lambda arg: ncp_array_cut), "1": MagicMock()}

        alg._fit_neutron_compton_profiles_to_row()
        fit_parameters_cut = alg._fit_parameters.copy()

        np.testing.assert_allclose(fit_parameters_cut[:, 1:-2], fit_parameters_masked[:, 1:-2], atol=1e-6)
        np.testing.assert_allclose(ncp_array_masked[:, cut_off_idx-6:], ncp_array_cut, atol=1e-6)


    def test_set_gaussian_resolution(self):
        alg = VesuvioAnalysisRoutine()
        alg._instrument_params = np.array([
            [144, 144, 54.1686, -0.41, 11.005, 0.720184],
            [145, 145, 52.3407, -0.53, 11.005, 0.717311],
            ])
        alg._resolution_params = load_resolution(alg._instrument_params)
        alg._masses = np.array([1, 12])

        alg._dataX = np.array([
            [227., 233., 239.], 
            [227., 233., 239.], 
        ])
        alg._set_kinematic_arrays(alg._dataX)
        alg._set_gaussian_resolution()
        expected_gaussian_resolution = np.array(
            [[[1.066538, 1.046284, 1.028226], 
              [7.070845, 6.854225, 6.662274]], 
             [[1.076188, 1.055946, 1.037921], 
              [7.219413, 7.003598, 6.812526]]]
        )
        np.testing.assert_allclose(expected_gaussian_resolution, alg._gaussian_resolution, atol=1e-6)


    def test_set_lorentzian_resolution(self):
        alg = VesuvioAnalysisRoutine()
        alg._instrument_params = np.array([
            [144, 144, 54.1686, -0.41, 11.005, 0.720184],
            [145, 145, 52.3407, -0.53, 11.005, 0.717311],
            ])
        alg._resolution_params = load_resolution(alg._instrument_params)
        alg._masses = np.array([1, 12])

        alg._dataX = np.array([
            [227., 233., 239.], 
            [227., 233., 239.], 
        ])
        alg._set_kinematic_arrays(alg._dataX)
        alg._set_lorentzian_resolution()
        expected_lorentzian_resolution = np.array(
            [[[0.119899, 0.119167, 0.118724], 
              [1.346864, 1.356234, 1.366692]], 
             [[0.124002, 0.123212, 0.122712], 
              [1.376647, 1.387298, 1.399038]]]
        )
        np.testing.assert_allclose(expected_lorentzian_resolution, alg._lorentzian_resolution, atol=1e-6)


    def test_get_gaussian_resolution(self):
        alg = VesuvioAnalysisRoutine()
        alg._gaussian_resolution = np.arange(18).reshape(2, 3, 3) 
        alg._y_space_arrays = np.arange(18).reshape(2, 3, 3)
        
        alg._row_being_fit = 0
        centers = np.array([0.5, 3.6, 7.6]).reshape(-1, 1)
        np.testing.assert_allclose(alg._get_gaussian_resolution(centers), np.array([0, 4, 8]).reshape(-1, 1))
        
        alg._row_being_fit = 1
        centers = np.array([11.3, 9.6, 14.7]).reshape(-1, 1)
        np.testing.assert_allclose(alg._get_gaussian_resolution(centers), np.array([11, 12, 15]).reshape(-1, 1))


    def test_get_lorentzian_resolution(self):
        alg = VesuvioAnalysisRoutine()
        alg._lorentzian_resolution = np.arange(18).reshape(2, 3, 3) 
        alg._y_space_arrays = np.arange(18).reshape(2, 3, 3)
        
        alg._row_being_fit = 0
        centers = np.array([0.5, 3.6, 7.6]).reshape(-1, 1)
        np.testing.assert_allclose(alg._get_lorentzian_resolution(centers), np.array([0, 4, 8]).reshape(-1, 1))
        
        alg._row_being_fit = 1
        centers = np.array([11.3, 9.6, 14.7]).reshape(-1, 1)
        np.testing.assert_allclose(alg._get_lorentzian_resolution(centers), np.array([11, 12, 15]).reshape(-1, 1))

    
if __name__ == "__main__":
    unittest.main()

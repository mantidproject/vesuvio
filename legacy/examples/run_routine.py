
from mvesuvio.oop.analysis_helpers import loadRawAndEmptyWsFromUserPath, cropAndMaskWorkspace
from mvesuvio.oop.AnalysisRoutine import AnalysisRoutine
from mvesuvio.oop.NeutronComptonProfile import NeutronComptonProfile
import numpy as np


def run_analysis():
     
    ws = loadRawAndEmptyWsFromUserPath(
        userWsRawPath='/home/ljg28444/Documents/user_0/some_new_experiment/some_new_exp/input_ws/some_new_exp_raw_forward.nxs',
        userWsEmptyPath='/home/ljg28444/Documents/user_0/some_new_experiment/some_new_exp/input_ws/some_new_exp_empty_forward.nxs',
        tofBinning = "110.,1.,420",
        name='exp',
        scaleRaw=1,
        scaleEmpty=1,
        subEmptyFromRaw=False
    )

    cropedWs = cropAndMaskWorkspace(ws, firstSpec=144, lastSpec=182,
                                    maskedDetectors=[173, 174, 179],
                                    maskTOFRange='120, 160')


    AR = AnalysisRoutine(cropedWs,
                         ip_file='/home/ljg28444/.mvesuvio/ip_files/ip2018_3.par',
                         number_of_iterations=1,
                         mask_spectra=[173, 174, 179],
                         multiple_scattering_correction=True,
                         vertical_width=0.1, horizontal_width=0.1, thickness=0.001,
                         gamma_correction=True,
                         mode_running='FORWARD',
                         transmission_guess=0.853,
                         multiple_scattering_order=2,
                         number_of_events=1.0e5)
        
    H = NeutronComptonProfile('H', mass=1.0079, intensity=1, width=4.7, center=0,
                              intensity_bounds=[0, np.nan], width_bounds=[3, 6], center_bounds=[-3, 1]) 
    C = NeutronComptonProfile('C', mass=12, intensity=1, width=12.71, center=0,
                              intensity_bounds=[0, np.nan], width_bounds=[12.71, 12.71], center_bounds=[-3, 1]) 
    S = NeutronComptonProfile('S', mass=16, intensity=1, width=8.76, center=0,
                              intensity_bounds=[0, np.nan], width_bounds=[8.76, 8.76], center_bounds=[-3, 1]) 
    Co = NeutronComptonProfile('Co', mass=27, intensity=1, width=13.897, center=0,
                              intensity_bounds=[0, np.nan], width_bounds=[13.897, 13.897], center_bounds=[-3, 1]) 

    AR.add_profiles(H, C, S, Co)
    AR.run()


run_analysis()

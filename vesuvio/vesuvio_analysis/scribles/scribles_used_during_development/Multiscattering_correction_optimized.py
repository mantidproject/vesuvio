# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

def create_slab_geometry(ws_name,vertical_width, horizontal_width, thickness):
        half_height, half_width, half_thick = 0.5*vertical_width, 0.5*horizontal_width, 0.5*thickness
        xml_str = \
        " <cuboid id=\"sample-shape\"> " \
        + "<left-front-bottom-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (half_width,-half_height,half_thick) \
        + "<left-front-top-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (half_width, half_height, half_thick) \
        + "<left-back-bottom-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (half_width, -half_height, -half_thick) \
        + "<right-front-bottom-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (-half_width, -half_height, half_thick) \
        + "</cuboid>"
        CreateSampleShape(ws_name, xml_str)
        return

def calculate_sample_properties(masses,mean_widths,mean_intensity_ratios, mode):
    """returns the one of the inputs necessary for the VesuvioCalculateGammaBackground
    or VesuvioCalculateMS"""
    
    if mode == "GammaBackground":      #Not used for backscattering
        profiles = ""
        for m, mass in enumerate(masses):
            width, intensity = str(mean_widths[m]), str(mean_intensity_ratios[m])
            profiles += "name=GaussianComptonProfile,Mass=" + str(mass) + ",Width=" + width + ",Intensity=" + intensity + ';' 
        sample_properties = profiles
        
    elif mode == "MultipleScattering":
        if hydrogen_peak:   
            # ADDITION OF THE HYDROGEN INTENSITY AS PROPORTIONAL TO A FITTED NCP (OXYGEN HERE)            
            masses = np.append(masses, 1.0079)
            mean_widths = np.append(mean_widths, 5.0)           
            mean_intensity_ratios = np.append(mean_intensity_ratios, hydrogen_to_mass0_ratio * mean_intensity_ratios[0])
            mean_intensity_ratios /= np.sum(mean_intensity_ratios)
            
        MS_properties = np.zeros(3*len(masses))
        MS_properties[::3] = masses
        MS_properties[1::3] = mean_intensity_ratios
        MS_properties[2::3] = mean_widths                    
        sample_properties = list(MS_properties)    
    else:
        print("\n Mode entered not valid")
    print ("\n The sample properties for ", mode, " are: ", sample_properties)
    return sample_properties

def correct_for_multiple_scattering(ws_name, sample_properties, transmission_guess, \
                                    multiple_scattering_order, number_of_events):
    """Uses the Mantid algorithm for the MS correction to create a Workspace for the MS"""
     
    print("Evaluating the Multiple Scattering Correction.")    
    MS_masses = sample_properties[::3]        #selects only the masses, every 3 numbers
    MS_amplitudes = sample_properties[1::3]   #same as above, but starts at first intensity
 
    dens, trans = VesuvioThickness(Masses=MS_masses, Amplitudes=MS_amplitudes, TransmissionGuess=transmission_guess,Thickness=0.1)         

    _TotScattering, _MulScattering = VesuvioCalculateMS(ws_name, NoOfMasses=len(MS_masses), SampleDensity=dens.cell(9,1), 
                                                                        AtomicProperties=sample_properties, BeamRadius=2.5,
                                                                        NumScatters=multiple_scattering_order, 
                                                                        NumEventsPerRun=int(number_of_events))
    data_normalisation = Integration(ws_name) 
    simulation_normalisation = Integration("_TotScattering")
    for workspace in ("_MulScattering","_TotScattering"):
        Divide(LHSWorkspace = workspace, RHSWorkspace = simulation_normalisation, OutputWorkspace = workspace)
        Multiply(LHSWorkspace = workspace, RHSWorkspace = data_normalisation, OutputWorkspace = workspace)
        RenameWorkspace(InputWorkspace = workspace, OutputWorkspace = str(ws_name)+workspace)
    DeleteWorkspaces([data_normalisation, simulation_normalisation, trans, dens])
    return


# def correct_for_multiple_scattering(main_ws, sample_properties, transmission_guess, \
#                                     multiple_scattering_order, number_of_events):
#     """Uses the Mantid algorithm for the MS correction to create a Workspace for the MS"""
#     
#     print ("Evaluating the Multiple Scattering Correction.") 
#     MS_masses = sample_properties[::3]        #selects only the masses, every 3 numbers
#     MS_amplitudes = sample_properties[1::3]   #same as above, but starts at first intensity
#         
#     dens, trans = VesuvioThickness(Masses=MS_masses, Amplitudes=MS_amplitudes, TransmissionGuess=transmission_guess,Thickness=0.1)   
#     
#     totscat, mulscat = VesuvioCalculateMS(main_ws, NoOfMasses=len(MS_masses), SampleDensity=dens.cell(9,1),\
#                                            AtomicProperties=sample_properties, BeamRadius=2.5, \
#                                            NumScatters=multiple_scattering_order, NumEventsPerRun=int(number_of_events))
#     
#     data_normalisation = Integration(main_ws)            #changed from original 
#     simulation_normalisation = Integration(totscat)
#     for ws, ws_name in zip((mulscat, totscat), (main_ws.name()+"MulScattering", main_ws.name()+"TotScattering")):
#         ws = Divide(LHSWorkspace = ws, RHSWorkspace = simulation_normalisation)
#         ws = Multiply(LHSWorkspace = ws, RHSWorkspace = data_normalisation)
#         RenameWorkspace(InputWorkspace = ws, OutputWorkspace = ws_name)
#     DeleteWorkspaces([data_normalisation, simulation_normalisation, trans, dens, mulscat, totscat])
#     #the only remaining workspaces are the _MulScattering and _TotScattering
#     return

main_ws = Load(Filename="C:/Users/guijo/Desktop/Work/CePtGe12_backward_100K_scipy/CePtGe12_100K_DD_.nxs", OutputWorkspace="CePtGe12_100K_DD_")

transmission_guess = 0.98                               #experimental value from VesuvioTransmission
multiple_scattering_order, number_of_events = 2, 1.e5
hydrogen_peak=True                                     # hydrogen multiple scattering
hydrogen_to_mass0_ratio = 0.5                             # hydrogen-to-mass[0] ratio obtaiend from the preliminary fit of forward scattering  0.77/0.02 =38.5
vertical_width, horizontal_width, thickness = 0.1, 0.1, 0.001 # expressed in meters
create_slab_geometry(main_ws.name(),vertical_width, horizontal_width, thickness)

masses=[140.1,195.1,72.6,27]             # Array of masses to be fitted
first_spec, last_spec = 15, 25
spec_offset = main_ws.getSpectrum(0).getSpectrumNo()  
iteration = 0

D = np.load(r"C:\Users\guijo\Desktop\Work\My_edited_scripts\tests_data\optimized_6.0_no_mulscat\with_res_spec3-134_iter1_par_chi_nit_run1.npz")
#only need the mean widths and intensity ratios for the multiscattering correction
mean_widths, mean_intensity_ratios = D["mean_widths"], D["mean_intensity_ratios"]
sample_properties = calculate_sample_properties(masses, mean_widths, mean_intensity_ratios, "MultipleScattering")
print(sample_properties)

first_idx, last_idx = first_spec - spec_offset, last_spec - spec_offset
main_ws = CropWorkspace(main_ws, StartWorkspaceIndex = first_idx, EndWorkspaceIndex = last_idx, OutputWorkspace="CePtGe12_100K_DD_")

correct_for_multiple_scattering(main_ws, sample_properties, transmission_guess, multiple_scattering_order, number_of_events)

Minus(LHSWorkspace = main_ws, RHSWorkspace = main_ws.name()+"_MulScattering", OutputWorkspace = main_ws.name()+str(iteration+1))

ws = mtd[main_ws.name()+str(iteration+1)]
dataY, dataE, dataX = ws.extractY(), ws.extractE(), ws.extractX()
savepath = r"C:\Users\guijo\Desktop\Work\My_edited_scripts\tests_data\extra\mulscat_opt_Hpeak_true.npz"
np.savez(savepath, dataX=dataX, dataY=dataY, dataE=dataE, sample_properties = np.array(sample_properties))
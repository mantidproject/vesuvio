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
        
def calculate_sample_properties(masses,mean_widths,mean_intensity_ratios, mode, verbose):
    if mode == "GammaBackground":
        profiles=""
        for m in range(len(masses)):
            mass, width, intensity=str(masses[m]), str(mean_widths[m]),str(mean_intensity_ratios[m])
            profiles+= "name=GaussianComptonProfile,Mass="+mass+",Width="+width+",Intensity="+intensity+';' 
        sample_properties = profiles
    elif mode == "MultipleScattering":
        MS_properties=[]
        if (hydrogen_peak):
            # ADDITION OF THE HYDROGEN INTENSITY AS PROPORTIONAL TO A FITTED NCP (OXYGEN HERE)
            
            mean_intensity_ratios_with_H = [0] * masses.__len__()
            masses_with_H = [0] * masses.__len__()
            mean_widths_with_H = [0] * masses.__len__()
            
            for m in range(masses.__len__()):
                mean_intensity_ratios_with_H[m] = mean_intensity_ratios[m]
                masses_with_H[m] = masses[m]
                mean_widths_with_H[m] = mean_widths[m]
                
            mean_intensity_ratios_with_H.append(hydrogen_to_mass0_ratio * mean_intensity_ratios[0])
            mean_intensity_ratios_with_H = list(map(lambda x: x / np.sum(mean_intensity_ratios_with_H), mean_intensity_ratios_with_H))
            
            masses_with_H.append(1.0079)
            mean_widths_with_H.append(5.0)
            
            NM = masses_with_H.__len__()

            for m in range(len(masses_with_H)):
                MS_properties.append(masses_with_H[m])
                MS_properties.append(mean_intensity_ratios_with_H[m])
                MS_properties.append(mean_widths_with_H[m])

        else:
            NM = masses.__len__()
            for m in range(len(masses)):
                MS_properties.append(masses[m])
                MS_properties.append(mean_intensity_ratios[m])
                MS_properties.append(mean_widths[m])

            
        sample_properties = MS_properties    
        
    if verbose:
        print("\n", "The sample properties for ",mode," are: ", sample_properties)
    return sample_properties
    
def correct_for_multiple_scattering(ws_name,first_spectrum,last_spectrum, sample_properties, 
                                                        transmission_guess, multiple_scattering_order, number_of_events):
    

    MS_masses = [0] * int(len(sample_properties)/3)
    MS_amplitudes = [0] * int(len(sample_properties)/3)
    
    for m in range(int(len(sample_properties)/3)):
        MS_masses[m]=sample_properties[3*m]
        MS_amplitudes[m] = sample_properties[3*m+1]


    if verbose:
        print("Evaluating the Multiple Scattering Correction.")
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
    DeleteWorkspace(data_normalisation)
    DeleteWorkspace(simulation_normalisation)
    DeleteWorkspace(trans)
    DeleteWorkspace(dens)
    return

main_ws = Load(Filename="C:/Users/guijo/Desktop/Work/CePtGe12_backward_100K_scipy/CePtGe12_100K_DD_.nxs", OutputWorkspace="CePtGe12_100K_DD_")
name='CePtGe12_100K_DD_'  

transmission_guess = 0.98                               #experimental value from VesuvioTransmission
multiple_scattering_order, number_of_events = 2, 1.e5
hydrogen_peak=True                                   # hydrogen multiple scattering
hydrogen_to_mass0_ratio = 0.5                             # hydrogen-to-mass[0] ratio obtaiend from the preliminary fit of forward scattering  0.77/0.02 =38.5
vertical_width, horizontal_width, thickness = 0.1, 0.1, 0.001 # expressed in meters
create_slab_geometry(name,vertical_width, horizontal_width, thickness)

verbose=True
first_spectrum, last_spectrum = 15, 25
spec_offset = main_ws.getSpectrum(0).getSpectrumNo()  

masses=[140.1,195.1,72.6,27]             # Array of masses to be fitted
iteration = 0

D = np.load(r"C:\Users\guijo\Desktop\Work\My_edited_scripts\tests_data\optimized_6.0_no_mulscat\with_res_spec3-134_iter1_par_chi_nit_run1.npz")
mean_widths, mean_intensity_ratios = D["mean_widths"], D["mean_intensity_ratios"]
sample_properties = calculate_sample_properties(masses, mean_widths, mean_intensity_ratios, "MultipleScattering", verbose)
print(sample_properties)

first_idx, last_idx = first_spectrum - spec_offset, last_spectrum - spec_offset
main_ws = CropWorkspace(main_ws, StartWorkspaceIndex = first_idx, EndWorkspaceIndex = last_idx, OutputWorkspace="CePtGe12_100K_DD_")

correct_for_multiple_scattering(name, first_spectrum,last_spectrum, sample_properties, transmission_guess, multiple_scattering_order, number_of_events)

Minus(LHSWorkspace= name, RHSWorkspace = str(name)+"_MulScattering", OutputWorkspace = name+str(iteration+1))

ws = mtd[main_ws.name()+str(iteration+1)]
dataY, dataE, dataX = ws.extractY(), ws.extractE(), ws.extractX()
savepath = r"C:\Users\guijo\Desktop\Work\My_edited_scripts\tests_data\extra\mulscat_ori_Hpeak_true.npz"
np.savez(savepath, dataX=dataX, dataY=dataY, dataE=dataE, sample_properties = np.array(sample_properties))
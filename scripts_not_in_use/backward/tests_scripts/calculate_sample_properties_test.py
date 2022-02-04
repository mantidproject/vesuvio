# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

def calculate_sample_properties(masses,mean_widths,mean_intensity_ratios, mode, verbose):
    if mode == "GammaBackground":
        profiles=""
        for m in range(len(masses)):
            mass, width, intensity=str(masses[m]), str(mean_widths[m]),str(mean_intensity_ratios[m])
            profiles+= "name=GaussianComptonProfile,Mass="+mass+",Width="+width+",Intensity="+intensity+';' 
        sample_properties = profiles
    elif mode == "MultipleScattering":
        MS_properties=[]
        if (hydrogen_peak):   #hydrogen peak is not an argument of the function???
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
        print ("\n", "The sample properties for ",mode," are: ", sample_properties)
    return sample_properties
    
    
    
def calculate_sample_properties_improved(masses,mean_widths,mean_intensity_ratios, mode, verbose):
    """returns the one of the inputs necessary for the VesuvioCalculateGammaBackground
    or VesuvioCalculateMS"""
    
    if mode == "GammaBackground":
        profiles = ""
        for m in range(len(masses)):
            mass, width, intensity=str(masses[m]), str(mean_widths[m]),str(mean_intensity_ratios[m])
            profiles += "name=GaussianComptonProfile,Mass="+mass+",Width="+width+",Intensity="+intensity+';' 
        sample_properties = profiles
        
    elif mode == "MultipleScattering":
        #MS_properties=[]
        if hydrogen_peak:   #if hydrogen_peak is set to True
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
    if verbose:
        print ("\n The sample properties for ", mode, " are: ", sample_properties)
    return sample_properties
    
    
hydrogen_peak=True                # hydrogen multiple scattering
hydrogen_to_mass0_ratio = 0.5              # hydrogen-to-mass[0] ratio obtaiend from the preliminary fit of forward scattering  0.77/0.02 =38.5
mode = "GammaBackground" #"MultipleScattering" #"GammaBackground"

masses = [140.1,195.1,72.6,27]                       
mean_widths = np.array([17.420225, 20,       12.2 ,      9.8     ])
mean_intensity_ratios =  np.array([0.00624116, 0.09919216, 0.21852755, 0.67603913])
calculate_sample_properties(masses, mean_widths, mean_intensity_ratios, mode, True)
calculate_sample_properties_improved(masses, mean_widths, mean_intensity_ratios, mode, True)

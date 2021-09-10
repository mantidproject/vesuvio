# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

def calculate_sample_properties(masses, mean_widths, mean_intensity_ratios, mode):
    """returns the one of the inputs necessary for the VesuvioCalculateGammaBackground
    or VesuvioCalculateMS"""
    masses = masses.reshape(4)
    
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

mean_widths = np.array([17.4625, 20.3696, 17.5842,  9.8])
mean_intensity_ratios = np.array([0.0071, 0.1131, 0.2492, 0.6307])
masses=np.array([140.1,195.1,72.6,27]).reshape(4, 1, 1)
hydrogen_peak = True
hydrogen_to_mass0_ratio = 0.5

mean_wid = np.copy(mean_widths)
mean_int = np.copy(mean_intensity_ratios)
sample_prop = calculate_sample_properties(masses, mean_wid, mean_int, "MultipleScattering")

np.testing.assert_allclose(mean_wid, mean_widths)
np.testing.assert_allclose(mean_int, mean_intensity_ratios)
print(mean_wid)
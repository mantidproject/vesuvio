# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np


def calculate_sample_properties(masses,mean_widths,mean_intensity_ratios, mode, verbose):   
    if mode == "GammaBackground":
        profiles = ""
        for m in range(len(masses)): #name=
            mass, width, intensity=str(masses[m]), str(mean_widths[m]),str(mean_intensity_ratios[m])
            profiles += "name=GaussianComptonProfile,Mass="+mass+",Width="+width+",Intensity="+intensity+';' 
        sample_properties = profiles
        
    elif mode == "MultipleScattering":
        MS_properties=[]
        if (hydrogen_peak):   #if this is set to True
            # ADDITION OF THE HYDROGEN INTENSITY AS PROPORTIONAL TO A FITTED NCP (OXYGEN HERE)          
            mean_intensity_ratios_with_H = list(mean_intensity_ratios)
            masses_with_H = list(masses)
            mean_widths_with_H = list(mean_widths)
            
            mean_intensity_ratios_with_H.append(hydrogen_to_mass0_ratio * mean_intensity_ratios[0])
            mean_intensity_ratios_with_H = list(map(lambda x: x / np.sum(mean_intensity_ratios_with_H), mean_intensity_ratios_with_H))         
            masses_with_H.append(1.0079)
            mean_widths_with_H.append(5.0)
            
            NM = len(masses_with_H)
            for m in range(len(masses_with_H)):
                MS_properties.append(masses_with_H[m])
                MS_properties.append(mean_intensity_ratios_with_H[m])
                MS_properties.append(mean_widths_with_H[m])
        else:
            NM = len(masses)
            for m in range(len(masses)):
                MS_properties.append(masses[m])
                MS_properties.append(mean_intensity_ratios[m])
                MS_properties.append(mean_widths[m])           
        sample_properties = MS_properties    
        
    if verbose:
        print ("\n", "The sample properties for ",mode," are: ", sample_properties)
    return sample_properties
    
def correct_for_gamma_background(ws_name):
    if verbose:
        print ("Evaluating the Gamma Background Correction.")
    # Create an empty workspace for the gamma correction
    CloneWorkspace(InputWorkspace=ws_name, OutputWorkspace="gamma_background_correction")
    ws=mtd["gamma_background_correction"]
    for spec in range(ws.getNumberHistograms()):
        profiles=''
        for m in range(masses.__len__()):
            mass,width,intensity=str(masses[m]), str(mean_widths[m]),str(mean_intensity_ratios[m])
            profiles+= "name=GaussianComptonProfile,Mass="+mass+",Width="+width+",Intensity="+intensity+';'
        background, corrected = VesuvioCalculateGammaBackground(InputWorkspace=ws_name, 
                                                                        ComptonFunction=profiles, WorkspaceIndexList=spec)
        for bin in range(ws.blocksize()):
            ws.dataY(spec)[bin],ws.dataE(spec)[bin]=background.dataY(0)[bin],background.dataE(0)[bin]
    RenameWorkspace(InputWorkspace= "gamma_background_correction", OutputWorkspace = str(ws_name)+"_gamma_background")
    DeleteWorkspace("background")
    DeleteWorkspace("corrected")
    return
    
def correct_for_gamma_background_improved(ws):
    if verbose:
        print ("Evaluating the Gamma Background Correction.")
    # Create an empty workspace for the gamma correction
    CloneWorkspace(InputWorkspace=ws, OutputWorkspace="gamma_background_correction")
    ws=mtd["gamma_background_correction"]
    
    profiles = calculate_sample_properties(masses, mean_widths, mean_intensity_ratios, "GammaBackground", verbose)
    for spec in range(ws.getNumberHistograms()):
        background, corrected = VesuvioCalculateGammaBackground(InputWorkspace=ws, ComptonFunction=profiles, WorkspaceIndexList=spec)
        ws.dataY(spec)[:], ws.dataE(spec)[:] = background.dataY(0)[:], background.dataE(0)[:]
        
    RenameWorkspace(InputWorkspace= "gamma_background_correction", OutputWorkspace = ws.name()+"_gamma_background")
    DeleteWorkspace("background")
    DeleteWorkspace("corrected")
    return 
    
    
def correct_for_gamma_background_improved_improved(ws):
    if verbose:
        print ("Evaluating the Gamma Background Correction.")
    # Create an empty workspace for the gamma correction
    #ws = CloneWorkspace(InputWorkspace=ws, OutputWorkspace="gamma_background_correction")
    
    profiles = calculate_sample_properties(masses, mean_widths, mean_intensity_ratios, "GammaBackground", verbose)
    background, corrected = VesuvioCalculateGammaBackground(InputWorkspace=ws, ComptonFunction=profiles, WorkspaceIndexList=spec)
 
#     for spec in range(ws.getNumberHistograms()):
#         background, corrected = VesuvioCalculateGammaBackground(InputWorkspace=ws, ComptonFunction=profiles, WorkspaceIndexList=spec)
#         ws.dataY(spec)[:], ws.dataE(spec)[:] = background.dataY(0)[:], background.dataE(0)[:]
#         
    RenameWorkspace(InputWorkspace= background, OutputWorkspace = ws.name()+"_gamma_background")
    DeleteWorkspace(background)
    DeleteWorkspace(corrected)
    return 

Load(Filename= r"C:/Users/guijo/Desktop/Work/CePtGe12_backward_100K_scipy/CePtGe12_100K_DD_.nxs", OutputWorkspace="CePtGe12_100K_DD_")
ws = mtd["CePtGe12_100K_DD_"]

#correct_for_gamma_background(ws)
verbose=True

masses = [140.1,195.1,72.6,27]                       
mean_widths = np.array([17.420225, 20,       12.2 ,      9.8     ])
mean_intensity_ratios =  np.array([0.00624116, 0.09919216, 0.21852755, 0.67603913])

correct_for_gamma_background_improved(ws)

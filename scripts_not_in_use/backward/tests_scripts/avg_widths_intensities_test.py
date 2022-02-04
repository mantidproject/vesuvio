# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np


masses = [140.1,195.1,72.6,27]

D = np.load(r"C:\Users\guijo\Desktop\Work\My_edited_scripts\optimized_mean_widths_and_intensities.npz")
intensities = D["intensities"]
print(intensities)
widths = D["widths"]
# intensities = np.array([[0.008,  0.0163, 0.0094, 0.0136, 0.0185],
#                          [0.1268, 0.2597, 0.1502, 0.2159, 0.2944],
#                          [0.2794, 0.5722, 0.3309, 0.4756, 0.6486],
#                          [1.2278, 1.4773, 1.6154, 0.7436, 1.5778]])
#                          
# widths = np.array([[18.6809, 17.,     17. ,    17.,     20.    ],
#                      [20. ,    20. ,    20.,     20. ,    25.    ],
#                      [12.2 ,   12.2 ,   12.2,    12.2 ,   18.    ],
#                      [ 9.8,    10.,      9.8 ,    9.8  ,  10.    ]])
                     
# positions = np.array([[  3.4472  30.       0.7944  30.     -30.    ],
#                      [ 30.      30.      23.0556  30.     -30.    ],
#                      [ 10.      -1.7361 -10.       2.673   10.    ],
#                      [  9.0805   0.3812   0.2036   5.9772  -0.1515]])
# 
# 
def calculate_mean_widths_and_intensities(masses,widths,intensities): #,spectra, verbose):
    
    spectra = range(3,3+len(widths[0]))
    
    print("len of spectra:\n", len(spectra), len(intensities[0]))
    print(intensities[:, 10:20])
    intensities = intensities[:, np.all(~np.isnan(intensities), axis=0)] #removes columns of all nans
    print(intensities[:,10:20])
    widths = widths[:, np.all(~np.isnan(widths), axis=0)]
    #taking out the nans like in original script gives exactly the same thing
    
    better_widths, better_intensities =np.zeros((len(masses),len(widths[0]))),np.zeros((len(masses),len(widths[0])))
    mean_widths,widths_std,mean_intensity_ratios,mean_intensity_ratios_std=np.zeros((len(masses))),np.zeros((len(masses))),np.zeros((len(masses))),np.zeros((len(masses)))
    
    for m in range(len(masses)):
        mean_widths[m]=np.nanmean(widths[m])
        widths_std[m]=np.nanstd(widths[m])
        for index in range(len(widths[0])): # over all spectra
            if  abs( widths[m][index]-mean_widths[m] ) > widths_std[m]:
                better_widths[m][index],better_intensities[m][index]= None, None
            else:
                better_widths[m][index],better_intensities[m][index]= widths[m][index],intensities[m][index]
        mean_widths[m]=np.nanmean(better_widths[m])
        widths_std[m]=np.nanstd(better_widths[m])
    #print("\n better intensities before norm \n", better_intensities)
    
    norm_values = []
    for spec in range(len(intensities[0])):   #originally spectra
        normalisation = better_intensities[:,spec].sum()
        norm_values.append(normalisation)
        better_intensities[:,spec]/=normalisation
    print("\nnorm values original:\n", norm_values[:10])

    for m in range(len(masses)):
        mean_intensity_ratios[m] = np.nanmean(better_intensities[m])
        mean_intensity_ratios_std[m] = np.nanstd(better_intensities[m])
        
#     print("\n intensities after norm: \n", better_intensities)
#     print("\n widths: \n", better_widths)    
#     print("\n intensity ratios and std: \n", mean_intensity_ratios, mean_intensity_ratios_std)
#     print("\n mean widths and std: \n", mean_widths, widths_std)
#     
    for m in range(len(masses)):
        print ("\n", "Mass: ", masses[m], " width: ", mean_widths[m], " \pm ", widths_std[m])
        print ("\n", "Mass: ", masses[m], " mean_intensity_ratio: ", mean_intensity_ratios[m], " \pm ", mean_intensity_ratios_std[m])
    return mean_widths, mean_intensity_ratios


def improved_version(masses,widths,intensities): #,spectra, verbose):

    mean_widths = np.nanmean(widths, axis=1).reshape(4,1)     #shape (1,4)
    widths_std = np.nanstd(widths, axis=1).reshape(4,1) 
    deviation = np.abs(widths - mean_widths)         #subtraction line by line basis
    
    print("widths before where:\n", widths[:,10:20])
    better_widths = np.where(deviation > widths_std, np.nan, widths)
    print("widths after where:\n", better_widths[:,10:20])
    better_intensities = np.where(deviation > widths_std, np.nan, intensities)
    
    mean_widths = np.nanmean(better_widths, axis=1)   #shape (1,4)
    widths_std = np.nanstd(better_widths, axis=1)
    #print("\n better intensities before norm \n", better_intensities)
   
    normalization_sum = np.sum(better_intensities, axis=0)        #only change from the original
    better_intensities /= normalization_sum
    print("norm values opt:\n", normalization_sum[10:20])    
    
    mean_intensity_ratios = np.nanmean(better_intensities, axis=1)  
    mean_intensity_ratios_std = np.nanstd(better_intensities, axis=1)
    
#     print("\n intensities after norm: \n", better_intensities)
#     print("\n widths: \n", better_widths)    
#     print("\n intensity ratios and std: \n", mean_intensity_ratios, mean_intensity_ratios_std)
#     print("\n mean widths and std: \n", mean_widths, widths_std)


    for m in range(len(masses)):
        print ("\n", "Mass: ", masses[m], " width: ", mean_widths[m], " \pm ", widths_std[m])
        print ("\n", "Mass: ", masses[m], " mean_intensity_ratio: ", mean_intensity_ratios[m], " \pm ", mean_intensity_ratios_std[m])
    return mean_widths, mean_intensity_ratios


# mean_widths = np.array([np.nanmean(widths, axis=1)])     #shape (1,4)
# widths_std = np.array([np.nanstd(widths, axis=1)])
# 
# print("\n \n widths \n", widths)
# print("mean widths \n", mean_widths.transpose())
# deviation = np.abs(widths - mean_widths.transpose())
# #subtraction line by line basis
# print("\n deviation \n", deviation)
# print("\n std \n", widths_std.transpose())
# print("\n where dev > std \n", np.abs(deviation)>widths_std.transpose())
# better_widths = np.where(deviation > widths_std.transpose(), None, widths)
# print("\n better widths \n", better_widths)
# 
# print("\n Norm intensities \n")
# print(intensities)
# normalization_sum = np.sum(intensities, axis=0)
# print(normalization_sum)
# intensities = intensities / normalization_sum
# print(intensities)

print("\n \n result from original function", calculate_mean_widths_and_intensities(masses,widths,intensities))
print("\n result from improved version", improved_version(masses, widths, intensities))

# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

def load_workspace_improved(ws_name, spectrum):
    
    """returns the data arrays for a given spectrum in a given workspace"""
    
    ws=mtd[str(ws_name)]
    spec_offset = ws.getSpectrum(0).getSpectrumNo()
    
    spec_idx = spectrum - spec_offset
    
    ws_y, ws_x, ws_e = ws.readY(spec_idx), ws.readX(spec_idx), ws.readE(spec_idx)
    
    hist_widths = ws_x[1:] - ws_x[0:-1]     #length decreases by one
    data_y = ws_y[:-1] / hist_widths
    data_e = ws_e[:-1] / hist_widths
    data_x = (ws_x[1:] + ws_x[0:-1]) / 2    #compute mean point of bins
   
    return data_x, data_y, data_e    #ws_x, ws_y, ws_e

def load_workspace(ws_name, spectrum):
    
    """returns the data arrays for a given spectrum in a given workspace"""
    
    ws=mtd[str(ws_name)]
  
    ws_len, ws_spectra = ws.blocksize()-1, ws.getNumberHistograms()
    ws_x,ws_y, ws_e = [0.]*ws_len, [0.]*ws_len,[0.]*ws_len
    for spec in range(ws_spectra):
        if ws.getSpectrum(spec).getSpectrumNo() == spectrum :
            for i in range(ws_len):
                # converting the histogram into points
                ws_y[i] = ( ws.readY(spec)[i] / (ws.readX(spec)[i+1] - ws.readX(spec)[i] ) )
                ws_e[i] = ( ws.readE(spec)[i] / (ws.readX(spec)[i+1] - ws.readX(spec)[i] ) )
                ws_x[i] = ( 0.5 * (ws.readX(spec)[i+1] + ws.readX(spec)[i] ) )
    ws_x, ws_y, ws_e = np.array(ws_x), np.array(ws_y), np.array(ws_e)
    
    return ws_x, ws_y, ws_e

data_x, data_y, data_e = load_workspace_improved("CePtGe12_100K_DD_", 5)
print("\n New Function")
print("data_x:", data_x[:3],"...", data_x[-3:])
print("data_y:", data_y[:3],"...", data_y[-3:])
print("data_e:", data_e[:3],"...", data_e[-3:])


data_x, data_y, data_e = load_workspace("CePtGe12_100K_DD_", 5)
print("\n New Function")
print("data_x:", data_x[:3],"...", data_x[-3:])
print("data_y:", data_y[:3],"...", data_y[-3:])
print("data_e:", data_e[:3],"...", data_e[-3:])


############## IMPROVED FUNCTION WORKING AS IT SHOULD
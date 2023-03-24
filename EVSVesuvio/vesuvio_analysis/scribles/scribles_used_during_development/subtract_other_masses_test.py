# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np



def sub_m_ori(ws_last_iteration, intensities, widths, positions, spectra, masses):
    first_ws = CloneWorkspace(InputWorkspace=ws_last_iteration)
    for index in range(len(spectra)):
        data_x, data_y, data_e = load_workspace(first_ws , spectra[index])
        if (data_y.all()==0):
            for bin in range(len(data_x)-1):
                first_ws.dataY(index)[bin] = 0
        else:
            for m in range(len(masses)-1):
                other_par = (intensities[m+1, index],widths[m+1, index],positions[m+1, index])
                ncp = calculate_ncp(other_par, spectra[index], [masses[m+1]], data_x)
                for bin in range(len(data_x)-1):
                    first_ws.dataY(index)[bin] -= ncp[bin]*(data_x[bin+1]-data_x[bin])

    return first_ws

def sub_m_opt(ws_last_iteration, ncp_all_m):

    ncp_all_m = ncp_all_m[1:, :, :]       #select all masses other than the first one
    ncp_tot = np.sum(ncp_all_m, axis=0)   #sum the ncp for remaining masses
    
    dataY, dataX, dataE = ws.extractY(), ws.extractX(), ws.extractE()
    
    dataY[:, :-1] -= ncp_tot * (dataX[:, 1:] - dataX[:, :-1])
    
    first_ws = CreateWorkspace(DataX=dataX.flatten(), DataY=dataY.flatten(), DataE=dataE.flatten(), Nspec=len(dataX))
    return first_ws
    

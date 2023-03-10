# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

def convert_to_y_space_and_symmetrise(ws_name,mass):  #Need to go over it
    ws = mtd[ws_name]
    print("\n First spectrum before conversion: ", ws.readY(0)[:5])   
    
    ws_y, ws_q = ConvertToYSpace(InputWorkspace=ws_name,Mass=mass,OutputWorkspace=ws_name+"_JoY",QWorkspace=ws_name+"_Q")
    print("\n First spectrum after conversion: ", ws_y.readY(0)[:5])    
    
    max_Y = np.ceil(2.5*mass+27)    #where from
    rebin_parameters = str(-max_Y)+","+str(2.*max_Y/120)+","+str(max_Y)   #first bin boundary, width, last bin boundary, so 120 bins over range
    ws = Rebin(InputWorkspace=ws_name+"_JoY", Params = rebin_parameters,FullBinsOnly=True, OutputWorkspace= ws_name+"_JoY")
    print("\n First spectrum after rebin: ", ws.readE(0)[:5])    
    tmp=CloneWorkspace(InputWorkspace=ws_name+"_JoY")
    for j in range(tmp.getNumberHistograms()):
        for k in range(tmp.blocksize()):
            tmp.dataE(j)[k] =0.
            if (tmp.dataY(j)[k]!=0):
                tmp.dataY(j)[k] =1.
    tmp=SumSpectra('tmp')
    print(tmp.readY(0)[:5])
    ws = SumSpectra(InputWorkspace=ws_name+"_JoY",OutputWorkspace=ws_name+"_JoY")
    print("\n dataE after sum spectra: ", ws.readE(0)[:5])

    ws = Divide(LHSWorkspace=ws_name+"_JoY", RHSWorkspace="tmp", OutputWorkspace =ws_name+"_JoY")
    ws=mtd[ws_name+"_JoY"]
    tmp=CloneWorkspace(InputWorkspace=ws_name+"_JoY")
    print("\n dataE before symetrise: ", tmp.readE(0)[:5])
    for k in range(tmp.blocksize()):
        tmp.dataE(0)[k] =(ws.dataE(0)[k]+ws.dataE(0)[ws.blocksize()-1-k])/2.
        tmp.dataY(0)[k] =(ws.dataY(0)[k]+ws.dataY(0)[ws.blocksize()-1-k])/2
    RenameWorkspace(InputWorkspace="tmp",OutputWorkspace=ws_name+"_JoY")
    normalise_workspace(ws_name+"_JoY")
    return max_Y
    
    
def convert_to_y_space_and_symmetrise_improved_improved(ws_name,mass):  #Need to go over it
    ws = mtd[ws_name]
    print("\n First spectrum before conversion: ", ws.readY(0)[:5])       
    ws_y, ws_q = ConvertToYSpace(InputWorkspace=ws_name,Mass=mass,OutputWorkspace=ws_name+"_JoY1",QWorkspace=ws_name+"_Q1")
    print("\n First spectrum after conversion: ", ws_y.readY(0)[:5])    
    max_Y = np.ceil(2.5*mass+27)    #where from
    rebin_parameters = str(-max_Y)+","+str(2.*max_Y/120)+","+str(max_Y)   #first bin boundary, width, last bin boundary, so 120 bins over range
    ws_y = Rebin(InputWorkspace=ws_y, Params = rebin_parameters, FullBinsOnly=True, OutputWorkspace=ws_name+"_JoY1")
    print("\n First spectrum after rebin: ", ws_y.readY(0)[:5])    
   
    #pass the y-data onto an array to easily manipulate
    matrix_Y = np.zeros((ws_y.getNumberHistograms(), ws_y.blocksize()))
    for spec_idx in range(len(matrix_Y)):
        matrix_Y[spec_idx, :] = ws_y.readY(spec_idx) 
    matrix_Y[matrix_Y != 0] = 1
    sum_Y = np.nansum(matrix_Y, axis=0)
    
    ws_y = SumSpectra(InputWorkspace=ws_y, OutputWorkspace=ws_name+"_JoY1")
    tmp=CloneWorkspace(InputWorkspace=ws_y)
    tmp.dataY(0)[:] = sum_Y
    tmp.dataE(0)[:] = np.zeros(tmp.blocksize())
    
    ws = Divide(LHSWorkspace=ws_y, RHSWorkspace=tmp, OutputWorkspace =ws_name+"_JoY1")
    print("dataE before sym: ", ws.readE(0)[:5])
    ws.dataY(0)[:] = (ws.readY(0)[:] + np.flip(ws.readY(0)[:])) / 2
    ws.dataE(0)[:] = (ws.readE(0)[:] + np.flip(ws.readE(0)[:])) / 2
    normalise_workspace(ws)
    return max_Y 
    
def normalise_workspace(ws_name):
    tmp_norm = Integration(ws_name)
    Divide(LHSWorkspace=ws_name,RHSWorkspace="tmp_norm",OutputWorkspace=ws_name)
    DeleteWorkspace("tmp_norm")
  
convert_to_y_space_and_symmetrise("CePtGe12_100K_DD_", 140)
convert_to_y_space_and_symmetrise_improved_improved("CePtGe12_100K_DD_", 140)
  
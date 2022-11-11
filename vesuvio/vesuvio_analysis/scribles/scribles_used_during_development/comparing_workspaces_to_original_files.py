# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np



f = open("workspace_names.txt", 'r')
data = f.read()
f.close()
names = data.split("\n")
print(names)


for ws in mtd:
    dataY = ws.extractY()
    dataX = ws.extractX()
    dataE = ws.extractE()
    savepath = r"C:\Users\guijo\Desktop\Work\My_edited_scripts\tests_data\comparing_workspaces_original4.2\expected_workspaces\_"+ws.name()+".npz"
    np.savez(savepath, dataX=dataX, dataY=dataY, dataE=dataE)
    
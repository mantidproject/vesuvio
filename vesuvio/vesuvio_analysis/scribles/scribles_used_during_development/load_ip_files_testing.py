# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

def load_ip_file(spectrum):
    """Instrument parameters of VESUVIO"""  
    
    #print "Loading parameters from file: ", namedtuple
    ipfile = r'C:\Users\guijo\Desktop\Work\ip2018.par'
    f = open(ipfile, 'r')
    data = f.read()
    lines = data.split('\n')
    for line in lines:       
        col = line.split('\t')
        if col[0].isdigit() and int(col[0]) == spectrum:
            angle = float(col[2])
            T0 = float(col[3])
            L0 = float(col[4])
            L1 = float(col[5])
    f.close()
    return angle, T0, L0, L1    
    
 
def load_ip_file_improved(spectrum):  
    ipfile = r'C:\Users\guijo\Desktop\Work\ip2018.par'
    f = open(ipfile, 'r')
    data = f.read()
    lines = data.split("\n")[1:-1]  #take out the first line of non numbers and the last empty line
    data = list(map(lambda line: list(map(float, line.split("\t"))), lines))
    data = np.array(data)
    row = data[data[:, 0]==spectrum]
    angle, T0, L0, L1 = row[0, 2:]   #0 because its an array inside of an array
    return angle, T0, L0, L1
      

"""
ACTUALLY THE 'IMPROVED' VERSION USUALLY TAKES MORE TIME, SO DO NOT REPLACE IT 
IN THE MAIN CODE
I AM SURPRISED
"""

def str_to_float(line):
    return list(map(float, line))
    
# data = list(map(lambda line: list(map(float, line.split("\t"))), lines))
# print(data[:3])
# 
# data = np.array(data)

print("original: ", load_ip_file(153))
print("optimized: ", load_ip_file_improved(153))

# data = list(map(str_to_float, lines[:]))
# print(data[:3])
#data = list(map(lambda row: 
# data = map(lambda i: int(i), lines)
# print(list(data)[:3])
#print(data)
# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

a = np.array([1,2,7,6,8,7,6,6,6,7,7,7])
b = np.array([1,1,1,2,1,1,1,1,1,1,1,1])
print(a[np.where(b == 2)])
print(2/a)

print(a*b) #so this corresponds to element-wise multiplication

c = (1,2,3,5)
print(c[3])
print(np.array([False, False]).all()==0)

print(a/b)

d = np.zeros((3,7))
d[0] = np.ones(7)
e = np.zeros(len(d[1]))
d_sum = np.sum(d, axis=0)
print(d)
print(d_sum)
print(len(e))

# ws = mtd["CePtGe12_100K_DD_"]
# print(ws.getSpectrum(5))
# 
a = np.empty(5)
a[:] = None
print(a[4])

##testing transpose
#ws = mtd["CePtGe12_100K_DD_"]

#MaskBins(ws, 142, 143)

#print(ws.getSpectrum(5))

a = np.array([[1,2,7],[6,8,7],[6,6,6], [7,7,7]])
b = np.array([[1,2,3,4]]).transpose()
print(b.shape)
c = a - b
print(c)

d=8
print(d)
d /= 2*2
print(d)
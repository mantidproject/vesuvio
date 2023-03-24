# import mantid algorithms, numpy and matplotlib
from mantid.simpleapi import *
import matplotlib.pyplot as plt
import numpy as np

sample_properties = [1, 2, 3, 1, 2, 3, 1, 2, 3]


# MS_masses = [0] * int(len(sample_properties)/3)
# MS_amplitudes = [0] * int((len(sample_properties)/3)
# 
MS_masses = sample_properties[::3]
MS_amplitudes = sample_properties[1::3]

# for m in range(len(sample_properties)/3):
#     MS_masses[m]=sample_properties[3*m]
#     MS_amplitudes[m] = sample_properties[3*m+1]
# 
print(MS_masses)
print(MS_amplitudes)

a = np.zeros(9)
a[::3] = 1
a = np.append(a, 2)
print(a)

a = np.array([1, 1, 1])
b = np.array([2, 2, 2])
c = np.array([3, 3, 3])

d = np.zeros(3*len(a))
d[::3] = a
d[1::3] = b
d[2::3] = c

print(d)

e = np.arange(10)
print(e)
e = np.flip(e)
print(e)

CreateWorkspace(DataX=np.arange(10), DataY=np.arange(10), Nspec=2, OutputWorkspace="test")
B = np.arange(20).reshape((5, 4))
print(B)
print(B[B<5])
print(B[2,1])
B[2, 1] =6
print(B)
C = np.full((5,1), np.nan)
print(C)
B = B>np.full((5,1), np.nan)
print(B[B<5])
print(3-np.nan)
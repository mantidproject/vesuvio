
# from mantid.simpleapi import *
from pathlib import Path
import numpy as np
currentPath = Path(__file__).absolute().parent 


# # Create a workspace to use
# ws = CreateSampleWorkspace()

# # Get the DetectorInfo object
# info = ws.detectorInfo()

# # Call setMasked
# info.setMasked(3, True)
# print(info.isMasked(0))


# # Append arrays

# A = np.empty((1, 3))

# B = np.array([1, 2, 3])[np.newaxis, :]

# A = np.append(A, B, axis=0)
# print(A)

# someList = []
# someList.append(np.arange(12).reshape(3, 4))
# print(np.array(someList).shape)

# try:
#     while True:
#         raise KeyError("Error!")
# except KeyError:
#     print("KeyError passed!")


# names = ["one", "two", "three"]
# numbers = np.arange(12).reshape((3, 4))
# for name, no in zip(names, numbers):
#     nolist = list(no)
#     print([name] + nolist)

A = np.arange(30)
firstHalf, secondHalf = np.split(A, 2)
print(secondHalf, np.flip(firstHalf))


L = [[1, 2], [3, 4]]
print(2 in L)

    



from mantid.simpleapi import *
from pathlib import Path
import numpy as np
currentPath = Path(__file__).absolute().parent 

# Create a workspace to use
ws = CreateSampleWorkspace()

# Get the DetectorInfo object
info = ws.detectorInfo()

# Call setMasked
info.setMasked(3, True)
print(info.isMasked(0))


# Append arrays

A = np.empty((1, 3))

B = np.array([1, 2, 3])[np.newaxis, :]

A = np.append(A, B, axis=0)
print(A)
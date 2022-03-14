import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit, cost, util
from iminuit.util import make_with_signature, describe, make_func_code
from pathlib import Path
repoPath = Path(__file__).absolute().parent
from mantid.simpleapi import Load, CropWorkspace
from scipy import optimize
from scipy import ndimage, signal
import time
import pandas as pd

ipPath = repoPath / "ip2018_3.par"

# Load into numpy array first because it deals better with formatting issues
ipData = np.loadtxt(ipPath, dtype=str)[1:].astype(float)
# Create dataframe
ipdf = pd.DataFrame(ipData, columns=["Det", "Plik", "theta", "t0", "L0", "L1"])
print(ipdf.head(5))


# Group detectors by L0 and L1


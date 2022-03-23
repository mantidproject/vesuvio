import numpy as np
from pathlib import Path
repoPath = Path(__file__).absolute().parent

oriPath = repoPath / "experiments" / "starch_80_RD" / "output_npz_for_testing" / "current_forward.npz"
copyPath = repoPath / "experiments" / "starch_80_RD copy" / "output_npz_for_testing" / "current_forward.npz"


oriData = np.load(oriPath)
copyData = np.load(copyPath)

for oriKey in oriData:
    np.testing.assert_array_equal(oriData[oriKey], copyData[oriKey])

print("\nPassed the test! Loading the workspaces from Vesuvio maintains exactly the same characteristics\n")
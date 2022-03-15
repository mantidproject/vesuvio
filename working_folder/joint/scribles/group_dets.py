import numpy as np
from pathlib import Path
repoPath = Path(__file__).absolute().parent
import matplotlib.pyplot as plt

ipPath = repoPath / "ip2018_3.par"

# Load into numpy array first because it deals better with formatting issues
ipData = np.loadtxt(ipPath, dtype=str)[1:].astype(float)
ipData = ipData[ipData[:,0]>=135]
print("front dets: ", len(ipData))

idx = np.argsort(ipData[:, -1])

dets = ipData[idx, 0]
L1sorted = ipData[idx, -1]
print(L1sorted[:5])

reltol = np.abs((L1sorted[1:] - L1sorted[:-1])/L1sorted[:-1])
anoIdx = np.argwhere(reltol>0.1)[0, 0]
print(dets[anoIdx-3:anoIdx+3])
print(L1sorted[anoIdx-3:anoIdx+3])

firstHalf = L1sorted[:anoIdx+1]
secondHalf = L1sorted[anoIdx+1:]

print("len of first half: ", len(firstHalf), "last item: ", firstHalf[-1])
print("len of first half: ", len(secondHalf), "first item: ", secondHalf[-1])

# plt.plot(L1sorted[:-1], reltol, "ko")
plt.plot(L1sorted, np.ones(L1sorted.size), "ko")
plt.xlabel("L1")
plt.show()
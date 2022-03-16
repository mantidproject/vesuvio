from argparse import ArgumentDefaultsHelpFormatter
import numpy as np
from pathlib import Path
repoPath = Path(__file__).absolute().parent
import matplotlib.pyplot as plt
import numba as nb
from numba import float64, int64, int64

ipPath = repoPath / "ip2018_3.par"

# Load into numpy array first because it deals better with formatting issues
ipData = np.loadtxt(ipPath, dtype=str)[1:].astype(float)
ipData = ipData[ipData[:,0]>=135]
print("front dets: ", len(ipData))

idx = np.argsort(ipData[:, -1])

L1 = ipData[:, -1] 
theta = ipData[:, 2] 

@nb.vectorize([float64(int64, int64)])
def detDistance(i, j):
    dTheta = (theta[i] - theta[j]) #   / theta[i]
    dL1 = (L1[i] - L1[j])   #    / L1[i]
    return np.sqrt(dTheta**2 + dL1**2)


def findMinimumPairs():
    # L1 = ipData[:, -1]
    # theta = ipData[:, 2]

    idx = np.arange(len(ipData))
    iIdx = idx[:, np.newaxis] * np.ones((1, len(ipData)))
    iIdx = iIdx.astype(int)
    print(iIdx.shape)
    jIdx = iIdx.T

    M = detDistance(iIdx, jIdx)
    M[M==0] = np.inf   # Mask zeros
    print(M[:5, :5])

    idxSortMin = np.argsort(np.min(M, axis=1))
    Msorted = M[idxSortMin, :]
    argMinsSorted = np.argmin(Msorted, axis=1)

    pairs, indivs = filterPairs(idxSortMin, argMinsSorted)

    return pairs, indivs


def filterPairs(idxs, args):   # Not robust, need to find something better
    pairs = []
    indivs = []
    used = []
    for i, j in zip(idxs, args):
        if j in used:
            indivs.append(i)
            continue
        if [j, i] not in pairs:
            pairs.append([i, j])
        used.append(j)
    return pairs, indivs

pairs, indivs = findMinimumPairs()
 

print(indivs)
print(pairs)

pairs = np.array(pairs)
indivs = np.array(indivs)

assert pairs.size == np.unique(pairs).size, "Pairs are being repeated!"
np.testing.assert_array_equal(np.sort(np.append(pairs.flatten(), indivs)), np.arange(len(ipData)))

for i, pair in enumerate(pairs):
    x = L1[pair]
    y = theta[pair]
    plt.scatter(x, y, label="pair"+str(i))

# for i, ind in enumerate(indivs):
#     x = L1[ind]
#     y = theta[ind]
#     plt.scatter(x, y, color="k",label="indiv"+str(i))
# plt.legend()
plt.xlabel("L1")
plt.ylabel("theta")


# def formPairs(ipData):
#     pairedDets = []
#     detPairs = []
#     mindistances = []
#     for i in range(len(ipData)):
        
#         if i in pairedDets:
#             continue

#         minDist = np.inf
#         minDet = 0
#         for j in range(i+1, len(ipData)):

#             if j in pairedDets:
#                 continue

#             d = detDistance(i, j, ipData)
#             if d < minDist:
#                 minDist = d
#                 minDet = j
        
#         mindistances.append(minDist)
#         if minDist > 1:
#             pairedDets.append(i)
#             detPairs.append([i])
#         else:
#             pairedDets.append(i)
#             pairedDets.append(j)
#             detPairs.append([i, minDet])
#     print("Mean Dist: ", np.mean(np.array(mindistances[:-1])))
#     return detPairs


# detPairs = formPairs(ipData)
# for pair in detPairs:
#     x = ipData[pair, -1]
#     y = ipData[pair, 2]
#     plt.scatter(x, y)

# ----------old code--------

# dets = ipData[idx, 0]
# L1sorted = ipData[idx, -1]
# thetaDets = ipData[idx, 2]
# print(L1sorted[:5])

# reltol = np.abs((L1sorted[1:] - L1sorted[:-1])/L1sorted[:-1])
# anoIdx = np.argwhere(reltol>0.1)[0, 0]
# print(dets[anoIdx-3:anoIdx+3])
# print(L1sorted[anoIdx-3:anoIdx+3])

# firstHalf = L1sorted[:anoIdx+1]
# secondHalf = L1sorted[anoIdx+1:]

# print("len of first half: ", len(firstHalf), "last item: ", firstHalf[-1])
# print("len of first half: ", len(secondHalf), "first item: ", secondHalf[-1])

# plt.plot(L1sorted[:-1], reltol, "ko")
# plt.plot(L1, theta, "ko")
# plt.xlabel("L1")
# plt.ylabel("Theta")
plt.show()
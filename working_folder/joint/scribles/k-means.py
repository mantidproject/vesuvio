import numpy as np
from pathlib import Path
repoPath = Path(__file__).absolute().parent
import matplotlib.pyplot as plt
import time 

ipPath = repoPath / "ip2018_3.par"

# Load into numpy array first because it deals better with formatting issues
ipData = np.loadtxt(ipPath, dtype=str)[1:].astype(float)
ipData = ipData[ipData[:,0]>=135]
print("front dets: ", len(ipData))


def groupDetectors(ipData, nGroups):
    """
    Uses the method of k-means to find clusters in theta-L1 space.
    Input: instrument parameters to extract L1 and theta of detectors.
    Output: list of group lists containing the idx of spectra.
    """
    assert nGroups > 0, "Number of groups must be bigger than zero."
    assert nGroups <= len(ipData)/2, "Number of groups should be less than half the spectra"
        

    L1 = ipData[:, -1]   
    theta = ipData[:, 2]  

    L1 /= np.sum(L1) * 2    # Bigger weight to L1
    theta /= np.sum(theta)


    points = np.vstack((L1, theta)).T
    assert points.shape == (len(L1), 2), "Wrong shape."

    centers = points[np.linspace(0, len(points)-1, nGroups).astype(int), :]

    plt.scatter(L1, theta, alpha=0.3, color="r", label="Detectors")
    plt.scatter(centers[:, 0], centers[:, 1], color="k", label="Starting centroids")
    plt.xlabel("L1")
    plt.ylabel("theta")
    plt.legend()
    plt.show()

    t0 = time.time()
    clusters, n = kMeansClustering(points, centers)
    t1 = time.time()
    print(f"Running time: {t1-t0} seconds")


    idxList = formIdxList(clusters, n, len(L1))
    # print(clusters)
    for i in range(n):
        clus = points[clusters==i]
        plt.scatter(clus[:, 0], clus[:, 1], label=f"group {i}")
    plt.xlabel("L1")
    plt.ylabel("theta")
    plt.legend()
    plt.show()

    return idxList


def pairDistance(p1, p2):
    "pairs have shape (1, 2)"
    return np.sqrt(np.sum(np.square(p1-p2)))


def closestCenter(points, centers):
    clusters = np.zeros(len(points))
    for p in range(len(points)):

        minCenter = 0
        minDist = pairDistance(points[p], centers[0])
        for i in range(1, len(centers)): 

            dist = pairDistance(points[p], centers[i])

            if dist < minDist:
                minDist = dist
                minCenter = i
        clusters[p] = minCenter
    return clusters, len(centers)


def calculateCenters(points, clusters, n):
    centers = np.zeros((n, 2))
    for i in range(n):
        centers[i] = np.mean(points[clusters==i, :], axis=0)
    return centers


def kMeansClustering(points, centers):
    prevCenters = centers
    while  True:
        clusters, n = closestCenter(points, prevCenters)
        centers = calculateCenters(points, clusters, n)
        # print(centers)
        
        if np.all(centers == prevCenters):
            break

        assert np.isfinite(centers).all(), f"Issue with starting centers! {centers}"

        prevCenters = centers
    clusters, n = closestCenter(points, centers)
    return clusters, n


def formIdxList(clusters, n, lenPoints):
    # Form list with groups of idxs
    idxList = []
    for i in range(n):
        idxs = np.argwhere(clusters==i).flatten()
        idxList.append(list(idxs))
    print("List of idexes that will be used for idexing: \n", idxList)

    # Check that idexes correspond to the same indexes as before
    flatList = []
    for group in idxList:
        for elem in group:
            flatList.append(elem)
    assert np.all(np.sort(np.array(flatList))==np.arange(lenPoints)), "Groupings did not work!"
    return idxList



groupDetectors(ipData, 16)
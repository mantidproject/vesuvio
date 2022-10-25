#%%
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


#%%
# Experimental sample from unknow dist
# Page 15 of Errors and Their Uncertainties
np.random.seed(1)
width = 5
data = np.random.normal(0, width, 2500)
numWidth = np.std(data, ddof=1)

fig, axs = plt.subplots(4, 2)
for i, nSize in enumerate([5, 10, 25, 50]):
    nGroups = len(data) / nSize
    groups = np.split(data, nGroups)
    means = np.mean(groups, axis=1)

    errOnMeans = np.std(groups, ddof=1, axis=1)/nSize**0.5  # Error on mean for each group
    errOnMeansParent = numWidth / nSize**0.5

    leg = f"N={nSize}\n"
    leg += r"$\overline{\alpha}$="+f"{np.mean(errOnMeans):.3f}"
    leg += r" $\pm$ "+f"{np.std(errOnMeans):.3f}"
    leg += "\n"+r"$\sigma / \sqrt{N}$="+f"{errOnMeansParent:.3f}"

    axs[i, 0].scatter(means, np.arange(len(means)), marker=".")
    axs[i, 1].hist(means, label=leg)

    axs[i, 0].set_xlim(-10, 10)
    axs[i, 1].set_xlim(-10, 10)

    axs[i, 1].legend()    

# plt.legend()
fig.suptitle("Distribution of means for different Group sizes")
plt.show()


#%%
# Understanding Bootstrap uncertainty
# np.random.seed(1)
width = 5  # Unknown dist with width 5 
N = 100
sample = np.random.normal(0, width, N)  # Sample from unknown gauss dist

sampleMean = np.mean(sample)
print(f"\nThe sample mean is: {sampleMean:.3f}")

# If we didnt know the distribution, we cannot use analytical formula to calculate error on the mean
# But we could use Bootstrap to calculate uncertainty

nSamples = 1000
bootMeans = np.zeros(nSamples)
for i in range(nSamples):
    idxs = np.random.randint(0, len(sample), size=len(sample))
    bootSamp = sample[idxs]
    bootMeans[i] = np.mean(bootSamp)

print(f"\nBootstrap mean: {np.mean(bootMeans):.3f}")
print(f"\nBootstrap uncertainty: {np.std(bootMeans, ddof=1):.3f}")
print(f"\nAnalytical STDOM sample: {np.std(sample, ddof=1)/N**0.5:.3f}")
print(f"\nAnalytical STDOM unknown: {width/N**0.5:.3f}")

plt.hist(bootMeans, histtype="step", label="Bootstrap")


# Draw from the actual dist
realMeans = np.zeros(nSamples)
for i in range(nSamples):
    realSample = np.random.normal(0, width, N)
    realMeans[i] = np.mean(realSample)

print(f"\nReal measurements uncertainty: {np.std(realMeans, ddof=1):.3f}")
plt.hist(realMeans, histtype="step", label="Real draws")
plt.legend()
plt.show()

# %%

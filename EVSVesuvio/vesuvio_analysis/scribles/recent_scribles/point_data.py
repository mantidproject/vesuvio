import numpy as np

A = np.arange(20)

hist = (A[1:] + A[:-1]) / 2

histWidhts = A[1:] - A[:-1]
assert min(histWidhts) == max(histWidhts), "Histogram widhts need to be the same length"

betterHist = A + histWidhts[0] / 2 

np.testing.assert_array_equal(hist, betterHist[:-1])
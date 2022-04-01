import numpy as np
np.random.seed(1)

A = np.arange(25).reshape((5,5))
idxs = np.random.randint(0, len(A[0]), A.shape)

print(A)
B = np.zeros(A.shape)
for i in range(len(A)):
    rowIdxs = np.random.randint(0, len(A[0]), len(A[0]))
    print(rowIdxs)
    B[i] = A[i, rowIdxs]
print(B)
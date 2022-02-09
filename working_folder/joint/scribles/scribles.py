import numpy as np

A = np.array([1])

print(A.reshape((A.size, 1, 1)))

B = np.arange(10)[np.newaxis, :]
print(np.sum(B, axis=0))
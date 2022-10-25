import numpy as np



A = np.arange(25).reshape((5, 5))
A[:, 3] = 0

nanCol = np.all(A==0, axis=0)
B = A[:, ~nanCol]
print(A)
print(B)
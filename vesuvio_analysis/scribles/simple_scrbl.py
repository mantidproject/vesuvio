import numpy as np

A = np.inf

print (10 < A)

B = np.arange(12).reshape((3, 4))
print(B.shape)
print(B[:, :B.shape[1]])
for i in range(0, B.shape[1]):
    print(B[:, i][:, np.newaxis])
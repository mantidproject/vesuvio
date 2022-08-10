import numpy as np

A = np.inf

print (10 < A)

B = np.arange(12).reshape((3, 4))
print(B.shape)
print(B[:, :B.shape[1]])
for i in range(0, B.shape[1]):
    print(B[:, i][:, np.newaxis])


a = np.arange(10)[:, np.newaxis]
a = np.mean(a, axis=0)
print(a.shape)
a[a==np.nan] = 0
print(a)
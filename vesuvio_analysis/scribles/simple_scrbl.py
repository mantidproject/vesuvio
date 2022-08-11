from logging import exception
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


mask = np.array([True, True, False, False, True])
B = np.arange(25).reshape((5, 5))
B = B[:, ~mask]
B -= 5

print(B)

for i, row in enumerate(B):
    B[i] = row[np.random.randint(0, len(row), len(row))]

# B = np.where(mask[np.newaxis, :], 0, B)
C = np.zeros((len(B), len(mask)))
C[:, ~mask] = B

print("\n\n", C)


class MyErr (Exception):
    pass

try: raise MyErr
except MyErr: print("\n\nexcepted!")
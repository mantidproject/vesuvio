import numpy as np
np.random.seed(1)

for i in range(1, 1):
    print("Loop activates.")


A = np.arange(25).reshape((5,5))
idxs = np.random.randint(0, len(A[0]), A.shape)

print(A)
B = np.zeros(A.shape)
for i in range(len(A)):
    rowIdxs = np.random.randint(0, len(A[0]), len(A[0]))
    print(rowIdxs)
    B[i] = A[i, rowIdxs]
print(B)


C = [[1, 2, 3], [4, 5, 6], [7, 8, 9, 10, 11]]

maxL = max([len(c) for c in C])
for l in C:
    while len(l)<maxL:
        l.append(0)

print(C)

D = np.arange(10)
print(D)
E = np.array([[2, 4, 6, 7]])
print(D[E])
print(np.delete(D, E, None))
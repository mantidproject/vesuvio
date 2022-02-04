import numpy as np

# A = np.arange(9)
# B = np.array([1, 4.7, 0])

# C = np.append([1, 4.7, 0], A)
# print(C)
# C[4::3] = np.arange(3)
# print(C)

# masses = np.arange(3).reshape((3, 1, 1))
# m = np.append([1.4], masses)
# print(m)

# bounds = np.array([
#     [0, np.nan], [12.71, 12.71], [-3, 1],
#     [0, np.nan], [8.76, 8.76], [-3, 1],
#     [0, np.nan], [13.897, 13.897], [-3, 1]
# ])

# bounds = np.append([[0, np.nan], [3, 6], [-3, 1]], bounds, axis=0)
# widths = np.arange(3)[:, np.newaxis] * np.ones((1, 2))
# bounds[4::3] = widths
# print(bounds)

# print(bounds)


# class Fruit:
#     def __init__(self) -> None:
#         pass

#     size = 10
#     volume = size * 3
#     height = 5

#     def changeSize(self):
#         self.size = 3
    
#     def multiply(self):
#         return self.size * self.height

#     def getMultiply(self):
#         return self.multiply()

#     def getVolume(self):
#         return self.size * 3

# apple = Fruit()
# print(apple.volume)
# print(apple.size)
# print("Change Size")
# apple.changeSize()
# print(apple.volume)
# print(apple.size)
# print("volume with function: ", apple.getVolume())

# print("getMultiply: ", apple.getMultiply)
# print("multiply: ", apple.multiply())

# A = 1
# B = np.arange(12).reshape((1, 4, 3))
# C = np.arange(12).reshape((1, 4, 3)) * 2
# D = np.append(B, C, axis=0)
# E= np.append([[[]]], B, axis=0)
# print(D)
# print(E)

from pathlib import Path

path = str(Path("./ip2018_3.par").absolute())
print(path)

from turtle import color


class Car:
    seats = 5

class Window(Car):
    state = "clear"

K = Car
W = Window

K.color = True

if K.color:
    print("color: ", K.color)
    print("window color: ", W.color)


import numpy as np

A = np.array([0, 1, 3, np.nan, 5])
print(np.nanmax(A))
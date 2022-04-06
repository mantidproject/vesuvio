
from turtle import color


class Car:
    seats = 5

K = Car
K.color = True

if K.color:
    print("color: ", K.color)
import multiprocessing as mp
import numpy as np

class MyVar:
    def __init__(self) -> None:
        pass
    some_variable = 3

var = MyVar()

def mySquare(x, y):
    return np.sum(x) * np.ones(len(x)) + y*var.some_variable

def main():
    print("No of processes available: ", mp.cpu_count())

    np.random.seed(1)
    arr_x = np.random.randint(0, 100, (100, 200))
    arr_y = np.arange(len(arr_x))[:, np.newaxis] * np.ones(arr_x.shape)
    arr_tot = np.stack((arr_x, arr_y), axis=1)

    if arr_tot.shape != (len(arr_x), 2, len(arr_x[0])):
        raise ValueError("Input array has wrong shape")
    pool = mp.Pool(processes=mp.cpu_count())
    result = np.array(
        pool.starmap(mySquare, arr_tot)
    )
    pool.close()
    print(result)

if __name__ == '__main__':
    main()

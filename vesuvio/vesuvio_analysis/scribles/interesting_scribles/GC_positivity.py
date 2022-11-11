import numpy as np
import matplotlib.pyplot as plt
import numba as nb
import time
from pathlib import Path
scriblePath = Path(__file__).absolute().parent

nbargs = {
    "parallel" : True,
    "fastmath" : True
}

@nb.njit(**nbargs)
def gramCharlier(x, c4, c6):
    return 1 + c4/32*(16*x**2-48*x+12)+c6/384*(64*x**3-480*x**2+720*x-120)


@nb.njit(**nbargs)
def numericalProcedure(x, c4, c6):
    GC = np.zeros((c4.size, c6.size, x.size))
    for i in range(c4.size):
        for j in range(c6.size):
            GC[i, j, :] = gramCharlier(x, c4[i], c6[j])
    return GC


def compile():
    x = np.linspace(0, 10, 10)
    c4 = np.linspace(-1, 1, 10)
    c6 = np.linspace(-1, 1, 10)
        
    for mode in ("Compiling", "Running"):
        t0 = time.time()
        numericalProcedure(x, c4, c6)
        t1 = time.time()
        print(mode+f" time: {t1-t0:.2f} seconds.")


def runNumProc(x, c4, c6):
    t0 = time.time()
    GCGrid = numericalProcedure(x, c4, c6)
    t1 = time.time()
    print(f"Main time: {t1-t0:.2f} seconds.")
    return GCGrid



x = np.linspace(0, 10, 1000)
c4 = np.linspace(-1, 1, 1000)
c6 = np.linspace(-1, 1, 1000)

filePath = scriblePath / "GCGrid.npz"

compile()
GCGrid = runNumProc(x, c4, c6)
GCPos = np.all(GCGrid>=0, axis=2)

np.savez(filePath, GCGrid=GCPos)
plt.imshow(GCPos)
plt.show()
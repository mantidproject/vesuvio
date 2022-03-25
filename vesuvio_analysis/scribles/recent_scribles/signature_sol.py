import numpy as np
import matplotlib.pyplot as plt
from iminuit import Minuit, cost, util
from iminuit.util import make_with_signature, describe, make_func_code
from scipy import signal, stats
from mantid.simpleapi import Load
from pathlib import Path
repoPath = Path(__file__).absolute().parent

def main():

    var = 5
    def func(x, a, b):
        return var * a * x + b

    x = np.linspace(0, 10, 20)
    yerr = np.random.random(x.size) * 5
    y = func(x, 2, 1) + np.random.random(x.size) * yerr


    func.func_code = make_func_code(["x", "A", "B"])
    print(describe(func))

    costfun = cost.LeastSquares(x, y, yerr, func)
    m = Minuit(costfun, A=0, B=0)
    m.simplex()
    m.migrad()
   
    leg = []
    for p, v in zip(m.parameters, m.values):
        leg.append(f"{p} = {v}\n")

    plt.errorbar(x, y, yerr, fmt=".", label="data")
    plt.plot(x, func(x, *m.values), label="".join(leg))
    plt.legend()
    plt.show()
    # sigFunc = make_with_signature(func, a="A", b="B")
    # print(describe(sigFunc))
    return

main()
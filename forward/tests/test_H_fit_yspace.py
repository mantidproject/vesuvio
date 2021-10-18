import unittest
import numpy as np
import numpy.testing as nptest
import matplotlib.pyplot as plt
from mantid.simpleapi import *    
from pathlib import Path
currentPath = Path(__file__).absolute().parent  # Path to the repository

resolution = Load(currentPath / "fixatures" / "resolution.nxs")
H_JoY_Sym = Load(currentPath / "fixatures" / "H_JoY_Sym.nxs")
simple_gaussian_fit = True

def fitProfileInYSpace(wsYSpaceSym, wsRes):
    for minimizer_sum in ('Levenberg-Marquardt','Simplex'):
        CloneWorkspace(InputWorkspace = ic.name+'joy_sum', OutputWorkspace = ic.name+minimizer_sum+'_joy_sum_fitted')
        
        if (simple_gaussian_fit):
            function='''composite=Convolution,FixResolution=true,NumDeriv=true;
            name=Resolution,Workspace=resolution_sum,WorkspaceIndex=0;
            name=UserFunction,Formula=y0+A*exp( -(x-x0)^2/2/sigma^2)/(2*3.1415*sigma^2)^0.5,
            y0=0,A=1,x0=0,sigma=5,   ties=()'''
        else:
            function='''composite=Convolution,FixResolution=true,NumDeriv=true;
            name=Resolution,Workspace=resolution_sum,WorkspaceIndex=0;
            name=UserFunction,Formula=y0+A*exp( -(x-x0)^2/2/sigma1^2)/(2*3.1415*sigma1^2)^0.5
            +A*0.054*exp( -(x-x0)^2/2/sigma2^2)/(2*3.1415*sigma2^2)^0.5,
            y0=0,x0=0,A=0.7143,sigma1=4.76, sigma2=5,   ties=(sigma1=4.76)'''

        Fit(Function=function, InputWorkspace=ic.name+minimizer_sum+'_joy_sum_fitted', Output=ic.name+minimizer_sum+'_joy_sum_fitted',Minimizer=minimizer_sum)
        
        ws=mtd[ic.name+minimizer_sum+'_joy_sum_fitted_Parameters']
        print('Using the minimizer: ',minimizer_sum)
        print('Hydrogen standard deviation: ',ws.cell(3,1),' +/- ',ws.cell(3,2))


def gaussian(x, y0, x0, A, sigma):
    return y0 + A / (2*np.pi)**0.5 / sigma * np.exp(-(x-x0)**2/2/sigma**2)


class TestSubMasses(unittest.TestCase):
    def setUp(self):
        dataFile = np.load(loadPath)


if __name__ == "__main__":
    unittest.main()
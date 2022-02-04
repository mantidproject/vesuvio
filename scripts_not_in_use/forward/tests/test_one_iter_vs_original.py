import unittest
import numpy as np
from numpy.core.fromnumeric import size
import numpy.testing as nptest
from pathlib import Path

import matplotlib.pyplot as plt
from jupyterthemes import jtplot
jtplot.style()
np.set_printoptions(suppress=True, precision=8, linewidth=150)
# plt.style.use('dark_background')

currentPath = Path(__file__).absolute().parent  # Path to the repository

pathToOriginal = currentPath / "fixatures" / "testing_full_scripts" / "original_144-182_1iter.npz" 
# pathToOriginal = currentPath / "fixatures" / "ori_spec3-134_iter4_ncp.npz"

pathToOptimized = currentPath / "fixatures" / "testing_full_scripts" / "optimized_144-182_1iter.npz" 

#--------------------- Problem to solve
# The same original script ran in Mantid 6.2 gives different results for
# the intensities, but it should give very similar results



class TestFitParameters(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        oriPars = originalResults["all_spec_best_par_chi_nit"][0]
        self.orispec = oriPars[:, 0]
        self.orichi2 = oriPars[:, -2]
        self.orinit = oriPars[:, -1]
        self.orimainPars = oriPars[:, 1:-2]
        self.oriintensities = self.orimainPars[:, 0::3]
        self.oriwidths = self.orimainPars[:, 1::3]
        self.oricenters = self.orimainPars[:, 2::3]

        optimizedResults = np.load(pathToOptimized)
        optPars = optimizedResults["all_spec_best_par_chi_nit"][0]
        self.optspec = optPars[:, 0]
        self.optchi2 = optPars[:, -2]
        self.optnit = optPars[:, -1]
        self.optmainPars = optPars[:, 1:-2]
        self.optintensities = self.optmainPars[:, 0::3]
        self.optwidths = self.optmainPars[:, 1::3]
        self.optcenters = self.optmainPars[:, 2::3]

        self.rtol = 0.0001
        self.equal_nan = True

    def test_mainPars(self):
        totalMask = np.isclose(
            self.orimainPars, self.optmainPars, rtol=self.rtol, equal_nan=self.equal_nan
            )
        totalDiffMask = ~ totalMask
        displayMask(totalDiffMask, self.rtol, "parameters")

        
        plotPars = False
        if plotPars:
            plt.figure()
            plt.imshow(totalMask, aspect="auto", cmap=plt.cm.RdYlGn, 
                        interpolation="nearest", norm=None)
            plt.title("Comparison between ori and opt pars")
            plt.xlabel("Parameters")
            plt.ylabel("Spectra")
            plt.show()

    def test_chi2(self):
        totalMask = np.isclose(
            self.orichi2, self.optchi2, rtol=self.rtol, equal_nan=self.equal_nan
            )
        totalDiffMask = ~ totalMask
        displayMask(totalDiffMask, self.rtol, "chi2")


    def test_nit(self):
        totalMask = np.isclose(
            self.orinit, self.optnit, rtol=self.rtol, equal_nan=self.equal_nan
            )
        totalDiffMask = ~ totalMask
        displayMask(totalDiffMask, self.rtol, "nit")


    def test_intensities(self):
        totalMask = np.isclose(
            self.oriintensities, self.optintensities, rtol=self.rtol, equal_nan=self.equal_nan
            )
        totalDiffMask = ~ totalMask
        displayMask(totalDiffMask, self.rtol, "intensities")


def displayMask(mask, rtol, string):
    noDiff = np.sum(mask)
    maskSize = mask.size
    print("\nNo of different "+string+f", rtol={rtol}:\n",
        noDiff, " out of ", maskSize,
        f"ie {100*noDiff/maskSize:.1f} %")    


class TestNcp(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        self.orincp = originalResults["all_tot_ncp"][0, :, :-1]
        
        optimizedResults = np.load(pathToOptimized)
        self.optncp = optimizedResults["all_tot_ncp"][0]

        self.rtol = 0.001
        self.equal_nan = True

    def test_ncp(self):
        correctNansOri = np.where(
            (self.orincp==0) & np.isnan(self.optncp), np.nan, self.orincp
        )

        totalMask = np.isclose(
            correctNansOri, self.optncp, rtol=self.rtol, equal_nan=self.equal_nan
            )
        totalDiffMask = ~ totalMask
        displayMask(totalDiffMask, self.rtol, "ncp")
        
        plotNcp = False
        if plotNcp:
            plt.figure()
            plt.imshow(totalMask, aspect="auto", cmap=plt.cm.RdYlGn, 
                        interpolation="nearest", norm=None)
            plt.title("Comparison between ori and opt ncp")
            plt.ylabel("Spectra")
            plt.show()


class TestMeanWidths(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        self.orimeanwidths = originalResults["all_mean_widths"][0]

        optimizedResults = np.load(pathToOptimized)
        self.optmeanwidths = optimizedResults["all_mean_widths"][0]
    
    def test_widths(self):
        print("\nMean widths:",
            "\nori: ", self.orimeanwidths, 
            "\nopt: ", self.optmeanwidths)


class TestMeanIntensities(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        self.orimeanintensities = originalResults["all_mean_intensities"][0]

        optimizedResults = np.load(pathToOptimized)
        self.optmeanintensities = optimizedResults["all_mean_intensities"][0]
    
    def test_intensities(self):
        print("\nMean intensity ratios:",
            "\nori: ", self.orimeanintensities, 
            "\nopt: ", self.optmeanintensities)


class TestFitWorkspaces(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        self.oriws = originalResults["all_fit_workspaces"][0]
        
        optimizedResults = np.load(pathToOptimized)
        self.optws = optimizedResults["all_fit_workspaces"][0]

        self.decimal = 8

    def test_ws(self):
        nptest.assert_almost_equal(self.optws, self.oriws, decimal=self.decimal)
        print(f"\nFitted ws is equal up to decimal={self.decimal}")


class TestSymSumYSpace(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        self.oridataY = originalResults["YSpaceSymSumDataY"]
        self.oridataE = originalResults["YSpaceSymSumDataE"]

        optimizedResults = np.load(pathToOptimized)
        self.optdataY = optimizedResults["YSpaceSymSumDataY"]
        self.optdataE = optimizedResults["YSpaceSymSumDataE"]
        self.rtol = 0.000001
        self.equal_nan = True
        self.decimal = 6

    def test_YSpaceDataY(self):
        nptest.assert_almost_equal(self.oridataY, self.optdataY, decimal=self.decimal)
        print(f"\nSummed Spectra in YSpace is equal up to decimal={self.decimal}")

        totalMask = np.isclose(
            self.oridataY, self.optdataY, rtol=self.rtol, equal_nan=self.equal_nan
            )
        totalDiffMask = ~ totalMask
        displayMask(totalDiffMask, self.rtol, "sym sum dataY YSpace")


        plotSumSpectra = False
        if plotSumSpectra:            
            plt.figure()
            plt.imshow(totalMask, aspect="auto", cmap=plt.cm.RdYlGn, 
                        interpolation="nearest", norm=None)
            plt.title("YSpace difference dataY")
            plt.ylabel("Summed Spectra")
            plt.show()     
           
    def test_YSpaceDataE(self):
        nptest.assert_almost_equal(self.oridataE, self.optdataE, decimal=self.decimal)
        print(f"\nErrors of Summed Spectra in YSpace is equal up to decimal={self.decimal}")


class TestResolution(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        self.orires = originalResults["resolution"]

        optimizedResults = np.load(pathToOptimized)
        self.optres = optimizedResults["resolution"]

        self.rtol = 0.0001
        self.equal_nan = True
        self.decimal = 8

    def test_resolution(self):
        totalMask = np.isclose(
            self.orires, self.optres, rtol=self.rtol, equal_nan=self.equal_nan
            )
        totalDiffMask = ~ totalMask
        displayMask(totalDiffMask, self.rtol, "values resolution YSpace")

        plotResolution = False
        if plotResolution:
            plt.figure()
            x = range(len(self.orires[0]))
            plt.plot(x, self.orires[0], label="oriRes")
            plt.plot(x, self.optres[0], label="optRes")
            plt.legend()
            plt.show()
        # nptest.assert_almost_equal(self.orires, self.optres, decimal=self.decimal)
        # print(f"\nResolution in YSpace is equal up to decimal={self.decimal}")


class TestHdataY(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        self.oriHdataY = originalResults["HdataY"]

        optimizedResults = np.load(pathToOptimized)
        self.optHdataY = optimizedResults["HdataY"]

        self.rtol = 0.0001
        self.equal_nan = True
        self.decimal = 4

    def test_HdataY(self):
        totalMask = np.isclose(
            self.oriHdataY, self.optHdataY, rtol=self.rtol, equal_nan=self.equal_nan
            )
        totalDiffMask = ~ totalMask
        displayMask(totalDiffMask, self.rtol, "values isolated H TOF")

        plotHdataY = False
        if plotHdataY:            
            plt.figure()
            plt.imshow(totalMask, aspect="auto", cmap=plt.cm.RdYlGn, 
                        interpolation="nearest", norm=None)
            plt.title("H peak TOF dataY")
            plt.ylabel("Spectra")
            plt.show()    

        nptest.assert_almost_equal(self.oriHdataY, self.optHdataY, decimal=self.decimal)
        print(f"\nIsolated H peak in TOF is equal up to decimal={self.decimal}")


class TestFinalRawDataY(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        self.oriFinalDataY = originalResults["finalRawDataY"]

        optimizedResults = np.load(pathToOptimized)
        self.optFinalDataY = optimizedResults["finalRawDataY"]

        self.rtol = 1e-6
        self.equal_nan = True
        self.decimal = 10

    def test_HdataY(self):
        totalMask = np.isclose(
            self.oriFinalDataY, self.optFinalDataY, rtol=self.rtol, equal_nan=self.equal_nan
            )
        totalDiffMask = ~ totalMask
        displayMask(totalDiffMask, self.rtol, "values final raw dataY")

        plotFinalRawDataY = False
        if plotFinalRawDataY:            
            plt.figure()
            plt.imshow(totalMask, aspect="auto", cmap=plt.cm.RdYlGn, 
                        interpolation="nearest", norm=None)
            plt.title("Final raw dataY TOF")
            plt.ylabel("Spectra")
            plt.show()    

        nptest.assert_almost_equal(self.oriFinalDataY, self.optFinalDataY, decimal=self.decimal)
        print(f"\nFinal DataY in TOF is equal up to decimal={self.decimal}")


class TestFinalRawDataE(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        self.oriFinalDataE = originalResults["finalRawDataE"]

        optimizedResults = np.load(pathToOptimized)
        self.optFinalDataE = optimizedResults["finalRawDataE"]

        self.rtol = 1e-6
        self.equal_nan = True
        self.decimal = 10

    def test_HdataY(self):
        totalMask = np.isclose(
            self.oriFinalDataE, self.optFinalDataE, rtol=self.rtol, equal_nan=self.equal_nan
            )
        totalDiffMask = ~ totalMask
        displayMask(totalDiffMask, self.rtol, "values final raw dataE")

        plotFinalRawDataY = False
        if plotFinalRawDataY:            
            plt.figure()
            plt.imshow(totalMask, aspect="auto", cmap=plt.cm.RdYlGn, 
                        interpolation="nearest", norm=None)
            plt.title("Final raw dataE TOF")
            plt.ylabel("Spectra")
            plt.show()    

        nptest.assert_almost_equal(self.oriFinalDataE, self.optFinalDataE, decimal=self.decimal)
        print(f"\nFinal DataE in TOF is equal up to decimal={self.decimal}")

class Testpopt(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        self.oripopt = originalResults["popt"]

        optimizedResults = np.load(pathToOptimized)
        self.optpopt = optimizedResults["popt"]
    
    def test_intensities(self):
        print("\nFit parameters LM:\nori:",
                self.oripopt[0], "\nopt:", self.optpopt[0])
        print("\nFit parameters Simplex:\nori:",
                self.oripopt[1], "\nopt:", self.optpopt[1])

class Testperr(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        self.oriperr = originalResults["perr"]

        optimizedResults = np.load(pathToOptimized)
        self.optperr = optimizedResults["perr"]
    
    def test_intensities(self):
        print("\nError in parameters LM:\nori:",
                self.oriperr[0], "\nopt:", self.optperr[0])
        print("\nError in parameters Simplex:\nori:",
                self.oriperr[1], "\nopt:", self.optperr[1])

if __name__ == "__main__":
    unittest.main()
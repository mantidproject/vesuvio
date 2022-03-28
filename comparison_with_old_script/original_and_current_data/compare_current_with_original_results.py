import unittest
import numpy as np
import numpy.testing as nptest
from pathlib import Path

import matplotlib.pyplot as plt
plt.style.use('seaborn-poster')
import matplotlib.ticker as mtick
# plt.style.use('fivethirtyeight')
plt.rcParams['axes.facecolor'] = (0.9, 0.9, 0.9)
plt.rcParams.update({"axes.grid" : True, "grid.color": "white"})

np.set_printoptions(suppress=True, precision=8, linewidth=150)
# plt.style.use('dark_background')

currentPath = Path(__file__).absolute().parent  # Path to the repository

testForward = False
if testForward:
    pathToOriginal = currentPath / "original_data" / "4iter_forward_GB_MS.npz" 
    pathToOptimized = currentPath / "current_data" / "4iter_forward_GM_MS.npz" 

else:
    pathToOriginal = currentPath / "original_data" / "4iter_backward_MS.npz" 
    pathToOptimized = currentPath / "current_data" / "4iter_backward_MS.npz" 


def displayMask(mask, rtol, string):
    noDiff = np.sum(mask)
    maskSize = mask.size
    print("\nNo of different "+string+f", rtol={rtol}:\n",
        noDiff, " out of ", maskSize,
        f"ie {100*noDiff/maskSize:.1f} %")    


def displayMaskAllIter(mask, rtol, string):
    print("\nNo of different "+string+f", rtol={rtol}:")
    for i, mask_i in enumerate(mask):
        noDiff = np.sum(mask_i)
        maskSize = mask_i.size
        print(f"iter {i}: ", noDiff, " out of ", maskSize, f"ie {100*noDiff/maskSize:.1f} %")    



class TestFitParameters(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        oriPars = originalResults["all_spec_best_par_chi_nit"]
        self.orispec = oriPars[:, :, 0]
        self.orichi2 = oriPars[:, :, -2]
        self.orinit = oriPars[:, :, -1]
        self.orimainPars = oriPars[:, :, 1:-2]
        self.oriintensities = self.orimainPars[:, :, 0::3]
        self.oriwidths = self.orimainPars[:, :, 1::3]
        self.oricenters = self.orimainPars[:, :, 2::3]

        optimizedResults = np.load(pathToOptimized)
        optPars = optimizedResults["all_spec_best_par_chi_nit"]
        self.optspec = optPars[:, :, 0]
        self.optchi2 = optPars[:, :, -2]
        self.optnit = optPars[:, :, -1]
        self.optmainPars = optPars[:, :, 1:-2]
        self.optintensities = self.optmainPars[:, :, 0::3]
        self.optwidths = self.optmainPars[:, :, 1::3]
        self.optcenters = self.optmainPars[:, :, 2::3]

        self.rtol = 0.0001
        self.equal_nan = True

    def test_mainPars(self):
        totalMask = np.isclose(
            self.orimainPars, self.optmainPars, rtol=self.rtol, equal_nan=self.equal_nan
            )
        totalDiffMask = ~ totalMask
        displayMaskAllIter(totalDiffMask, self.rtol, "parameters")

        
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
        displayMaskAllIter(totalDiffMask, self.rtol, "chi2")


    def test_nit(self):
        totalMask = np.isclose(
            self.orinit, self.optnit, rtol=self.rtol, equal_nan=self.equal_nan
            )
        totalDiffMask = ~ totalMask
        displayMaskAllIter(totalDiffMask, self.rtol, "nit")


    def test_intensities(self):
        totalMask = np.isclose(
            self.oriintensities, self.optintensities, rtol=self.rtol, equal_nan=self.equal_nan
            )
        totalDiffMask = ~ totalMask
        displayMaskAllIter(totalDiffMask, self.rtol, "intensities")


class TestNcp(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        self.orincp = originalResults["all_tot_ncp"][:, :, :-1]
        
        optimizedResults = np.load(pathToOptimized)
        self.optncp = optimizedResults["all_tot_ncp"]

        self.rtol = 0.0001
        self.equal_nan = True

    def test_ncp(self):
        correctNansOri = np.where(
            (self.orincp==0) & np.isnan(self.optncp), np.nan, self.orincp
        )

        totalMask = np.isclose(
            correctNansOri, self.optncp, rtol=self.rtol, equal_nan=self.equal_nan
            )
        totalDiffMask = ~ totalMask
        displayMaskAllIter(totalDiffMask, self.rtol, "ncp")
        
        plotNcp = False
        if plotNcp:
            noOfIter = len(self.orincp)
            fig, axs = plt.subplots(1, noOfIter)
            for i, ax in enumerate(axs):
                ax.imshow(totalMask[i], aspect="auto", cmap=plt.cm.RdYlGn, 
                        interpolation="nearest", norm=None)
            fig.suptitle("Comparison between ori and opt ncp")
            axs[0].set_ylabel("Spectra")
            # axs.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))
            plt.show()


def plot_values_and_errors(oriwidths, optwidths, name):
    noOfMasses = len(oriwidths[0])
    fig, axs = plt.subplots(2, noOfMasses, figsize=(12, 8))
    x = range(len(oriwidths))
    relativeDifference = abs(optwidths - oriwidths) / oriwidths

    for i in range(noOfMasses):
        axs[0, i].plot(x, optwidths[:, i], "ro-", label="opt", alpha=0.6)
        axs[0, i].plot(x, oriwidths[:, i], "bo--", label="ori", alpha=0.6)
        axs[1, i].plot(x, relativeDifference[:, i], "bo--", label="relErr")
        axs[1, i].yaxis.set_major_formatter(mtick.FormatStrFormatter('%.2e'))

    axs[0, 0].set_ylabel(name+" Values")
    axs[1, 0].set_ylabel("Relative Error")
    fig.suptitle("Evolution of mean and rel errors of "+name+" over iterations")
    plt.legend(loc="upper left", bbox_to_anchor = (1,1))
    plt.show()


class TestMeanWidths(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        self.orimeanwidths = originalResults["all_mean_widths"]

        optimizedResults = np.load(pathToOptimized)
        self.optmeanwidths = optimizedResults["all_mean_widths"]
    
    def test_widths(self):
        print("\nFinal mean widths:",
            "\nori: ", self.orimeanwidths[-1], 
            "\nopt: ", self.optmeanwidths[-1])

        plot_values_and_errors(self.orimeanwidths, self.optmeanwidths, "Widths")


class TestMeanIntensities(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        self.orimeanintensities = originalResults["all_mean_intensities"]

        optimizedResults = np.load(pathToOptimized)
        self.optmeanintensities = optimizedResults["all_mean_intensities"]
    
    def test_intensities(self):
        print("\nFinal mean intensity ratios:",
            "\nori: ", self.orimeanintensities[-1], 
            "\nopt: ", self.optmeanintensities[-1])

        plot_values_and_errors(self.orimeanintensities, self.optmeanintensities, "Intensity Ratios")


class TestFitWorkspaces(unittest.TestCase):
    def setUp(self):
        originalResults = np.load(pathToOriginal)
        self.oriws = originalResults["all_fit_workspaces"]
        
        optimizedResults = np.load(pathToOptimized)
        self.optws = optimizedResults["all_fit_workspaces"]

        self.decimal = 8
        self.rtol = 0.0001
        self.equal_nan = True

    def test_ws(self):
        totalMask = np.isclose(
            self.oriws, self.optws, rtol=self.rtol, equal_nan=self.equal_nan
            )
        totalDiffMask = ~ totalMask
        displayMaskAllIter(totalDiffMask, self.rtol, "workspaces to be fitted")


# class TestResolution(unittest.TestCase):
#     def setUp(self):
#         originalResults = np.load(pathToOriginal)
#         self.orires = originalResults["resolution"]

#         optimizedResults = np.load(pathToOptimized)
#         self.optres = optimizedResults["resolution"]

#         self.rtol = 0.0001
#         self.equal_nan = True
#         self.decimal = 8

#     def test_resolution(self):
#         totalMask = np.isclose(
#             self.orires, self.optres, rtol=self.rtol, equal_nan=self.equal_nan
#             )
#         totalDiffMask = ~ totalMask
#         displayMask(totalDiffMask, self.rtol, "values resolution YSpace")

#         plotResolution = False
#         if plotResolution:
#             plt.figure()
#             x = range(len(self.orires[0]))
#             plt.plot(x, self.orires[0], label="oriRes")
#             plt.plot(x, self.optres[0], label="optRes")
#             plt.legend()
#             plt.show()
#         # nptest.assert_almost_equal(self.orires, self.optres, decimal=self.decimal)
#         # print(f"\nResolution in YSpace is equal up to decimal={self.decimal}")


# class TestHdataY(unittest.TestCase):
#     def setUp(self):
#         originalResults = np.load(pathToOriginal)
#         self.oriHdataY = originalResults["HdataY"]

#         optimizedResults = np.load(pathToOptimized)
#         self.optHdataY = optimizedResults["HdataY"]

#         self.rtol = 0.0001
#         self.equal_nan = True
#         self.decimal = 4

#     def test_HdataY(self):
#         totalMask = np.isclose(
#             self.oriHdataY, self.optHdataY, rtol=self.rtol, equal_nan=self.equal_nan
#             )
#         totalDiffMask = ~ totalMask
#         displayMask(totalDiffMask, self.rtol, "values isolated H TOF")

#         plotHdataY = False
#         if plotHdataY:            
#             plt.figure()
#             plt.imshow(totalMask, aspect="auto", cmap=plt.cm.RdYlGn, 
#                         interpolation="nearest", norm=None)
#             plt.title("H peak TOF dataY")
#             plt.ylabel("Spectra")
#             plt.show()    

#         nptest.assert_almost_equal(self.oriHdataY, self.optHdataY, decimal=self.decimal)
#         print(f"\nIsolated H peak in TOF is equal up to decimal={self.decimal}")


# class TestFinalRawDataY(unittest.TestCase):
#     def setUp(self):
#         originalResults = np.load(pathToOriginal)
#         self.oriFinalDataY = originalResults["finalRawDataY"]

#         optimizedResults = np.load(pathToOptimized)
#         self.optFinalDataY = optimizedResults["finalRawDataY"]

#         self.rtol = 1e-6
#         self.equal_nan = True
#         self.decimal = 7

#     def test_finalDataY(self):
#         totalMask = np.isclose(
#             self.oriFinalDataY, self.optFinalDataY, rtol=self.rtol, equal_nan=self.equal_nan
#             )
#         totalDiffMask = ~ totalMask
#         displayMask(totalDiffMask, self.rtol, "values final raw dataY")

#         plotFinalRawDataY = False
#         if plotFinalRawDataY:            
#             plt.figure()
#             plt.imshow(totalMask, aspect="auto", cmap=plt.cm.RdYlGn, 
#                         interpolation="nearest", norm=None)
#             plt.title("Final raw dataY TOF")
#             plt.ylabel("Spectra")
#             plt.show()    

#         nptest.assert_almost_equal(self.oriFinalDataY, self.optFinalDataY, decimal=self.decimal)
#         print(f"\nFinal DataY in TOF is equal up to decimal={self.decimal}")


# class TestFinalRawDataE(unittest.TestCase):
#     def setUp(self):
#         originalResults = np.load(pathToOriginal)
#         self.oriFinalDataE = originalResults["finalRawDataE"]

#         optimizedResults = np.load(pathToOptimized)
#         self.optFinalDataE = optimizedResults["finalRawDataE"]

#         self.rtol = 1e-6
#         self.equal_nan = True
#         self.decimal = 7

#     def test_finalDataE(self):
#         totalMask = np.isclose(
#             self.oriFinalDataE, self.optFinalDataE, rtol=self.rtol, equal_nan=self.equal_nan
#             )
#         totalDiffMask = ~ totalMask
#         displayMask(totalDiffMask, self.rtol, "values final raw dataE")

#         plotFinalRawDataY = False
#         if plotFinalRawDataY:            
#             plt.figure()
#             plt.imshow(totalMask, aspect="auto", cmap=plt.cm.RdYlGn, 
#                         interpolation="nearest", norm=None)
#             plt.title("Final raw dataE TOF")
#             plt.ylabel("Spectra")
#             plt.show()    

#         nptest.assert_almost_equal(self.oriFinalDataE, self.optFinalDataE, decimal=self.decimal)
#         print(f"\nFinal DataE in TOF is equal up to decimal={self.decimal}")

# Don't do the tests below anymore because changed the averaging and symetrizing in yspace

# class TestSymSumYSpace(unittest.TestCase):
#     def setUp(self):
#         originalResults = np.load(pathToOriginal)
#         self.oridataY = originalResults["YSpaceSymSumDataY"]
#         self.oridataE = originalResults["YSpaceSymSumDataE"]

#         optimizedResults = np.load(pathToOptimized)
#         self.optdataY = optimizedResults["YSpaceSymSumDataY"]
#         self.optdataE = optimizedResults["YSpaceSymSumDataE"]
#         self.rtol = 0.000001
#         self.equal_nan = True
#         self.decimal = 6

#     def test_YSpaceDataY(self):
#         nptest.assert_almost_equal(self.oridataY, self.optdataY, decimal=self.decimal)
#         print(f"\nSummed Spectra in YSpace is equal up to decimal={self.decimal}")

#         totalMask = np.isclose(
#             self.oridataY, self.optdataY, rtol=self.rtol, equal_nan=self.equal_nan
#             )
#         totalDiffMask = ~ totalMask
#         displayMask(totalDiffMask, self.rtol, "sym sum dataY YSpace")


#         plotSumSpectra = False
#         if plotSumSpectra:            
#             plt.figure()
#             plt.imshow(totalMask, aspect="auto", cmap=plt.cm.RdYlGn, 
#                         interpolation="nearest", norm=None)
#             plt.title("YSpace difference dataY")
#             plt.ylabel("Summed Spectra")
#             plt.show()     
           
#     def test_YSpaceDataE(self):
#         nptest.assert_almost_equal(self.oridataE, self.optdataE, decimal=self.decimal)
#         print(f"\nErrors of Summed Spectra in YSpace is equal up to decimal={self.decimal}")


# class Testpopt(unittest.TestCase):
#     def setUp(self):
#         originalResults = np.load(pathToOriginal)
#         self.oripopt = originalResults["popt"]

#         optimizedResults = np.load(pathToOptimized)
#         self.optpopt = optimizedResults["popt"]
    
#     def test_intensities(self):
#         print("\nFit parameters LM:\nori:",
#                 self.oripopt[0], "\nopt:", self.optpopt[0])
#         print("\nFit parameters Simplex:\nori:",
#                 self.oripopt[1], "\nopt:", self.optpopt[1])

# class Testperr(unittest.TestCase):
#     def setUp(self):
#         originalResults = np.load(pathToOriginal)
#         self.oriperr = originalResults["perr"]

#         optimizedResults = np.load(pathToOptimized)
#         self.optperr = optimizedResults["perr"]
    
#     def test_intensities(self):
#         print("\nError in parameters LM:\nori:",
#                 self.oriperr[0], "\nopt:", self.optperr[0])
#         print("\nError in parameters Simplex:\nori:",
#                 self.oriperr[1], "\nopt:", self.optperr[1])

if __name__ == "__main__":
    unittest.main()
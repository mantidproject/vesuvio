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

from EVSVesuvio.scripts import handle_config
from EVSVesuvio.analysis_runner import import_from_path
from pathlib import Path
from os import path

scriptName = handle_config.read_config_var('caching.experiment')
experimentsPath = Path(handle_config.read_config_var('caching.location')) / "experiments" / scriptName # Path to the repository

oriResultsPatth = experimentsPath / "original_results"
currResultsPatth = experimentsPath / "output_files"

def get_files_to_compare():
    for ori_f in oriResultsPatth.iterdir():
        for curr_f in currResultsPatth.iterdir():
            if ori_f.name == curr_f.name:
                if input(f"Selected ws: {ori_f.name}, continue? (y/n)") == 'y':
                    pathToOriginal = ori_f
                    pathToOptimized = curr_f
                    return pathToOriginal, pathToOptimized
    raise FileNotFoundError("Did not find two matching files!")
                
pathToOriginal, pathToOptimized = get_files_to_compare()

# testForward = True 
# if testForward:
#     pathToOriginal = oriResultsPatth / "4iter_forward_GB_MS.npz" 
#     pathToOptimized = currResultsPatth / "spec_144-182_iter_3_MS_GC.npz" 
#
# else:
#     pathToOriginal = oriResultsPatth / "4iter_backward_MS.npz" 
#     pathToOptimized = currResultsPatth / "spec_3-134_iter_3_MS.npz" 
#

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
        self.orincp = originalResults["all_tot_ncp"][:,:,:-1]
        
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


if __name__ == "__main__":
    unittest.main()


import numpy as np
import matplotlib .pyplot as plt
from pathlib import Path
from scipy import stats
from vesuvio_analysis.core_functions.analysis_functions import filterWidthsAndIntensities, calculateMeansAndStds, filterWidthsAndIntensities
from vesuvio_analysis.core_functions.analysis_functions import loadInstrParsFileIntoArray
from vesuvio_analysis.core_functions.bootstrap import setOutputDirs

currentPath = Path(__file__).parent.absolute() 
experimentsPath = currentPath / ".." / ".. " / "experiments"
IPFilesPath = currentPath / ".." / "ip_files" 



def calcBootMeans(bestPars):
    """Performs the means and std on each bootstrap sample"""
    bootWidths = bestPars[:, :, 1::3]
    bootIntensities = bestPars[:, :, 0::3]

    bootMeanW = np.zeros((len(bootWidths[0,0,:]), len(bootWidths)))
    bootStdW = np.zeros(bootMeanW.shape)
    bootMeanI = np.zeros(bootMeanW.shape)
    bootStdI = np.zeros(bootMeanW.shape)

    for j, (widths, intensities) in enumerate(zip(bootWidths, bootIntensities)):

        meanW, stdW, meanI, stdI = calculateMeansAndStds(widths.T, intensities.T)

        bootMeanW[:, j] = meanW
        bootStdW[:, j] = stdW
        bootMeanI[:, j] = meanI
        bootStdI[:, j] = stdI

    return bootMeanW, bootMeanI, bootStdW, bootStdI


def filteredBootMeans(bestPars):
    bootWidths = bestPars[:, :, 1::3]
    bootIntensities = bestPars[:, :, 0::3]

    for i, (widths, intensities) in enumerate(zip(bootWidths, bootIntensities)):
        filteredWidths, filteredIntensities = filterWidthsAndIntensities(widths.T, intensities.T)

        bootWidths[i] = filteredWidths.T
        bootIntensities[i] = filteredIntensities.T
    
    filteredBestPars = bestPars.copy()
    filteredBestPars[:, :, 1::3] = bootWidths
    filteredBestPars[:, :, 0::3] = bootIntensities
    return filteredBestPars


def plotHists(ax, samples, nBins, title, disableCI=False, disableLeg=False):
    ax.set_title(f"Histogram of {title}")
    for i, bootHist in enumerate(samples):

        if np.all(bootHist==0) or np.all(np.isnan(bootHist)):
            continue
        
        mean = np.nanmean(bootHist)
        bounds = np.percentile(bootHist, [5, 95])
        errors = bounds - mean

        leg = f"Row {i}: {mean:>6.3f} +{errors[1]:.3f} {errors[0]:.3f}"
        ax.hist(bootHist, nBins, histtype="step", label=leg)

        ax.axvline(mean, 0.9, 0.97, color="k", ls="--", alpha=0.4)
        
        if disableCI:
            pass
        else:
            ax.axvspan(bounds[0], bounds[1], alpha=0.2, color="r")
    
    if disableLeg:
        pass
    else:
        ax.legend(loc="upper center")


def plotMeansOverNoSamples(sampleNos, samples, title):

    sampleMeans = np.zeros((len(samples), len(sampleNos)))
    sampleErrors = np.zeros((len(samples), 2, len(sampleNos)))
    for i, N in enumerate(sampleNos):
        subSample = samples[:, :N]

        mean = np.mean(subSample, axis=1)

        bounds = np.percentile(subSample, [5, 95], axis=1).T
        assert bounds.shape == (len(subSample), 2), f"Wrong shape: {bounds.shape}"
        errors = bounds - mean[:, np.newaxis]

        sampleMeans[:, i] = mean
        sampleErrors[:, :, i] = errors

    sampleMeans = sampleMeans - sampleMeans[:, 0][:, np.newaxis]
    for i, (means, errors) in enumerate(zip(sampleMeans, sampleErrors)):
        plt.plot(sampleNos, means, label=f"idx {i}")
        plt.fill_between(sampleNos, errors[0, :], errors[1, :], alpha=0.2)
    
    plt.title(f"Evolution of {title} over the sample size.")
    plt.legend()
    plt.show()


def plotRawHists(bootSamples, idx, specRange, IPPath):

    firstIdx = specRange[0]
    lastIdx = specRange[1]
    samples = bootSamples[:, firstIdx:lastIdx, idx].T
    nBins = 100
    print(f"\nShape: {samples.shape}\n")
    # assert samples.shape == (len(samples[0, 0, :]), len(samples)), f"Wrong shape: {samples.shape}"
    
    print(f"\nNaNs positions: {np.argwhere(samples==np.nan)}\n")
    fig, ax = plt.subplots()
    plotHists(ax, samples, nBins, f"idx {idx}", disableCI=True, disableLeg=True)

    meanSamples = np.nanmean(samples, axis=0).flatten()
    ax.hist(meanSamples, nBins, color="r", histtype="step", linewidth=2)
    ax.axvline(np.nanmean(meanSamples), 0.9, 0.97, color="r", ls="--", linewidth=2)
    # ax.axvline(np.percentile(meanSamples, 0.5), 0.9, 0.97, color="r", ls="--", linewidth=2)

    # Calculate correlation with scattering angle
    ipMatrix = np.loadtxt(IPPath, dtype=str)[1:].astype(float)

    thetas = ipMatrix[firstIdx : lastIdx, 2]    # Scattering angle on third column
    assert thetas.shape == (len(samples),), f"Wrong shape: {thetas.shape}"

    histMeansCorr = True
    if histMeansCorr:
        deltaMeans = np.nanmean(samples, axis=1) #- np.nanmean(meanSamples)
        # deltaMeans = np.percentile(samples, 0.5, axis=1) - np.percentile(meanSamples, 0.5)

        # Remove masked spectra:
        nanMask = np.isnan(deltaMeans)
        deltaMeans = deltaMeans[~nanMask]
        thetas = thetas[~nanMask]

        print(thetas[:10])
        corr = stats.pearsonr(thetas, deltaMeans)
        ax.set_title(f"Correlation scatt angle: {corr[0]:.3f}")
    else:
        bounds = np.percentile(samples, [5, 95], axis=1).T
        assert bounds.shape == (len(samples), 2), f"Wrong shape: {bounds.shape}"
        histWidths = bounds[:, 1] - bounds[:, 0]

        nanMask = np.isnan(histWidths)
        histWidths = histWidths[~nanMask]
        thetas = thetas[~nanMask]

        corr = stats.pearsonr(thetas, histWidths)
        ax.set_title(f"Correlation scatt angle: {corr[0]:.3f}")


    plt.show()


def checkBootSamplesVSParent(bestPars, parentPars):
    """
    For an unbiased estimator, the mean of the bootstrap samples will converge to 
    the mean of the experimental sample (here called parent).
    """

    bootWidths = bestPars[:, :, 1::3]
    bootIntensities = bestPars[:, :, 0::3]

    # TODO: Need to decide on wether to use mean, median or mode
    meanBootWidths = np.mean(bootWidths, axis=0)
    meanBootIntensities = np.mean(bootIntensities, axis=0)

    avgWidths, stdWidths, avgInt, stdInt = calculateMeansAndStds(meanBootWidths.T, meanBootIntensities.T)

    parentWidths = parentPars[:, 1::3]
    parentIntensities = parentPars[:, 0::3]

    avgWidthsP, stdWidthsP, avgIntP, stdIntP = calculateMeansAndStds(parentWidths.T, parentIntensities.T)
  
    print("\nComparing Bootstrap means with parent means:\n")
    printResults(avgWidths, stdWidths, "Boot Widths")
    printResults(avgWidthsP, stdWidthsP, "Parent Widths")
    printResults(avgInt, stdInt, "Boot Intensities")
    printResults(avgIntP, stdIntP, "Parent Intensities")
    

def plot2DHists(bootSamples, nBins, mode):
    """bootSamples has histogram rows for each parameter"""

    plotSize = len(bootSamples)
    fig, axs = plt.subplots(plotSize, plotSize, figsize=(8, 8))

    for i in range(plotSize):
        for j in range(plotSize):
            
            # axs[i, j].axis("off")

            if j>i:
                axs[i, j].set_visible(False)

            elif i==j:
                if i>0:
                    orientation="horizontal"
                else:
                    orientation="vertical"

                axs[i, j].hist(bootSamples[i], nBins, orientation=orientation)

            else:
                axs[i, j].hist2d(bootSamples[j], bootSamples[i], nBins)
                
            if j==0:
                axs[i, j].set_ylabel(f"idx {i}")  
            else:
                axs[i, j].get_yaxis().set_ticks([])

            if i==plotSize-1:
                axs[i, j].set_xlabel(f"idx{j}")
            else:
                axs[i, j].get_xaxis().set_ticks([])
            
            axs[i, j].set_title(f"r = {stats.pearsonr(bootSamples[i], bootSamples[j])[0]:.3f}")

    
    fig.suptitle(f"1D and 2D Histograms of {mode}")

    plt.show()


def addParentMeans(ax, means):
    for mean in means:
        ax.axvline(mean, 0, 0.97, color="k", ls=":")
        

def printResults(arrM, arrE, mode):
    print(f"\n{mode}:\n")
    for i, (m, e) in enumerate(zip(arrM, arrE)):
        print(f"{mode} {i}: {m:>6.3f} \u00B1 {e:<6.3f}")


# def dataPaths(sampleName, firstSpec, lastSpec, msIter, MS, GC, nSamples, speed):
#     # Build Filename based on ic
#     corr = ""
#     if MS & (msIter>1):
#         corr+="_MS"
#     if GC & (msIter>1):
#         corr+="_GC"

#     fileName = f"spec_{firstSpec}-{lastSpec}_iter_{msIter}{corr}"
#     fileNameZ = fileName + ".npz"

#     bootOutPath = experimentsPath / sampleName / "jackknife_data"
    
#     bootName = fileName + f"_nsampl_{nSamples}"
#     bootNameYFit = fileName + "_ySpaceFit" + f"_nsampl_{nSamples}"

#     bootNameZ = bootName + ".npz"
#     bootNameYFitZ = bootNameYFit + ".npz"

#     loadPath = bootOutPath / speed / bootNameZ
#     bootData = np.load(loadPath)

#     loadYFitPath = bootOutPath / speed / bootNameYFitZ

#     return loadPath, loadYFitPath



# sampleName = "D_HMT"
# firstSpec = 3
# lastSpec = 134
# msIter = 4
# MS = True
# GC = False
# nSamples = 1000
# nBins = 30
# speed = "slow"
# ySpaceFit = False

# sampleName = "starch_80_RD"
# firstSpec = 3
# lastSpec = 134
# msIter = 4
# MS = True
# GC = False
# nSamples = 144
# nBins = 20 #int(nSamples/25)
# speed = "slow"
# ySpaceFit = False
# IPPath = IPFilesPath / "ip2018_3.par"

def runAnalysisOfStoredBootstrap(bckwdIC, fwdIC, bootIC):


    setOutputDirs([bckwdIC, fwdIC], bootIC)
    
    for IC in [bckwdIC, fwdIC]:
        if not(IC.bootSavePath.is_file()):
            print("Bootstrap data files not found, unable to run analysis!")
            print(f"{IC.bootSavePath.name}")
        
        bootData = np.load(IC.bootSavePath)
        bootPars = bootData["boot_samples"][:, :, 1:-2]
        parentPars = bootData["parent_result"][:, 1:-2]

        print("Success! Data files found:")
        print(f"{IC.bootSavePath.name}")
        print("no of samples: ", len(bootPars))



# dataPath, dataYFitPath = dataPaths(sampleName, firstSpec, lastSpec, msIter, MS, GC, nSamples, speed)

# bootData = np.load(dataPath)
# bootPars = bootData["boot_samples"][:, :, 1:-2]
# parentPars = bootData["parent_result"][:, 1:-2]

# print(f"\nNumber of samples: {len(bootData)}")

# assert ~np.all(bootPars[-1] == parentPars), "Error in Jackknife due to last column."

# checkBootSamplesVSParent(bootPars, parentPars)


# filteredBootPars = bootPars.copy()
# # filteredBootPars = filteredBootMeans(bootPars)
# # plotRawHists(filteredBootPars, 1, [0, 38], IPPath)


# meanWp, meanIp, stdWp, stdIp = calcBootMeans(parentPars[np.newaxis, :, :])
# meanWp = meanWp.flatten()
# meanIp = meanIp.flatten()

# meanW, meanI, stdW, stdI = calcBootMeans(bootPars)

# # plotMeansOverNoSamples(np.linspace(50, 2500, 20).astype(int), meanW, "Widths")


# fig, axs = plt.subplots(1, 2, figsize=(15, 3))
# for ax, means, title, meanp in zip(axs.flatten(), [meanW, meanI], ["Widths", "Intensities"], [meanWp, meanIp]):
#     plotHists(ax, means, nBins, title, disableCI=True)
#     # addParentMeans(ax, meanp)
# plt.show()


# if ySpaceFit:
#     bootYFitData = np.load(dataYFitPath)
#     bootYFitVals = bootYFitData["boot_vals"]    # Previously boot_samples
#     mFitVals = bootYFitVals[:, 0, :-1].T  # Last value is chi

#     # Plot each parameter in an individual histogram
#     fig, axs = plt.subplots(len(mFitVals), 1, figsize=(8, 10))
#     for i, (ax, hist) in enumerate(zip(axs.flatten(), mFitVals)):
#         plotHists(ax, hist[np.newaxis, :], nBins, f"idx {i}")
#     plt.show()


# plot2DHists(meanW, nBins, "Widths")    
# plot2DHists(meanI, nBins, "Intensities")   


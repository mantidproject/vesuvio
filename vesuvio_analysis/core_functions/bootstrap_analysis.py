
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



def calcMeansWithOriginalProc(bestPars):
    """Performs the means and std on each bootstrap sample according to original procedure"""
    
    bootWidths = bestPars[:, :, 1::3]
    bootIntensities = bestPars[:, :, 0::3]

    bootMeanW = np.zeros((len(bootWidths[0,0,:]), len(bootWidths)))
    bootStdW = np.zeros(bootMeanW.shape)
    bootMeanI = np.zeros(bootMeanW.shape)
    bootStdI = np.zeros(bootMeanW.shape)

    for j, (widths, intensities) in enumerate(zip(bootWidths, bootIntensities)):

        meanW, stdW, meanI, stdI = calculateMeansAndStds(widths.T, intensities.T)

        bootMeanW[:, j] = meanW       # Interested only in the means 
        bootMeanI[:, j] = meanI

    return bootMeanW, bootMeanI


def filteredBootMeans(bestPars):
    """Use same filtering function used on original procedure"""

    # Extract Widths and Intensities from bootstrap samples
    bootWidths = bestPars[:, :, 1::3]
    bootIntensities = bestPars[:, :, 0::3]

    # Perform the filter
    for i, (widths, intensities) in enumerate(zip(bootWidths, bootIntensities)):
        filteredWidths, filteredIntensities = filterWidthsAndIntensities(widths.T, intensities.T)
        
        bootWidths[i] = filteredWidths.T
        bootIntensities[i] = filteredIntensities.T
    
    # Convert back to format of bootstrap samples
    filteredBestPars = bestPars.copy()
    filteredBestPars[:, :, 1::3] = bootWidths
    filteredBestPars[:, :, 0::3] = bootIntensities
    return filteredBestPars


# def plotHists(ax, samples, disableCI=False, disableLeg=False):
#     """Plots each row of 2D samples array."""

#     # ax.set_title(f"Histogram of {title}")
#     for i, bootHist in enumerate(samples):

#         if np.all(bootHist==0) or np.all(np.isnan(bootHist)):
#             continue
        
#         mean = np.nanmean(bootHist)
#         bounds = np.percentile(bootHist, [5, 95])
#         errors = bounds - mean
#         leg = f"Row {i}: {mean:>6.3f} +{errors[1]:.3f} {errors[0]:.3f}"

#         ax.hist(bootHist, histtype="step", label=leg)

#         ax.axvline(mean, 0.9, 0.97, color="k", ls="--", alpha=0.4)
        
#         if disableCI:
#             pass
#         else:
#             ax.axvspan(bounds[0], bounds[1], alpha=0.2, color="r")
    
#     if disableLeg:
#         pass
#     else:
#         ax.legend(loc="upper center")


def plotMeansOverNoSamples(ax, bootMeans):

    nSamples = len(bootMeans[0])
    assert nSamples >= 10, "To plot evolution of means, need at least 10 samples!"
    noOfPoints = int(nSamples / 10)
    sampleSizes = np.linspace(10, nSamples, noOfPoints).astype(int)

    sampleMeans = np.zeros((len(bootMeans), len(sampleSizes)))
    sampleErrors = np.zeros((len(bootMeans), 2, len(sampleSizes)))

    for i, N in enumerate(sampleSizes):
        subSample = bootMeans[:, :N]

        mean = np.mean(subSample, axis=1)

        bounds = np.percentile(subSample, [5, 95], axis=1).T
        assert bounds.shape == (len(subSample), 2), f"Wrong shape: {bounds.shape}"
        errors = bounds - mean[:, np.newaxis]

        sampleMeans[:, i] = mean
        sampleErrors[:, :, i] = errors

    firstValues = sampleMeans[:, 0][:, np.newaxis]
    meansRelDiff = (sampleMeans - firstValues) / firstValues
    
    errorsRel = sampleErrors / sampleMeans[:, np.newaxis, :]

    for i, (means, errors) in enumerate(zip(meansRelDiff, errorsRel)):
        ax.plot(sampleSizes, means, label=f"idx {i}")
        ax.fill_between(sampleSizes, errors[0, :], errors[1, :], alpha=0.1)
    
    ax.legend()


# def plotHistsAndMeanHists(sampleHists, meanHist, ax):

#     # firstIdx = specRange[0]
#     # lastIdx = specRange[1]
#     # samples = bootSamples[:, firstIdx:lastIdx, idx].T
#     # samples, meanSamples = selectRawSamplesPerIdx(bootSamples, idx)
#     # nBins = int(len(samples[0] / 10))

#     plotHists(ax, sampleHists, disableCI=True, disableLeg=True)
#     # meanSamples = np.nanmean(samples, axis=0).flatten()
#     ax.hist(meanHist, color="r", histtype="step", linewidth=2)
#     ax.axvline(np.nanmean(meanHist), 0.9, 0.97, color="r", ls="--", linewidth=2)
#     # ax.axvline(np.percentile(meanSamples, 0.5), 0.9, 0.97, color="r", ls="--", linewidth=2)


# def selectRawSamplesPerIdx(bootSamples, idx):
#     """
#     Returns samples and mean for a single width or intensity at a given index.
#     Samples shape: (No of spectra, No of boot samples)
#     Samples mean: Mean of spectra (mean of histograms)
#     """
#     samples = bootSamples[:, :, idx].T
#     return samples


def calcCorrWithScatAngle(samples, IC):
    """Calculate correlation coefficient between histogram means and scattering angle."""
    
    ipMatrix = np.loadtxt(IC.instrParsPath, dtype=str)[1:].astype(float)
    
    firstSpec, lastSpec = IC.bootSavePath.name.split("_")[1].split("-")
    allSpec = ipMatrix[:, 0]
    selectedSpec = (allSpec>=int(firstSpec)) & (allSpec>=int(lastSpec))
    
    thetas = ipMatrix[selectedSpec, 2]  # Scattering angle on third column

    # thetas = ipMatrix[firstIdx : lastIdx, 2]    # Scattering angle on third column
    assert thetas.shape == (len(samples),), f"Wrong shape: {thetas.shape}"

    histMeans = np.nanmean(samples, axis=1) 

    # Remove masked spectra:
    nanMask = np.isnan(histMeans)
    histMeans = histMeans[~nanMask]
    thetas = thetas[~nanMask]

    corr = stats.pearsonr(thetas, histMeans)
    return corr


def checkBootSamplesVSParent(bestPars, parentPars):
    """
    For an unbiased estimator, the mean of the bootstrap samples will converge to 
    the mean of the experimental sample (here called parent).
    """

    bootWidths = bestPars[:, :, 1::3]
    bootIntensities = bestPars[:, :, 0::3]

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
    

def plot2DHists(bootSamples, mode):
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

                axs[i, j].hist(bootSamples[i], orientation=orientation)

            else:
                axs[i, j].hist2d(bootSamples[j], bootSamples[i])
                
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


# def addParentMeans(ax, means):
#     for mean in means:
#         ax.axvline(mean, 0, 0.97, color="k", ls=":")
        

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

def runAnalysisOfStoredBootstrap(bckwdIC, fwdIC, bootIC, analysisIC):


    setOutputDirs([bckwdIC, fwdIC], bootIC)
    
    #
    for IC in [bckwdIC, fwdIC]:
        if not(IC.bootSavePath.is_file()):
            print("Bootstrap data files not found, unable to run analysis!")
            print(f"{IC.bootSavePath.name}")
            continue       # If files are not found for backward, look for forward
        
        bootData = np.load(IC.bootSavePath)

        bootParsRaw = bootData["boot_samples"][:, :, 1:-2]
        parentParsRaw = bootData["parent_result"][:, 1:-2]
        
        nSamples = len(bootParsRaw)

        print(f"\nData files found:\n{IC.bootSavePath.name}")
        print(f"\nNumber of samples in the file: {nSamples}")
        assert ~np.all(bootParsRaw[-1] == parentParsRaw), "Error in Jackknife due to last column."

        checkBootSamplesVSParent(bootParsRaw, parentParsRaw)    # Prints comparison

        bootPars = bootParsRaw.copy()      # By default, do not filter means
        if analysisIC.filterAvg:
            bootPars = filteredBootMeans(bootParsRaw.copy())
        
        if analysisIC.plotRawWidthsIntensities:

            fig, axs = plt.subplots(2, len(IC.masses))
            for i, j in enumerate(range(1, 3*len(IC.masses), 3)):
                axs[0, i].set_title(f"Width {i}")
                idxSamples = selectRawSamplesPerIdx(bootPars, j)
                plotHists(axs[0, i], idxSamples, disableCI=True, disableLeg=True)

            for i, j in enumerate(range(0, 3*len(IC.masses), 3)):
                axs[1, i].set_title(f"Intensity {i}")
                idxSamples = selectRawSamplesPerIdx(bootPars, j)
                plotHists(axs[1, i], idxSamples, disableCI=True, disableLeg=True)

            plt.show()


        # # Modify so that the filtering doesn't happen by default
        # meanWp, meanIp, stdWp, stdIp = calcMeansWithOriginalProc(parentPars[np.newaxis, :, :])
        # meanWp = meanWp.flatten()
        # meanIp = meanIp.flatten()
        
        meanWidths = np.zeros((len(IC.masses), nSamples))
        for i, j in enumerate(range(1, 3*len(IC.masses), 3)):
            idxSamples = selectRawSamplesPerIdx(bootPars, j)
            meanWidths[i] = np.nanmean(idxSamples, axis=0)

        meanIntensities = np.zeros((len(IC.masses), nSamples))
        for i, j in enumerate(range(0, 3*len(IC.masses), 3)):
            idxSamples = selectRawSamplesPerIdx(bootPars, j)
            meanIntensities[i] = np.nanmean(idxSamples, axis=0)

        if analysisIC.filterAvg == True:     # Check that treatment of data is 
            meanWOri, meanIOri = calcMeansWithOriginalProc(bootParsRaw)
            np.testing.assert_array_almost_equal(meanWOri, meanWidths)
            np.testing.assert_array_almost_equal(meanIOri, meanIntensities)
            
        print("\n\n Test passed! Mean Widths match!")

        fig, axs = plt.subplots(2, 1)
        axs[0].set_title("Histograms of mean Widths")
        plotHists(axs[0], meanWidths, disableAvg=True, disableCI=True)

        axs[1].set_title("Histograms of mean Intensitiess")
        plotHists(axs[1], meanIntensities, disableAvg=True, disableCI=True)
        
        plt.show()

        if analysisIC.plotMeansEvolution:
            fig, axs = plt.subplots(2, 1)
            axs[0].set_title("Evolution of mean Widths over sample size")
            plotMeansOverNoSamples(axs[0], meanWidths)

            axs[1].set_title("Evolution of mean Intensities over sample size")
            plotMeansOverNoSamples(axs[1], meanIntensities)

            plt.show()

        if analysisIC.plot2DHists:
            plot2DHists(meanWidths, "Widths")    
            plot2DHists(meanIntensities, "Intensities")   
                    
        # if analysisIC.plotYFitHists:

# if ySpaceFit:
#     bootYFitData = np.load(dataYFitPath)
#     bootYFitVals = bootYFitData["boot_vals"]    # Previously boot_samples
#     mFitVals = bootYFitVals[:, 0, :-1].T  # Last value is chi

#     # Plot each parameter in an individual histogram
#     fig, axs = plt.subplots(len(mFitVals), 1, figsize=(8, 10))
#     for i, (ax, hist) in enumerate(zip(axs.flatten(), mFitVals)):
#         plotHists(ax, hist[np.newaxis, :])
#     plt.show()


# plot2DHists(meanW, nBins, "Widths")    
# plot2DHists(meanI, nBins, "Intensities")   

def plotHists(ax, samples, disableCI=False, disableLeg=False, disableAvg=False):
    """Plots each row of 2D samples array as a histogram."""

    # ax.set_title(f"Histogram of {title}")
    for i, bootHist in enumerate(samples):

        if np.all(bootHist==0) or np.all(np.isnan(bootHist)):
            continue
        
        mean = np.nanmean(bootHist)
        bounds = np.percentile(bootHist, [5, 95])
        errors = bounds - mean
        leg = f"Row {i}: {mean:>6.3f} +{errors[1]:.3f} {errors[0]:.3f}"

        ax.hist(bootHist, histtype="step", label=leg)
        ax.axvline(mean, 0.9, 0.97, color="k", ls="--", alpha=0.4)
        
        if not(disableCI):
            ax.axvspan(bounds[0], bounds[1], alpha=0.1, color="r")
    
    if not(disableAvg):
        # Plot average over histograms
        avgHist = np.nanmean(samples, axis=0).flatten()
        ax.hist(avgHist, color="r", histtype="step", linewidth=2)
        ax.axvline(np.nanmean(avgHist), 0.9, 0.97, color="r", ls="--", linewidth=2)
    
    if not(disableLeg):
        ax.legend(loc="upper center")


def selectRawSamplesPerIdx(bootSamples, idx):
    """
    Returns samples and mean for a single width or intensity at a given index.
    Samples shape: (No of spectra, No of boot samples)
    Samples mean: Mean of spectra (mean of histograms)
    """
    samples = bootSamples[:, :, idx].T
    return samples

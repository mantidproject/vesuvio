
from email.policy import default
import numpy as np
import matplotlib .pyplot as plt
from pathlib import Path
from scipy import stats
from vesuvio_analysis.core_functions.analysis_functions import calculateMeansAndStds, filterWidthsAndIntensities
from vesuvio_analysis.core_functions.ICHelpers import setBootstrapDirs
from vesuvio_analysis.core_functions.fit_in_yspace import selectModelAndPars

currentPath = Path(__file__).parent.absolute() 
experimentsPath = currentPath / ".." / ".. " / "experiments"
IPFilesPath = currentPath / ".." / "ip_files" 


def runAnalysisOfStoredBootstrap(bckwdIC, fwdIC, yFitIC, bootIC, analysisIC, userCtr):

    if not(analysisIC.runAnalysis):
        return

    setBootstrapDirs([bckwdIC, fwdIC], bootIC, userCtr)   # Same function used to store data, to check below if dirs exist

    for IC in [bckwdIC, fwdIC]:

        if not(IC.bootSavePath.is_file()):
            print("Bootstrap data files not found, unable to run analysis!")
            print(f"{IC.bootSavePath.name}")
            continue    # If main results are not present, assume ysapce results are also missing

        bootParsRaw, parentParsRaw, nSamples, corrResiduals = readBootData(IC.bootSavePath)
        checkResiduals(corrResiduals)
        checkBootSamplesVSParent(bootParsRaw, parentParsRaw, IC)    # Prints comparison

        bootPars = bootParsRaw.copy()      # By default do not filter means, copy to avoid accidental changes
        if analysisIC.filterAvg:
            bootPars = filteredBootMeans(bootParsRaw.copy(), IC)
        
        # Plots histograms of all spectra for a given width or intensity
        plotRawWidthsAndIntensities(analysisIC, IC, bootPars, parentParsRaw)
        
        # Calculate bootstrap histograms for mean widths and intensities 
        meanWidths, meanIntensities = calculateMeanWidthsIntensities(bootPars, IC, nSamples)

        # If filer is on, check that it matches original procedure
        checkMeansProcedure(analysisIC, IC, meanWidths, meanIntensities, bootParsRaw)

        plotMeanWidthsAndIntensities(analysisIC, IC, meanWidths, meanIntensities, parentParsRaw)
        plotMeansEvolution(analysisIC, meanWidths, meanIntensities)
        plot2DHistsWidthsAndIntensities(analysisIC, meanWidths, meanIntensities)


        if not(IC.bootYFitSavePath.is_file()):
            print("Bootstrap data file for y-space fit not found, unable to run analysis!")
            print(f"{IC.bootYFitSavePath.name}")
            continue

        fitIdx = 0   # 0 for Minuit, 1 for LM

        bootYFitData = np.load(IC.bootYFitSavePath)
        try:
            bootYFitVals = bootYFitData["boot_samples"]   
        except KeyError:
            bootYFitVals = bootYFitData["boot_vals"]      # To account for some previous samples
        minuitFitVals = bootYFitVals[:, fitIdx, :-1].T   # Select Minuit values and Discard last value chi2
        
        try:
            parentPopt = bootYFitData["parent_popt"][fitIdx]
            parentPerr = bootYFitData["parent_perr"][fitIdx]
            printYFitParentPars(yFitIC, parentPopt, parentPerr)  # TODO: Test this function
        except KeyError:
            pass

        plotMeansEvolutionYFit(analysisIC, minuitFitVals)
        plotYFitHists(analysisIC, yFitIC, minuitFitVals)
        plot2DHistsYFit(analysisIC, minuitFitVals)

def readBootData(dataPath):
        bootData = np.load(dataPath)

        bootParsRaw = bootData["boot_samples"][:, :, 1:-2]
        parentParsRaw = bootData["parent_result"][:, 1:-2]
        nSamples = len(bootParsRaw)
        try:
            corrResiduals = bootData["corr_residuals"]
        except KeyError:
            corrResiduals = np.array([np.nan]) 
            print("\nCorrelation of coefficients not found!\n")


        print(f"\nData files found:\n{dataPath.name}")
        print(f"\nNumber of samples in the file: {nSamples}")
        assert ~np.all(bootParsRaw[-1] == parentParsRaw), "Error in Jackknife due to last column."
        return bootParsRaw, parentParsRaw, nSamples, corrResiduals


def checkResiduals(corrRes):
    if np.all(np.isnan(corrRes)):
        return
    
    corrCoef = corrRes[:, 0]
    nCorrelated = np.sum(corrCoef>0.5)
    print(f"\nNumber of spectra with pearson r > 0.5: {nCorrelated}")
    return


def checkBootSamplesVSParent(bestPars, parentPars, IC):
    """
    For an unbiased estimator, the mean of the bootstrap samples will converge to 
    the mean of the experimental sample (here called parent).
    """

    bootWidths = bestPars[:, :, 1::3]
    bootIntensities = bestPars[:, :, 0::3]

    meanBootWidths = np.mean(bootWidths, axis=0)
    meanBootIntensities = np.mean(bootIntensities, axis=0)

    avgWidths, stdWidths, avgInt, stdInt = calculateMeansAndStds(meanBootWidths.T, meanBootIntensities.T, IC)

    parentWidths = parentPars[:, 1::3]
    parentIntensities = parentPars[:, 0::3]

    avgWidthsP, stdWidthsP, avgIntP, stdIntP = calculateMeansAndStds(parentWidths.T, parentIntensities.T, IC)
  
    print("\nComparing Bootstrap means with parent means:\n")
    printResults(avgWidths, stdWidths, "Boot Widths")
    printResults(avgWidthsP, stdWidthsP, "Parent Widths")
    printResults(avgInt, stdInt, "Boot Intensities")
    printResults(avgIntP, stdIntP, "Parent Intensities")


def printResults(arrM, arrE, mode):
    print(f"\n{mode}:\n")
    for i, (m, e) in enumerate(zip(arrM, arrE)):
        print(f"{mode} {i}: {m:>6.3f} \u00B1 {e:<6.3f}")


def filteredBootMeans(bestPars, IC):  # Pass IC just to check flag for preliminary procedure
    """Use same filtering function used on original procedure"""

    # Extract Widths and Intensities from bootstrap samples
    bootWidths = bestPars[:, :, 1::3]
    bootIntensities = bestPars[:, :, 0::3]

    # Perform the filter
    for i, (widths, intensities) in enumerate(zip(bootWidths, bootIntensities)):
        filteredWidths, filteredIntensities = filterWidthsAndIntensities(widths.T, intensities.T, IC)
        
        bootWidths[i] = filteredWidths.T
        bootIntensities[i] = filteredIntensities.T
    
    # Convert back to format of bootstrap samples
    filteredBestPars = bestPars.copy()
    filteredBestPars[:, :, 1::3] = bootWidths
    filteredBestPars[:, :, 0::3] = bootIntensities
    return filteredBestPars


def plotRawWidthsAndIntensities(analysisIC, IC, bootPars, parentPars):
    """
    Plots histograms of each width and intensity seperatly.
    Plots histogram of means over spectra for each width or intensity.
    """

    if not(analysisIC.plotRawWidthsIntensities):
        return

    parentWidths, parentIntensities = extractParentMeans(parentPars, IC)
    noOfMasses = len(parentWidths)

    fig, axs = plt.subplots(2, noOfMasses)

    for axIdx, startIdx, kind, parentMeans in zip([0, 1], [1, 0], ["Width", "Intensity"], [parentWidths, parentIntensities]):

        for i, j in enumerate(range(startIdx, 3*noOfMasses, 3)):
            axs[axIdx, i].set_title(f"{kind} {i}")
            idxSamples = selectRawSamplesPerIdx(bootPars, j)
            plotHists(axs[axIdx, i], idxSamples, disableCI=True, disableLeg=True)
            axs[axIdx, i].axvline(parentMeans[i], 0.75, 0.97, color="b", ls="-", alpha=0.4)
            
    plt.show()
    return


def extractParentMeans(parentPars, IC):
    """Uses original treatment of widths and intensities to calculate parent means."""
    # Modify so that the filtering doesn't happen by default
    meanWp, meanIp = calcMeansWithOriginalProc(parentPars[np.newaxis, :, :], IC)
    meanWp = meanWp.flatten()
    meanIp = meanIp.flatten()
    return meanWp, meanIp


def calcMeansWithOriginalProc(bestPars, IC):
    """Performs the means and std on each bootstrap sample according to original procedure"""
    
    bootWidths = bestPars[:, :, 1::3]
    bootIntensities = bestPars[:, :, 0::3]

    bootMeanW = np.zeros((len(bootWidths[0,0,:]), len(bootWidths)))
    bootStdW = np.zeros(bootMeanW.shape)
    bootMeanI = np.zeros(bootMeanW.shape)
    bootStdI = np.zeros(bootMeanW.shape)

    for j, (widths, intensities) in enumerate(zip(bootWidths, bootIntensities)):

        meanW, stdW, meanI, stdI = calculateMeansAndStds(widths.T, intensities.T, IC)

        bootMeanW[:, j] = meanW       # Interested only in the means 
        bootMeanI[:, j] = meanI

    return bootMeanW, bootMeanI


def calculateMeanWidthsIntensities(bootPars, IC, nSamples):
    """
    Calculates means for each Bootstrap sample.
    Returns means with size equal to the number of Boot samples.
    """

    # Calculate bootstrap histograms for mean widths and intensities 
    meanWidths = np.zeros((len(IC.masses), nSamples))
    for i, j in enumerate(range(1, 3*len(IC.masses), 3)):
        idxSamples = selectRawSamplesPerIdx(bootPars, j)
        meanWidths[i, :] = np.nanmean(idxSamples, axis=0)

    meanIntensities = np.zeros((len(IC.masses), nSamples))
    for i, j in enumerate(range(0, 3*len(IC.masses), 3)):
        idxSamples = selectRawSamplesPerIdx(bootPars, j)
        meanIntensities[i, :] = np.nanmean(idxSamples, axis=0)   

    return meanWidths, meanIntensities


def checkMeansProcedure(analysisIC, IC, meanWidths, meanIntensities, bootParsRaw):
    """Checks that filtering and averaging of Bootstrap samples follows the original procedure"""
    
    if not(analysisIC.filterAvg):     # When filtering not present, no comparison available
        return
    
    else:         # Check that treatment of data matches original
        meanWOri, meanIOri = calcMeansWithOriginalProc(bootParsRaw, IC)
        np.testing.assert_array_almost_equal(meanWOri, meanWidths)
        np.testing.assert_array_almost_equal(meanIOri, meanIntensities)
        return
       

def plotMeanWidthsAndIntensities(analysisIC, IC, meanWidths, meanIntensities, parentParsRaw):
    """
    Most informative histograms, shows all mean widhts and intensities of Bootstrap samples
    """

    if not(analysisIC.plotMeanWidthsIntensities):
        return

    parentWidths, parentIntensities = extractParentMeans(parentParsRaw, IC)


    print("\n\n Test passed! Mean Widths match!")

    fig, axs = plt.subplots(2, 1)
    axs[0].set_title("Histograms of mean Widths")
    axs[1].set_title("Histograms of mean Intensitiess")

    for ax, means, parentMeans in zip(axs.flatten(), [meanWidths, meanIntensities], [parentWidths, parentIntensities]):
        plotHists(ax, means, disableAvg=True, disableCI=True)
        for pMean in parentMeans:
            ax.axvline(pMean, 0.75, 0.97, color="b", ls="-", alpha=0.4)

    plt.show()
    return


def plotMeansEvolution(IC, meanWidths, meanIntensities):
    """Shows how results of Bootstrap change depending on number of bootstrap samples"""
    
    if not(IC.plotMeansEvolution):
        return

    fig, axs = plt.subplots(2, 1)
    axs[0].set_title("Evolution of mean Widths")
    plotMeansOverNoSamples(axs[0], meanWidths)

    axs[1].set_title("Evolution of mean Intensities")
    plotMeansOverNoSamples(axs[1], meanIntensities)

    plt.show()
    return


    
def plotMeansEvolutionYFit(analysisIC, minuitFitVals):

    if not(analysisIC.plotMeansEvolution):
        return
    
    fig, ax = plt.subplots()
    ax.set_title("Evolution of y-space fit parameters")
    plotMeansOverNoSamples(ax, minuitFitVals)

    plt.show()
    return


def plotMeansOverNoSamples(ax, bootMeans):

    nSamples = len(bootMeans[0])
    assert nSamples >= 10, "To plot evolution of means, need at least 10 samples!"
    noOfPoints = int(nSamples / 10)
    sampleSizes = np.linspace(10, nSamples, noOfPoints).astype(int)

    sampleMeans = np.zeros((len(bootMeans), len(sampleSizes)))
    sampleErrors = np.zeros((len(bootMeans), 2, len(sampleSizes)))

    for i, N in enumerate(sampleSizes):
        subSample = bootMeans[:, :N].copy()

        mean = np.mean(subSample, axis=1)

        bounds = np.percentile(subSample, [5, 95], axis=1).T
        assert bounds.shape == (len(subSample), 2), f"Wrong shape: {bounds.shape}"
        errors = bounds - mean[:, np.newaxis]

        sampleMeans[:, i] = mean
        sampleErrors[:, :, i] = errors

    firstValues = sampleMeans[:, 0][:, np.newaxis]
    meansRelDiff = (sampleMeans - firstValues) / firstValues * 100  # %
    
    errorsRel = sampleErrors / sampleMeans[:, np.newaxis, :] * 100  # %

    for i, (means, errors) in enumerate(zip(meansRelDiff, errorsRel)):
        ax.plot(sampleSizes, means, label=f"idx {i}")
        ax.fill_between(sampleSizes, errors[0, :], errors[1, :], alpha=0.1)
    
    ax.legend()
    ax.set_xlabel("Number of Bootstrap samples")
    ax.set_ylabel("Percent change (%)")


def plot2DHistsWidthsAndIntensities(IC, meanWidths, meanIntensities):

    if not(IC.plot2DHists):
        return

    assert meanWidths.shape == meanIntensities.shape, "Widths and Intensities need to be the same shape."
    
    plot2DHists(meanWidths, "Widths")    
    plot2DHists(meanIntensities, "Intensities")   
    return


def plot2DHists(bootSamples, mode):
    """bootSamples has histogram rows for each parameter"""

    plotSize = len(bootSamples)
    fig, axs = plt.subplots(plotSize, plotSize, figsize=(6, 10), tight_layout=True)

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


def printYFitParentPars(yFitIC, parentPopt, parentPerr):

    model, defaultPars, sharedPars = selectModelAndPars(yFitIC.fitModel)

    print("\nParent parameters of y-sapce fit:\n")
    for p, m, e in zip(defaultPars, parentPopt, parentPerr):
        print(f"{p:5s}:  {m:8.3f} +/- {e:8.3f}")



def plotYFitHists(analysisIC, yFitIC, yFitHists):
    """Histogram for each parameter of model used for fit in y-space."""

    if not(analysisIC.plotYFitHists):
        return

    # Plot each parameter in an individual histogram
    fig, axs = plt.subplots(2, int(np.ceil(len(yFitHists)/2)), figsize=(12, 7), tight_layout=True)

    # To label each histogram, extract signature of function used for the fit
    model, defaultPars, sharedPars = selectModelAndPars(yFitIC.fitModel)

    for i, (ax, hist, par) in enumerate(zip(axs.flatten(), yFitHists, defaultPars)):
        ax.set_title(f"Fit Parameter: {par}")
        plotHists(ax, hist[np.newaxis, :], disableAvg=True)
    
    # Hide plots not in use:
    for ax in axs.flat:
        if not ax.lines:   # If empty list
            ax.set_visible(False)

    plt.show()
    return


def plot2DHistsYFit(analysisIC, minuitFitVals):

    if not(analysisIC.plot2DHists):
        return
    
    plot2DHists(minuitFitVals, "Y-space Fit parameters")
    return


def plotHists(ax, samples, disableCI=False, disableLeg=False, disableAvg=False):
    """Plots each row of 2D samples array as a histogram."""

    # ax.set_title(f"Histogram of {title}")
    for i, bootHist in enumerate(samples):

        if np.all(bootHist==0) or np.all(np.isnan(bootHist)):
            continue
        
        mean = np.nanmean(bootHist)
        bounds = np.percentile(bootHist, [16, 68+16])   # 1 std: 68%, 2 std: 95%
        errors = bounds - mean
        leg = f"Row {i}: {mean:>6.3f} +{errors[1]:.3f} {errors[0]:.3f}"

        ax.hist(bootHist, histtype="step", label=leg, linewidth=1)
        ax.axvline(mean, 0.9, 0.97, color="k", ls="--", alpha=0.4)
        
        if not(disableCI):
            ax.axvspan(bounds[0], bounds[1], alpha=0.1, color="b")
    
    if not(disableAvg):
        # Plot average over histograms
        avgHist = np.nanmean(samples, axis=0).flatten()
        ax.hist(avgHist, color="r", histtype="step", linewidth=2)
        ax.axvline(np.nanmean(avgHist), 0.75, 0.97, color="r", ls="--", linewidth=2)
    
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


# TODO: Make use of this function?
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
import numpy as np
from mantid.simpleapi import *
from scipy import optimize
from functools import partial

# Format print output of arrays
np.set_printoptions(suppress=True, precision=4, linewidth=100, threshold=sys.maxsize)


def iterativeFitForDataReduction(ic):
    printInitialParameters(ic)

    initialWs = loadVesuvioDataWorkspaces(ic)   
    cropedWs = cropAndMaskWorkspace(ic, initialWs)
    wsToBeFitted = CloneWorkspace(InputWorkspace=cropedWs, OutputWorkspace=cropedWs.name()+"0")
   
    createSlabGeometry(ic)

    # Initialize arrays to store script results
    fittingResults = resultsObject(ic)

    for iteration in range(ic.noOfMSIterations):
        # Workspace from previous iteration
        wsToBeFitted = mtd[ic.name+str(iteration)]
        SumSpectra(InputWorkspace=wsToBeFitted, OutputWorkspace=wsToBeFitted.name()+"_sum")

        print("\nFitting spectra ", ic.firstSpec, " - ", ic.lastSpec, "\n.............")
        
        # The fitting procedure stores results in the fittingResults object
        fitNcpToWorkspace(ic, wsToBeFitted, fittingResults)
        
        fittingResults.calculateMeansAndStd()
        fittingResults.printCurrentIrerationResults()

        # When last iteration, skip MS and GC
        if iteration == ic.noOfMSIterations - 1:
          break 

        meanWidths = fittingResults.all_mean_widths[iteration]
        meanIntensityRatios = fittingResults.all_mean_intensities[iteration]
        
        CloneWorkspace(InputWorkspace=ic.name, OutputWorkspace="tmpNameWs")

        if ic.MSCorrectionFlag:
            createWorkspacesForMSCorrection(ic, meanWidths, meanIntensityRatios)
            Minus(LHSWorkspace="tmpNameWs", RHSWorkspace=ic.name+"_MulScattering",
                    OutputWorkspace="tmpNameWs")

        if ic.GammaCorrectionFlag:  
            createWorkspacesForGammaCorrection(ic, meanWidths, meanIntensityRatios)
            Minus(LHSWorkspace="tmpNameWs", RHSWorkspace=ic.name+"_gamma_background", 
                    OutputWorkspace="tmpNameWs")

        RenameWorkspace(InputWorkspace="tmpNameWs", OutputWorkspace=ic.name+str(iteration+1))

    wsFinal = mtd[ic.name+str(ic.noOfMSIterations - 1)]
    fittingResults.save()
    return wsFinal, fittingResults


def printInitialParameters(ic):
    print("\nRUNNING ", ic.modeRunning, " SCATTERING.")
    if ic.modeRunning == "BACKWARD":
        print("\n\nH to first mass ratio: ", ic.HToMass0Ratio)
    print("\n\nInitial fitting parameters:\n", 
            ic.initPars.reshape((ic.masses.size, 3)),
            "\n\nInitial fitting bounds:\n", 
            ic.bounds, "\n")


def loadVesuvioDataWorkspaces(ic):
    """Loads raw and empty workspaces from either LoadVesuvio or user specified path"""
    # if ic.loadWsFromUserPathFlag:
    wsToBeFitted =  loadRawAndEmptyWsFromUserPath(ic)
    # else:
    #     wsToBeFitted = loadRawAndEmptyWsFromLoadVesuvio(ic)
    return wsToBeFitted


def loadRawAndEmptyWsFromUserPath(ic):

    print('\n', 'Loading the sample runs: ', ic.runs, '\n')
    Load(Filename=ic.userWsRawPath, OutputWorkspace=ic.name+"raw")
    Rebin(InputWorkspace=ic.name+'raw', Params=ic.tof_binning,
          OutputWorkspace=ic.name+'raw')
    SumSpectra(InputWorkspace=ic.name+'raw', OutputWorkspace=ic.name+'raw'+'_sum')
    wsToBeFitted = CloneWorkspace(InputWorkspace=ic.name+'raw', OutputWorkspace=ic.name+"uncroped_unmasked")

    if ic.mode=="DoubleDifference":
        print('\n', 'Loading the empty runs: ', ic.empty_runs, '\n')
        Load(Filename=ic.userWsEmptyPath, OutputWorkspace=ic.name+"empty")
        Rebin(InputWorkspace=ic.name+'empty', Params=ic.tof_binning,
            OutputWorkspace=ic.name+'empty')
        wsToBeFitted = Minus(LHSWorkspace=ic.name+'raw', RHSWorkspace=ic.name+'empty',
                            OutputWorkspace=ic.name+"uncroped_unmasked")
    return wsToBeFitted


# def loadRawAndEmptyWsFromLoadVesuvio(ic):
    
#     print('\n', 'Loading the sample runs: ', ic.runs, '\n')
#     LoadVesuvio(Filename=ic.runs, SpectrumList=ic.spectra, Mode=ic.mode,
#                 InstrumentParFile=ic.ipfile, OutputWorkspace=ic.name+'raw')
#     Rebin(InputWorkspace=ic.name+'raw', Params=ic.tof_binning,
#           OutputWorkspace=ic.name+'raw')
#     SumSpectra(InputWorkspace=ic.name+'raw', OutputWorkspace=ic.name+'raw'+'_sum')
#     wsToBeFitted = CloneWorkspace(InputWorkspace=ic.name+'raw', OutputWorkspace=ic.name+"uncroped_unmasked")

#     if ic.mode=="DoubleDifference":
#         print('\n', 'Loading the empty runs: ', ic.empty_runs, '\n')
#         LoadVesuvio(Filename=ic.empty_runs, SpectrumList=ic.spectra, Mode=ic.mode,
#                     InstrumentParFile=ic.ipfile, OutputWorkspace=ic.name+'empty')
#         Rebin(InputWorkspace=ic.name+'empty', Params=ic.tof_binning,
#             OutputWorkspace=ic.name+'empty')
#         wsToBeFitted = Minus(LHSWorkspace=ic.name+'raw', RHSWorkspace=ic.name+'empty', 
#                             OutputWorkspace=ic.name+"uncroped_unmasked")
#     return wsToBeFitted


def cropAndMaskWorkspace(ic, ws):
    """Returns cloned and cropped workspace with modified name"""
    # Read initial Spectrum number
    wsFirstSpec = ws.getSpectrumNumbers()[0]
    assert ic.firstSpec >= wsFirstSpec, "Can't crop workspace, firstSpec < first spectrum in workspace."
    
    initialIdx = ic.firstSpec - wsFirstSpec
    lastIdx = ic.lastSpec - wsFirstSpec
    
    newWsName = ws.name().split("uncroped")[0]  # Retrieve original name
    cropedWs = CropWorkspace(
        InputWorkspace=ws, 
        StartWorkspaceIndex=initialIdx, EndWorkspaceIndex=lastIdx, 
        OutputWorkspace=newWsName
        )
    MaskDetectors(Workspace=cropedWs, WorkspaceIndexList=ic.maskedDetectorIdx)
    return cropedWs


def createSlabGeometry(ic):
    half_height, half_width, half_thick = 0.5*ic.vertical_width, 0.5*ic.horizontal_width, 0.5*ic.thickness
    xml_str = \
        " <cuboid id=\"sample-shape\"> " \
        + "<left-front-bottom-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (half_width, -half_height, half_thick) \
        + "<left-front-top-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (half_width, half_height, half_thick) \
        + "<left-back-bottom-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (half_width, -half_height, -half_thick) \
        + "<right-front-bottom-point x=\"%f\" y=\"%f\" z=\"%f\" /> " % (-half_width, -half_height, half_thick) \
        + "</cuboid>"
    CreateSampleShape(ic.name, xml_str)


def fitNcpToWorkspace(ic, ws, fittingResults):
    """Firstly calculates matrices for all spectrums,
    then iterates over each spectrum
    """
    wsDataY = ws.extractY()       #DataY unaltered
    fittingResults.addDataY(wsDataY)

    dataY, dataX, dataE = loadWorkspaceIntoArrays(ws)                     
    resolutionPars, instrPars, kinematicArrays, ySpacesForEachMass = prepareFitArgs(ic, dataX)
    
    # Fit all spectrums
    fitPars = np.array(list(map(
        partial(fitNcpToSingleSpec, ic=ic), 
        dataY, dataE, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays
    )))
    fittingResults.addFitPars(fitPars)

    # Create ncpTotal workspaces
    mainPars = fittingResults.getMainPars()
    ncpForEachMass = np.array(list(map(
        partial(buildNcpFromSpec, ic=ic), 
        mainPars , ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays
    )))
    ncpTotal = np.sum(ncpForEachMass, axis=1)  
    createNcpWorkspaces(ncpForEachMass, ncpTotal, ws)  
    # Adds individual and total ncp
    fittingResults.addNCP(ncpForEachMass, ncpTotal)

 
class resultsObject:
    """Used to store fitting results of the script at each iteration"""

    def __init__(self, ic):
        """Initializes arrays full of zeros"""

        self.all_fit_workspaces = None 
        self.all_spec_best_par_chi_nit = None 
        self.all_tot_ncp = None 
        self.all_ncp_for_each_mass = None
        
        self.all_mean_widths = None
        self.all_mean_intensities = None 
        self.all_std_widths = None 
        self.all_std_intensities = None 

        # Pass all attributes of ic into attributes to be used whithin this object
        self.maskedDetectorIdx = ic.maskedDetectorIdx
        self.masses = ic.masses
        self.noOfMasses = ic.noOfMasses
        self.resultsSavePath = ic.resultsSavePath


    def addDataY(self, dataY):
        if self.all_fit_workspaces is None:
            self.all_fit_workspaces = dataY[np.newaxis, :, :]
        else:
            self.all_fit_workspaces = np.append(self.all_fit_workspaces, dataY[np.newaxis, :, :], axis=0)
        # self.all_fit_workspaces[MSiter] = dataY
    

    def addFitPars(self, fitPars):
        if self.all_spec_best_par_chi_nit is None:
            self.all_spec_best_par_chi_nit = fitPars[np.newaxis, :, :]
        else:
            self.all_spec_best_par_chi_nit = np.append(self.all_spec_best_par_chi_nit, fitPars[np.newaxis, :, :], axis=0)
       
 
    def getMainPars(self):
        # Since only want a reading, use copy() not to risk changing the array
        return self.all_spec_best_par_chi_nit[-1][:, 1:-2].copy()
    

    def addNCP(self, ncpForEachMass, ncpTotal):
        if self.all_tot_ncp is None:
            self.all_ncp_for_each_mass = ncpForEachMass[np.newaxis, :, :, :]
            self.all_tot_ncp = ncpTotal[np.newaxis, :, :]
        else:
            self.all_ncp_for_each_mass = np.append(self.all_ncp_for_each_mass, ncpForEachMass[np.newaxis, :, :, :], axis=0)
            self.all_tot_ncp = np.append(self.all_tot_ncp, ncpTotal[np.newaxis, :, :], axis=0)


    def calculateMeansAndStd(self):
        # Copy arrays to avoid changing stored values
        # Transpose to horizontal shape
        intensities = self.all_spec_best_par_chi_nit[-1][:, 1:-2:3].copy().T
        widths = self.all_spec_best_par_chi_nit[-1][:, 2:-2:3].copy().T
        noOfMasses = self.noOfMasses

        # Replace zeros from masked spectra with nans
        widths[:, self.maskedDetectorIdx] = np.nan
        intensities[:, self.maskedDetectorIdx] = np.nan

        meanWidths = np.nanmean(widths, axis=1).reshape(noOfMasses, 1)  
        stdWidths = np.nanstd(widths, axis=1).reshape(noOfMasses, 1)

        # Subtraction row by row
        widthDeviation = np.abs(widths - meanWidths)
        # Where True, replace by nan
        betterWidths = np.where(widthDeviation > stdWidths, np.nan, widths)
        betterIntensities = np.where(widthDeviation > stdWidths, np.nan, intensities)

        meanWidths = np.nanmean(betterWidths, axis=1)  
        stdWidths = np.nanstd(betterWidths, axis=1)

        # Not nansum(), to propagate nan
        normalization = np.sum(betterIntensities, axis=0)
        intensityRatios = betterIntensities / normalization

        meanIntensityRatios = np.nanmean(intensityRatios, axis=1)
        stdIntensityRatios = np.nanstd(intensityRatios, axis=1)

        # Append in the same fashion as above
        if self.all_mean_intensities is None:
            self.all_mean_widths = meanWidths[np.newaxis, :]
            self.all_std_widths = stdWidths[np.newaxis, :]
            self.all_mean_intensities = meanIntensityRatios[np.newaxis, :]
            self.all_std_intensities = stdIntensityRatios[np.newaxis, :]
        else:
            self.all_mean_widths = np.append(self.all_mean_widths, meanWidths[np.newaxis, :], axis=0) 
            self.all_std_widths = np.append(self.all_std_widths, stdWidths[np.newaxis, :], axis=0)
            self.all_mean_intensities = np.append(self.all_mean_intensities, meanIntensityRatios[np.newaxis, :], axis=0)
            self.all_std_intensities = np.append(self.all_std_intensities, stdIntensityRatios[np.newaxis, :], axis=0)


    def printCurrentIrerationResults(self):

        print("\n\nSpec ------- Main Pars ----------- Chi Nit:\n")
        print(self.all_spec_best_par_chi_nit[-1])

        for i, mass in enumerate(self.masses):
            print(f"\nMass = {mass:.2f} amu:")
            print(f"Width:     {self.all_mean_widths[-1, i]:.3f} \u00B1 {self.all_std_widths[-1, i]:.3f} ")
            print(f"Intensity: {self.all_mean_intensities[-1, i]:.3f} \u00B1 {self.all_std_intensities[-1, i]:.3f} ")


    def save(self):
        """Saves all of the arrays stored in this object"""

        # TODO: Take out nans next time when running original results
        # Because original results were recently saved with nans, mask spectra with nans
        self.all_spec_best_par_chi_nit[:, self.maskedDetectorIdx, :] = np.nan
        self.all_ncp_for_each_mass[:, self.maskedDetectorIdx, :, :] = np.nan
        self.all_tot_ncp[:, self.maskedDetectorIdx, :] = np.nan

        savePath = self.resultsSavePath
        np.savez(savePath,
                 all_fit_workspaces=self.all_fit_workspaces,
                 all_spec_best_par_chi_nit=self.all_spec_best_par_chi_nit,
                 all_mean_widths=self.all_mean_widths,
                 all_mean_intensities=self.all_mean_intensities,
                 all_std_widths=self.all_std_widths,
                 all_std_intensities=self.all_std_intensities,
                 all_tot_ncp=self.all_tot_ncp,
                 all_ncp_for_each_mass=self.all_ncp_for_each_mass)

    

def loadWorkspaceIntoArrays(ws):
    """Output: dataY, dataX and dataE as arrays and converted to point data"""
    dataY = ws.extractY()
    dataE = ws.extractE()
    dataX = ws.extractX()

    histWidths = dataX[:, 1:] - dataX[:, :-1]
    dataY = dataY[:, :-1] / histWidths
    dataE = dataE[:, :-1] / histWidths
    dataX = (dataX[:, 1:] + dataX[:, :-1]) / 2
    return dataY, dataX, dataE


def prepareFitArgs(ic, dataX):
    instrPars = loadInstrParsFileIntoArray(ic.InstrParsPath, ic.firstSpec, ic.lastSpec)       
    resolutionPars = loadResolutionPars(instrPars)                                   

    v0, E0, delta_E, delta_Q = calculateKinematicsArrays(dataX, instrPars)   
    kinematicArrays = np.array([v0, E0, delta_E, delta_Q])
    ySpacesForEachMass = convertDataXToYSpacesForEachMass(dataX, ic.masses, delta_Q, delta_E)        
    
    kinematicArrays = reshapeArrayPerSpectrum(kinematicArrays)
    ySpacesForEachMass = reshapeArrayPerSpectrum(ySpacesForEachMass)
    return resolutionPars, instrPars, kinematicArrays, ySpacesForEachMass


def loadInstrParsFileIntoArray(InstrParsPath, firstSpec, lastSpec):
    """Loads instrument parameters into array, from the file in the specified path"""

    # For some odd reason, np.loadtxt() is only working with Path object
    # So transform string path into Path object
    # IPFileName = IPFileString.split("/")[-1]
    # InstrParsPath = repoPath / IPFileName

    data = np.loadtxt(InstrParsPath, dtype=str)[1:].astype(float)

    spectra = data[:, 0]
    select_rows = np.where((spectra >= firstSpec) & (spectra <= lastSpec))
    instrPars = data[select_rows]
    return instrPars


def loadResolutionPars(instrPars):
    """Resolution of parameters to propagate into TOF resolution
       Output: matrix with each parameter in each column"""
    spectrums = instrPars[:, 0] 
    L = len(spectrums)
    #for spec no below 135, back scattering detectors, mode is double difference
    #for spec no 135 or above, front scattering detectors, mode is single difference
    dE1 = np.where(spectrums < 135, 88.7, 73)       #meV, STD
    dE1_lorz = np.where(spectrums < 135, 40.3, 24)  #meV, HFHM
    dTOF = np.repeat(0.37, L)      #us
    dTheta = np.repeat(0.016, L)   #rad
    dL0 = np.repeat(0.021, L)      #meters
    dL1 = np.repeat(0.023, L)      #meters
    
    resolutionPars = np.vstack((dE1, dTOF, dTheta, dL0, dL1, dE1_lorz)).transpose() 
    return resolutionPars 


def calculateKinematicsArrays(dataX, instrPars):          
    """Kinematics quantities calculated from TOF data"""   

    mN, Ef, en_to_vel, vf, hbar = loadConstants()    
    det, plick, angle, T0, L0, L1 = np.hsplit(instrPars, 6)     #each is of len(dataX)
    t_us = dataX - T0                                           #T0 is electronic delay due to instruments
    v0 = vf * L0 / ( vf * t_us - L1 )
    E0 =  np.square( v0 / en_to_vel )            #en_to_vel is a factor used to easily change velocity to energy and vice-versa
    
    delta_E = E0 - Ef  
    delta_Q2 = 2. * mN / hbar**2 * ( E0 + Ef - 2. * np.sqrt(E0*Ef) * np.cos(angle/180.*np.pi) )
    delta_Q = np.sqrt( delta_Q2 )
    return v0, E0, delta_E, delta_Q              #shape(no of spectrums, no of bins)


def reshapeArrayPerSpectrum(A):
    """Exchanges the first two indices of an array A,
    ao rearranges array to match iteration per spectrum of main fitting map()
    """
    return np.stack(np.split(A, len(A), axis=0), axis=2)[0]


def convertDataXToYSpacesForEachMass(dataX, masses, delta_Q, delta_E):
    "Calculates y spaces from TOF data, each row corresponds to one mass" 
    
    #prepare arrays to broadcast
    dataX = dataX[np.newaxis, :, :]
    delta_Q = delta_Q[np.newaxis, :, :]
    delta_E = delta_E[np.newaxis, :, :]  

    mN, Ef, en_to_vel, vf, hbar = loadConstants()
    masses = masses.reshape(masses.size, 1, 1)

    energyRecoil = np.square( hbar * delta_Q ) / 2. / masses              
    ySpacesForEachMass = masses / hbar**2 /delta_Q * (delta_E - energyRecoil)    #y-scaling  
    return ySpacesForEachMass


def fitNcpToSingleSpec(dataY, dataE, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays, ic):
    """Fits the NCP and returns the best fit parameters for one spectrum"""

    if np.all(dataY == 0) : 
        return np.zeros(len(ic.initPars)+3)  
    
    scaledPars = ic.initPars * ic.scalingFactors
    scaledBounds = ic.bounds * ic.scalingFactors[:, np.newaxis]

    result = optimize.minimize(
        errorFunction, 
        scaledPars, 
        args=(dataY, dataE, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays, ic),
        method='SLSQP', 
        bounds = scaledBounds, 
        constraints=ic.constraints
        )

    fitScaledPars = result["x"]
    fitPars = fitScaledPars / ic.scalingFactors

    noDegreesOfFreedom = len(dataY) - len(fitPars)
    specFitPars = np.append(instrPars[0], fitPars)
    return np.append(specFitPars, [result["fun"] / noDegreesOfFreedom, result["nit"]])


def errorFunction(scaledPars, dataY, dataE, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays, ic):
    """Error function to be minimized, operates in TOF space"""

    unscaledPars = scaledPars / ic.scalingFactors
    ncpForEachMass, ncpTotal = calculateNcpSpec(ic, unscaledPars, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays)

    if np.all(dataE == 0) | np.all(np.isnan(dataE)):
        # This condition is not usually satisfied but in the exceptional case that it is,
        # we can use a statistical weight to make sure the chi2 used is not too small for the 
        # optimization algorithm. This is not used in the original script
        chi2 = (ncpTotal - dataY)**2 / dataY**2
    else:
        chi2 =  (ncpTotal - dataY)**2 / dataE**2    
    return np.sum(chi2)


def calculateNcpSpec(ic, unscaledPars, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays):    
    """Creates a synthetic C(t) to be fitted to TOF values of a single spectrum, from J(y) and resolution functions
       Shapes: datax (1, n), ySpacesForEachMass (4, n), res (4, 2), deltaQ (1, n), E0 (1,n),
       where n is no of bins"""
    
    masses, intensities, widths, centers = prepareArraysFromPars(ic, unscaledPars) 
    v0, E0, deltaE, deltaQ = kinematicArrays
    
    gaussRes, lorzRes = caculateResolutionForEachMass(
        masses, ySpacesForEachMass, centers, resolutionPars, instrPars, kinematicArrays
        )
    totalGaussWidth = np.sqrt(widths**2 + gaussRes**2)                 
    
    JOfY = pseudoVoigt(ySpacesForEachMass - centers, totalGaussWidth, lorzRes)  
    
    FSE =  - numericalThirdDerivative(ySpacesForEachMass, JOfY) * widths**4 / deltaQ * 0.72 
    
    ncpForEachMass = intensities * (JOfY + FSE) * E0 * E0**(-0.92) * masses / deltaQ   
    ncpTotal = np.sum(ncpForEachMass, axis=0)
    return ncpForEachMass, ncpTotal


def prepareArraysFromPars(ic, initPars):
    """Extracts the intensities, widths and centers from the fitting parameters
        Reshapes all of the arrays to collumns, for the calculation of the ncp,"""

    masses = ic.masses[:, np.newaxis]    
    intensities = initPars[::3].reshape(masses.shape)
    widths = initPars[1::3].reshape(masses.shape)
    centers = initPars[2::3].reshape(masses.shape)  
    return masses, intensities, widths, centers 


def caculateResolutionForEachMass(masses, ySpacesForEachMass, centers, resolutionPars, instrPars, kinematicArrays):    
    """Calculates the gaussian and lorentzian resolution
    output: two column vectors, each row corresponds to each mass"""
    
    v0, E0, delta_E, delta_Q = kinematicsAtYCenters(ySpacesForEachMass, centers, kinematicArrays)
    
    gaussianResWidth = calcGaussianResolution(masses, v0, E0, delta_E, delta_Q, resolutionPars, instrPars)
    lorentzianResWidth = calcLorentzianResolution(masses, v0, E0, delta_E, delta_Q, resolutionPars, instrPars)
    return gaussianResWidth, lorentzianResWidth


def kinematicsAtYCenters(ySpacesForEachMass, centers, kinematicArrays):
    """v0, E0, deltaE, deltaQ at the peak of the ncpTotal for each mass"""

    shapeOfArrays = centers.shape
    proximityToYCenters = np.abs(ySpacesForEachMass - centers)
    yClosestToCenters = proximityToYCenters.min(axis=1).reshape(shapeOfArrays)
    yCentersMask = proximityToYCenters == yClosestToCenters

    v0, E0, deltaE, deltaQ = kinematicArrays

    # Expand arrays to match shape of yCentersMask
    v0 = v0 * np.ones(shapeOfArrays)
    E0 = E0 * np.ones(shapeOfArrays)
    deltaE = deltaE * np.ones(shapeOfArrays)
    deltaQ = deltaQ * np.ones(shapeOfArrays)

    v0 = v0[yCentersMask].reshape(shapeOfArrays)
    E0 = E0[yCentersMask].reshape(shapeOfArrays)
    deltaE = deltaE[yCentersMask].reshape(shapeOfArrays)
    deltaQ = deltaQ[yCentersMask].reshape(shapeOfArrays)
    return v0, E0, deltaE, deltaQ


def calcGaussianResolution(masses, v0, E0, delta_E, delta_Q, resolutionPars, instrPars):
    # Currently the function that takes the most time in the fitting
    assert masses.shape == (masses.size, 1), f"masses.shape: {masses.shape}. The shape of the masses array needs to be a collumn!"

    det, plick, angle, T0, L0, L1 = instrPars
    dE1, dTOF, dTheta, dL0, dL1, dE1_lorz = resolutionPars
    mN, Ef, en_to_vel, vf, hbar = loadConstants()

    angle = angle * np.pi/180

    dWdE1 = 1. + (E0 / Ef)**1.5 * (L1 / L0)
    dWdTOF = 2. * E0 * v0 / L0
    dWdL1 = 2. * E0**1.5 / Ef**0.5 / L0
    dWdL0 = 2. * E0 / L0

    dW2 = dWdE1**2*dE1**2 + dWdTOF**2*dTOF**2 + dWdL1**2*dL1**2 + dWdL0**2*dL0**2
    # conversion from meV^2 to A^-2, dydW = (M/q)^2
    dW2 *= (masses / hbar**2 / delta_Q)**2

    dQdE1 = 1. - (E0 / Ef)**1.5 * L1/L0 - np.cos(angle) * ((E0 / Ef)**0.5 - L1/L0 * E0/Ef)
    dQdTOF = 2.*E0 * v0/L0
    dQdL1 = 2.*E0**1.5 / L0 / Ef**0.5
    dQdL0 = 2.*E0 / L0
    dQdTheta = 2. * np.sqrt(E0 * Ef) * np.sin(angle)

    dQ2 = dQdE1**2*dE1**2 + (dQdTOF**2*dTOF**2 + dQdL1**2*dL1**2 + dQdL0 **
                             2*dL0**2)*np.abs(Ef/E0*np.cos(angle)-1) + dQdTheta**2*dTheta**2
    dQ2 *= (mN / hbar**2 / delta_Q)**2

    # in A-1    #same as dy^2 = (dy/dw)^2*dw^2 + (dy/dq)^2*dq^2
    gaussianResWidth = np.sqrt(dW2 + dQ2)
    return gaussianResWidth


def calcLorentzianResolution(masses, v0, E0, delta_E, delta_Q, resolutionPars, instrPars):
    assert masses.shape == (masses.size, 1), "The shape of the masses array needs to be a collumn!"
        
    det, plick, angle, T0, L0, L1 = instrPars
    dE1, dTOF, dTheta, dL0, dL1, dE1_lorz = resolutionPars
    mN, Ef, en_to_vel, vf, hbar = loadConstants()

    angle = angle * np.pi / 180

    dWdE1_lor = (1. + (E0/Ef)**1.5 * (L1/L0))**2
    # conversion from meV^2 to A^-2
    dWdE1_lor *= (masses / hbar**2 / delta_Q)**2

    dQdE1_lor = (1. - (E0/Ef)**1.5 * L1/L0 - np.cos(angle)
                 * ((E0/Ef)**0.5 + L1/L0 * E0/Ef))**2
    dQdE1_lor *= (mN / hbar**2 / delta_Q)**2

    lorentzianResWidth = np.sqrt(dWdE1_lor + dQdE1_lor) * dE1_lorz   # in A-1
    return lorentzianResWidth


def loadConstants():
    """Output: the mass of the neutron, final energy of neutrons (selected by gold foil),
    factor to change energies into velocities, final velocity of neutron and hbar"""
    mN=1.008    #a.m.u.
    Ef=4906.         # meV
    en_to_vel = 4.3737 * 1.e-4
    vf = np.sqrt(Ef) * en_to_vel  # m/us
    hbar = 2.0445
    return mN, Ef, en_to_vel, vf, hbar


def pseudoVoigt(x, sigma, gamma):
    """Convolution between Gaussian with std sigma and Lorentzian with HWHM gamma"""
    fg, fl = 2.*sigma*np.sqrt(2.*np.log(2.)), 2.*gamma
    f = 0.5346 * fl + np.sqrt(0.2166*fl**2 + fg**2)
    eta = 1.36603 * fl/f - 0.47719 * (fl/f)**2 + 0.11116 * (fl/f)**3
    sigma_v, gamma_v = f/(2.*np.sqrt(2.*np.log(2.))), f / 2.
    pseudo_voigt = eta * \
        lorentizian(x, gamma_v) + (1.-eta) * gaussian(x, sigma_v)
    # TODO: Ask about this comment
    # norm = np.sum(pseudo_voigt)*(x[1]-x[0])
    return pseudo_voigt  # /np.abs(norm)


def gaussian(x, sigma):
    """Gaussian function centered at zero"""
    gaussian = np.exp(-x**2/2/sigma**2)
    gaussian /= np.sqrt(2.*np.pi)*sigma
    return gaussian


def lorentizian(x, gamma):
    """Lorentzian centered at zero"""
    lorentzian = gamma/np.pi / (x**2 + gamma**2)
    return lorentzian


def numericalThirdDerivative(x, fun):
    k6 = (- fun[:, 12:] + fun[:, :-12]) * 1
    k5 = (+ fun[:, 11:-1] - fun[:, 1:-11]) * 24
    k4 = (- fun[:, 10:-2] + fun[:, 2:-10]) * 192
    k3 = (+ fun[:,  9:-3] - fun[:, 3:-9]) * 488
    k2 = (+ fun[:,  8:-4] - fun[:, 4:-8]) * 387
    k1 = (- fun[:,  7:-5] + fun[:, 5:-7]) * 1584

    dev = k1 + k2 + k3 + k4 + k5 + k6
    dev /= np.power(x[:, 7:-5] - x[:, 6:-6], 3)
    dev /= 12**3

    derivative = np.zeros(fun.shape)
    derivative[:, 6:-6] = dev
    # Padded with zeros left and right to return array with same shape
    return derivative


def buildNcpFromSpec(initPars, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays, ic):
    """input: all row shape
       output: row shape with the ncpTotal for each mass"""

    if np.all(initPars==0):  
        return np.zeros(ySpacesForEachMass.shape) 
    
    ncpForEachMass, ncpTotal = calculateNcpSpec(ic, initPars, ySpacesForEachMass, resolutionPars, instrPars, kinematicArrays)        
    return ncpForEachMass


def createNcpWorkspaces(ncpForEachMass, ncpTotal, ws):
    """Creates workspaces from ncp array data"""

    # Need to rearrage array of yspaces into seperate arrays for each mass
    ncpForEachMass = switchFirstTwoAxis(ncpForEachMass)
    dataX = ws.extractX()
    dataE = np.zeros(dataX.shape)

    # Original script does not have this step to multiply by histogram widths
    histWidths = dataX[:, 1:] - dataX[:, :-1]
    dataX = dataX[:, :-1]       # Cut last column to match ncpTotal length
    dataY = ncpTotal * histWidths
    dataE = dataE[:, :-1] * histWidths

    ncpTotWs = CreateWorkspace(DataX=dataX.flatten(), DataY=dataY.flatten(), DataE=dataE.flatten(),
                     Nspec=len(dataX), OutputWorkspace=ws.name()+"_tof_fitted_profiles")
    SumSpectra(InputWorkspace=ncpTotWs, OutputWorkspace=ncpTotWs.name()+"_sum" )

    for i, ncp_m in enumerate(ncpForEachMass):
        ncpMWs = CreateWorkspace(DataX=dataX.flatten(), DataY=ncp_m.flatten(), Nspec=len(dataX),
                        OutputWorkspace=ws.name()+"_tof_fitted_profile_"+str(i+1))
        SumSpectra(InputWorkspace=ncpMWs, OutputWorkspace=ncpMWs.name()+"_sum" )



def switchFirstTwoAxis(A):
    """Exchanges the first two indices of an array A,
    rearranges matrices per spectrum for iteration of main fitting procedure
    """
    return np.stack(np.split(A, len(A), axis=0), axis=2)[0]


def createWorkspacesForMSCorrection(ic, meanWidths, meanIntensityRatios):
    """Creates _MulScattering and _TotScattering workspaces used for the MS correction"""

    sampleProperties = calcMSCorrectionSampleProperties(ic, meanWidths, meanIntensityRatios)
    print("\nThe sample properties for Multiple Scattering correction are:\n\n", 
            sampleProperties, "\n")
    createMulScatWorkspaces(ic, ic.name, sampleProperties)


def calcMSCorrectionSampleProperties(ic, meanWidths, meanIntensityRatios):
    masses = ic.masses.flatten()

    # If H not present ie backward scattering, add it to sample properties
    if (ic.modeRunning == "BACKWARD") and ic.addHToMS:   
        masses = np.append(masses, 1.0079)
        meanWidths = np.append(meanWidths, 5.0)
        HIntensity = ic.HToMass0Ratio * meanIntensityRatios[0]
        meanIntensityRatios = np.append(meanIntensityRatios, HIntensity)
        meanIntensityRatios /= np.sum(meanIntensityRatios)

    MSProperties = np.zeros(3*len(masses))
    MSProperties[::3] = masses
    MSProperties[1::3] = meanIntensityRatios
    MSProperties[2::3] = meanWidths
    sampleProperties = list(MSProperties)   

    return sampleProperties


def createMulScatWorkspaces(ic, wsName, sampleProperties):
    """Uses the Mantid algorithm for the MS correction to create two Workspaces _TotScattering and _MulScattering"""

    print("\nEvaluating the Multiple Scattering Correction...\n")
    # selects only the masses, every 3 numbers
    MS_masses = sampleProperties[::3]
    # same as above, but starts at first intensities
    MS_amplitudes = sampleProperties[1::3]

    dens, trans = VesuvioThickness(
        Masses=MS_masses, Amplitudes=MS_amplitudes, TransmissionGuess=ic.transmission_guess, Thickness=0.1
        )

    _TotScattering, _MulScattering = VesuvioCalculateMS(
        wsName, 
        NoOfMasses=len(MS_masses), 
        SampleDensity=dens.cell(9, 1),
        AtomicProperties=sampleProperties, 
        BeamRadius=2.5,
        NumScatters=ic.multiple_scattering_order,
        NumEventsPerRun=int(ic.number_of_events)
        )

    data_normalisation = Integration(wsName)
    simulation_normalisation = Integration("_TotScattering")
    for workspace in ("_MulScattering", "_TotScattering"):
        Divide(LHSWorkspace=workspace, RHSWorkspace=simulation_normalisation, 
               OutputWorkspace=workspace)
        Multiply(LHSWorkspace=workspace, RHSWorkspace=data_normalisation, 
                 OutputWorkspace=workspace)
        RenameWorkspace(InputWorkspace=workspace,
                        OutputWorkspace=str(wsName)+workspace)
    DeleteWorkspaces(
        [data_normalisation, simulation_normalisation, trans, dens]
        )
  # The only remaining workspaces are the _MulScattering and _TotScattering


def createWorkspacesForGammaCorrection(ic, meanWidths, meanIntensityRatios):
    """Creates _gamma_background correction workspace to be subtracted from the main workspace"""

    # I do not know why, but setting these instrument parameters is required
    SetInstrumentParameter(ic.name, ParameterName='hwhm_lorentz', 
                            ParameterType='Number', Value='24.0')
    SetInstrumentParameter(ic.name, ParameterName='sigma_gauss', 
                            ParameterType='Number', Value='73.0')

    profiles = calcGammaCorrectionProfiles(ic.masses, meanWidths, meanIntensityRatios)
    background, corrected = VesuvioCalculateGammaBackground(
        InputWorkspace=ic.name, ComptonFunction=profiles
        )
    RenameWorkspace(InputWorkspace= background, OutputWorkspace = ic.name+"_gamma_background")
    Scale(InputWorkspace = ic.name+"_gamma_background", OutputWorkspace = ic.name+"_gamma_background", 
        Factor=0.9, Operation="Multiply")

    DeleteWorkspace(corrected)


def calcGammaCorrectionProfiles(masses, meanWidths, meanIntensityRatios):
    masses = masses.flatten()
    profiles = ""
    for mass, width, intensity in zip(masses, meanWidths, meanIntensityRatios):
        profiles += "name=GaussianComptonProfile,Mass="   \
                    + str(mass) + ",Width=" + str(width)  \
                    + ",Intensity=" + str(intensity) + ';'
    print("\n The sample properties for Gamma Correction are:\n",
            profiles)
    return profiles
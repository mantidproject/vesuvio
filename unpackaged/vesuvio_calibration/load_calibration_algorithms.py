"""
  This script imports and subscribes calibration algorithms for the VESUVIO spectrometer.

  Two Mantid algorithms are provided: EVSCalibrationFit and EVSCalibrationAnalysis.

  EVSCalibrationFit is used to fit m peaks to n spectra. The positions of the peaks are esitmated using the supplied instrument parameter
  file and the d-spacings of the sample (if provided). Support is provided for both Voigt and Gaussian functions.

  EVSCalibrationAnalysis uses the output from EVSCalibrationFit to calculate instrument parameters.

  The procedures used here are based upon those described in: Calibration of an electron volt neutron spectrometer, Nuclear Instruments and
  Methods in Physics Research A (15 October 2010), doi:10.1016/j.nima.2010.09.079 by J. Mayers, M. A. Adams
"""

from mantid.api import AlgorithmFactory
from unpackaged.vesuvio_calibration.calibration_scripts.calibrate_vesuvio_fit import EVSCalibrationFit
from unpackaged.vesuvio_calibration.calibration_scripts.calibrate_vesuvio_analysis import EVSCalibrationAnalysis

AlgorithmFactory.subscribe(EVSCalibrationFit)
AlgorithmFactory.subscribe(EVSCalibrationAnalysis)

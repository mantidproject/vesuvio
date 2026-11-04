from enum import StrEnum


class Tags(StrEnum):
    BACKWARD = "back"
    FORWARD = "front"


class ScatteringType(StrEnum):
    BACKWARD = "backward"
    FORWARD = "forward"


class Masking(StrEnum):
    NAN = "nan"
    NCP = "ncp"


class Mode(StrEnum):
    SINGLE_DIFFERENCE = "SingleDifference"
    DOUBLE_DIFFERENCE = "DoubleDifference"
    THICK_DIFFERENCE = "ThickDifference"
    FOIL_OUT = "FoilOut"
    FOIL_IN = "FoilIn"
    FOIL_IN_OUT = "FoilInOut"


class PeakType(StrEnum):
    RESONANCE = "Resonance"
    BRAGG = "Bragg"
    RECOIL = "Recoil"


class SharedParameterFitType(StrEnum):
    INDIVIDUAL = "Individual"
    SHARED = "Shared"
    BOTH = "Both"


class ModeRunning(StrEnum):
    BACKWARD = "BACKWARD"
    FORWARD = "FORWARD"


class FitFunction(StrEnum):
    GAUSSIAN = "Gaussian"
    VOIGT = "Voigt"

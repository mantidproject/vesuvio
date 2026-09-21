from enum import StrEnum


class Tags(StrEnum):
    BACKWARD = "back"
    FORWARD = "front"


class FitModels(StrEnum):
    GAUSS = "gauss"
    GAUSS_CNTR = "gauss_cntr"
    GCC4 = "gcc4"
    GCC4_CNTR = "gcc4_cntr"
    GCC6 = "gcc6"
    GCC6_CNTR = "gcc6_cntr"
    GCC4C6 = "gcc4c6"
    GCC4C6_CNTR = "gcc4c6_cntr"
    DOUBLEWELL = "doublewell"
    GAUSS2D = "gauss2d"
    GAUSS3D = "gauss3d"


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

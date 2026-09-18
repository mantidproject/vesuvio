from enum import StrEnum


class Tags(StrEnum):
    Backward = "back"
    Forward = "front"


class FitModels(StrEnum):
    gauss = "gauss"
    gauss_cntr = "gauss_cntr"
    gcc4 = "gcc4"
    gcc4_cntr = "gcc4_cntr"
    gcc6 = "gcc6"
    gcc6_cntr = "gcc6_cntr"
    gcc4c6 = "gcc4c6"
    gcc4c6_cntr = "gcc4c6_cntr"
    doublewell = "doublewell"
    gauss2d = "gauss2d"
    gauss3d = "gauss3d"


class Masking(StrEnum):
    nan = "nan"
    ncp = "ncp"


class Mode(StrEnum):
    SingleDifference = "SingleDifference"
    DoubleDifference = "DoubleDifference"
    ThickDifference = "ThickDifference"
    FoilOut = "FoilOut"
    FoilIn = "FoilIn"
    FoilInOut = "FoilInOut"

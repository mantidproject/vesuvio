from enum import StrEnum


RESOLUTION_WORKSPACE_SUFFIX = "_ws_resolution"
JOY_WORKSPACE_SUFFIX = "_joy"
JOY_AVG_WORKSPACE_SUFFIX = "_joy_avg"
Q_WORKSPACE_SUFFIX = "_q"
REBIN_WORKSPACE_SUFFIX = "_rebin"
CORRECTED_WORKSPACE_SUFFIX = "_corr"

SUM_WORKSPACE_SUFFIX = "_sum"
FSE_FIT_WORKSPACE_SUFFIX = "_fse_fit"
FSE_RESIDUALS_WORKSPACE_SUFFIX = "_fse_residuals"
FSE_CONTRIBUTION_WORKSPACE_SUFFIX = "_fse_contribution"


class FSEFitWith(StrEnum):
    FREE_SHARED = "FREE_SHARED"
    FREE_UNSHARED = "FREE_UNSHARED"
    TIED_TO_MODEL = "TIED_TO_MODEL"


WORKSPACE_FSE_SUFFIX = {
    FSEFitWith.FREE_SHARED: "free_shared",
    FSEFitWith.FREE_UNSHARED: "free_unshared",
    FSEFitWith.TIED_TO_MODEL: "tied",
}


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

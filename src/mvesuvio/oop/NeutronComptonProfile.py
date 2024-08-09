from dataclasses import dataclass

@dataclass(frozen=False)
class NeutronComptonProfile:
    label: str
    mass: float

    intensity: float
    width: float
    center: float

    intensity_bounds: tuple[float, float]
    width_bounds: tuple[float, float]
    center_bounds: tuple[float, float]

    mean_intensity: float = None
    mean_width: float = None
    mean_center: float = None

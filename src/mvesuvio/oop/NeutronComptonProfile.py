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


# class NeutronComptonProfile:
#
#     def __init__(self, label, mass, intensity, width, center,
#                  intensity_bounds, width_bounds, center_bounds):
#         self._mass = mass 
#         self._label = label 
#         self._intensity = intensity
#         self._width = width
#         self._center = center
#         self._intensity_bounds = intensity_bounds
#         self._width_bounds = width_bounds
#         self._center_bounds = center_bounds 
#
#     @property
#     def label(self):
#         return self._label
#
#     @property
#     def mass(self):
#         return self._mass
#
#     @property
#     def width(self):
#         return self._width
#
#     @property 
#     def intensity(self):
#         return self._intensity
#
#     @property
#     def center(self):
#         return self._center
#
#     @property
#     def width_bounds(self):
#         return self._width_bounds
#
#     @property
#     def intensity_bounds(self):
#         return self._intensity_bounds
#
#     @property
#     def center_bounds(self):
#         return self._center_bounds
#

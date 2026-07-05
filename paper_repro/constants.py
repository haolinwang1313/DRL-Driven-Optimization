from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

MORPHOLOGY_FEATURES = [
    "FAR",
    "SD",
    "AF",
    "AR_ew",
    "AR_ns",
    "SVF",
    "BD",
    "OSR",
    "SC",
    "PAR",
    "theta",
    "OSLI",
]

PERFORMANCE_TARGETS = ["EUIt", "EG", "H"]

BLOCK_INDEX_TO_COORD = {
    0: (0, 0),
    1: (1, 0),
    2: (2, 0),
    3: (0, 1),
    4: (1, 1),
    5: (2, 1),
    6: (0, 2),
    7: (1, 2),
    8: (2, 2),
}


@dataclass(frozen=True)
class Prototype:
    name: str
    category: str
    footprint_area: float
    min_floors: int
    max_floors: int
    width_m: float
    depth_m: float
    courtyard_fraction: float = 0.0

    @property
    def perimeter_m(self) -> float:
        outer = 2.0 * (self.width_m + self.depth_m)
        if self.courtyard_fraction <= 0:
            return outer
        inner_area = self.footprint_area * self.courtyard_fraction / max(1e-6, 1.0 - self.courtyard_fraction)
        inner_side = inner_area ** 0.5
        return outer + 4.0 * inner_side


PROTOTYPES = {
    "P-1": Prototype("P-1", "Point", 2000.0, 1, 3, 40.0, 50.0),
    "P-2": Prototype("P-2", "Point", 1500.0, 4, 12, 30.0, 50.0),
    "P-3": Prototype("P-3", "Point", 1000.0, 13, 30, 25.0, 40.0),
    "P-4": Prototype("P-4", "Point", 1000.0, 13, 30, 20.0, 50.0),
    "S-1": Prototype("S-1", "Slab", 2000.0, 1, 3, 25.0, 80.0),
    "S-2": Prototype("S-2", "Slab", 1500.0, 4, 12, 20.0, 75.0),
    "S-3": Prototype("S-3", "Slab", 1000.0, 13, 30, 16.0, 62.5),
    "C-1": Prototype("C-1", "Courtyard", 2000.0, 1, 3, 50.0, 50.0, courtyard_fraction=0.25),
    "C-2": Prototype("C-2", "Courtyard", 1500.0, 4, 12, 45.0, 40.0, courtyard_fraction=0.2),
}

SIM_STACK_PACKAGES = {
    "ladybug-core": "ladybug",
    "ladybug-radiance": "ladybug_radiance",
    "honeybee-core": "honeybee",
    "honeybee-energy": "honeybee_energy",
    "honeybee-radiance": "honeybee_radiance",
    "dragonfly-core": "dragonfly",
    "dragonfly-energy": "dragonfly_energy",
    "dragonfly-radiance": "dragonfly_radiance",
    "uwg": "uwg",
}

SIM_STACK_EXECUTABLES = {
    "energyplus": "energyplus",
    "radiance_rtrace": "rtrace",
    "radiance_oconv": "oconv",
}

COMMON_USER_EXECUTABLE_HINTS = {
    "energyplus": [
        Path.home() / "opt" / "EnergyPlus" / "energyplus",
        Path.home() / "opt" / "EnergyPlus" / "bin" / "energyplus",
    ],
    "radiance_rtrace": [
        Path.home() / "opt" / "Radiance" / "bin" / "rtrace",
        Path.home() / "ray" / "bin" / "rtrace",
    ],
    "radiance_oconv": [
        Path.home() / "opt" / "Radiance" / "bin" / "oconv",
        Path.home() / "ray" / "bin" / "oconv",
    ],
}

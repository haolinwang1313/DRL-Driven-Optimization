from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import cos, radians, sin, sqrt
from typing import Iterable

import numpy as np
import pandas as pd

from paper_repro.constants import BLOCK_INDEX_TO_COORD, MORPHOLOGY_FEATURES, PROTOTYPES, Prototype


@dataclass
class CellAssignment:
    block_index: int
    prototype_name: str
    floors: int

    @property
    def prototype(self) -> Prototype:
        return PROTOTYPES[self.prototype_name]


@dataclass
class BlockMorphology:
    open_space_index: int
    theta_deg: float
    assignments: list[CellAssignment]
    block_size_m: float = 240.0
    land_unit_size_m: float = 80.0
    floor_height_m: float = 3.0

    @property
    def block_area(self) -> float:
        return self.block_size_m**2

    def to_record(self) -> dict:
        return {
            "open_space_index": self.open_space_index,
            "theta_deg": self.theta_deg,
            "block_size_m": self.block_size_m,
            "land_unit_size_m": self.land_unit_size_m,
            "floor_height_m": self.floor_height_m,
            "assignments": [asdict(item) for item in self.assignments],
        }


def _projected_widths(width_m: float, depth_m: float, theta_deg: float) -> tuple[float, float]:
    theta = radians(abs(theta_deg))
    width_x = abs(width_m * cos(theta)) + abs(depth_m * sin(theta))
    width_y = abs(width_m * sin(theta)) + abs(depth_m * cos(theta))
    return width_x, width_y


def _open_space_bonus(osli: int) -> float:
    x, y = BLOCK_INDEX_TO_COORD[osli]
    return (x + y) / 4.0


def block_to_features(block: BlockMorphology) -> dict[str, float]:
    footprints = []
    gross_areas = []
    heights = []
    perimeters = []
    wall_surfaces = []
    ar_ew_values = []
    ar_ns_values = []
    for assignment in block.assignments:
        prototype = assignment.prototype
        footprint = prototype.footprint_area
        gross_area = footprint * assignment.floors
        height = assignment.floors * block.floor_height_m
        proj_x, proj_y = _projected_widths(prototype.width_m, prototype.depth_m, block.theta_deg)
        canyon_width_x = max(block.land_unit_size_m - proj_x, 1.0)
        canyon_width_y = max(block.land_unit_size_m - proj_y, 1.0)

        footprints.append(footprint)
        gross_areas.append(gross_area)
        heights.append(height)
        perimeters.append(prototype.perimeter_m)
        wall_surfaces.append(prototype.perimeter_m * height)
        ar_ew_values.append(height / canyon_width_x)
        ar_ns_values.append(height / canyon_width_y)

    total_footprint = float(np.sum(footprints))
    total_gross = float(np.sum(gross_areas))
    total_surface = float(np.sum(wall_surfaces) + 2.0 * total_footprint)
    open_area = max(block.block_area - total_footprint, 1.0)
    average_height = float(np.mean(heights))
    mean_ar_ew = float(np.mean(ar_ew_values))
    mean_ar_ns = float(np.mean(ar_ns_values))
    density_component = total_footprint / block.block_area
    openness = open_area / block.block_area
    open_space_bonus = _open_space_bonus(block.open_space_index)
    svf = 0.18 + 0.62 * openness + 0.14 * open_space_bonus - 0.08 * (mean_ar_ew + mean_ar_ns) / 2.0
    svf = float(np.clip(svf, 0.05, 0.95))

    features = {
        "FAR": total_gross / block.block_area,
        "SD": max(heights) - average_height,
        "AF": total_gross / max(total_footprint, 1.0),
        "AR_ew": mean_ar_ew,
        "AR_ns": mean_ar_ns,
        "SVF": svf,
        "BD": density_component,
        "OSR": open_area / max(total_gross, 1.0),
        "SC": total_surface / max(total_gross, 1.0),
        "PAR": float(np.sum(perimeters) / max(total_footprint, 1.0)),
        "theta": block.theta_deg,
        "OSLI": float(block.open_space_index),
    }
    return features


def random_block(rng: np.random.Generator, block_size_m: float, land_unit_size_m: float, floor_height_m: float) -> BlockMorphology:
    open_space_index = int(rng.integers(0, 9))
    theta_deg = float(rng.uniform(-45.0, 45.0))
    assignments: list[CellAssignment] = []
    available_prototypes = list(PROTOTYPES.values())
    for block_index in range(9):
        if block_index == open_space_index:
            continue
        prototype = available_prototypes[int(rng.integers(0, len(available_prototypes)))]
        floors = int(rng.integers(prototype.min_floors, prototype.max_floors + 1))
        assignments.append(CellAssignment(block_index=block_index, prototype_name=prototype.name, floors=floors))
    return BlockMorphology(
        open_space_index=open_space_index,
        theta_deg=theta_deg,
        assignments=assignments,
        block_size_m=block_size_m,
        land_unit_size_m=land_unit_size_m,
        floor_height_m=floor_height_m,
    )


def generate_morphology_dataset(
    n_samples: int,
    seed: int,
    block_size_m: float,
    land_unit_size_m: float,
    floor_height_m: float,
    return_blocks: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, list[dict]]:
    rng = np.random.default_rng(seed)
    rows = []
    block_records: list[dict] = []
    for sample_id in range(n_samples):
        block = random_block(rng, block_size_m=block_size_m, land_unit_size_m=land_unit_size_m, floor_height_m=floor_height_m)
        features = block_to_features(block)
        rows.append({"sample_id": sample_id, **features})
        if return_blocks:
            block_records.append({"sample_id": sample_id, **block.to_record()})
    frame = pd.DataFrame(rows)
    feature_frame = frame[["sample_id", *MORPHOLOGY_FEATURES]]
    if return_blocks:
        return feature_frame, block_records
    return feature_frame


def feature_bounds(frame: pd.DataFrame) -> dict[str, tuple[float, float]]:
    bounds = {}
    for feature in MORPHOLOGY_FEATURES:
        bounds[feature] = (float(frame[feature].min()), float(frame[feature].max()))
    return bounds


def features_from_action(action: Iterable[float]) -> dict[str, float]:
    return {name: float(value) for name, value in zip(MORPHOLOGY_FEATURES, action, strict=True)}


def write_block_records(block_records: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        for record in block_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

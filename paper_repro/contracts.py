from __future__ import annotations

from pathlib import Path
from typing import Iterable

import json
import pandas as pd

from paper_repro.constants import MORPHOLOGY_FEATURES, PERFORMANCE_TARGETS

SIMULATED_SAMPLE_COLUMNS = ["sample_id", *MORPHOLOGY_FEATURES, *PERFORMANCE_TARGETS, "weather_station", "simulation_mode"]
OPTIMIZATION_RESULT_COLUMNS = [
    "method",
    "scenario",
    "seed",
    *MORPHOLOGY_FEATURES,
    *PERFORMANCE_TARGETS,
    "reward",
]


def ensure_columns(frame: pd.DataFrame, required: Iterable[str]) -> pd.DataFrame:
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    return frame


def write_csv(frame: pd.DataFrame, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    return path


def write_json(payload: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path

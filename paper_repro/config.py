from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    raw: dict[str, Any]

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        with Path(path).open("r", encoding="utf-8") as handle:
            return cls(yaml.safe_load(handle))

    def __getitem__(self, item: str) -> Any:
        return self.raw[item]

    @property
    def artifact_root(self) -> Path:
        return Path(self.raw["project"]["artifact_root"])

    def path(self, *parts: str) -> Path:
        return Path(*parts)

    def ensure_artifact_dirs(self) -> dict[str, Path]:
        report_cfg = self.raw["report"]
        dirs = {name: Path(path) for name, path in report_cfg.items() if name.endswith("_dir")}
        dirs["artifact_root"] = self.artifact_root
        for path in dirs.values():
            path.mkdir(parents=True, exist_ok=True)
        return dirs

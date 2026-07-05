from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def test_public_catalog_paths_and_hashes_match() -> None:
    catalog = yaml.safe_load((ROOT / "data/catalog.yaml").read_text(encoding="utf-8"))
    for entry in catalog["datasets"]:
        if entry.get("status") == "not_included" or entry.get("redistributable") is False:
            continue
        path = ROOT / entry["path"]
        assert path.exists(), entry["path"]
        assert _sha256(path) == entry["sha256"], entry["path"]


def test_canonical_dataset_shape_and_boundary() -> None:
    samples = pd.read_csv(ROOT / "data/generated/canonical_2000/simulated_samples.csv")
    assert len(samples) == 2000
    assert set(["FAR", "SD", "AF", "AR_ew", "AR_ns", "SVF", "BD", "OSR", "SC", "PAR", "theta", "OSLI"]).issubset(samples.columns)
    assert set(["EUIt", "EG", "H"]).issubset(samples.columns)
    assert samples["simulation_mode"].eq("fallback_analytic").all()
    assert samples["EUIt"].between(60, 90).all()
    assert samples["EG"].between(1.0, 3.5).all()
    assert samples["H"].between(5.0, 8.5).all()

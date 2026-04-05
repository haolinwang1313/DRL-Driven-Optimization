from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

from paper_repro.constants import MORPHOLOGY_FEATURES, PERFORMANCE_TARGETS
from paper_repro.surrogate import load_surrogate


def analyze(checkpoint_a: Path, checkpoint_b: Path, dataset_path: Path) -> dict[str, dict]:
    data = pd.read_csv(dataset_path)
    bundle_a = load_surrogate(checkpoint_a)
    bundle_b = load_surrogate(checkpoint_b)

    rng = np.random.default_rng(20260404)
    random_frame = pd.DataFrame(
        {feature: rng.uniform(data[feature].min(), data[feature].max(), size=5000) for feature in MORPHOLOGY_FEATURES}
    )

    truth = data[PERFORMANCE_TARGETS]
    payload: dict[str, dict] = {}
    for name, bundle in {"checkpoint_a": bundle_a, "checkpoint_b": bundle_b}.items():
        train_pred = bundle.predict(data[MORPHOLOGY_FEATURES], clip=True)
        raw_pred = bundle.predict(random_frame, clip=False)
        entry: dict[str, dict] = {"training_fit": {}, "random_raw": {}}
        for target in PERFORMANCE_TARGETS:
            lo = float(truth[target].min())
            hi = float(truth[target].max())
            err = train_pred[target] - truth[target]
            entry["training_fit"][target] = {
                "mae": float(err.abs().mean()),
                "rmse": float(np.sqrt((err**2).mean())),
                "corr": float(np.corrcoef(train_pred[target], truth[target])[0, 1]),
            }
            entry["random_raw"][target] = {
                "mean": float(raw_pred[target].mean()),
                "std": float(raw_pred[target].std()),
                "below_min_frac": float((raw_pred[target] < lo).mean()),
                "above_max_frac": float((raw_pred[target] > hi).mean()),
            }
        triple = (
            (raw_pred["EUIt"] < truth["EUIt"].min())
            & (raw_pred["EG"] > truth["EG"].max())
            & (raw_pred["H"] > truth["H"].max())
        )
        entry["random_raw"]["triple_better_than_bounds_frac"] = float(triple.mean())
        payload[name] = entry
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-a", required=True)
    parser.add_argument("--checkpoint-b", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    payload = analyze(Path(args.checkpoint_a), Path(args.checkpoint_b), Path(args.dataset))
    if args.output:
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        raise

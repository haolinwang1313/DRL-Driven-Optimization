from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def merge_shards(optimization_dir: Path, shard_prefixes: list[str], output_prefix: str) -> dict[str, str]:
    result_frames: list[pd.DataFrame] = []
    merged_logs: dict[str, list[dict]] = {}
    merged_logs_all: dict[str, dict[str, list[dict]]] = {}

    for prefix in shard_prefixes:
        for csv_path in sorted(optimization_dir.glob(f"ddpg_results_{prefix}_*.csv")):
            result_frames.append(pd.read_csv(csv_path))
        for json_path in sorted(optimization_dir.glob(f"ddpg_logs_{prefix}_*.json")):
            payload = _load_json(json_path)
            for scenario, rows in payload.items():
                merged_logs.setdefault(scenario, []).extend(rows)
        for json_path in sorted(optimization_dir.glob(f"ddpg_logs_all_{prefix}_*.json")):
            payload = _load_json(json_path)
            for scenario, seed_map in payload.items():
                scenario_bucket = merged_logs_all.setdefault(scenario, {})
                for seed, rows in seed_map.items():
                    scenario_bucket[str(seed)] = rows

    if not result_frames:
        raise FileNotFoundError(f"No ddpg shard csv files found for prefixes: {shard_prefixes}")

    merged_frame = pd.concat(result_frames, ignore_index=True).sort_values(["scenario", "seed"]).reset_index(drop=True)
    results_path = optimization_dir / f"ddpg_results_{output_prefix}.csv"
    logs_path = optimization_dir / f"ddpg_logs_{output_prefix}.json"
    logs_all_path = optimization_dir / f"ddpg_logs_all_{output_prefix}.json"

    merged_frame.to_csv(results_path, index=False)
    logs_path.write_text(json.dumps(merged_logs, ensure_ascii=False, indent=2), encoding="utf-8")
    logs_all_path.write_text(json.dumps(merged_logs_all, ensure_ascii=False, indent=2), encoding="utf-8")

    return {
        "results_path": str(results_path),
        "logs_path": str(logs_path),
        "logs_all_path": str(logs_all_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge sharded DDPG outputs")
    parser.add_argument("--optimization-dir", default="artifacts/publication/optimization")
    parser.add_argument("--prefix", action="append", required=True, help="Shard prefix, e.g. guard_bp")
    parser.add_argument("--output-prefix", required=True, help="Merged output prefix")
    args = parser.parse_args()
    payload = merge_shards(Path(args.optimization_dir), args.prefix, args.output_prefix)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

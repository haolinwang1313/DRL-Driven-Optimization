from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean
from typing import Any

from paper_repro.config import Config


PREFERRED_LOG_NAMES = (
    "ddpg_logs_all_guardrail_full.json",
    "ddpg_logs_all_remote_match.json",
    "ddpg_logs_all_hp2000_full.json",
    "ddpg_logs_all.json",
)


@dataclass(frozen=True)
class ScenarioSummary:
    path: Path
    scenario: str
    count: int
    min_return: float
    max_return: float
    mean_return: float
    final_mean_return: float


def _optimization_dir(root: Path) -> Path:
    return root if root.name == "optimization" else root / "optimization"


def _preferred_logs(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    optimization_dir = _optimization_dir(root)
    for name in PREFERRED_LOG_NAMES:
        path = optimization_dir / name
        if path.exists():
            return [path]
    full = sorted(optimization_dir.glob("ddpg_logs_all*_full.json"))
    if full:
        return [full[0]]
    return sorted(optimization_dir.glob("ddpg_logs_all*.json"))


def _default_logs(config_path: Path) -> list[Path]:
    resolved_config = config_path.resolve()
    repo_root = resolved_config.parent.parent
    config = Config.from_yaml(resolved_config)
    roots = [repo_root / config.artifact_root]
    roots.extend(sorted((repo_root / "artifacts" / "reviewer_round_02").glob("**/optimization")))
    paths: list[Path] = []
    for root in roots:
        paths.extend(_preferred_logs(root))
    return list(dict.fromkeys(path.resolve() for path in paths))


def _load_log(path: Path) -> dict[str, dict[str, list[dict[str, Any]]]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected scenario dictionary")
    return payload


def _summaries(path: Path) -> list[ScenarioSummary]:
    payload = _load_log(path)
    summaries: list[ScenarioSummary] = []
    for scenario, seed_logs in sorted(payload.items()):
        returns: list[float] = []
        final_returns: list[float] = []
        for entries in seed_logs.values():
            if not entries:
                continue
            seed_returns = [float(entry["cumulative_reward"]) for entry in entries]
            returns.extend(seed_returns)
            final_returns.append(seed_returns[-1])
        if not returns:
            continue
        summaries.append(
            ScenarioSummary(
                path=path,
                scenario=str(scenario),
                count=len(returns),
                min_return=min(returns),
                max_return=max(returns),
                mean_return=fmean(returns),
                final_mean_return=fmean(final_returns),
            )
        )
    return summaries


def validate(paths: list[Path], *, max_steps: int, tolerance: float) -> int:
    if not paths:
        print("No DDPG seeded log files found.")
        return 2
    exit_code = 0
    for path in paths:
        for summary in _summaries(path):
            print(
                f"{path} | {summary.scenario}: count={summary.count} "
                f"min={summary.min_return:.8f} max={summary.max_return:.8f} "
                f"mean={summary.mean_return:.8f} final_mean={summary.final_mean_return:.8f}"
            )
            if summary.min_return < -tolerance or summary.max_return > max_steps + tolerance:
                exit_code = 1
    return exit_code


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate DDPG cumulative episode returns against the current reward contract.")
    parser.add_argument("--config", default="configs/revision.yaml")
    parser.add_argument("--log-root", default="", help="Optional exact log file, optimization dir, or artifact root to check.")
    parser.add_argument("--max-steps", type=int, default=40)
    parser.add_argument("--tolerance", type=float, default=1e-5)
    args = parser.parse_args()

    config_path = Path(args.config)
    if args.log_root:
        paths = _preferred_logs(Path(args.log_root))
    else:
        paths = _default_logs(config_path)
    raise SystemExit(validate(paths, max_steps=args.max_steps, tolerance=args.tolerance))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json

from paper_repro.config import Config
from paper_repro.pipeline import (
    bootstrap_pipeline,
    dataset_pipeline,
    full_reproduce,
    optimizer_pipeline,
    publication_diagnostics_pipeline,
    physical_stack_candidate_pipeline,
    publication_review_pipeline,
    publication_sync_pipeline,
    report_pipeline,
    surrogate_pipeline,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Paper02 reproduction CLI")
    parser.add_argument("--config", default="configs/default.yaml")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bootstrap = subparsers.add_parser("bootstrap-sim-stack")
    bootstrap.add_argument("--install-missing", action="store_true")

    subparsers.add_parser("build-dataset")
    subparsers.add_parser("train-surrogate")
    subparsers.add_parser("select-surrogate")
    run_optimizers = subparsers.add_parser("run-optimizers")
    run_optimizers.add_argument("--ddpg-only", action="store_true")
    run_optimizers.add_argument("--nsga2-only", action="store_true")
    run_optimizers.add_argument("--cmaes-only", action="store_true")
    run_optimizers.add_argument("--random-only", action="store_true")
    run_optimizers.add_argument("--scenario", action="append")
    run_optimizers.add_argument("--seed-start", type=int, default=0)
    run_optimizers.add_argument("--seed-end", type=int)
    run_optimizers.add_argument("--output-suffix", default="")
    subparsers.add_parser("make-paper-figures")
    sync_publication = subparsers.add_parser("sync-publication-results")
    sync_publication.add_argument("--server-config")
    subparsers.add_parser("publication-diagnostics")
    subparsers.add_parser("reevaluate-candidates")
    physical_probe = subparsers.add_parser("physical-reevaluate-candidates")
    physical_probe.add_argument("--input-csv")
    physical_probe.add_argument("--limit", type=int, default=5)
    physical_probe.add_argument("--server-config")
    physical_probe.add_argument("--output-suffix", default="")
    physical_probe.add_argument("--async", dest="async_mode", action="store_true")
    physical_probe.add_argument("--wait-seconds", type=int, default=0)
    physical_probe.add_argument("--job-id")
    subparsers.add_parser("publication-review")

    full = subparsers.add_parser("full-reproduce")
    full.add_argument("--install-missing", action="store_true")
    return parser


def _run(command: str, config_path: str, install_missing: bool = False, **extra) -> dict:
    config = Config.from_yaml(config_path)
    if command == "bootstrap-sim-stack":
        return bootstrap_pipeline(config, install_missing=install_missing)
    if command == "build-dataset":
        frame = dataset_pipeline(config)
        return {"rows": len(frame)}
    if command == "train-surrogate":
        _, summary = surrogate_pipeline(config)
        return summary
    if command == "select-surrogate":
        _, summary = surrogate_pipeline(config)
        return summary
    if command == "run-optimizers":
        _, metrics = optimizer_pipeline(
            config,
            ddpg_only=extra.get("ddpg_only", False),
            nsga2_only=extra.get("nsga2_only", False),
            cmaes_only=extra.get("cmaes_only", False),
            random_only=extra.get("random_only", False),
            scenarios=extra.get("scenario"),
            seed_start=extra.get("seed_start", 0),
            seed_end=extra.get("seed_end"),
            output_suffix=extra.get("output_suffix", ""),
        )
        return metrics
    if command == "make-paper-figures":
        return report_pipeline(config)
    if command == "sync-publication-results":
        return publication_sync_pipeline(config, server_cfg_path=extra.get("server_config"))
    if command == "publication-diagnostics":
        return publication_diagnostics_pipeline(config)
    if command == "reevaluate-candidates":
        return publication_diagnostics_pipeline(config)
    if command == "physical-reevaluate-candidates":
        return physical_stack_candidate_pipeline(
            config,
            input_csv=extra.get("input_csv"),
            limit=extra.get("limit", 5),
            server_cfg_path=extra.get("server_config"),
            output_suffix=extra.get("output_suffix", ""),
            async_mode=extra.get("async_mode", False),
            wait_seconds=extra.get("wait_seconds", 0),
            job_id=extra.get("job_id"),
        )
    if command == "publication-review":
        return publication_review_pipeline(config)
    if command == "full-reproduce":
        return full_reproduce(config, install_missing=install_missing)
    raise ValueError(f"Unknown command: {command}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    payload = _run(
        args.command,
        args.config,
        install_missing=getattr(args, "install_missing", False),
        ddpg_only=getattr(args, "ddpg_only", False),
        nsga2_only=getattr(args, "nsga2_only", False),
        cmaes_only=getattr(args, "cmaes_only", False),
        random_only=getattr(args, "random_only", False),
        scenario=getattr(args, "scenario", None),
        seed_start=getattr(args, "seed_start", 0),
        seed_end=getattr(args, "seed_end", None),
        output_suffix=getattr(args, "output_suffix", ""),
        server_config=getattr(args, "server_config", None),
        input_csv=getattr(args, "input_csv", None),
        limit=getattr(args, "limit", 5),
        async_mode=getattr(args, "async_mode", False),
        wait_seconds=getattr(args, "wait_seconds", 0),
        job_id=getattr(args, "job_id", None),
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def bootstrap_sim_stack_main() -> None:
    print(json.dumps(_run("bootstrap-sim-stack", "configs/default.yaml"), indent=2, ensure_ascii=False))


def build_dataset_main() -> None:
    print(json.dumps(_run("build-dataset", "configs/default.yaml"), indent=2, ensure_ascii=False))


def train_surrogate_main() -> None:
    print(json.dumps(_run("train-surrogate", "configs/default.yaml"), indent=2, ensure_ascii=False))


def select_surrogate_main() -> None:
    print(json.dumps(_run("select-surrogate", "configs/revision.yaml"), indent=2, ensure_ascii=False))


def run_optimizers_main() -> None:
    print(json.dumps(_run("run-optimizers", "configs/default.yaml"), indent=2, ensure_ascii=False))


def make_paper_figures_main() -> None:
    print(json.dumps(_run("make-paper-figures", "configs/default.yaml"), indent=2, ensure_ascii=False))


def sync_publication_results_main() -> None:
    print(json.dumps(_run("sync-publication-results", "configs/revision.yaml"), indent=2, ensure_ascii=False))


def publication_diagnostics_main() -> None:
    print(json.dumps(_run("publication-diagnostics", "configs/revision.yaml"), indent=2, ensure_ascii=False))


def publication_review_main() -> None:
    print(json.dumps(_run("publication-review", "configs/revision.yaml"), indent=2, ensure_ascii=False))


def full_reproduce_main() -> None:
    print(json.dumps(_run("full-reproduce", "configs/default.yaml"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

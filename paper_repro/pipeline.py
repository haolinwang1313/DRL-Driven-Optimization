from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from paper_repro.bootstrap import bootstrap_sim_stack
from paper_repro.config import Config
from paper_repro.constants import MORPHOLOGY_FEATURES, PERFORMANCE_TARGETS
from paper_repro.contracts import write_csv, write_json
from paper_repro.figures import generate_all_figures
from paper_repro.metrics import (
    compute_convergence_diagnostics,
    compute_hv_igd_by_method,
    compute_seeded_convergence_diagnostics,
    load_benchmark_results,
    summarize_objectives,
    summarize_preference_utilities,
)
from paper_repro.optimizers import run_ddpg, run_nsga2, run_random_search
from paper_repro.publication import sync_publication_results, validate_publication_results
from paper_repro.reviewer import run_revision_review
from paper_repro.runtime import resolve_device
from paper_repro.simulation import build_simulated_dataset, reevaluate_candidates
from paper_repro.surrogate import load_surrogate, train_surrogate


def bootstrap_pipeline(config: Config, install_missing: bool = False) -> dict:
    return bootstrap_sim_stack(config, install_missing=install_missing)


def dataset_pipeline(config: Config) -> pd.DataFrame:
    return build_simulated_dataset(config)


def surrogate_pipeline(config: Config, dataset: pd.DataFrame | None = None):
    if dataset is None:
        dataset = pd.read_csv(Path(config["report"]["data_dir"]) / "simulated_samples.csv")
    return train_surrogate(config, dataset)


def optimizer_pipeline(
    config: Config,
    ddpg_only: bool = False,
    nsga2_only: bool = False,
    random_only: bool = False,
    scenarios: list[str] | None = None,
    seed_start: int = 0,
    seed_end: int | None = None,
    output_suffix: str = "",
):
    model_path = Path(config["report"]["models_dir"]) / "surrogate.pt"
    surrogate = load_surrogate(model_path, device=resolve_device(config))
    ddpg_results = pd.DataFrame()
    ddpg_logs = {}
    nsga_results = pd.DataFrame()
    nsga_calibration = {}
    random_results = pd.DataFrame()
    random_summary = {}
    if random_only:
        random_results, random_summary = run_random_search(
            config,
            surrogate,
            scenarios=scenarios,
            seed_start=seed_start,
            seed_end=seed_end,
            output_suffix=output_suffix,
        )
        return random_results, {"random_search": random_summary}
    if not nsga2_only:
        ddpg_results, ddpg_logs = run_ddpg(
            config,
            surrogate,
            scenarios=scenarios,
            seed_start=seed_start,
            seed_end=seed_end,
            output_suffix=output_suffix,
        )
        if ddpg_only:
            return ddpg_results, {"ddpg_logged_scenarios": list(ddpg_logs)}
    if not ddpg_only:
        nsga_results, nsga_calibration = run_nsga2(config, surrogate)
        if nsga2_only:
            if "best_score" in nsga_calibration:
                return nsga_results, {"nsga2_best_score": nsga_calibration["best_score"]}
            return nsga_results, {"nsga2_calibration": nsga_calibration}
    combined = pd.concat([ddpg_results, nsga_results], ignore_index=True)
    write_csv(combined, Path(config["report"]["optimization_dir"]) / "optimization_results.csv")

    benchmark_path = Path(config["project"]["benchmark_dataset"])
    benchmark_summary = summarize_objectives(load_benchmark_results(benchmark_path), "scenario") if benchmark_path.exists() else {}
    local_summary = summarize_objectives(combined.assign(group=combined["scenario"].where(combined["method"] != "NSGA-II", "NSGA-II")), "group")
    hv_igd = compute_hv_igd_by_method(combined)
    metrics_payload = {
        "local_summary": local_summary,
        "benchmark_summary": benchmark_summary,
        "hv_igd": hv_igd.to_dict(orient="records"),
        "ddpg_logged_scenarios": list(ddpg_logs),
    }
    if "best_score" in nsga_calibration:
        metrics_payload["nsga2_best_score"] = nsga_calibration["best_score"]
    else:
        metrics_payload["nsga2_calibration"] = nsga_calibration
    write_json(metrics_payload, Path(config["report"]["optimization_dir"]) / "paper_metrics.json")
    return combined, metrics_payload


def report_pipeline(config: Config) -> dict:
    outputs = generate_all_figures(config)
    report_dir = Path(config["report"]["reports_dir"])
    optimization_dir = Path(config["report"]["optimization_dir"])
    metrics_path = optimization_dir / "paper_metrics.json"
    metrics_text = metrics_path.read_text(encoding="utf-8") if metrics_path.exists() else "{}"
    report_text = "\n".join(
        [
            "# Reproduction Report",
            "",
            "## Assumptions",
            "- AR was reconstructed as AR_ew and AR_ns using directional canyon aspect ratios.",
            "- Dataset.xlsx is used as a benchmark only and is not used during surrogate training.",
            "- The current end-to-end run uses the fallback analytic simulator unless a physical stack is fully available.",
            "",
            "## Outputs",
            f"- Figures directory: `{outputs['figures_dir']}`",
            f"- Metrics JSON: `{metrics_path}`",
            "",
            "## Metrics Snapshot",
            "```json",
            metrics_text,
            "```",
        ]
    )
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "reproduction_report.md"
    report_path.write_text(report_text, encoding="utf-8")
    return {"report_path": str(report_path), **outputs}


def publication_sync_pipeline(config: Config, server_cfg_path: str | None = None) -> dict:
    sync_payload = sync_publication_results(config, server_cfg_path=server_cfg_path)
    validation_payload = validate_publication_results(config)
    return {"sync": sync_payload, "validation": validation_payload}


def publication_diagnostics_pipeline(config: Config) -> dict:
    validate_publication_results(config)
    optimization_dir = Path(config["report"]["optimization_dir"])
    models_dir = Path(config["report"]["models_dir"])
    diagnostics_dir = Path(config["publication"]["diagnostics_dir"])
    reevaluation_dir = Path(config["publication"]["reevaluation_dir"])
    diagnostics_dir.mkdir(parents=True, exist_ok=True)
    reevaluation_dir.mkdir(parents=True, exist_ok=True)

    cv_predictions = pd.read_csv(models_dir / "cv_predictions.csv")
    ddpg_results = pd.read_csv(optimization_dir / "ddpg_results.csv")
    nsga_results = pd.read_csv(optimization_dir / "nsga2_results.csv")
    combined = pd.concat([ddpg_results, nsga_results], ignore_index=True)

    ddpg_logs = json.loads((optimization_dir / "ddpg_logs.json").read_text(encoding="utf-8"))
    convergence = compute_convergence_diagnostics(ddpg_logs)
    convergence_seeded = {}
    convergence_seeded_path = None
    ddpg_all_logs_path = optimization_dir / "ddpg_logs_all.json"
    if ddpg_all_logs_path.exists():
        ddpg_all_logs = json.loads(ddpg_all_logs_path.read_text(encoding="utf-8"))
        convergence_seeded_frame, convergence_seeded = compute_seeded_convergence_diagnostics(ddpg_all_logs)
        convergence_seeded_path = diagnostics_dir / "ddpg_seed_convergence.csv"
        write_csv(convergence_seeded_frame, convergence_seeded_path)
    preference_summary = summarize_preference_utilities(combined, config["optimization"]["utility_weights"])
    dataset = pd.read_csv(Path(config["report"]["data_dir"]) / "simulated_samples.csv")

    top_candidates = []
    for selection_scenario, weights in config["optimization"]["utility_weights"].items():
        selected = summarize_preference_utilities(combined, {selection_scenario: weights})[selection_scenario]
        top_candidates.extend(selected)
    top_frame = pd.DataFrame(top_candidates).drop_duplicates(subset=["method", "scenario", "seed"])
    reevaluated = reevaluate_candidates(config, top_frame[MORPHOLOGY_FEATURES].reset_index(drop=True), deterministic=True)
    reevaluated["method"] = top_frame["method"].to_numpy()
    reevaluated["scenario"] = top_frame["scenario"].to_numpy()
    reevaluated["seed"] = top_frame["seed"].to_numpy()
    for target in PERFORMANCE_TARGETS:
        reevaluated[f"surrogate_{target}"] = top_frame[target].to_numpy()
        reevaluated[f"reeval_abs_error_{target}"] = (reevaluated[target] - reevaluated[f"surrogate_{target}"]).abs()
    reevaluation_csv = reevaluation_dir / "top_candidate_reevaluation.csv"
    write_csv(reevaluated, reevaluation_csv)

    target_slice_rows = []
    for target in PERFORMANCE_TARGETS:
        absolute_error = (cv_predictions[f"pred_{target}"] - cv_predictions[f"true_{target}"]).abs()
        q_low = cv_predictions[f"true_{target}"].quantile(0.1)
        q_high = cv_predictions[f"true_{target}"].quantile(0.9)
        low_mask = cv_predictions[f"true_{target}"] <= q_low
        high_mask = cv_predictions[f"true_{target}"] >= q_high
        target_slice_rows.append(
            {
                "target": target,
                "mae_all": float(absolute_error.mean()),
                "mae_low_tail": float(absolute_error[low_mask].mean()),
                "mae_high_tail": float(absolute_error[high_mask].mean()),
                "q10": float(q_low),
                "q90": float(q_high),
            }
        )
    write_csv(pd.DataFrame(target_slice_rows), diagnostics_dir / "surrogate_extreme_region_errors.csv")

    coverage_summary = {
        "dataset_rows": int(len(dataset)),
        "feature_ranges": {
            feature: {"min": float(dataset[feature].min()), "max": float(dataset[feature].max())}
            for feature in MORPHOLOGY_FEATURES
        },
        "targets": {
            target: {"min": float(dataset[target].min()), "max": float(dataset[target].max())}
            for target in PERFORMANCE_TARGETS
        },
    }
    write_json(coverage_summary, diagnostics_dir / "dataset_coverage_summary.json")

    nonlinear_path = diagnostics_dir / "nonlinear_response_profiles.png"
    nonlinear_status = "generated"
    try:
        model_bundle = load_surrogate(models_dir / "surrogate.pt")
        base_point = dataset[MORPHOLOGY_FEATURES].median().to_numpy(dtype=float)
        nonlinear_pairs = [("OSR", "EUIt"), ("FAR", "EG"), ("SVF", "H"), ("theta", "H")]
        fig, axes = plt.subplots(2, 2, figsize=(10, 8))
        for axis, (feature, target) in zip(axes.flatten(), nonlinear_pairs, strict=True):
            values = np.linspace(dataset[feature].quantile(0.05), dataset[feature].quantile(0.95), 100)
            responses = []
            feature_idx = MORPHOLOGY_FEATURES.index(feature)
            for value in values:
                probe = base_point.copy()
                probe[feature_idx] = value
                responses.append(model_bundle.predict_action(probe)[PERFORMANCE_TARGETS.index(target)])
            axis.plot(values, responses, color="#1D4ED8", linewidth=1.6)
            axis.set_xlabel(feature)
            axis.set_ylabel(target)
            axis.set_title(f"Partial response: {feature} -> {target}")
        fig.tight_layout()
        fig.savefig(nonlinear_path, dpi=220, bbox_inches="tight")
        plt.close(fig)
    except Exception as exc:  # noqa: BLE001
        nonlinear_status = f"skipped: {type(exc).__name__}: {exc}"

    diagnostics_payload = {
        "cv_rows": int(len(cv_predictions)),
        "ddpg_rows": int(len(ddpg_results)),
        "nsga_rows": int(len(nsga_results)),
        "hv_igd": compute_hv_igd_by_method(combined).to_dict(orient="records"),
        "preference_summary": preference_summary,
        "convergence": convergence,
        "convergence_seeded": convergence_seeded,
        "convergence_seeded_csv": str(convergence_seeded_path) if convergence_seeded_path is not None else "",
        "reevaluation_csv": str(reevaluation_csv),
        "nonlinear_response_profiles": str(nonlinear_path),
        "nonlinear_status": nonlinear_status,
        "dataset_coverage_summary": str(diagnostics_dir / "dataset_coverage_summary.json"),
    }
    write_json(diagnostics_payload, diagnostics_dir / "publication_diagnostics.json")
    return diagnostics_payload


def publication_review_pipeline(config: Config) -> dict:
    validate_publication_results(config)
    return run_revision_review(config)


def full_reproduce(config: Config, install_missing: bool = False) -> dict:
    bootstrap = bootstrap_pipeline(config, install_missing=install_missing)
    dataset = dataset_pipeline(config)
    _, surrogate_summary = surrogate_pipeline(config, dataset=dataset)
    _, metrics = optimizer_pipeline(config)
    report = report_pipeline(config)
    return {"bootstrap": bootstrap, "surrogate": surrogate_summary, "metrics": metrics, "report": report}

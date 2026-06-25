from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from paper_repro.constants import MORPHOLOGY_FEATURES, PERFORMANCE_TARGETS
from paper_repro.contracts import OPTIMIZATION_RESULT_COLUMNS, write_csv, write_json
from paper_repro.physical_stack import project_candidates_to_nearest_blocks
from paper_repro.round2 import (
    ROUND2_FEATURES,
    ROUND2_TARGETS,
    UTILITY_SCENARIOS,
    _append_utilities,
    _feasible_pool_random_resampling,
    _load_baseline_optimization,
    _load_round2_dataset,
    build_fixed_reference,
    dedupe_objective_tuples,
    evaluate_archive_metrics,
    parse_physical_results_frame,
    prepare_round2_workspace,
    sha256_path,
    theoretical_max_hv,
)

CANONICAL_REFERENCE_PROTOCOL = "benchmark-reference-v2"
ROUND2_LOCK_VERSION = "reviewer-round2-canonical-result-lock-v1"
TUPLE_ROUNDING_PRECISION = 12
FORBIDDEN_PHYSICAL_WORDING = {
    "successful physical validation",
    "physical closure",
    "external confirmation",
    "physical support for optimizer ranking",
    "strong validation",
}


def _group_method_scenario(group_name: str) -> tuple[str, str]:
    if "::" in group_name:
        method, scenario = group_name.split("::", 1)
        return method, scenario
    return group_name, group_name


def _row_key(frame: pd.DataFrame, columns: list[str], *, decimals: int = TUPLE_ROUNDING_PRECISION) -> pd.Series:
    return frame[columns].apply(
        lambda row: "|".join(f"{round(float(value), decimals):.{decimals}f}" for value in row),
        axis=1,
    )


def _unique_candidate_count(frame: pd.DataFrame) -> int:
    return int(_row_key(frame, ROUND2_FEATURES).nunique()) if not frame.empty else 0


def _unique_objective_count(frame: pd.DataFrame) -> int:
    return int(_row_key(frame, list(ROUND2_TARGETS)).nunique()) if not frame.empty else 0


def canonical_reference_hash(reference_front: np.ndarray, *, decimals: int = TUPLE_ROUNDING_PRECISION) -> str:
    rounded = [[round(float(value), decimals) for value in row] for row in np.asarray(reference_front, dtype=float)]
    rounded.sort()
    payload = json.dumps(rounded, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_round2_group_archives(base_config: Any, paths: Any) -> tuple[dict[str, pd.DataFrame], dict[str, Any]]:
    dataset = _load_round2_dataset(base_config)
    ddpg_baseline, nsga_baseline, baseline_combined = _load_baseline_optimization(base_config)
    utility_weights = base_config["optimization"]["utility_weights"]
    cma_results_path = paths.optimization_dir / "cmaes_results_round2.csv"
    cma_archive_path = paths.optimization_dir / "cmaes_archive_round2.csv"
    random_results_path = paths.optimization_dir / "random_search_results_round2.csv"
    random_archive_path = paths.optimization_dir / "random_search_archive_round2.csv"
    required = [cma_results_path, cma_archive_path, random_results_path, random_archive_path]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(
            "Canonical result lock requires existing optimizer archives. Missing: " + ", ".join(missing)
        )

    cma_results = pd.read_csv(cma_results_path)
    cma_archive = pd.read_csv(cma_archive_path)
    random_results = pd.read_csv(random_results_path)
    random_archive = pd.read_csv(random_archive_path)
    feasible_pool_random = _feasible_pool_random_resampling(
        dataset,
        utility_weights,
        master_seed=int(base_config["round2"]["master_seed"]),
        evaluation_budget=int(base_config["optimization"]["random_search"]["evaluation_budget"]),
        seeds_per_scenario=int(base_config["optimization"]["random_search"]["seeds_per_scenario"]),
    )

    group_archives: dict[str, pd.DataFrame] = {}
    for scenario_name in UTILITY_SCENARIOS:
        group_archives[f"DDPG::{scenario_name}"] = ddpg_baseline.loc[ddpg_baseline["scenario"] == scenario_name].copy()
        group_archives[f"CMA-ES::{scenario_name}"] = cma_archive.loc[cma_archive["scenario"] == scenario_name].copy()
        group_archives[f"RandomSearch::{scenario_name}"] = random_archive.loc[random_archive["scenario"] == scenario_name].copy()
        group_archives[f"FeasiblePoolRandom::{scenario_name}"] = feasible_pool_random.loc[
            feasible_pool_random["scenario"] == scenario_name
        ].copy()
    group_archives["NSGA-II"] = nsga_baseline.copy()

    sources = {
        "ddpg_results": Path(base_config["round2"]["baseline_runs"]["ddpg_results"]),
        "nsga2_results": Path(base_config["round2"]["baseline_runs"]["nsga2_results"]),
        "cmaes_results_round2": cma_results_path,
        "cmaes_archive_round2": cma_archive_path,
        "random_search_results_round2": random_results_path,
        "random_search_archive_round2": random_archive_path,
        "canonical_dataset": Path(base_config["round2"]["canonical_dataset"]),
    }
    metadata = {
        "dataset": dataset,
        "baseline_combined": baseline_combined,
        "utility_weights": utility_weights,
        "cma_results": cma_results,
        "random_results": random_results,
        "feasible_pool_random": feasible_pool_random,
        "source_files": {
            name: {
                "path": str(path),
                "sha256": sha256_path(path),
            }
            for name, path in sources.items()
        },
    }
    return group_archives, metadata


def _build_reference_payload(
    base_config: Any,
    group_archives: dict[str, pd.DataFrame],
    reference: dict[str, Any],
    source_files: dict[str, Any],
) -> dict[str, Any]:
    group_rows = []
    for group_name in sorted(group_archives):
        frame = group_archives[group_name]
        method, scenario = _group_method_scenario(group_name)
        group_rows.append(
            {
                "group": group_name,
                "method": method,
                "scenario": scenario,
                "rows": int(len(frame)),
                "unique_candidates": _unique_candidate_count(frame),
                "unique_objective_tuples": _unique_objective_count(frame),
            }
        )
    return {
        "protocol_name": CANONICAL_REFERENCE_PROTOCOL,
        "implementation_version": ROUND2_LOCK_VERSION,
        "objective_orientation": {"EUIt": "minimize", "EG": "maximize", "H": "maximize"},
        "canonical_candidate_universe": "Union of per-group retained archives from DDPG, NSGA-II, CMA-ES, RandomSearch, and deterministic FeasiblePoolRandom.",
        "source_files": source_files,
        "representation_groups": group_rows,
        "duplicate_policy": "Duplicates remain in metric evaluation; lineage hashes round tuples to 12 decimals for stable identity.",
        "non_dominated_policy": "Compute the non-dominated front inside each group in minimization space [EUIt, -EG, -H], then union those fronts to derive the fixed ideal, nadir, and reference front.",
        "ideal_vector": [float(value) for value in reference["ideal"]],
        "nadir_vector": [float(value) for value in reference["nadir"]],
        "reference_point": [float(value) for value in reference["reference_point"]],
        "normalized_reference_front_hash": canonical_reference_hash(reference["reference_front"]),
        "reference_front_rows": int(len(reference["reference_front"])),
        "tuple_rounding_precision": TUPLE_ROUNDING_PRECISION,
        "random_seed": int(base_config["round2"]["master_seed"]),
        "protocol_version": ROUND2_LOCK_VERSION,
    }


def build_projected_metric_rows(
    dataset: pd.DataFrame,
    group_archives: dict[str, pd.DataFrame],
    reference: dict[str, Any],
) -> tuple[dict[str, pd.DataFrame], dict[str, dict[str, Any]], pd.DataFrame]:
    projected_groups: dict[str, pd.DataFrame] = {}
    metadata: dict[str, dict[str, Any]] = {}
    for group_name, frame in group_archives.items():
        projected = project_candidates_to_nearest_blocks(frame, dataset)
        projected_unique = projected.drop_duplicates(subset=["matched_sample_id"], keep="first").reset_index(drop=True)
        method, scenario = _group_method_scenario(group_name)
        projected_rows = dataset.loc[dataset["sample_id"].isin(projected_unique["matched_sample_id"].astype(int))].copy()
        projected_rows["method"] = method
        projected_rows["scenario"] = scenario
        projected_rows["seed"] = -1
        projected_rows["reward"] = np.nan
        projected_groups[group_name] = projected_rows[OPTIMIZATION_RESULT_COLUMNS]
        metadata[group_name] = {
            "source_archive_size": int(len(frame)),
            "unique_matched_sample_count": int(projected_unique["matched_sample_id"].nunique()),
            "projection_duplicate_collapse_rate": 1.0 - projected_unique["matched_sample_id"].nunique() / max(len(frame), 1),
            "analytic_target": "nearest-block stored analytic output",
        }
    metrics = evaluate_archive_metrics(projected_groups, reference)
    return projected_groups, metadata, metrics


def build_equal_size_tables(
    group_archives: dict[str, pd.DataFrame],
    reference: dict[str, Any],
    *,
    sizes: list[int],
    repetitions: int,
    master_seed: int,
    legacy_repetition_frame: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    repetition_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    group_order = list(group_archives)
    legacy_lookup: dict[tuple[str, int], pd.DataFrame] = {}
    if legacy_repetition_frame is not None and not legacy_repetition_frame.empty:
        for (group_name, requested_size), frame in legacy_repetition_frame.groupby(["group", "requested_size"], sort=False):
            legacy_lookup[(str(group_name), int(requested_size))] = frame.sort_values("replicate", kind="mergesort").reset_index(drop=True)
    sequential_rng = np.random.default_rng(master_seed) if legacy_lookup else None

    for group_index, group_name in enumerate(group_order):
        frame = group_archives[group_name].reset_index(drop=True)
        method, scenario = _group_method_scenario(group_name)
        source_archive_size = int(len(frame))
        candidate_keys = _row_key(frame, ROUND2_FEATURES).to_numpy()
        objective_keys = _row_key(frame, list(ROUND2_TARGETS)).to_numpy()
        unique_before_sampling = int(pd.unique(candidate_keys).size) if len(candidate_keys) else 0
        for size_index, requested_size in enumerate(sizes):
            effective_sample_size = min(int(requested_size), source_archive_size)
            is_oversized_request = source_archive_size < int(requested_size)
            is_ddpg = method == "DDPG"

            if is_oversized_request and sequential_rng is not None and not is_ddpg:
                for _ in range(repetitions):
                    sequential_rng.choice(source_archive_size, size=effective_sample_size, replace=False)

            if is_oversized_request:
                summary_rows.append(
                    {
                        "group": group_name,
                        "method": method,
                        "scenario": scenario,
                        "requested_sample_size": int(requested_size),
                        "effective_sample_size": np.nan,
                        "source_archive_size": source_archive_size,
                        "repetitions": 0,
                        "replacement": False,
                        "status": "not_applicable",
                        "not_applicable_reason": "requested size exceeds available retained rows",
                        "unique_before_sampling": unique_before_sampling,
                        "HV_mean": np.nan,
                        "HV_std": np.nan,
                        "HV_q05": np.nan,
                        "HV_median": np.nan,
                        "HV_q95": np.nan,
                        "IGD_mean": np.nan,
                        "IGD_std": np.nan,
                        "IGD_q05": np.nan,
                        "IGD_median": np.nan,
                        "IGD_q95": np.nan,
                        "sampled_unique_objective_tuples_mean": np.nan,
                        "sampled_unique_objective_tuples_std": np.nan,
                    }
                )
                continue

            group_rows: list[dict[str, Any]] = []
            if is_ddpg:
                metrics = evaluate_archive_metrics({group_name: frame}, reference).iloc[0]
                for repetition in range(repetitions):
                    row = {
                        "group": group_name,
                        "method": method,
                        "scenario": scenario,
                        "requested_sample_size": int(requested_size),
                        "effective_sample_size": int(effective_sample_size),
                        "source_archive_size": source_archive_size,
                        "repetition": repetition,
                        "replacement": False,
                        "status": "valid",
                        "random_seed": int(master_seed),
                        "unique_before_sampling": unique_before_sampling,
                        "unique_after_sampling": unique_before_sampling,
                        "sampled_unique_objective_tuples": int(pd.unique(objective_keys).size) if len(objective_keys) else 0,
                        "HV": float(metrics["HV"]),
                        "IGD": float(metrics["IGD"]),
                    }
                    repetition_rows.append(row)
                    group_rows.append(row)
            else:
                legacy_rows = legacy_lookup.get((group_name, int(requested_size)))
                for repetition in range(repetitions):
                    if sequential_rng is not None:
                        indices = sequential_rng.choice(source_archive_size, size=effective_sample_size, replace=False)
                    else:
                        random_seed = master_seed + group_index * 100_000 + size_index * 10_000 + repetition
                        indices = np.random.default_rng(random_seed).choice(
                            source_archive_size,
                            size=effective_sample_size,
                            replace=False,
                        )
                    sampled = frame.iloc[np.asarray(indices, dtype=int)].reset_index(drop=True)
                    assert len(sampled) == effective_sample_size
                    assert sampled.index.nunique() == effective_sample_size
                    if legacy_rows is not None:
                        metric_row = legacy_rows.iloc[repetition]
                        hv_value = float(metric_row["HV"])
                        igd_value = float(metric_row["IGD"])
                    else:
                        metrics = evaluate_archive_metrics({group_name: sampled}, reference).iloc[0]
                        hv_value = float(metrics["HV"])
                        igd_value = float(metrics["IGD"])
                    row = {
                        "group": group_name,
                        "method": method,
                        "scenario": scenario,
                        "requested_sample_size": int(requested_size),
                        "effective_sample_size": int(effective_sample_size),
                        "source_archive_size": source_archive_size,
                        "repetition": repetition,
                        "replacement": False,
                        "status": "valid",
                        "random_seed": int(master_seed if sequential_rng is not None else master_seed + group_index * 100_000 + size_index * 10_000 + repetition),
                        "unique_before_sampling": unique_before_sampling,
                        "unique_after_sampling": int(pd.unique(candidate_keys[np.asarray(indices, dtype=int)]).size),
                        "sampled_unique_objective_tuples": int(pd.unique(objective_keys[np.asarray(indices, dtype=int)]).size),
                        "HV": hv_value,
                        "IGD": igd_value,
                    }
                    repetition_rows.append(row)
                    group_rows.append(row)

            valid_frame = pd.DataFrame(group_rows)
            summary_rows.append(
                {
                    "group": group_name,
                    "method": method,
                    "scenario": scenario,
                    "requested_sample_size": int(requested_size),
                    "effective_sample_size": int(effective_sample_size),
                    "source_archive_size": source_archive_size,
                    "repetitions": int(len(valid_frame)),
                    "replacement": False,
                    "status": "valid",
                    "not_applicable_reason": "",
                    "unique_before_sampling": unique_before_sampling,
                    "HV_mean": float(valid_frame["HV"].mean()),
                    "HV_std": float(valid_frame["HV"].std(ddof=1)) if len(valid_frame) > 1 else np.nan,
                    "HV_q05": float(valid_frame["HV"].quantile(0.05)),
                    "HV_median": float(valid_frame["HV"].median()),
                    "HV_q95": float(valid_frame["HV"].quantile(0.95)),
                    "IGD_mean": float(valid_frame["IGD"].mean()),
                    "IGD_std": float(valid_frame["IGD"].std(ddof=1)) if len(valid_frame) > 1 else np.nan,
                    "IGD_q05": float(valid_frame["IGD"].quantile(0.05)),
                    "IGD_median": float(valid_frame["IGD"].median()),
                    "IGD_q95": float(valid_frame["IGD"].quantile(0.95)),
                    "sampled_unique_objective_tuples_mean": float(valid_frame["sampled_unique_objective_tuples"].mean()),
                    "sampled_unique_objective_tuples_std": (
                        float(valid_frame["sampled_unique_objective_tuples"].std(ddof=1))
                        if len(valid_frame) > 1
                        else np.nan
                    ),
                }
            )

    return pd.DataFrame(repetition_rows), pd.DataFrame(summary_rows)


def _build_canonical_metric_rows(
    group_archives: dict[str, pd.DataFrame],
    reference_hash: str,
    full_archive: pd.DataFrame,
    unique_objective_metrics: pd.DataFrame,
    projected_groups: dict[str, pd.DataFrame],
    projected_metadata: dict[str, dict[str, Any]],
    projected_metrics: pd.DataFrame,
    equal_size_summary: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, metric_row in full_archive.iterrows():
        group_name = str(metric_row["group"])
        method, scenario = _group_method_scenario(group_name)
        frame = group_archives[group_name]
        rows.append(
            {
                "representation_family": "descriptor_clipped_full_archive",
                "group": group_name,
                "method": method,
                "scenario": scenario,
                "source_archive_size": int(len(frame)),
                "unique_candidate_size": _unique_candidate_count(frame),
                "evaluated_archive_size": int(len(frame)),
                "non_dominated_size": int(metric_row["non_dominated_rows"]),
                "requested_sample_size": np.nan,
                "effective_sample_size": np.nan,
                "HV": float(metric_row["HV"]),
                "IGD": float(metric_row["IGD"]),
                "reference_protocol": CANONICAL_REFERENCE_PROTOCOL,
                "reference_hash": reference_hash,
                "status": "valid",
                "note": "fixed benchmark reference",
            }
        )
    for _, metric_row in unique_objective_metrics.iterrows():
        group_name = str(metric_row["group"])
        method, scenario = _group_method_scenario(group_name)
        source_frame = group_archives[group_name]
        unique_frame = dedupe_objective_tuples(source_frame[["method", "scenario", "seed", *ROUND2_FEATURES, *ROUND2_TARGETS, "reward"]])
        rows.append(
            {
                "representation_family": "descriptor_unique_objective_archive",
                "group": group_name,
                "method": method,
                "scenario": scenario,
                "source_archive_size": int(len(source_frame)),
                "unique_candidate_size": _unique_candidate_count(unique_frame),
                "evaluated_archive_size": int(len(unique_frame)),
                "non_dominated_size": int(metric_row["non_dominated_rows"]),
                "requested_sample_size": np.nan,
                "effective_sample_size": np.nan,
                "HV": float(metric_row["HV"]),
                "IGD": float(metric_row["IGD"]),
                "reference_protocol": CANONICAL_REFERENCE_PROTOCOL,
                "reference_hash": reference_hash,
                "status": "valid",
                "note": "fixed benchmark reference",
            }
        )
    for _, metric_row in projected_metrics.iterrows():
        group_name = str(metric_row["group"])
        method, scenario = _group_method_scenario(group_name)
        projected_frame = projected_groups[group_name]
        meta = projected_metadata[group_name]
        rows.append(
            {
                "representation_family": "projected_feasible_morphology_archive",
                "group": group_name,
                "method": method,
                "scenario": scenario,
                "source_archive_size": int(meta["source_archive_size"]),
                "unique_candidate_size": int(meta["unique_matched_sample_count"]),
                "evaluated_archive_size": int(len(projected_frame)),
                "non_dominated_size": int(metric_row["non_dominated_rows"]),
                "requested_sample_size": np.nan,
                "effective_sample_size": np.nan,
                "HV": float(metric_row["HV"]),
                "IGD": float(metric_row["IGD"]),
                "reference_protocol": CANONICAL_REFERENCE_PROTOCOL,
                "reference_hash": reference_hash,
                "status": "valid",
                "note": "fixed benchmark reference on projected feasible morphology archive",
            }
        )
    for _, metric_row in equal_size_summary.iterrows():
        rows.append(
            {
                "representation_family": "descriptor_equal_size_archive",
                "group": str(metric_row["group"]),
                "method": str(metric_row["method"]),
                "scenario": str(metric_row["scenario"]),
                "source_archive_size": int(metric_row["source_archive_size"]),
                "unique_candidate_size": int(metric_row["unique_before_sampling"]),
                "evaluated_archive_size": (
                    int(metric_row["effective_sample_size"])
                    if pd.notna(metric_row["effective_sample_size"])
                    else np.nan
                ),
                "non_dominated_size": np.nan,
                "requested_sample_size": int(metric_row["requested_sample_size"]),
                "effective_sample_size": metric_row["effective_sample_size"],
                "HV": metric_row["HV_mean"],
                "IGD": metric_row["IGD_mean"],
                "reference_protocol": CANONICAL_REFERENCE_PROTOCOL,
                "reference_hash": reference_hash,
                "status": str(metric_row["status"]),
                "note": str(metric_row["not_applicable_reason"]),
            }
        )
    return pd.DataFrame(rows)


def validate_result_registry(entries: list[dict[str, Any]]) -> list[str]:
    issues: list[str] = []
    for entry in entries:
        source = Path(str(entry["source_file"]))
        if not source.exists():
            issues.append(f"missing source file for {entry['result_id']}: {source}")
            continue
        actual_sha = sha256_path(source)
        if actual_sha != entry["source_sha256"]:
            issues.append(f"sha mismatch for {entry['result_id']}: {source}")
        if entry.get("value_status") == "superseded" and entry.get("valid_for_main_text"):
            issues.append(f"superseded result cannot be valid_for_main_text: {entry['result_id']}")
        value_lower = str(entry.get("value", "")).strip().lower()
        if entry["result_id"].startswith("physical_") and value_lower in FORBIDDEN_PHYSICAL_WORDING:
            issues.append(f"forbidden physical wording in registry: {entry['result_id']}")
    return issues


def _registry_entry(
    *,
    result_id: str,
    scientific_meaning: str,
    value: Any,
    unit: str,
    representation_family: str,
    source_file: str | Path,
    source_row_filter: str,
    generation_command: str,
    reference_hash: str | None,
    claim_boundary: str,
    valid_for_main_text: bool,
    valid_for_appendix: bool,
    supersedes: list[str] | None = None,
) -> dict[str, Any]:
    source = Path(source_file)
    return {
        "result_id": result_id,
        "scientific_meaning": scientific_meaning,
        "value": value,
        "unit": unit,
        "representation_family": representation_family,
        "source_file": str(source),
        "source_row_filter": source_row_filter,
        "source_sha256": sha256_path(source),
        "generation_command": generation_command,
        "reference_protocol": CANONICAL_REFERENCE_PROTOCOL if reference_hash else "",
        "reference_hash": reference_hash or "",
        "valid_for_main_text": bool(valid_for_main_text),
        "valid_for_appendix": bool(valid_for_appendix),
        "supersedes": supersedes or [],
        "claim_boundary": claim_boundary,
    }


def _build_metric_lineage_records(
    reference_hash: str,
    equal_size_summary_v2: pd.DataFrame,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = [
        {
            "source_file": "research/reviewer-round-02/optimizer-projection-summary.csv",
            "source_metric": "projected_HV/projected_IGD",
            "representation_family": "projected_feasible_morphology_archive",
            "status": "valid_but_local_reference",
            "reference_hash": "",
            "replacement_file": "research/reviewer-round-02/canonical_benchmark_metrics.csv",
            "root_cause": "run_feasibility_audit rebuilt a projected-only reference front before computing HV/IGD.",
        },
        {
            "source_file": "research/reviewer-round-02/benchmark-metric-definition-audit.csv",
            "source_metric": "projected_feasible_block_archive",
            "representation_family": "projected_feasible_morphology_archive",
            "status": "valid_fixed_reference",
            "reference_hash": reference_hash,
            "replacement_file": "research/reviewer-round-02/canonical_benchmark_metrics.csv",
            "root_cause": "run_benchmark_fairness evaluated projected archives against the fixed benchmark reference.",
        },
        {
            "source_file": "research/reviewer-round-02/benchmark-equal-size-summary.csv",
            "source_metric": "actual_size column",
            "representation_family": "descriptor_equal_size_archive",
            "status": "metadata_error",
            "reference_hash": reference_hash,
            "replacement_file": "research/reviewer-round-02/benchmark_equal_size_summary_v2.csv",
            "root_cause": "The summary stored source archive size in a column that reads like sampled row count.",
        },
    ]
    invalid_rows = equal_size_summary_v2.loc[equal_size_summary_v2["status"] == "not_applicable", ["group", "requested_sample_size"]]
    for row in invalid_rows.itertuples():
        records.append(
            {
                "source_file": "research/reviewer-round-02/benchmark-equal-size-summary.csv",
                "source_metric": f"{row.group} requested_size={int(row.requested_sample_size)}",
                "representation_family": "descriptor_equal_size_archive",
                "status": "obsolete",
                "reference_hash": reference_hash,
                "replacement_file": "research/reviewer-round-02/benchmark_equal_size_summary_v2.csv",
                "root_cause": "Older tables reported oversized equal-size requests as if they were valid metric rows.",
            }
        )
    return records


def _render_metric_lineage_md(records: list[dict[str, Any]]) -> str:
    lines = [
        "# Metric Lineage Audit",
        "",
        "## Root cause",
        "- `optimizer_projection_summary.csv` was generated with a projected-only local reference front.",
        "- `benchmark-metric-definition-audit.csv` evaluated projected archives against the fixed benchmark reference.",
        "- `benchmark-equal-size-summary.csv` overloaded `actual_size` with source archive size and silently kept oversized requests.",
        "",
        "## Lineage records",
    ]
    for record in records:
        lines.extend(
            [
                f"- `{record['source_file']}` / `{record['source_metric']}`",
                f"  status: `{record['status']}`",
                f"  replacement: `{record['replacement_file']}`",
                f"  note: {record['root_cause']}",
            ]
        )
    return "\n".join(lines)


def _build_optimizer_output_contract(
    base_config: Any,
    group_archives: dict[str, pd.DataFrame],
    projected_metadata: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    nsga_budget = int(base_config["optimization"]["nsga2"]["evaluation_budget"])
    random_budget = int(base_config["optimization"]["random_search"]["evaluation_budget"])
    cma_budget = int(base_config["optimization"]["cmaes"]["evaluation_budget"])
    ddpg_budget = int(base_config["optimization"]["ddpg"]["max_episodes"]) * int(
        base_config["optimization"]["ddpg"]["max_steps_per_episode"]
    )
    for group_name, frame in sorted(group_archives.items()):
        method, scenario = _group_method_scenario(group_name)
        seeds = int(frame["seed"].nunique())
        projected_unique = int(projected_metadata.get(group_name, {}).get("unique_matched_sample_count", np.nan))
        unique_objectives = _unique_objective_count(frame)
        if method == "DDPG":
            retained_object = "best scalarized candidate"
            queries_per_seed = ddpg_budget
            archive_role = "diagnostic"
            naturally_pareto = False
        elif method == "NSGA-II":
            retained_object = "final population"
            queries_per_seed = nsga_budget
            archive_role = "primary"
            naturally_pareto = True
        elif method == "CMA-ES":
            retained_object = "top-reward queried archive"
            queries_per_seed = cma_budget
            archive_role = "diagnostic"
            naturally_pareto = False
        elif method == "RandomSearch":
            retained_object = "top-reward queried archive"
            queries_per_seed = random_budget
            archive_role = "diagnostic"
            naturally_pareto = False
        else:
            retained_object = "feasible-pool best"
            queries_per_seed = random_budget
            archive_role = "diagnostic"
            naturally_pareto = False
        rows.append(
            {
                "method": method,
                "scenario": scenario,
                "seeds": seeds,
                "queries_per_seed": queries_per_seed,
                "retained_rows_per_seed": int(len(frame) / max(seeds, 1)),
                "retained_object": retained_object,
                "total_retained_rows": int(len(frame)),
                "unique_objective_tuples": unique_objectives,
                "unique_feasible_projections": projected_unique,
                "naturally_pareto_archive_producing": naturally_pareto,
                "archive_hv_igd_role": archive_role,
            }
        )
    return pd.DataFrame(rows)


def _render_hv_ceiling_md(
    group_archives: dict[str, pd.DataFrame],
    full_archive: pd.DataFrame,
    decomposition_frame: pd.DataFrame,
    reference_point: list[float],
) -> str:
    max_hv = theoretical_max_hv(reference_point)
    nds = NonDominatedSorting()
    lines = [
        "# HV Ceiling Interpretation",
        "",
        f"- Fixed reference point: `{reference_point}`.",
        f"- Theoretical maximum HV: `{max_hv:.6f}`.",
        "- HV values near the ceiling indicate reference-volume saturation, not archive richness by themselves.",
        "",
        "## Group diagnostics",
    ]
    for _, row in full_archive.iterrows():
        group_name = str(row["group"])
        group = decomposition_frame.loc[decomposition_frame["group"] == group_name].copy()
        unique_frame = group[["method", "scenario", "seed", *ROUND2_FEATURES, *ROUND2_TARGETS]].copy()
        unique_frame["reward"] = np.nan
        unique_frame = dedupe_objective_tuples(unique_frame)
        matrix = np.column_stack(
            [
                unique_frame["EUIt"].to_numpy(dtype=float),
                -unique_frame["EG"].to_numpy(dtype=float),
                -unique_frame["H"].to_numpy(dtype=float),
            ]
        )
        front_idx = nds.do(matrix, only_non_dominated_front=True) if len(matrix) else []
        method, scenario = _group_method_scenario(group_name)
        lines.extend(
            [
                f"- `{group_name}`",
                f"  method/scenario: `{method}` / `{scenario}`",
                f"  HV = `{float(row['HV']):.6f}`, distance_to_hv_ceiling = `{(max_hv - float(row['HV'])):.6f}`, fraction_of_theoretical_max = `{(float(row['HV']) / max_hv):.6f}`",
                f"  clipped_utopia_rows = `{int(group['is_exact_utopian_tuple'].sum())}` / `{len(group)}`",
                f"  unique_objective_tuples = `{_unique_objective_count(group)}`",
                f"  unique_non_dominated_tuples = `{len(front_idx)}`",
            ]
        )
    lines.extend(
        [
            "",
            "## Interpretation rules",
            "- CMA-ES ceiling-hitting rows mostly collapse onto one or two clipped objective tuples, so they do not imply a richer Pareto archive.",
            "- NSGA-II stays near the ceiling because it covers the normalized reference box well; diversity still needs separate tuple-count or spread diagnostics.",
        ]
    )
    return "\n".join(lines)


def _build_physical_evidence_levels() -> dict[str, str]:
    return {
        "execution_closure": "complete",
        "metric_agreement": "weak",
        "ranking_transfer": "unsupported",
        "optimizer_superiority_under_physical_evaluation": "unsupported",
    }


def _build_registry_entries(
    paths: Any,
    reference_hash: str,
    equal_size_summary_v2: pd.DataFrame,
    physical_evidence_levels: dict[str, str],
) -> list[dict[str, Any]]:
    dependencies = pd.read_csv(paths.data_dir / "descriptor_dependencies.csv")
    surrogate_summary = pd.read_csv(paths.research_root / "surrogate-validation-summary.csv")
    fig9 = pd.read_csv(paths.research_root / "fig9d_utility_recalc.csv")
    canonical_metrics = pd.read_csv(paths.research_root / "canonical_benchmark_metrics.csv")
    physical_metrics = pd.read_csv(paths.research_root / "physical-validation-metrics.csv")
    climate_rank = pd.read_csv(paths.research_root / "climate-rank-stability.csv")
    climate_summary = pd.read_csv(paths.research_root / "climate-sensitivity-summary.csv")
    runtime_audit = pd.read_csv(paths.optimization_dir / "runtime_audit.csv")
    publication_diagnostics = json.loads(
        Path("artifacts/publication/diagnostics/publication_diagnostics.json").read_text(encoding="utf-8")
    )
    entries = [
        _registry_entry(
            result_id="sample_coverage.components_for_95_variance",
            scientific_meaning="Principal components required to explain 95% of descriptor variance.",
            value=int(dependencies.loc[dependencies["metric"] == "components_for_95_variance", "value"].iloc[0]),
            unit="components",
            representation_family="sample_coverage",
            source_file=paths.data_dir / "descriptor_dependencies.csv",
            source_row_filter="metric == components_for_95_variance",
            generation_command="python tools/run_round2_feasibility_audit.py --config configs/reviewer_round2_experiments.yaml --run-id 20260625_round2_closure",
            reference_hash=None,
            claim_boundary="Describes descriptor-space coverage only; not an optimizer comparison.",
            valid_for_main_text=True,
            valid_for_appendix=True,
        ),
        _registry_entry(
            result_id="surrogate_validation.repeated_kfold.EG_mean_nmae",
            scientific_meaning="Repeated 5x5 CV mean nMAE for EG.",
            value=float(
                surrogate_summary.loc[
                    (surrogate_summary["validation_family"] == "repeated_kfold")
                    & (surrogate_summary["target"] == "EG"),
                    "mean_nMAE",
                ].iloc[0]
            ),
            unit="nMAE",
            representation_family="repeated_cv",
            source_file=paths.research_root / "surrogate-validation-summary.csv",
            source_row_filter="validation_family == repeated_kfold and target == EG",
            generation_command="python tools/run_round2_surrogate_validation.py --config configs/reviewer_round2_experiments.yaml --run-id 20260625_round2_closure",
            reference_hash=None,
            claim_boundary="Supports surrogate accuracy on the selected checkpoint only.",
            valid_for_main_text=True,
            valid_for_appendix=True,
        ),
        _registry_entry(
            result_id="surrogate_validation.leave_one_osli_out.EG_mean_nmae",
            scientific_meaning="Leave-one-OSLI-out mean nMAE for EG.",
            value=float(
                surrogate_summary.loc[
                    (surrogate_summary["validation_family"] == "leave_one_osli_out")
                    & (surrogate_summary["target"] == "EG"),
                    "mean_nMAE",
                ].iloc[0]
            ),
            unit="nMAE",
            representation_family="leave_one_osli_out",
            source_file=paths.research_root / "surrogate-validation-summary.csv",
            source_row_filter="validation_family == leave_one_osli_out and target == EG",
            generation_command="python tools/run_round2_surrogate_validation.py --config configs/reviewer_round2_experiments.yaml --run-id 20260625_round2_closure",
            reference_hash=None,
            claim_boundary="Supports held-out morphology-stratum accuracy only.",
            valid_for_main_text=True,
            valid_for_appendix=True,
        ),
        _registry_entry(
            result_id="surrogate_validation.outer_shell_holdout.EG_mean_nmae",
            scientific_meaning="Outer-shell holdout mean nMAE for EG.",
            value=float(
                surrogate_summary.loc[
                    (surrogate_summary["validation_family"] == "outer_shell_holdout")
                    & (surrogate_summary["target"] == "EG"),
                    "mean_nMAE",
                ].iloc[0]
            ),
            unit="nMAE",
            representation_family="outer_shell_holdout",
            source_file=paths.research_root / "surrogate-validation-summary.csv",
            source_row_filter="validation_family == outer_shell_holdout and target == EG",
            generation_command="python tools/run_round2_surrogate_validation.py --config configs/reviewer_round2_experiments.yaml --run-id 20260625_round2_closure",
            reference_hash=None,
            claim_boundary="Describes shell-boundary extrapolation on the selected dataset only.",
            valid_for_main_text=True,
            valid_for_appendix=True,
        ),
        _registry_entry(
            result_id="surrogate_validation.feature_tail_holdout.EG_mean_nmae",
            scientific_meaning="Feature-tail holdout mean nMAE for EG.",
            value=float(
                surrogate_summary.loc[
                    (surrogate_summary["validation_family"] == "feature_tail_holdout")
                    & (surrogate_summary["target"] == "EG"),
                    "mean_nMAE",
                ].iloc[0]
            ),
            unit="nMAE",
            representation_family="feature_tail_holdout",
            source_file=paths.research_root / "surrogate-validation-summary.csv",
            source_row_filter="validation_family == feature_tail_holdout and target == EG",
            generation_command="python tools/run_round2_surrogate_validation.py --config configs/reviewer_round2_experiments.yaml --run-id 20260625_round2_closure",
            reference_hash=None,
            claim_boundary="Describes tail-region surrogate behavior only.",
            valid_for_main_text=True,
            valid_for_appendix=True,
        ),
        _registry_entry(
            result_id="training_dynamics.balanced_performance.plateau_episode",
            scientific_meaning="Balanced DDPG plateau episode from legacy publication diagnostics.",
            value=float(publication_diagnostics["convergence"]["Balanced_Performance"]["plateau_episode"]),
            unit="episode",
            representation_family="training_dynamics",
            source_file="artifacts/publication/diagnostics/publication_diagnostics.json",
            source_row_filter="convergence.Balanced_Performance.plateau_episode",
            generation_command="python -m paper_repro.cli publication-diagnostics --config configs/revision.yaml",
            reference_hash=None,
            claim_boundary="Historical training-dynamics context; not a canonical benchmark-fairness metric.",
            valid_for_main_text=False,
            valid_for_appendix=True,
        ),
        _registry_entry(
            result_id="post_hoc_utility.balanced_performance.ddpg_minus_nsgaii",
            scientific_meaning="Balanced-scenario post-hoc utility gap between DDPG and NSGA-II.",
            value=float(fig9.loc[fig9["scenario"] == "Balanced_Performance", "DDPG_minus_NSGAII"].iloc[0]),
            unit="utility_delta",
            representation_family="post_hoc_utility",
            source_file=paths.research_root / "fig9d_utility_recalc.csv",
            source_row_filter="scenario == Balanced_Performance",
            generation_command="python tools/run_round2_benchmark_fairness.py --config configs/reviewer_round2_experiments.yaml --run-id 20260625_round2_closure --resume",
            reference_hash=None,
            claim_boundary="Post-hoc utility is not the DDPG training reward and should not be described as such.",
            valid_for_main_text=False,
            valid_for_appendix=True,
        ),
        _registry_entry(
            result_id="canonical_full_archive.nsga_hv",
            scientific_meaning="NSGA-II full-archive HV under the fixed benchmark reference.",
            value=float(
                canonical_metrics.loc[
                    (canonical_metrics["representation_family"] == "descriptor_clipped_full_archive")
                    & (canonical_metrics["group"] == "NSGA-II"),
                    "HV",
                ].iloc[0]
            ),
            unit="HV",
            representation_family="descriptor_clipped_full_archive",
            source_file=paths.research_root / "canonical_benchmark_metrics.csv",
            source_row_filter="representation_family == descriptor_clipped_full_archive and group == NSGA-II",
            generation_command="python tools/lock_round2_canonical_results.py --config configs/reviewer_round2_experiments.yaml --run-id 20260625_round2_closure",
            reference_hash=reference_hash,
            claim_boundary="Comparable only within the fixed benchmark reference defined in canonical-benchmark-reference.json.",
            valid_for_main_text=True,
            valid_for_appendix=True,
            supersedes=["research/reviewer-round-02/hv_igd_full_archive.csv"],
        ),
        _registry_entry(
            result_id="equal_size.nsga_requested_20.hv_mean",
            scientific_meaning="NSGA-II equal-size HV mean at requested sample size 20.",
            value=float(
                equal_size_summary_v2.loc[
                    (equal_size_summary_v2["group"] == "NSGA-II")
                    & (equal_size_summary_v2["requested_sample_size"] == 20),
                    "HV_mean",
                ].iloc[0]
            ),
            unit="HV",
            representation_family="descriptor_equal_size_archive",
            source_file=paths.research_root / "benchmark_equal_size_summary_v2.csv",
            source_row_filter="group == NSGA-II and requested_sample_size == 20",
            generation_command="python tools/lock_round2_canonical_results.py --config configs/reviewer_round2_experiments.yaml --run-id 20260625_round2_closure",
            reference_hash=reference_hash,
            claim_boundary="Fairness downsampling diagnostic only; only valid where requested size does not exceed the retained archive.",
            valid_for_main_text=True,
            valid_for_appendix=True,
        ),
        _registry_entry(
            result_id="projected_metrics.nsga_hv_fixed_reference",
            scientific_meaning="NSGA-II projected feasible HV under the fixed benchmark reference.",
            value=float(
                canonical_metrics.loc[
                    (canonical_metrics["representation_family"] == "projected_feasible_morphology_archive")
                    & (canonical_metrics["group"] == "NSGA-II"),
                    "HV",
                ].iloc[0]
            ),
            unit="HV",
            representation_family="projected_feasible_morphology_archive",
            source_file=paths.research_root / "canonical_benchmark_metrics.csv",
            source_row_filter="representation_family == projected_feasible_morphology_archive and group == NSGA-II",
            generation_command="python tools/lock_round2_canonical_results.py --config configs/reviewer_round2_experiments.yaml --run-id 20260625_round2_closure",
            reference_hash=reference_hash,
            claim_boundary="Projection metrics diagnose descriptor-to-feasible collapse; they do not validate physical optimizer ranking.",
            valid_for_main_text=False,
            valid_for_appendix=True,
            supersedes=["research/reviewer-round-02/optimizer-projection-summary.csv"],
        ),
        _registry_entry(
            result_id="physical_stress_test.euit_spearman",
            scientific_meaning="EUIt rank correlation in the limited physics-based cross-model stress test.",
            value=float(physical_metrics.loc[physical_metrics["target"] == "EUIt", "Spearman_rho"].iloc[0]),
            unit="spearman_rho",
            representation_family="physical_evaluated_subset",
            source_file=paths.research_root / "physical-validation-metrics.csv",
            source_row_filter="target == EUIt",
            generation_command="python tools/run_round2_physical_validation.py --config configs/reviewer_round2_experiments.yaml --run-id 20260625_round2_closure --resume",
            reference_hash=None,
            claim_boundary="Supports only a limited physics-based cross-model stress test with weak metric agreement and unsupported ranking transfer.",
            valid_for_main_text=False,
            valid_for_appendix=True,
        ),
        _registry_entry(
            result_id="physical_evidence.metric_agreement",
            scientific_meaning="Canonical evidence-level label for physical metric agreement.",
            value=physical_evidence_levels["metric_agreement"],
            unit="label",
            representation_family="physical_evaluated_subset",
            source_file=paths.research_root / "physical-validation-metrics.csv",
            source_row_filter="canonical label derived from physical-validation-metrics.csv",
            generation_command="python tools/lock_round2_canonical_results.py --config configs/reviewer_round2_experiments.yaml --run-id 20260625_round2_closure",
            reference_hash=None,
            claim_boundary="Physical evidence wording must stay at weak agreement and unsupported ranking transfer.",
            valid_for_main_text=True,
            valid_for_appendix=True,
        ),
        _registry_entry(
            result_id="climate_rank_stability.eg_harbin_spearman",
            scientific_meaning="EG rank stability under severe-cold climate sensitivity.",
            value=float(
                climate_rank.loc[
                    (climate_rank["station"] == "Harbin") & (climate_rank["rank_metric"] == "EG"),
                    "spearman",
                ].iloc[0]
            ),
            unit="spearman_rho",
            representation_family="limited_four_block_cross_climate_physical_sensitivity_analysis",
            source_file=paths.research_root / "climate-rank-stability.csv",
            source_row_filter="station == Harbin and rank_metric == EG",
            generation_command="python tools/run_round2_climate_sensitivity.py --config configs/reviewer_round2_experiments.yaml --run-id 20260625_round2_closure --resume",
            reference_hash=None,
            claim_boundary="Supports only limited four-block cross-climate physical sensitivity analysis, not surrogate generalization.",
            valid_for_main_text=False,
            valid_for_appendix=True,
        ),
        _registry_entry(
            result_id="climate_delta.harbin_mean_delta_euit",
            scientific_meaning="Mean EUIt delta under severe-cold climate sensitivity.",
            value=float(climate_summary.loc[climate_summary["station"] == "Harbin", "mean_delta_EUIt"].iloc[0]),
            unit="kWh/m2/y",
            representation_family="limited_four_block_cross_climate_physical_sensitivity_analysis",
            source_file=paths.research_root / "climate-sensitivity-summary.csv",
            source_row_filter="station == Harbin",
            generation_command="python tools/run_round2_climate_sensitivity.py --config configs/reviewer_round2_experiments.yaml --run-id 20260625_round2_closure --resume",
            reference_hash=None,
            claim_boundary="Describes climate sensitivity of four locked physical cases only.",
            valid_for_main_text=False,
            valid_for_appendix=True,
        ),
        _registry_entry(
            result_id="runtime_audit.feasible_pool_random_seconds_mean",
            scientific_meaning="Mean local runtime for FeasiblePoolRandom runtime audit.",
            value=float(runtime_audit.loc[runtime_audit["method"] == "FeasiblePoolRandom", "seconds"].mean()),
            unit="seconds",
            representation_family="runtime_audit",
            source_file=paths.optimization_dir / "runtime_audit.csv",
            source_row_filter="method == FeasiblePoolRandom",
            generation_command="python tools/run_round2_benchmark_fairness.py --config configs/reviewer_round2_experiments.yaml --run-id 20260625_round2_closure --resume",
            reference_hash=None,
            claim_boundary="Runtime audit is implementation context, not scientific proof.",
            valid_for_main_text=False,
            valid_for_appendix=True,
        ),
    ]
    return entries


def _render_canonical_result_lock_md(
    canonical_metrics: pd.DataFrame,
    equal_size_summary_v2: pd.DataFrame,
    lineage_records: list[dict[str, Any]],
) -> str:
    main_text_rows = canonical_metrics.loc[
        (
            (canonical_metrics["representation_family"] == "descriptor_clipped_full_archive")
            & (canonical_metrics["group"] == "NSGA-II")
        )
        | (
            (canonical_metrics["representation_family"] == "descriptor_equal_size_archive")
            & (canonical_metrics["status"] == "valid")
            & (canonical_metrics["requested_sample_size"] == 20)
        )
    ]
    appendix_rows = canonical_metrics.loc[
        canonical_metrics["status"].eq("valid")
        & ~canonical_metrics.index.isin(main_text_rows.index)
    ]
    lines = [
        "# Canonical Result Lock",
        "",
        "## Main-text-eligible values",
    ]
    for row in main_text_rows.itertuples():
        label = f"{row.representation_family} / {row.group}"
        if pd.notna(row.requested_sample_size):
            label += f" / requested={int(row.requested_sample_size)}"
        lines.append(f"- `{label}`: HV = `{row.HV}`; IGD = `{row.IGD}`")
    lines.extend(
        [
            "",
            "## Appendix-only values",
        ]
    )
    for row in appendix_rows.itertuples():
        lines.append(f"- `{row.representation_family} / {row.group}`: HV = `{row.HV}`; IGD = `{row.IGD}`")
    lines.extend(
        [
            "",
            "## Superseded values",
        ]
    )
    for record in lineage_records:
        if record["status"] in {"obsolete", "metadata_error", "valid_but_local_reference"}:
            lines.append(f"- `{record['source_file']}` / `{record['source_metric']}` -> `{record['status']}`")
    lines.extend(
        [
            "",
            "## Valid but not cross-table comparable",
            "- Local-reference projected HV/IGD values remain valid as diagnostics only and must keep the `local_reference` label.",
            "- Physical-evaluated subsets are too small and structurally asymmetric for primary HV/IGD comparison.",
            "- DDPG, CMA-ES, RandomSearch, and FeasiblePoolRandom full-archive rows remain diagnostic because their retained outputs are not symmetric Pareto archives.",
            "",
            "## Conclusions that must be removed",
            "- Any wording that treats the physical run as successful validation, physical closure, or external confirmation.",
            "- Any claim that physical evaluation supports optimizer superiority or benchmark ranking transfer.",
            "- Any claim that CMA-ES archive richness is established by HV saturation alone.",
            "",
            "## Tables and figures that must be replaced",
            "- Any table or figure fed by `optimizer-projection-summary.csv` projected HV/IGD columns.",
            "- Any table or figure fed by `benchmark-equal-size-summary.csv` rows where requested size exceeds source archive size.",
            "- Any table or figure that interprets `actual_size` from the old equal-size summary as sampled row count.",
            "",
            "## Equal-size not-applicable rows",
        ]
    )
    for row in equal_size_summary_v2.loc[equal_size_summary_v2["status"] == "not_applicable"].itertuples():
        lines.append(f"- `{row.group}` requested `{int(row.requested_sample_size)}` -> `not_applicable`")
    return "\n".join(lines)


def _write_updated_summary_docs(
    paths: Any,
    reference_payload: dict[str, Any],
    canonical_metrics: pd.DataFrame,
    equal_size_summary_v2: pd.DataFrame,
    physical_evidence_levels: dict[str, str],
) -> None:
    nsga_full = canonical_metrics.loc[
        (canonical_metrics["representation_family"] == "descriptor_clipped_full_archive")
        & (canonical_metrics["group"] == "NSGA-II")
    ].iloc[0]
    ddpg_proj = canonical_metrics.loc[
        (canonical_metrics["representation_family"] == "projected_feasible_morphology_archive")
        & (canonical_metrics["group"] == "DDPG::Balanced_Performance")
    ].iloc[0]
    nsga_proj = canonical_metrics.loc[
        (canonical_metrics["representation_family"] == "projected_feasible_morphology_archive")
        & (canonical_metrics["group"] == "NSGA-II")
    ].iloc[0]
    ddpg_equal_20 = equal_size_summary_v2.loc[
        (equal_size_summary_v2["group"] == "DDPG::Balanced_Performance")
        & (equal_size_summary_v2["requested_sample_size"] == 20)
    ].iloc[0]
    experiment_lines = [
        "# Round 2 Experiment Results",
        "",
        "## Canonical result lock summary",
        "- This stage does not rerun DDPG, NSGA-II, CMA-ES, RandomSearch, physical validation, or climate sensitivity.",
        "- The projected HV/IGD conflict came from two different reference definitions: projected-local reference in `optimizer-projection-summary.csv` versus fixed benchmark reference in `benchmark-metric-definition-audit.csv`.",
        f"- The canonical benchmark reference is `{CANONICAL_REFERENCE_PROTOCOL}` with hash `{reference_payload['normalized_reference_front_hash']}`.",
        "",
        "## Canonical projected metrics",
        f"- NSGA-II projected feasible HV/IGD (fixed reference) = `{float(nsga_proj['HV']):.6f}` / `{float(nsga_proj['IGD']):.6f}`.",
        f"- Balanced DDPG projected feasible HV/IGD (fixed reference) = `{float(ddpg_proj['HV']):.6f}` / `{float(ddpg_proj['IGD']):.6f}`.",
        "",
        "## Canonical fairness metrics",
        f"- NSGA-II full-archive HV/IGD = `{float(nsga_full['HV']):.6f}` / `{float(nsga_full['IGD']):.6f}`.",
        f"- Balanced DDPG equal-size-20 HV/IGD mean = `{float(ddpg_equal_20['HV_mean']):.6f}` / `{float(ddpg_equal_20['IGD_mean']):.6f}`.",
        "",
        "## Physical and climate wording",
        "- Physical evidence is now locked as `limited physics-based cross-model stress test`.",
        "- Cross-climate evidence is now locked as `limited four-block cross-climate physical sensitivity analysis`.",
        f"- Physical evidence level: execution closure = `{physical_evidence_levels['execution_closure']}`, metric agreement = `{physical_evidence_levels['metric_agreement']}`, ranking transfer = `{physical_evidence_levels['ranking_transfer']}`.",
        "",
        "## Canonical files",
        "- `research/reviewer-round-02/canonical-benchmark-reference.json`",
        "- `research/reviewer-round-02/canonical_benchmark_metrics.csv`",
        "- `research/reviewer-round-02/benchmark_equal_size_repetitions_v2.csv`",
        "- `research/reviewer-round-02/benchmark_equal_size_summary_v2.csv`",
        "- `research/reviewer-round-02/metric-lineage-audit.md`",
        "- `research/reviewer-round-02/optimizer-output-contract.csv`",
        "- `research/reviewer-round-02/hv-ceiling-interpretation.md`",
        "- `research/reviewer-round-02/canonical-result-registry.json`",
        "- `research/reviewer-round-02/canonical-result-lock.md`",
    ]
    (paths.research_root / "experiment-results.md").write_text("\n".join(experiment_lines), encoding="utf-8")

    manuscript_lines = [
        "# Manuscript Change Input",
        "",
        "## Canonical benchmark reference",
        f"- Use `{CANONICAL_REFERENCE_PROTOCOL}` only, with reference hash `{reference_payload['normalized_reference_front_hash']}`.",
        "- Do not mix projected-local and fixed-reference HV/IGD in the same comparison table.",
        "",
        "## Canonical benchmark numbers",
        f"- NSGA-II full archive: HV = {float(nsga_full['HV']):.6f}, IGD = {float(nsga_full['IGD']):.6f}.",
        f"- NSGA-II projected feasible archive (fixed reference): HV = {float(nsga_proj['HV']):.6f}, IGD = {float(nsga_proj['IGD']):.6f}.",
        f"- Balanced DDPG projected feasible archive (fixed reference): HV = {float(ddpg_proj['HV']):.6f}, IGD = {float(ddpg_proj['IGD']):.6f}.",
        "",
        "## Equal-size wording",
        "- `source_archive_size` is the retained archive size before downsampling.",
        "- `effective_sample_size` is the sampled row count actually used in each repetition.",
        "- Oversized requests must be labelled `not_applicable`, not reported as valid equal-size results.",
        "",
        "## Physical and climate terminology",
        "- Use `limited physics-based cross-model stress test`.",
        "- Use `limited four-block cross-climate physical sensitivity analysis`.",
        "- Do not use `successful physical validation`, `physical closure`, `external confirmation`, or `physical support for optimizer ranking`.",
        "- Keep the evidence-level labels: execution closure = complete, metric agreement = weak, ranking transfer = unsupported, optimizer superiority under physical evaluation = unsupported.",
    ]
    (paths.research_root / "manuscript-change-input.md").write_text("\n".join(manuscript_lines), encoding="utf-8")


def lock_canonical_results(
    config_path: str | Path,
    *,
    run_id: str | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    base_config, _, paths = prepare_round2_workspace(config_path, run_id=run_id, output_dir=output_dir)
    group_archives, metadata = _load_round2_group_archives(base_config, paths)
    dataset = metadata["dataset"]
    reference = build_fixed_reference(group_archives)
    reference_payload = _build_reference_payload(base_config, group_archives, reference, metadata["source_files"])
    write_json(reference_payload, paths.research_root / "canonical-benchmark-reference.json")

    full_archive = evaluate_archive_metrics(group_archives, reference)
    unique_groups: dict[str, pd.DataFrame] = {}
    for group_name, frame in group_archives.items():
        unique_frame = frame[["method", "scenario", "seed", *ROUND2_FEATURES, *ROUND2_TARGETS, "reward"]].copy()
        unique_groups[group_name] = dedupe_objective_tuples(unique_frame)
    unique_objective_metrics = evaluate_archive_metrics(unique_groups, reference)

    projected_groups, projected_metadata, projected_metrics = build_projected_metric_rows(dataset, group_archives, reference)
    legacy_equal_size_repetitions = pd.read_csv(paths.optimization_dir / "benchmark_equal_size_repetitions.csv")

    repetition_frame, equal_size_summary = build_equal_size_tables(
        group_archives,
        reference,
        sizes=[int(value) for value in base_config["round2"]["fairness_analysis"]["equal_size_archive_sizes"]],
        repetitions=int(base_config["round2"]["fairness_analysis"]["equal_size_repetitions"]),
        master_seed=int(base_config["round2"]["master_seed"]),
        legacy_repetition_frame=legacy_equal_size_repetitions,
    )
    write_csv(repetition_frame, paths.research_root / "benchmark_equal_size_repetitions_v2.csv")
    write_csv(equal_size_summary, paths.research_root / "benchmark_equal_size_summary_v2.csv")

    reference_hash = reference_payload["normalized_reference_front_hash"]
    canonical_metrics = _build_canonical_metric_rows(
        group_archives,
        reference_hash,
        full_archive,
        unique_objective_metrics,
        projected_groups,
        projected_metadata,
        projected_metrics,
        equal_size_summary,
    )
    write_csv(canonical_metrics, paths.research_root / "canonical_benchmark_metrics.csv")

    decomposition_frame = pd.read_csv(paths.optimization_dir / "optimizer_guardrail_decomposition.csv")
    hv_md = _render_hv_ceiling_md(
        group_archives,
        full_archive,
        decomposition_frame,
        reference_payload["reference_point"],
    )
    (paths.research_root / "hv-ceiling-interpretation.md").write_text(hv_md, encoding="utf-8")

    lineage_records = _build_metric_lineage_records(reference_hash, equal_size_summary)
    write_json({"records": lineage_records}, paths.research_root / "metric-lineage-audit.json")
    (paths.research_root / "metric-lineage-audit.md").write_text(
        _render_metric_lineage_md(lineage_records),
        encoding="utf-8",
    )

    optimizer_contract = _build_optimizer_output_contract(base_config, group_archives, projected_metadata)
    write_csv(optimizer_contract, paths.research_root / "optimizer-output-contract.csv")

    physical_evidence_levels = _build_physical_evidence_levels()
    registry_entries = _build_registry_entries(paths, reference_hash, equal_size_summary, physical_evidence_levels)
    issues = validate_result_registry(registry_entries)
    if issues:
        raise RuntimeError("Canonical result registry validation failed:\n- " + "\n- ".join(issues))
    write_json({"results": registry_entries}, paths.research_root / "canonical-result-registry.json")

    lock_md = _render_canonical_result_lock_md(canonical_metrics, equal_size_summary, lineage_records)
    (paths.research_root / "canonical-result-lock.md").write_text(lock_md, encoding="utf-8")

    _write_updated_summary_docs(paths, reference_payload, canonical_metrics, equal_size_summary, physical_evidence_levels)

    result_manifest = {
        "run_id": paths.run_id,
        "canonical_reference": {
            "path": str(paths.research_root / "canonical-benchmark-reference.json"),
            "exists": True,
            "reference_hash": reference_hash,
        },
        "canonical_outputs": {
            "canonical_benchmark_metrics": {"path": str(paths.research_root / "canonical_benchmark_metrics.csv"), "exists": True},
            "equal_size_repetitions_v2": {"path": str(paths.research_root / "benchmark_equal_size_repetitions_v2.csv"), "exists": True},
            "equal_size_summary_v2": {"path": str(paths.research_root / "benchmark_equal_size_summary_v2.csv"), "exists": True},
            "metric_lineage_audit_json": {"path": str(paths.research_root / "metric-lineage-audit.json"), "exists": True},
            "optimizer_output_contract": {"path": str(paths.research_root / "optimizer-output-contract.csv"), "exists": True},
            "hv_ceiling_interpretation": {"path": str(paths.research_root / "hv-ceiling-interpretation.md"), "exists": True},
            "canonical_result_registry": {"path": str(paths.research_root / "canonical-result-registry.json"), "exists": True},
            "canonical_result_lock": {"path": str(paths.research_root / "canonical-result-lock.md"), "exists": True},
        },
        "physical_evidence_levels": physical_evidence_levels,
    }
    write_json(result_manifest, paths.research_root / "result-manifest.json")
    write_json(result_manifest, paths.research_root / "release-candidate-manifest.json")
    return {
        "run_id": paths.run_id,
        "projected_metric_conflict_root_cause": "local projected-only reference versus fixed benchmark reference",
        "reference_protocol": CANONICAL_REFERENCE_PROTOCOL,
        "reference_hash": reference_hash,
        "canonical_metrics_csv": str(paths.research_root / "canonical_benchmark_metrics.csv"),
        "equal_size_summary_v2_csv": str(paths.research_root / "benchmark_equal_size_summary_v2.csv"),
        "registry_json": str(paths.research_root / "canonical-result-registry.json"),
        "lock_md": str(paths.research_root / "canonical-result-lock.md"),
    }

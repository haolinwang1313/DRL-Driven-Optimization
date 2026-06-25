from __future__ import annotations

import json
import math
import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from PIL import Image
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting

from paper_repro.config import Config
from paper_repro.constants import MORPHOLOGY_FEATURES, PERFORMANCE_TARGETS
from paper_repro.metrics import compute_seeded_convergence_diagnostics
from paper_repro.round2 import _feasible_pool_random_resampling
from paper_repro.surrogate import load_surrogate

CM_TO_IN = 1 / 2.54
DOUBLE_COL_IN = 17.5 * CM_TO_IN
SINGLE_COL_IN = 8.5 * CM_TO_IN
GALLERY_PAGE_IN = (8.27, 11.69)
OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "sky": "#56B4E9",
    "pink": "#CC79A7",
    "yellow": "#F0E442",
    "black": "#000000",
    "gray": "#6B7280",
}
TARGET_LABELS = {
    "EUIt": "EUIt (kWh/m²/y)",
    "EG": "EG (10⁶ kWh/y)",
    "H": "H (h)",
}
FEATURE_LABELS = {
    "FAR": "FAR",
    "SD": "SD (m)",
    "AF": "AF (floors)",
    "AR_ew": "AR_e-w",
    "AR_ns": "AR_n-s",
    "SVF": "SVF",
    "BD": "BD",
    "OSR": "OSR",
    "SC": "SC",
    "PAR": "PAR",
    "theta": "θ (deg)",
    "OSLI": "OSLI",
}
FORBIDDEN_TEXT = [
    "physical closure",
    "external confirmation",
    "generalization proof",
    "10^6 - d",
]
SECRET_LIKE_TOKENS = ("password", "token", "secret", "host", "user", "pid", "identity", "ssh", "remote")
BENCHMARK_REFERENCE_PROTOCOL = "benchmark-reference-v2"


@dataclass(frozen=True)
class PackageSpec:
    file_name: str
    representation_family: str
    source_keys: tuple[str, ...]
    valid_for_main_text: bool
    valid_for_appendix: bool
    claim_boundary: str
    description: str
    data_dictionary: dict[str, str]
    reference_required: bool = False


@dataclass(frozen=True)
class FigureSpec:
    figure_id: str
    semantic_name: str
    category: str
    planned_location: str
    source_files: tuple[str, ...]
    panel_descriptions: tuple[str, ...]
    claim_boundary: str
    builder: Callable[[dict[str, pd.DataFrame], dict[str, Any]], tuple[plt.Figure, dict[str, Any]]]


def _repo_root(repo_root: str | Path | None) -> Path:
    return Path(repo_root or Path.cwd()).resolve()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sha256_path(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_dump(payload: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _relative_path(path: str | Path, repo_root: Path) -> str:
    candidate = Path(path)
    return candidate.resolve().relative_to(repo_root).as_posix() if candidate.is_absolute() else candidate.as_posix()


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _set_publication_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 8,
            "axes.labelsize": 8,
            "axes.titlesize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "axes.linewidth": 0.8,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 3.5,
            "ytick.major.size": 3.5,
            "xtick.minor.size": 2.0,
            "ytick.minor.size": 2.0,
            "figure.dpi": 600,
            "savefig.dpi": 600,
            "savefig.pad_inches": 0.03,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.grid": False,
        }
    )


def _style_axis(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(which="both", direction="in", top=False, right=False)


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(0.02, 0.98, label, transform=ax.transAxes, ha="left", va="top", fontweight="bold", fontsize=9)


def _ensure_equal_limits(ax: plt.Axes, x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))
    span = hi - lo
    pad = max(span * 0.03, 1e-6)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    return tuple(ax.get_xlim())


def _metric_stats(truth: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    residual = pred - truth
    mae = float(np.mean(np.abs(residual)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    target_range = max(float(np.max(truth) - np.min(truth)), 1e-8)
    corr = float(pd.Series(truth).corr(pd.Series(pred), method="pearson"))
    spearman = float(pd.Series(truth).corr(pd.Series(pred), method="spearman"))
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((truth - float(np.mean(truth))) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-8)
    return {
        "MAE": mae,
        "RMSE": rmse,
        "nMAE": mae / target_range,
        "R2": r2,
        "Spearman": spearman,
        "Pearson": corr,
    }


def _sanitize_strings(frame: pd.DataFrame) -> pd.DataFrame:
    cleaned = frame.copy()
    for column in cleaned.columns:
        if pd.api.types.is_string_dtype(cleaned[column]) or cleaned[column].dtype == object:
            cleaned[column] = cleaned[column].map(
                lambda value: value.replace("\\", "/") if isinstance(value, str) and not re.match(r"^[A-Za-z]:/", value) else value
            )
    return cleaned


def _assert_no_absolute_local_paths(frame: pd.DataFrame) -> None:
    pattern = re.compile(r"^[A-Za-z]:[\\/]")
    for column in frame.columns:
        cleaned = frame[column].dropna().astype(str)
        if cleaned.empty:
            continue
        for value in cleaned:
            if pattern.match(value) or value.startswith("/"):
                raise RuntimeError(f"absolute local path leaked into data package column {column}: {value}")


def _assert_no_secret_like_columns(frame: pd.DataFrame) -> None:
    for column in frame.columns:
        lowered = column.lower()
        if any(token in lowered for token in SECRET_LIKE_TOKENS):
            raise RuntimeError(f"secret-like column leaked into data package: {column}")


def _read_text_extract(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""


def _load_registry(repo_root: Path) -> dict[str, Any]:
    registry_path = repo_root / "research" / "reviewer-round-02" / "canonical-result-registry.json"
    return _read_json(registry_path)


def _validate_registry_sources(repo_root: Path, registry: dict[str, Any]) -> list[dict[str, Any]]:
    validated = []
    for entry in registry["results"]:
        source = repo_root / Path(entry["source_file"])
        if not source.exists():
            raise FileNotFoundError(f"missing registry source file: {source}")
        actual_sha = _sha256_path(source)
        if actual_sha != entry["source_sha256"]:
            raise RuntimeError(f"registry sha mismatch: {source}")
        validated.append({**entry, "source_path": source})
    return validated


def _resolve_source_roots(repo_root: Path) -> dict[str, Path]:
    config = Config.from_yaml(repo_root / "configs" / "reviewer_round2_experiments.yaml")
    registry = _load_registry(repo_root)
    registry_entries = _validate_registry_sources(repo_root, registry)
    round2_source = next(
        Path(entry["source_file"]) for entry in registry_entries if "artifacts/reviewer_round_02/" in entry["source_file"].replace("\\", "/")
    )
    parts = round2_source.parts
    root_parts = []
    for part in parts:
        root_parts.append(part)
        if part == "20260625_round2_closure":
            break
    round2_root = repo_root / Path(*root_parts)
    reference = _read_json(repo_root / "research" / "reviewer-round-02" / "canonical-benchmark-reference.json")
    compare_root = repo_root / Path(reference["source_files"]["ddpg_results"]["path"]).parent.parent
    selection_root = repo_root / Path(config["round2"]["surrogate_selection_root"])
    return {
        "round2_root": round2_root,
        "compare_root": compare_root,
        "selection_root": selection_root,
    }


def _load_ddpg_logs_all(compare_root: Path) -> dict[str, dict[str, list[dict[str, float]]]]:
    optimization_dir = compare_root / "optimization"
    direct = optimization_dir / "ddpg_logs_all.json"
    if direct.exists():
        return _read_json(direct)
    full_candidates = sorted(optimization_dir.glob("ddpg_logs_all*_full.json"))
    if full_candidates:
        return _read_json(full_candidates[0])
    raise FileNotFoundError(f"missing DDPG seeded logs under {optimization_dir}")


def _pca_cumulative(dataset: pd.DataFrame) -> pd.DataFrame:
    values = dataset[MORPHOLOGY_FEATURES].to_numpy(dtype=float)
    normalized = (values - values.min(axis=0, keepdims=True)) / np.maximum(values.max(axis=0, keepdims=True) - values.min(axis=0, keepdims=True), 1e-8)
    centered = normalized - normalized.mean(axis=0, keepdims=True)
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    explained = (singular_values**2) / max(float(np.sum(singular_values**2)), 1e-8)
    cumulative = np.cumsum(explained)
    return pd.DataFrame(
        {
            "record_type": "pca_cumulative",
            "component_index": np.arange(1, len(cumulative) + 1, dtype=int),
            "explained_variance_ratio": explained,
            "cumulative_explained_variance": cumulative,
        }
    )


def _nearest_neighbor_distances(dataset: pd.DataFrame) -> pd.DataFrame:
    values = dataset[MORPHOLOGY_FEATURES].to_numpy(dtype=float)
    normalized = (values - values.min(axis=0, keepdims=True)) / np.maximum(values.max(axis=0, keepdims=True) - values.min(axis=0, keepdims=True), 1e-8)
    distance = np.linalg.norm(normalized[:, None, :] - normalized[None, :, :], axis=2)
    np.fill_diagonal(distance, np.inf)
    return pd.DataFrame(
        {
            "record_type": "nearest_neighbor_distance",
            "sample_id": dataset["sample_id"].to_numpy(dtype=int),
            "normalized_nearest_neighbor_distance": distance.min(axis=1),
        }
    )


def _osli_frequency(dataset: pd.DataFrame) -> pd.DataFrame:
    counts = dataset["OSLI"].round().astype(int).value_counts().sort_index()
    return pd.DataFrame(
        {
            "record_type": "osli_frequency",
            "OSLI": counts.index.to_numpy(dtype=int),
            "count": counts.to_numpy(dtype=int),
        }
    )


def _build_descriptor_coverage(round2_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    coverage = pd.read_csv(round2_root / "data" / "sampling_coverage_summary.csv").copy()
    dataset = pd.read_csv(round2_root / "data" / "simulated_samples.csv")
    coverage["record_type"] = "feature_summary"
    pca = _pca_cumulative(dataset)
    nn = _nearest_neighbor_distances(dataset)
    osli = _osli_frequency(dataset)
    frame = pd.concat([coverage, pca, nn, osli], ignore_index=True, sort=False)
    return frame, [round2_root / "data" / "sampling_coverage_summary.csv", round2_root / "data" / "simulated_samples.csv"]


def _build_descriptor_dependencies(round2_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    frame = pd.read_csv(round2_root / "data" / "descriptor_dependencies.csv").copy()
    expression = {
        "FAR_minus_BD_times_AF": "FAR - BD × AF",
        "OSR_minus_(1_minus_BD)_over_FAR": "OSR - (1 - BD) / FAR",
        "components_for_95_variance": "PCA components for 95% variance",
    }
    frame["expression"] = frame["dependency_name"].map(expression).fillna(frame["dependency_name"])
    return frame, [round2_root / "data" / "descriptor_dependencies.csv"]


def _build_surrogate_parity(round2_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    frame = pd.read_csv(round2_root / "models" / "surrogate_validation_predictions.csv")
    repeated = frame.loc[frame["validation_family"] == "repeated_kfold"].copy()
    grouped = repeated.groupby("sample_id", as_index=False).agg(
        analytic_EUIt=("true_EUIt", "mean"),
        analytic_EG=("true_EG", "mean"),
        analytic_H=("true_H", "mean"),
        predicted_EUIt=("pred_EUIt", "mean"),
        predicted_EG=("pred_EG", "mean"),
        predicted_H=("pred_H", "mean"),
        distance_to_center=("distance_to_center", "mean"),
        nearest_train_distance=("nearest_train_distance", "mean"),
        contributing_oof_rows=("sample_id", "size"),
    )
    for target in PERFORMANCE_TARGETS:
        grouped[f"residual_{target}"] = grouped[f"predicted_{target}"] - grouped[f"analytic_{target}"]
    return grouped, [round2_root / "models" / "surrogate_validation_predictions.csv"]


def _build_surrogate_validation_regimes(repo_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    source = repo_root / "research" / "reviewer-round-02" / "surrogate-validation-summary.csv"
    frame = pd.read_csv(source).copy()
    frame["metric_direction"] = "nMAE lower / R² higher / Spearman higher"
    return frame, [source]


def _build_ddpg_training_curves(compare_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    logs = _load_ddpg_logs_all(compare_root)
    rows: list[pd.DataFrame] = []
    for scenario, seed_logs in logs.items():
        for seed, entries in seed_logs.items():
            frame = pd.DataFrame(entries)
            if frame.empty:
                continue
            frame["scenario"] = scenario
            frame["seed"] = int(seed)
            rows.append(frame[["scenario", "seed", "episode", "cumulative_reward", "EUIt", "EG", "H"]])
    merged = pd.concat(rows, ignore_index=True)
    summary = (
        merged.groupby(["scenario", "episode"], as_index=False)
        .agg(
            n_seeds=("seed", "nunique"),
            reward_mean=("cumulative_reward", "mean"),
            reward_std=("cumulative_reward", "std"),
            EUIt_mean=("EUIt", "mean"),
            EUIt_std=("EUIt", "std"),
            EG_mean=("EG", "mean"),
            EG_std=("EG", "std"),
            H_mean=("H", "mean"),
            H_std=("H", "std"),
        )
        .fillna(0.0)
    )
    return summary, [next((compare_root / "optimization").glob("ddpg_logs_all*_full.json"))]


def _build_ddpg_seed_diagnostics(compare_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    logs = _load_ddpg_logs_all(compare_root)
    seeded, _ = compute_seeded_convergence_diagnostics(logs)
    seeded["late_regression"] = seeded["best_final_gap_ratio"] > 0.2
    return seeded, [next((compare_root / "optimization").glob("ddpg_logs_all*_full.json"))]


def _build_benchmark_utility(round2_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    utility = pd.read_csv(round2_root / "optimization" / "utility_sensitivity.csv")
    summary = (
        utility.groupby(["method", "scenario", "seed", "utility_scenario"], as_index=False)
        .agg(
            best_legacy_utility=("legacy_utility", "max"),
            best_fixed_domain_utility=("fixed_domain_utility", "max"),
            analytic_EUIt=("EUIt", "min"),
            analytic_EG=("EG", "max"),
            analytic_H=("H", "max"),
        )
    )
    summary = summary.rename(columns={"utility_scenario": "evaluation_scenario"})
    return summary, [round2_root / "optimization" / "utility_sensitivity.csv"]


def _build_benchmark_equal_size_20(repo_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    path = repo_root / "research" / "reviewer-round-02" / "benchmark_equal_size_repetitions_v2.csv"
    frame = pd.read_csv(path)
    filtered = frame.loc[(frame["requested_sample_size"] == 20) & (frame["status"] == "valid")].copy()
    reference = _read_json(repo_root / "research" / "reviewer-round-02" / "canonical-benchmark-reference.json")
    filtered["reference_protocol"] = BENCHMARK_REFERENCE_PROTOCOL
    filtered["reference_hash"] = reference["normalized_reference_front_hash"]
    return filtered, [path, repo_root / "research" / "reviewer-round-02" / "canonical-benchmark-reference.json"]


def _build_output_contract(repo_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    path = repo_root / "research" / "reviewer-round-02" / "optimizer-output-contract.csv"
    return pd.read_csv(path), [path]


def _parse_hv_ceiling(repo_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    path = repo_root / "research" / "reviewer-round-02" / "hv-ceiling-interpretation.md"
    lines = path.read_text(encoding="utf-8").splitlines()
    rows: list[dict[str, Any]] = []
    current: dict[str, Any] | None = None
    for line in lines:
        if line.startswith("- `") and line.endswith("`"):
            if current:
                rows.append(current)
            group = line.split("`")[1]
            method, scenario = (group.split("::", 1) + [group])[:2] if "::" in group else (group, group)
            current = {"group": group, "method": method, "scenario": scenario}
        elif current and "HV =" in line:
            match = re.search(
                r"HV = `([0-9.]+)`, distance_to_hv_ceiling = `([0-9.]+)`, fraction_of_theoretical_max = `([0-9.]+)`",
                line,
            )
            if match:
                current["HV"] = float(match.group(1))
                current["distance_to_hv_ceiling"] = float(match.group(2))
                current["fraction_of_theoretical_max"] = float(match.group(3))
        elif current and "clipped_utopia_rows" in line:
            match = re.search(r"clipped_utopia_rows = `([0-9]+)` / `([0-9]+)`", line)
            if match:
                current["clipped_utopia_rows"] = int(match.group(1))
                current["total_rows"] = int(match.group(2))
                current["clipped_utopia_fraction"] = int(match.group(1)) / max(int(match.group(2)), 1)
        elif current and "unique_objective_tuples" in line:
            current["unique_objective_tuples"] = int(re.search(r"`([0-9]+)`", line).group(1))
        elif current and "unique_non_dominated_tuples" in line:
            current["unique_non_dominated_tuples"] = int(re.search(r"`([0-9]+)`", line).group(1))
    if current:
        rows.append(current)
    reference = _read_json(repo_root / "research" / "reviewer-round-02" / "canonical-benchmark-reference.json")
    frame = pd.DataFrame(rows)
    frame["reference_protocol"] = BENCHMARK_REFERENCE_PROTOCOL
    frame["reference_hash"] = reference["normalized_reference_front_hash"]
    frame["theoretical_max_hv"] = math.prod(reference["reference_point"])
    return frame, [path, repo_root / "research" / "reviewer-round-02" / "canonical-benchmark-reference.json"]


def _build_feasible_projection_summary(repo_root: Path, round2_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    canonical = pd.read_csv(repo_root / "research" / "reviewer-round-02" / "canonical_benchmark_metrics.csv")
    raw = pd.read_csv(repo_root / "research" / "reviewer-round-02" / "optimizer-projection-summary.csv")
    before = canonical.loc[canonical["representation_family"] == "descriptor_clipped_full_archive", ["group", "HV", "IGD"]].rename(
        columns={"HV": "descriptor_HV", "IGD": "descriptor_IGD"}
    )
    after = canonical.loc[
        canonical["representation_family"] == "projected_feasible_morphology_archive",
        ["group", "HV", "IGD", "reference_hash"],
    ].rename(columns={"HV": "projected_HV_fixed_reference", "IGD": "projected_IGD_fixed_reference"})
    raw["group"] = np.where(raw["method"] == "NSGA-II", "NSGA-II", raw["method"] + "::" + raw["scenario"])
    merged = raw.merge(before, on="group", how="left").merge(after, on="group", how="left")
    merged["representation_family_before"] = "descriptor_clipped_full_archive"
    merged["representation_family_after"] = "projected_feasible_morphology_archive"
    return merged, [repo_root / "research" / "reviewer-round-02" / "canonical_benchmark_metrics.csv", repo_root / "research" / "reviewer-round-02" / "optimizer-projection-summary.csv"]


def _build_feasible_projection_metrics(round2_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    frame = pd.read_csv(round2_root / "optimization" / "optimizer_feasibility_audit.csv")
    keep = frame[
        [
            "candidate_index",
            "matched_sample_id",
            "projection_distance",
            "method",
            "scenario",
            "seed",
            "EUIt",
            "EG",
            "H",
            "projected_EUIt",
            "projected_EG",
            "projected_H",
        ]
    ].copy()
    keep["group"] = np.where(keep["method"] == "NSGA-II", "NSGA-II", keep["method"] + "::" + keep["scenario"])
    return keep, [round2_root / "optimization" / "optimizer_feasibility_audit.csv"]


def _build_physical_direct_cases(round2_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    frame = pd.read_csv(round2_root / "physical" / "physical_validation_results.csv")
    direct = frame.loc[frame["selection_stratum"] != "optimizer_linked"].copy()
    keep = direct[
        [
            "matched_sample_id",
            "selection_stratum",
            "projection_distance",
            "EUIt",
            "EG",
            "H",
            "physical_EUIt",
            "physical_EG_total_production",
            "physical_H_proxy",
            "energyplus_ok",
            "radiance_ok",
        ]
    ].rename(
        columns={
            "EUIt": "analytic_EUIt",
            "EG": "analytic_EG",
            "H": "analytic_H",
            "physical_EG_total_production": "physical_EG_GHI_proxy",
            "physical_H_proxy": "physical_H",
        }
    )
    keep["case_family"] = "direct_feasible"
    return keep, [round2_root / "physical" / "physical_validation_results.csv"]


def _build_physical_stress_metrics(repo_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    path = repo_root / "research" / "reviewer-round-02" / "physical-validation-metrics.csv"
    frame = pd.read_csv(path).copy()
    frame["stress_test_label"] = "limited physics-based cross-model stress test"
    return frame, [path]


def _build_optimizer_linked_gaps(repo_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    path = repo_root / "research" / "reviewer-round-02" / "physical-validation-optimizer-mapping.csv"
    return pd.read_csv(path), [path]


def _build_climate_case_results(repo_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    path = repo_root / "research" / "reviewer-round-02" / "climate-sensitivity-results.csv"
    frame = pd.read_csv(path).copy()
    keep = frame[
        [
            "matched_sample_id",
            "station",
            "climate_bucket",
            "EUIt",
            "EG",
            "H",
            "physical_EUIt",
            "physical_EG_total_production",
            "physical_H_proxy",
            "delta_EUIt_vs_baseline",
            "delta_EG_vs_baseline",
            "delta_H_vs_baseline",
        ]
    ].rename(columns={"physical_EG_total_production": "physical_EG", "physical_H_proxy": "physical_H"})
    return keep, [path]


def _build_climate_summary(repo_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    path = repo_root / "research" / "reviewer-round-02" / "climate-sensitivity-summary.csv"
    return pd.read_csv(path), [path]


def _build_climate_rank_stability(repo_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    path = repo_root / "research" / "reviewer-round-02" / "climate-rank-stability.csv"
    return pd.read_csv(path), [path]


def _build_scale_study(selection_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    summary = pd.read_csv(selection_root / "data" / "dataset_scale_summary.csv")
    comparison = pd.read_csv(selection_root / "models" / "surrogate_comparison.csv")
    winners = pd.read_csv(selection_root / "models" / "surrogate_regime_winners.csv")
    merged = comparison.merge(summary[["dataset_scale", "rows"]], on="dataset_scale", how="left")
    merged["is_selected"] = False
    selected_keys = set(zip(winners["dataset_scale"], winners["candidate"]))
    merged["is_selected"] = merged.apply(lambda row: (row["dataset_scale"], row["candidate"]) in selected_keys, axis=1)
    return merged, [selection_root / "data" / "dataset_scale_summary.csv", selection_root / "models" / "surrogate_comparison.csv", selection_root / "models" / "surrogate_regime_winners.csv"]


def _build_morphology_signatures(config: Config, round2_root: Path) -> tuple[pd.DataFrame, list[Path]]:
    dataset = pd.read_csv(config["round2"]["canonical_dataset"])
    utility_weights = config["optimization"]["utility_weights"]
    feasible_pool_random = _feasible_pool_random_resampling(
        dataset,
        utility_weights,
        master_seed=int(config["round2"]["master_seed"]),
        evaluation_budget=int(config["optimization"]["random_search"]["evaluation_budget"]),
        seeds_per_scenario=int(config["optimization"]["random_search"]["seeds_per_scenario"]),
    )
    groups = {
        "NSGA-II": pd.read_csv(config["round2"]["baseline_runs"]["nsga2_results"]),
        "DDPG::Balanced_Performance": pd.read_csv(config["round2"]["baseline_runs"]["ddpg_results"]).loc[
            lambda df: df["scenario"] == "Balanced_Performance"
        ],
        "DDPG::Energy_Saving_Focus": pd.read_csv(config["round2"]["baseline_runs"]["ddpg_results"]).loc[
            lambda df: df["scenario"] == "Energy_Saving_Focus"
        ],
        "DDPG::Energy_Generation_Focus": pd.read_csv(config["round2"]["baseline_runs"]["ddpg_results"]).loc[
            lambda df: df["scenario"] == "Energy_Generation_Focus"
        ],
        "FeasiblePoolRandom::Balanced_Performance": feasible_pool_random.loc[feasible_pool_random["scenario"] == "Balanced_Performance"],
    }
    rows = []
    for group, frame in groups.items():
        label = {
            "NSGA-II": "NSGA-II",
            "DDPG::Balanced_Performance": "DDPG Balanced",
            "DDPG::Energy_Saving_Focus": "DDPG Saving",
            "DDPG::Energy_Generation_Focus": "DDPG Generation",
            "FeasiblePoolRandom::Balanced_Performance": "FeasiblePoolRandom Balanced",
        }[group]
        for feature in MORPHOLOGY_FEATURES:
            rows.append(
                {
                    "group": group,
                    "group_label": label,
                    "feature": feature,
                    "median": float(frame[feature].median()),
                    "q25": float(frame[feature].quantile(0.25)),
                    "q75": float(frame[feature].quantile(0.75)),
                }
            )
    sources = [
        Path(config["round2"]["baseline_runs"]["nsga2_results"]),
        Path(config["round2"]["baseline_runs"]["ddpg_results"]),
        round2_root / "optimization" / "random_search_results_round2.csv",
    ]
    return pd.DataFrame(rows), sources


PACKAGE_SPECS: tuple[PackageSpec, ...] = (
    PackageSpec(
        "descriptor_coverage.csv",
        "sample_coverage",
        ("descriptor_coverage",),
        True,
        True,
        "Supports descriptor-space coverage and PCA breadth only; not an optimizer ranking.",
        "Coverage, PCA, OSLI frequency, and nearest-neighbor summaries for the 2000-row canonical descriptor dataset.",
        {
            "record_type": "feature_summary, pca_cumulative, nearest_neighbor_distance, or osli_frequency.",
            "feature": "Morphology descriptor name for feature-summary rows.",
            "count": "Number of rows contributing to the summary.",
            "q05": "5th percentile of the descriptor.",
            "q25": "25th percentile of the descriptor.",
            "median": "Median descriptor value.",
            "q75": "75th percentile of the descriptor.",
            "q95": "95th percentile of the descriptor.",
            "component_index": "Principal component index for PCA rows.",
            "cumulative_explained_variance": "Cumulative explained variance after each principal component.",
            "normalized_nearest_neighbor_distance": "Nearest-neighbor distance after min-max normalization across the 12 descriptors.",
            "OSLI": "Rounded open-space-location index.",
        },
    ),
    PackageSpec(
        "descriptor_dependencies.csv",
        "sample_coverage",
        ("descriptor_dependencies",),
        True,
        True,
        "Records descriptor algebra and PCA threshold facts only.",
        "Descriptor dependency diagnostics and algebraic residual summaries.",
        {
            "dependency_name": "Canonical dependency or diagnostic key.",
            "metric": "Statistic applied to the dependency.",
            "value": "Observed diagnostic value.",
            "expression": "Readable dependency expression.",
        },
    ),
    PackageSpec(
        "surrogate_parity_mean_predictions.csv",
        "repeated_cv",
        ("surrogate_parity",),
        True,
        True,
        "Cross-validated surrogate predictions remain analytic-target surrogates, not EnergyPlus truth.",
        "Repeated-kfold out-of-fold sample-level mean predictions for parity plots.",
        {
            "sample_id": "Canonical descriptor sample identifier.",
            "analytic_EUIt": "Analytic response-generator EUIt target.",
            "predicted_EUIt": "Mean repeated-CV surrogate prediction for EUIt.",
            "analytic_EG": "Analytic response-generator EG target.",
            "predicted_EG": "Mean repeated-CV surrogate prediction for EG.",
            "analytic_H": "Analytic response-generator H target.",
            "predicted_H": "Mean repeated-CV surrogate prediction for H.",
            "residual_EUIt": "Predicted minus analytic EUIt.",
            "residual_EG": "Predicted minus analytic EG.",
            "residual_H": "Predicted minus analytic H.",
        },
    ),
    PackageSpec(
        "surrogate_validation_regimes.csv",
        "surrogate_validation",
        ("surrogate_validation_regimes",),
        True,
        True,
        "These rows assess surrogate accuracy against the analytic response generator only.",
        "Validation-family summary used for robustness heatmaps.",
        {
            "validation_family": "Validation regime label.",
            "target": "Target response variable.",
            "mean_nMAE": "Mean normalized mean absolute error.",
            "mean_R2": "Mean coefficient of determination.",
            "mean_Spearman_rho": "Mean Spearman correlation.",
            "metric_direction": "Interpretation of higher/lower values.",
        },
    ),
    PackageSpec(
        "ddpg_training_curves_summary.csv",
        "training_dynamics",
        ("ddpg_training_curves",),
        True,
        True,
        "One episode equals 40 sequential surrogate queries; this is serialized black-box search, not physical time evolution.",
        "Seed-aggregated DDPG training curves across scenarios and targets.",
        {
            "scenario": "DDPG scalarization scenario.",
            "episode": "Episode index from 1 to 600.",
            "reward_mean": "Mean cumulative reward across seeds.",
            "reward_std": "Standard deviation of cumulative reward across seeds.",
            "EUIt_mean": "Mean surrogate EUIt response at episode end.",
            "EG_mean": "Mean surrogate EG response at episode end.",
            "H_mean": "Mean surrogate H response at episode end.",
        },
    ),
    PackageSpec(
        "ddpg_seed_diagnostics.csv",
        "training_dynamics",
        ("ddpg_seed_diagnostics",),
        False,
        True,
        "Seed-level DDPG diagnostics are appendix-only and do not imply physical validation.",
        "Per-seed DDPG plateau, regression, and reward diagnostics.",
        {
            "scenario": "DDPG scalarization scenario.",
            "seed": "Seed index within the scenario.",
            "reward_best": "Best cumulative reward seen in training.",
            "reward_final": "Final cumulative reward at episode 600.",
            "plateau_episode": "First episode reaching 95% of the best rolling reward.",
            "best_final_gap_ratio": "Relative regression from best to final reward.",
            "late_regression": "True when the best-to-final regression exceeds 20%.",
        },
    ),
    PackageSpec(
        "benchmark_utility.csv",
        "post_hoc_utility",
        ("benchmark_utility",),
        True,
        True,
        "Post-hoc fixed-domain utility is a bounded analytic comparison, not the DDPG training reward.",
        "Per-seed best utility summaries for fixed-domain and legacy post-hoc utility.",
        {
            "method": "Optimizer family.",
            "scenario": "Source retained-output scenario.",
            "evaluation_scenario": "Utility weighting scenario used for comparison.",
            "seed": "Seed identifier.",
            "best_fixed_domain_utility": "Best fixed-domain utility found within the retained output.",
            "best_legacy_utility": "Best legacy utility found within the retained output.",
        },
    ),
    PackageSpec(
        "benchmark_equal_size_20.csv",
        "descriptor_equal_size_archive",
        ("benchmark_equal_size_20",),
        True,
        True,
        "Only truthful requested-size-20 rows are retained; oversized DDPG and FeasiblePoolRandom requests are excluded.",
        "All valid equal-size-20 benchmark repetitions with the canonical benchmark reference.",
        {
            "group": "Method/scenario group name.",
            "repetition": "Equal-size repetition index.",
            "HV": "Hypervolume under benchmark-reference-v2.",
            "IGD": "IGD under benchmark-reference-v2.",
            "sampled_unique_objective_tuples": "Unique objective tuples in the sampled subset.",
            "reference_hash": "Canonical benchmark reference hash.",
        },
        reference_required=True,
    ),
    PackageSpec(
        "benchmark_output_contract_counts.csv",
        "output_contract",
        ("benchmark_output_contract",),
        True,
        True,
        "These counts describe retained output structure and comparability limits, not direct optimizer superiority.",
        "Optimizer output contract counts and retained-object metadata.",
        {
            "method": "Optimizer family.",
            "queries_per_seed": "Budgeted surrogate queries per seed.",
            "retained_rows_per_seed": "Rows retained per seed.",
            "unique_objective_tuples": "Unique clipped objective tuples.",
            "unique_feasible_projections": "Unique projected feasible blocks.",
            "archive_hv_igd_role": "Whether archive-level HV/IGD is primary or diagnostic.",
        },
    ),
    PackageSpec(
        "benchmark_hv_ceiling.csv",
        "hv_ceiling",
        ("benchmark_hv_ceiling",),
        False,
        True,
        "HV saturation indicates reference-volume coverage and must be interpreted with duplicate and tuple-count diagnostics.",
        "Canonical HV ceiling diagnostics parsed from the locked round-2 interpretation.",
        {
            "group": "Method/scenario group name.",
            "fraction_of_theoretical_max": "HV divided by the theoretical maximum 1.331.",
            "clipped_utopia_fraction": "Fraction of rows clipped to the utopia corner.",
            "unique_objective_tuples": "Unique clipped objective tuples.",
            "unique_non_dominated_tuples": "Unique non-dominated objective tuples.",
        },
        reference_required=True,
    ),
    PackageSpec(
        "feasible_projection_summary.csv",
        "projected_feasible_morphology_archive",
        ("feasible_projection_summary",),
        True,
        True,
        "Projection diagnostics measure descriptor-to-feasible representation sensitivity only; they are not physical validation.",
        "Method-level projection collapse and canonical before/after HV/IGD summaries.",
        {
            "group": "Method/scenario group name.",
            "projection_distance_mean": "Mean nearest-block projection distance.",
            "duplicate_collapse_rate": "Fraction of retained rows collapsing onto duplicate feasible blocks.",
            "descriptor_HV": "Descriptor-space HV under benchmark-reference-v2.",
            "projected_HV_fixed_reference": "Projected-feasible HV under benchmark-reference-v2.",
            "reference_hash": "Canonical benchmark reference hash.",
        },
        reference_required=True,
    ),
    PackageSpec(
        "feasible_projection_metrics.csv",
        "projected_feasible_morphology_archive",
        ("feasible_projection_metrics",),
        True,
        True,
        "Candidate-level projection distances are descriptor-space diagnostics only.",
        "Candidate-level projection rows used for distance distributions.",
        {
            "group": "Method/scenario group name.",
            "projection_distance": "Nearest feasible-block projection distance.",
            "matched_sample_id": "Projected feasible block identifier.",
            "EUIt": "Descriptor-space analytic EUIt.",
            "projected_EUIt": "Projected feasible-block analytic EUIt.",
        },
    ),
    PackageSpec(
        "physical_direct_cases.csv",
        "physical_evaluated_subset",
        ("physical_direct_cases",),
        True,
        True,
        "These 18 rows support only the limited physics-based cross-model stress test.",
        "Direct feasible physical-evaluation cases only; optimizer-linked cases are excluded.",
        {
            "matched_sample_id": "Canonical feasible block identifier.",
            "case_family": "Unified family label for the 18 direct feasible physical cases.",
            "analytic_EUIt": "Analytic response-generator EUIt.",
            "physical_EUIt": "Physics-based EUIt stress-test result.",
            "analytic_EG": "Analytic response-generator EG proxy.",
            "physical_EG_GHI_proxy": "Physics-based simplified rooftop-PV proxy.",
            "analytic_H": "Analytic response-generator H target.",
            "physical_H": "Physics-based January 20 windowsill direct-sun-hours proxy.",
        },
    ),
    PackageSpec(
        "physical_stress_metrics.csv",
        "physical_evaluated_subset",
        ("physical_stress_metrics",),
        True,
        True,
        "Metric agreement remains weak and ranking transfer unsupported.",
        "Summary metrics for the limited physics-based cross-model stress test.",
        {
            "target": "Target response variable.",
            "MAE": "Mean absolute error.",
            "nMAE": "Normalized mean absolute error.",
            "Spearman_rho": "Spearman rank correlation.",
            "rank_preservation": "Pairwise rank-preservation fraction.",
        },
    ),
    PackageSpec(
        "optimizer_linked_physical_gaps.csv",
        "physical_evaluated_subset",
        ("optimizer_linked_gaps",),
        False,
        True,
        "Optimizer-linked cases are appendix-only and must not be mixed into direct-case parity plots.",
        "Gap decomposition for the six optimizer-linked physical cases.",
        {
            "optimizer_source": "Optimizer family of the linked candidate.",
            "scenario": "Scenario label for the linked candidate.",
            "projection_gap_EUIt": "Gap from surrogate candidate to projected analytic block.",
            "analytic_to_physical_gap_EUIt": "Gap from projected analytic block to physical EUIt result.",
            "total_gap_EUIt": "Total surrogate-to-physical EUIt gap.",
        },
    ),
    PackageSpec(
        "climate_case_results.csv",
        "limited_four_block_cross_climate_physical_sensitivity_analysis",
        ("climate_case_results",),
        True,
        True,
        "These rows cover exactly four direct feasible blocks across three additional climates and do not prove surrogate generalization.",
        "Per-case cross-climate physical sensitivity rows for Beijing, Guangzhou, and Harbin.",
        {
            "matched_sample_id": "Canonical block identifier.",
            "station": "Additional climate station.",
            "delta_EUIt_vs_baseline": "EUIt delta relative to Dongtai baseline weather.",
            "delta_EG_vs_baseline": "EG proxy delta relative to Dongtai baseline weather.",
            "delta_H_vs_baseline": "H delta relative to Dongtai baseline weather.",
        },
    ),
    PackageSpec(
        "climate_summary.csv",
        "limited_four_block_cross_climate_physical_sensitivity_analysis",
        ("climate_summary",),
        True,
        True,
        "Mean climate deltas summarize only four direct-feasible cases.",
        "Station-level mean climate sensitivity summary.",
        {
            "station": "Additional climate station.",
            "mean_delta_EUIt": "Mean EUIt delta relative to Dongtai.",
            "mean_delta_EG": "Mean EG proxy delta relative to Dongtai.",
            "mean_delta_H": "Mean H delta relative to Dongtai.",
        },
    ),
    PackageSpec(
        "climate_rank_stability.csv",
        "limited_four_block_cross_climate_physical_sensitivity_analysis",
        ("climate_rank_stability",),
        True,
        True,
        "Rank stability remains target-dependent and is not a climate-generalization proof.",
        "Target-wise climate rank stability for the four direct-feasible cases.",
        {
            "station": "Additional climate station.",
            "rank_metric": "Target used for ranking.",
            "spearman": "Spearman rank correlation against Dongtai ordering.",
            "kendall": "Kendall rank correlation against Dongtai ordering.",
        },
    ),
    PackageSpec(
        "scale_study.csv",
        "scale_study",
        ("scale_study",),
        False,
        True,
        "Scale-study rows trace surrogate-selection evidence only and are not physical validation.",
        "Dataset-scale surrogate-selection comparison across 500/1000/1500/2000 rows.",
        {
            "dataset_scale": "Dataset size regime.",
            "candidate": "Surrogate candidate preset.",
            "mean_target_nmae": "Average normalized MAE across targets.",
            "mean_tail_nmae": "Average tail-region normalized MAE across targets.",
            "mean_r2": "Average R² across targets.",
            "selection_objective": "Selection objective used during surrogate choice.",
            "is_selected": "True for the regime winner.",
        },
    ),
    PackageSpec(
        "morphology_signatures.csv",
        "morphology_descriptor_signatures",
        ("morphology_signatures",),
        False,
        True,
        "Median morphology signatures are descriptive summaries of retained outputs, not stable design rules.",
        "Median and interquartile morphology descriptor signatures for representative groups.",
        {
            "group_label": "Display label for the strategy group.",
            "feature": "Morphology descriptor.",
            "median": "Median retained value for the descriptor.",
            "q25": "25th percentile of the descriptor.",
            "q75": "75th percentile of the descriptor.",
        },
    ),
)


def _package_builders(repo_root: Path, roots: dict[str, Path], config: Config) -> dict[str, tuple[pd.DataFrame, list[Path]]]:
    return {
        "descriptor_coverage": _build_descriptor_coverage(roots["round2_root"]),
        "descriptor_dependencies": _build_descriptor_dependencies(roots["round2_root"]),
        "surrogate_parity": _build_surrogate_parity(roots["round2_root"]),
        "surrogate_validation_regimes": _build_surrogate_validation_regimes(repo_root),
        "ddpg_training_curves": _build_ddpg_training_curves(roots["compare_root"]),
        "ddpg_seed_diagnostics": _build_ddpg_seed_diagnostics(roots["compare_root"]),
        "benchmark_utility": _build_benchmark_utility(roots["round2_root"]),
        "benchmark_equal_size_20": _build_benchmark_equal_size_20(repo_root),
        "benchmark_output_contract": _build_output_contract(repo_root),
        "benchmark_hv_ceiling": _parse_hv_ceiling(repo_root),
        "feasible_projection_summary": _build_feasible_projection_summary(repo_root, roots["round2_root"]),
        "feasible_projection_metrics": _build_feasible_projection_metrics(roots["round2_root"]),
        "physical_direct_cases": _build_physical_direct_cases(roots["round2_root"]),
        "physical_stress_metrics": _build_physical_stress_metrics(repo_root),
        "optimizer_linked_gaps": _build_optimizer_linked_gaps(repo_root),
        "climate_case_results": _build_climate_case_results(repo_root),
        "climate_summary": _build_climate_summary(repo_root),
        "climate_rank_stability": _build_climate_rank_stability(repo_root),
        "scale_study": _build_scale_study(roots["selection_root"]),
        "morphology_signatures": _build_morphology_signatures(config, roots["round2_root"]),
    }


def build_round2_figure_data_package(data_root: str | Path, *, repo_root: str | Path | None = None, strict: bool = False) -> dict[str, Any]:
    root = _repo_root(repo_root)
    data_root_path = (root / Path(data_root)).resolve()
    data_root_path.mkdir(parents=True, exist_ok=True)
    config = Config.from_yaml(root / "configs" / "reviewer_round2_experiments.yaml")
    registry = _load_registry(root)
    _validate_registry_sources(root, registry)
    roots = _resolve_source_roots(root)
    builders = _package_builders(root, roots, config)
    manifest_entries = []
    for spec in PACKAGE_SPECS:
        frame, upstream_sources = builders[spec.source_keys[0]]
        frame = _sanitize_strings(frame)
        _assert_no_absolute_local_paths(frame)
        _assert_no_secret_like_columns(frame)
        out_path = data_root_path / spec.file_name
        out_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(out_path, index=False)
        entry = {
            "file_name": spec.file_name,
            "sha256": _sha256_path(out_path),
            "rows": int(len(frame)),
            "columns": [{"name": key, "description": value} for key, value in spec.data_dictionary.items()],
            "description": spec.description,
            "representation_family": spec.representation_family,
            "valid_for_main_text": spec.valid_for_main_text,
            "valid_for_appendix": spec.valid_for_appendix,
            "claim_boundary": spec.claim_boundary,
            "generation_command": "uv run python tools/build_round2_revision_figures.py --data-root paper/manuscript/figure_data/round2 --output-dir paper/manuscript/figures/round2_candidate --build-gallery",
            "source_files": [
                {
                    "path": _relative_path(source, root),
                    "sha256": _sha256_path(root / Path(source)) if not Path(source).is_absolute() else _sha256_path(source),
                }
                for source in upstream_sources
            ],
            "reference_protocol": BENCHMARK_REFERENCE_PROTOCOL if spec.reference_required else "",
            "reference_hash": (
                _read_json(root / "research" / "reviewer-round-02" / "canonical-benchmark-reference.json")["normalized_reference_front_hash"]
                if spec.reference_required
                else ""
            ),
            "status": "valid",
        }
        manifest_entries.append(entry)
    manifest = {
        "generated_at": _utc_now(),
        "build_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip(),
        "canonical_reference_hash": _read_json(root / "research" / "reviewer-round-02" / "canonical-benchmark-reference.json")[
            "normalized_reference_front_hash"
        ],
        "files": manifest_entries,
    }
    manifest_path = data_root_path / "manifest.json"
    _json_dump(manifest, manifest_path)
    readme_lines = [
        "# Round 2 Figure Data Package",
        "",
        f"- Generated at: `{manifest['generated_at']}`",
        f"- Build commit: `{manifest['build_commit']}`",
        f"- Canonical reference hash: `{manifest['canonical_reference_hash']}`",
        "",
        "## Files",
    ]
    for entry in manifest_entries:
        source_lines = ", ".join(f"`{source['path']}` ({source['sha256']})" for source in entry["source_files"])
        readme_lines.extend(
            [
                f"### {entry['file_name']}",
                f"- Description: {entry['description']}",
                f"- Representation family: `{entry['representation_family']}`",
                f"- SHA-256: `{entry['sha256']}`",
                f"- Source files: {source_lines}",
                f"- Generation command: `{entry['generation_command']}`",
                f"- Valid for main text: `{entry['valid_for_main_text']}`",
                f"- Valid for appendix: `{entry['valid_for_appendix']}`",
                f"- Claim boundary: {entry['claim_boundary']}",
                "",
            ]
        )
    (data_root_path / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")
    if strict:
        for entry in manifest_entries:
            if entry["reference_protocol"] and not entry["reference_hash"]:
                raise RuntimeError(f"missing reference hash for {entry['file_name']}")
    return manifest


def _load_package_manifest(data_root: Path) -> dict[str, Any]:
    return _read_json(data_root / "manifest.json")


def _load_package_frames(data_root: Path, manifest: dict[str, Any], *, strict: bool = False) -> dict[str, pd.DataFrame]:
    frames = {}
    for entry in manifest["files"]:
        if strict and entry.get("status") != "valid":
            raise RuntimeError(f"package source rejected because status != valid: {entry['file_name']}")
        path = data_root / entry["file_name"]
        if not path.exists():
            raise FileNotFoundError(f"missing figure-data file: {path}")
        if _sha256_path(path) != entry["sha256"]:
            raise RuntimeError(f"figure-data sha mismatch: {path}")
        frames[entry["file_name"]] = pd.read_csv(path)
    return frames


def _package_entry(manifest: dict[str, Any], file_name: str) -> dict[str, Any]:
    for entry in manifest["files"]:
        if entry["file_name"] == file_name:
            return entry
    raise KeyError(file_name)


def _save_figure_outputs(fig: plt.Figure, base_path: Path, *, formats: tuple[str, ...], dpi: int) -> dict[str, str]:
    outputs = {}
    base_path.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        out = base_path.with_suffix(f".{fmt}")
        if fmt == "pdf":
            fig.savefig(out, format="pdf", dpi=dpi)
        else:
            fig.savefig(out, format=fmt, dpi=dpi)
        outputs[fmt] = str(out)
    plt.close(fig)
    return outputs


def _build_main_m1(frames: dict[str, pd.DataFrame], _: dict[str, Any]) -> tuple[plt.Figure, dict[str, Any]]:
    coverage = frames["descriptor_coverage.csv"]
    parity = frames["surrogate_parity_mean_predictions.csv"]
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_IN, 11.0 * CM_TO_IN))
    pca = coverage.loc[coverage["record_type"] == "pca_cumulative"].sort_values("component_index")
    ax = axes[0, 0]
    ax.plot(pca["component_index"], pca["cumulative_explained_variance"], color=OKABE_ITO["blue"], marker="o", linewidth=1.4, markersize=3.5)
    ax.axhline(0.95, color=OKABE_ITO["red"], linestyle="--", linewidth=1.0)
    ax.axvline(6, color=OKABE_ITO["gray"], linestyle=":", linewidth=1.0)
    _style_axis(ax)
    _panel_label(ax, "a")
    ax.set_xlabel("Number of principal components")
    ax.set_ylabel("Cumulative explained variance")
    ax.set_ylim(0.0, 1.02)
    for ax, target, label in zip(axes.flatten()[1:], PERFORMANCE_TARGETS, ["b", "c", "d"], strict=True):
        truth = parity[f"analytic_{target}"].to_numpy(dtype=float)
        pred = parity[f"predicted_{target}"].to_numpy(dtype=float)
        stats = _metric_stats(truth, pred)
        ax.scatter(truth, pred, s=10, c=OKABE_ITO["blue"], alpha=0.55, edgecolors="none")
        limits = _ensure_equal_limits(ax, truth, pred)
        ax.plot(limits, limits, color=OKABE_ITO["black"], linestyle="--", linewidth=0.9)
        _style_axis(ax)
        _panel_label(ax, label)
        ax.set_xlabel(f"Analytic target {TARGET_LABELS[target].split(' ', 1)[-1]}")
        ax.set_ylabel(f"Cross-validated surrogate prediction {TARGET_LABELS[target].split(' ', 1)[-1]}")
        text = f"R²={stats['R2']:.3f}\nMAE={stats['MAE']:.3f}\nnMAE={stats['nMAE']:.3f}\nρ={stats['Spearman']:.3f}"
        ax.text(0.96, 0.04, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=6.7)
    fig.tight_layout(pad=0.4, w_pad=0.8, h_pad=0.8)
    return fig, {
        "parity_axes": {
            target: {
                "xlim": list(axes.flatten()[idx].get_xlim()),
                "ylim": list(axes.flatten()[idx].get_ylim()),
            }
            for idx, target in zip([1, 2, 3], PERFORMANCE_TARGETS, strict=True)
        }
    }


def _build_main_m2(frames: dict[str, pd.DataFrame], _: dict[str, Any]) -> tuple[plt.Figure, dict[str, Any]]:
    frame = frames["surrogate_validation_regimes.csv"].copy()
    family_order = ["repeated_kfold", "leave_one_osli_out", "outer_shell_holdout", "feature_tail_holdout"]
    target_order = ["EUIt", "EG", "H"]
    metrics = [("mean_nMAE", "a"), ("mean_Spearman_rho", "b")]
    fig, axes = plt.subplots(1, 2, figsize=(DOUBLE_COL_IN, 5.2 * CM_TO_IN))
    for ax, (metric, label) in zip(axes, metrics, strict=True):
        pivot = frame.pivot(index="validation_family", columns="target", values=metric).reindex(index=family_order, columns=target_order)
        im = ax.imshow(pivot.to_numpy(dtype=float), cmap="Blues" if metric == "mean_nMAE" else "Greens", aspect="auto")
        _panel_label(ax, label)
        ax.set_xticks(np.arange(len(target_order)))
        ax.set_xticklabels([target.replace("EUIt", "EUIt").replace("EG", "EG").replace("H", "H") for target in target_order])
        ax.set_yticks(np.arange(len(family_order)))
        ax.set_yticklabels(["Repeated 5×5 CV", "Leave-one-OSLI-out", "Outer-shell holdout", "Feature-tail holdout"])
        for i in range(len(family_order)):
            for j in range(len(target_order)):
                ax.text(j, i, f"{pivot.iloc[i, j]:.3f}", ha="center", va="center", fontsize=6.5)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.ax.tick_params(labelsize=6.5)
        cbar.set_label("Lower is better" if metric == "mean_nMAE" else "Higher is better", fontsize=6.5)
    fig.tight_layout(pad=0.4, w_pad=0.8)
    return fig, {}


def _build_main_m3(frames: dict[str, pd.DataFrame], _: dict[str, Any]) -> tuple[plt.Figure, dict[str, Any]]:
    frame = frames["ddpg_training_curves_summary.csv"].copy()
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_IN, 11.0 * CM_TO_IN), sharex=True)
    panels = [
        ("reward_mean", "reward_std", "Cumulative reward", "a"),
        ("EUIt_mean", "EUIt_std", TARGET_LABELS["EUIt"], "b"),
        ("EG_mean", "EG_std", TARGET_LABELS["EG"], "c"),
        ("H_mean", "H_std", TARGET_LABELS["H"], "d"),
    ]
    colors = {
        "Balanced_Performance": OKABE_ITO["blue"],
        "Energy_Saving_Focus": OKABE_ITO["green"],
        "Energy_Generation_Focus": OKABE_ITO["red"],
    }
    markers = {"Balanced_Performance": "o", "Energy_Saving_Focus": "^", "Energy_Generation_Focus": "s"}
    for ax, (mean_col, std_col, ylabel, label) in zip(axes.flatten(), panels, strict=True):
        for scenario, group in frame.groupby("scenario", sort=True):
            ax.plot(group["episode"], group[mean_col], color=colors[scenario], linewidth=1.2, marker=markers[scenario], markersize=0, linestyle="-")
            ax.fill_between(group["episode"], group[mean_col] - group[std_col].fillna(0.0), group[mean_col] + group[std_col].fillna(0.0), color=colors[scenario], alpha=0.16)
        _style_axis(ax)
        _panel_label(ax, label)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Episode")
    handles = [
        Line2D([0], [0], color=colors[key], marker=markers[key], linewidth=1.2, markersize=4, label=key.replace("_", " "))
        for key in ["Balanced_Performance", "Energy_Saving_Focus", "Energy_Generation_Focus"]
    ]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.02), ncol=3, frameon=False)
    fig.tight_layout(pad=0.4, w_pad=0.8, h_pad=0.8, rect=(0, 0, 1, 0.96))
    return fig, {"episode_horizon": 600, "queries_per_episode": 40}


def _build_main_m4(frames: dict[str, pd.DataFrame], manifest: dict[str, Any]) -> tuple[plt.Figure, dict[str, Any]]:
    utility = frames["benchmark_utility.csv"].copy()
    eq20 = frames["benchmark_equal_size_20.csv"].copy()
    contract = frames["benchmark_output_contract_counts.csv"].copy()
    reference_hash = _package_entry(manifest, "benchmark_equal_size_20.csv")["reference_hash"]
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_IN, 11.2 * CM_TO_IN))
    scenarios = ["Balanced_Performance", "Energy_Saving_Focus", "Energy_Generation_Focus"]
    labels = ["Balanced", "Saving", "Generation"]
    methods = ["DDPG", "NSGA-II"]
    ax = axes[0, 0]
    x = np.arange(len(scenarios))
    for offset, method, color, marker in [(-0.1, "DDPG", OKABE_ITO["blue"], "o"), (0.1, "NSGA-II", OKABE_ITO["black"], "s")]:
        subset = utility.loc[utility["method"] == method].copy()
        values = []
        errors = []
        for scenario in scenarios:
            scenario_group = subset.loc[subset["evaluation_scenario"] == scenario, "best_fixed_domain_utility"].to_numpy(dtype=float)
            values.append(float(np.mean(scenario_group)))
            errors.append([float(np.quantile(scenario_group, 0.05)), float(np.quantile(scenario_group, 0.95))])
        values_arr = np.asarray(values, dtype=float)
        low = values_arr - np.asarray([item[0] for item in errors])
        high = np.asarray([item[1] for item in errors]) - values_arr
        ax.errorbar(x + offset, values_arr, yerr=np.vstack([low, high]), color=color, marker=marker, linestyle="-", linewidth=1.1, markersize=4, capsize=2.2, label=method)
    _style_axis(ax)
    _panel_label(ax, "a")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Fixed-domain post-hoc utility")
    ax.legend(frameon=False, loc="lower right")

    group_positions = {"DDPG": 0, "NSGA-II": 1, "CMA-ES": 2, "RandomSearch": 3, "FeasiblePoolRandom": 4}
    for axis, metric, label in [(axes[0, 1], "HV", "b"), (axes[1, 0], "IGD", "c")]:
        for scenario, color, marker in zip(scenarios, [OKABE_ITO["blue"], OKABE_ITO["green"], OKABE_ITO["red"]], ["o", "^", "s"], strict=True):
            subset = eq20.loc[(eq20["scenario"] == scenario) | ((eq20["group"] == "NSGA-II") & (scenario == "Balanced_Performance"))]
            plotted = []
            for group_name, group_frame in subset.groupby("group"):
                method = "NSGA-II" if group_name == "NSGA-II" else group_frame["method"].iloc[0]
                if method == "NSGA-II" and scenario != "Balanced_Performance":
                    continue
                value = group_frame[metric].to_numpy(dtype=float)
                q05 = float(np.quantile(value, 0.05))
                q95 = float(np.quantile(value, 0.95))
                x_pos = group_positions[method] + {"Balanced_Performance": -0.18, "Energy_Saving_Focus": 0.0, "Energy_Generation_Focus": 0.18}.get(scenario, 0.0)
                mean = float(np.mean(value))
                lower_err = max(mean - q05, 0.0)
                upper_err = max(q95 - mean, 0.0)
                axis.errorbar([x_pos], [mean], yerr=[[lower_err], [upper_err]], color=color if method != "NSGA-II" else OKABE_ITO["black"], marker=marker if method != "NSGA-II" else "D", linestyle="", capsize=2.0, markersize=4.2)
                plotted.append((x_pos, mean))
            if metric == "HV":
                axis.axhline(1.331, color=OKABE_ITO["gray"], linestyle="--", linewidth=0.9)
                axis.text(4.3, 1.331, " ceiling=1.331", va="bottom", ha="left", fontsize=6.5, color=OKABE_ITO["gray"])
        _style_axis(axis)
        _panel_label(axis, label)
        axis.set_xticks(list(group_positions.values()))
        axis.set_xticklabels(list(group_positions))
        axis.set_ylabel(metric)
        axis.tick_params(axis="x", rotation=20)
    axes[0, 1].annotate("CMA-ES: clipped-utopia duplicates dominate", xy=(2, 1.331), xytext=(2.5, 1.16), arrowprops={"arrowstyle": "->", "linewidth": 0.8}, fontsize=6.3)

    ax = axes[1, 1]
    melt = contract.melt(
        id_vars=["method", "scenario"],
        value_vars=["total_retained_rows", "unique_objective_tuples", "unique_feasible_projections"],
        var_name="count_type",
        value_name="count_value",
    )
    count_labels = {
        "total_retained_rows": "retained rows",
        "unique_objective_tuples": "unique objective tuples",
        "unique_feasible_projections": "unique feasible blocks",
    }
    for count_type, marker, linestyle in [("total_retained_rows", "o", "-"), ("unique_objective_tuples", "^", "--"), ("unique_feasible_projections", "s", ":")]:
        subset = melt.loc[melt["count_type"] == count_type]
        for scenario, color, offset in [("Balanced_Performance", OKABE_ITO["blue"], -0.15), ("Energy_Saving_Focus", OKABE_ITO["green"], 0.0), ("Energy_Generation_Focus", OKABE_ITO["red"], 0.15)]:
            points = subset.loc[(subset["scenario"] == scenario) | ((subset["method"] == "NSGA-II") & (scenario == "Balanced_Performance"))]
            xs = []
            ys = []
            for _, row in points.iterrows():
                x_pos = group_positions[row["method"]] + (offset if row["method"] != "NSGA-II" else 0.0)
                xs.append(x_pos)
                ys.append(row["count_value"])
            ax.plot(xs, ys, color=color, marker=marker, linestyle=linestyle, linewidth=0.9, markersize=3.5)
    _style_axis(ax)
    _panel_label(ax, "d")
    ax.set_yscale("log")
    ax.set_xticks(list(group_positions.values()))
    ax.set_xticklabels(list(group_positions), rotation=20)
    ax.set_ylabel("Count (log scale)")
    ax.text(0.02, 0.02, f"reference hash: {reference_hash[:12]}…", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.2)
    fig.tight_layout(pad=0.4, w_pad=0.8, h_pad=0.8)
    return fig, {"reference_hash": reference_hash, "reference_protocol": BENCHMARK_REFERENCE_PROTOCOL}


def _build_main_m5(frames: dict[str, pd.DataFrame], manifest: dict[str, Any]) -> tuple[plt.Figure, dict[str, Any]]:
    summary = frames["feasible_projection_summary.csv"].copy()
    metrics = frames["feasible_projection_metrics.csv"].copy()
    reference_hash = _package_entry(manifest, "feasible_projection_summary.csv")["reference_hash"]
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_IN, 11.0 * CM_TO_IN))
    order = [
        "DDPG::Balanced_Performance",
        "DDPG::Energy_Saving_Focus",
        "DDPG::Energy_Generation_Focus",
        "NSGA-II",
        "CMA-ES::Balanced_Performance",
        "CMA-ES::Energy_Saving_Focus",
        "CMA-ES::Energy_Generation_Focus",
        "RandomSearch::Balanced_Performance",
        "RandomSearch::Energy_Saving_Focus",
        "RandomSearch::Energy_Generation_Focus",
    ]
    ax = axes[0, 0]
    positions = np.arange(len(order))
    for pos, group in zip(positions, order, strict=True):
        subset = metrics.loc[metrics["group"] == group, "projection_distance"].to_numpy(dtype=float)
        if subset.size == 0:
            continue
        ax.violinplot(subset, positions=[pos], widths=0.7, showmeans=True, showextrema=False)
    _style_axis(ax)
    _panel_label(ax, "a")
    ax.set_xticks(positions)
    ax.set_xticklabels([item.replace("::", "\n").replace("_", " ") for item in order], rotation=0, fontsize=6.2)
    ax.set_ylabel("Projection distance")

    ax = axes[0, 1]
    subset = summary.set_index("group").reindex(order).dropna(subset=["duplicate_collapse_rate"])
    ax.bar(np.arange(len(subset)), subset["duplicate_collapse_rate"], color=OKABE_ITO["blue"], alpha=0.8, hatch="//")
    ax2 = ax.twinx()
    ax2.plot(np.arange(len(subset)), subset["unique_matched_sample_count"], color=OKABE_ITO["red"], marker="o", linewidth=1.1)
    _style_axis(ax)
    _panel_label(ax, "b")
    ax.set_xticks(np.arange(len(subset)))
    ax.set_xticklabels([item.replace("::", "\n").replace("_", " ") for item in subset.index], fontsize=6.2)
    ax.set_ylabel("Duplicate collapse rate")
    ax2.set_ylabel("Unique feasible blocks")
    ax.annotate("NSGA-II: 2000 descriptor rows → 51 blocks", xy=(3, subset.loc["NSGA-II", "duplicate_collapse_rate"]), xytext=(4.5, 0.72), arrowprops={"arrowstyle": "->", "linewidth": 0.8}, fontsize=6.3)

    for axis, metric_before, metric_after, label in [
        (axes[1, 0], "descriptor_HV", "projected_HV_fixed_reference", "c"),
        (axes[1, 1], "descriptor_IGD", "projected_IGD_fixed_reference", "d"),
    ]:
        for index, row in subset.iterrows():
            x1, x2 = 0, 1
            axis.plot([x1, x2], [row[metric_before], row[metric_after]], color=OKABE_ITO["gray"], linewidth=0.6, alpha=0.7)
            axis.scatter([x1, x2], [row[metric_before], row[metric_after]], c=[OKABE_ITO["blue"], OKABE_ITO["red"]], s=12)
        _style_axis(axis)
        _panel_label(axis, label)
        axis.set_xticks([0, 1])
        axis.set_xticklabels(["Descriptor", "Projected feasible"])
        axis.set_ylabel("HV" if "HV" in metric_before else "IGD")
        axis.text(0.02, 0.02, f"reference hash: {reference_hash[:12]}…", transform=axis.transAxes, ha="left", va="bottom", fontsize=6.2)
    fig.tight_layout(pad=0.4, w_pad=0.9, h_pad=0.8)
    return fig, {"reference_hash": reference_hash, "reference_protocol": BENCHMARK_REFERENCE_PROTOCOL}


def _build_main_m6(frames: dict[str, pd.DataFrame], _: dict[str, Any]) -> tuple[plt.Figure, dict[str, Any]]:
    direct = frames["physical_direct_cases.csv"].copy()
    metrics = frames["physical_stress_metrics.csv"].set_index("target")
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_IN, 5.3 * CM_TO_IN))
    columns = [
        ("analytic_EUIt", "physical_EUIt", "EUIt", "a"),
        ("analytic_EG", "physical_EG_GHI_proxy", "EG_GHI_proxy", "b"),
        ("analytic_H", "physical_H", "H", "c"),
    ]
    parity_limits = {}
    for ax, (x_col, y_col, metric_key, label) in zip(axes, columns, strict=True):
        x = direct[x_col].to_numpy(dtype=float)
        y = direct[y_col].to_numpy(dtype=float)
        ax.scatter(x, y, s=18, c=OKABE_ITO["blue"], alpha=0.7, edgecolors="white", linewidths=0.3)
        limits = _ensure_equal_limits(ax, x, y)
        ax.plot(limits, limits, color=OKABE_ITO["black"], linestyle="--", linewidth=0.9)
        coef = np.polyfit(x, y, deg=1)
        xs = np.linspace(limits[0], limits[1], 100)
        ax.plot(xs, coef[0] * xs + coef[1], color=OKABE_ITO["red"], linewidth=0.9)
        row = metrics.loc[metric_key]
        text = f"n={int(row['count'])}\nMAE={row['MAE']:.3f}\nnMAE={row['nMAE']:.3f}\nρ={row['Spearman_rho']:.3f}\nrank={row['rank_preservation']:.2f}"
        ax.text(0.96, 0.04, text, transform=ax.transAxes, ha="right", va="bottom", fontsize=6.5)
        _style_axis(ax)
        _panel_label(ax, label)
        unit_label = TARGET_LABELS["EUIt" if metric_key == "EUIt" else ("EG" if metric_key == "EG_GHI_proxy" else "H")]
        ax.set_xlabel(f"Analytic response-generator value {unit_label}")
        ax.set_ylabel(f"Physics-based stress-test value {unit_label if metric_key != 'EG_GHI_proxy' else 'EG (10⁶ kWh/y)'}")
        parity_limits[metric_key] = {"xlim": list(ax.get_xlim()), "ylim": list(ax.get_ylim())}
    fig.tight_layout(pad=0.4, w_pad=0.8)
    return fig, {"parity_axes": parity_limits, "direct_case_count": int(len(direct)), "stress_test_label": "limited physics-based cross-model stress test"}


def _build_main_m7(frames: dict[str, pd.DataFrame], _: dict[str, Any]) -> tuple[plt.Figure, dict[str, Any]]:
    cases = frames["climate_case_results.csv"].copy()
    summary = frames["climate_summary.csv"].copy()
    stability = frames["climate_rank_stability.csv"].copy()
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_IN, 11.0 * CM_TO_IN))
    metrics = [("delta_EUIt_vs_baseline", TARGET_LABELS["EUIt"], "a"), ("delta_EG_vs_baseline", TARGET_LABELS["EG"], "b"), ("delta_H_vs_baseline", TARGET_LABELS["H"], "c")]
    stations = ["Beijing", "Guangzhou", "Harbin"]
    x = np.arange(len(stations))
    for ax, (column, ylabel, label) in zip(axes.flatten()[:3], metrics, strict=True):
        grouped = cases.groupby("station")[column]
        means = [float(grouped.mean().loc[station]) for station in stations]
        lowers = [float(grouped.min().loc[station]) for station in stations]
        uppers = [float(grouped.max().loc[station]) for station in stations]
        ax.bar(x, means, color=[OKABE_ITO["blue"], OKABE_ITO["green"], OKABE_ITO["red"]], alpha=0.85, edgecolor="black", linewidth=0.3)
        ax.errorbar(x, means, yerr=[np.asarray(means) - np.asarray(lowers), np.asarray(uppers) - np.asarray(means)], fmt="none", ecolor=OKABE_ITO["black"], linewidth=0.8, capsize=2.0)
        _style_axis(ax)
        _panel_label(ax, label)
        ax.set_xticks(x)
        ax.set_xticklabels(stations)
        ax.set_ylabel(f"Mean Δ relative to Dongtai {ylabel}")
    ax = axes[1, 1]
    pivot = stability.pivot(index="station", columns="rank_metric", values="spearman").reindex(index=stations, columns=["EUIt", "EG", "H"])
    im = ax.imshow(pivot.to_numpy(dtype=float), cmap="RdYlGn", vmin=-1.0, vmax=1.0, aspect="auto")
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            ax.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6.5)
    _panel_label(ax, "d")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(["EUIt", "EG", "H"])
    ax.set_yticks(np.arange(len(stations)))
    ax.set_yticklabels(stations)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=6.5)
    cbar.set_label("Spearman rank stability", fontsize=6.5)
    fig.tight_layout(pad=0.4, w_pad=0.8, h_pad=0.8)
    return fig, {"direct_blocks": 4, "additional_climates": 3, "analysis_label": "limited four-block cross-climate physical sensitivity analysis"}


def _build_appendix_a1(frames: dict[str, pd.DataFrame], _: dict[str, Any]) -> tuple[plt.Figure, dict[str, Any]]:
    coverage = frames["descriptor_coverage.csv"].copy()
    feature_rows = coverage.loc[coverage["record_type"] == "feature_summary"].copy()
    nn_rows = coverage.loc[coverage["record_type"] == "nearest_neighbor_distance"].copy()
    osli = coverage.loc[coverage["record_type"] == "osli_frequency"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_IN, 5.3 * CM_TO_IN))
    ax = axes[0]
    ordered = feature_rows.set_index("feature").loc[MORPHOLOGY_FEATURES].reset_index()
    ax.errorbar(np.arange(len(ordered)), ordered["median"], yerr=[ordered["median"] - ordered["q25"], ordered["q75"] - ordered["median"]], fmt="o", color=OKABE_ITO["blue"], capsize=2.0, linewidth=0.9)
    ax.set_xticks(np.arange(len(ordered)))
    ax.set_xticklabels([FEATURE_LABELS[feature] for feature in ordered["feature"]], rotation=45, ha="right")
    ax.set_ylabel("Descriptor value")
    _style_axis(ax)
    _panel_label(ax, "a")
    ax = axes[1]
    ax.bar(osli["OSLI"], osli["count"], color=OKABE_ITO["green"], edgecolor="black", linewidth=0.3)
    ax.set_xlabel("OSLI")
    ax.set_ylabel("Count")
    _style_axis(ax)
    _panel_label(ax, "b")
    ax = axes[2]
    ax.hist(nn_rows["normalized_nearest_neighbor_distance"], bins=24, color=OKABE_ITO["red"], alpha=0.8, edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Normalized nearest-neighbor distance")
    ax.set_ylabel("Count")
    _style_axis(ax)
    _panel_label(ax, "c")
    fig.tight_layout(pad=0.4, w_pad=0.8)
    return fig, {}


def _build_appendix_a2(frames: dict[str, pd.DataFrame], _: dict[str, Any]) -> tuple[plt.Figure, dict[str, Any]]:
    parity = frames["surrogate_parity_mean_predictions.csv"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_IN, 5.0 * CM_TO_IN))
    for ax, target, label in zip(axes, PERFORMANCE_TARGETS, ["a", "b", "c"], strict=True):
        residual = parity[f"residual_{target}"].to_numpy(dtype=float)
        ax.hist(residual, bins=28, color=OKABE_ITO["blue"], alpha=0.78, edgecolor="white", linewidth=0.3)
        ax.axvline(0.0, color=OKABE_ITO["black"], linestyle="--", linewidth=0.9)
        _style_axis(ax)
        _panel_label(ax, label)
        ax.set_xlabel(f"{TARGET_LABELS[target]} residual")
        ax.set_ylabel("Count")
    fig.tight_layout(pad=0.4, w_pad=0.8)
    return fig, {}


def _build_appendix_a3(frames: dict[str, pd.DataFrame], _: dict[str, Any]) -> tuple[plt.Figure, dict[str, Any]]:
    scale = frames["scale_study.csv"].copy()
    winners = scale.loc[scale["is_selected"]].sort_values("dataset_scale")
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_IN, 10.8 * CM_TO_IN), sharex=True)
    for ax, column, ylabel, label in [
        (axes[0, 0], "mean_target_nmae", "Mean target nMAE", "a"),
        (axes[0, 1], "mean_tail_nmae", "Mean tail nMAE", "b"),
        (axes[1, 0], "mean_r2", "Mean R²", "c"),
        (axes[1, 1], "selection_objective", "Selection objective", "d"),
    ]:
        for candidate, group in scale.groupby("candidate"):
            ax.plot(group["dataset_scale"], group[column], linewidth=0.8, alpha=0.35, color=OKABE_ITO["gray"])
        ax.plot(winners["dataset_scale"], winners[column], color=OKABE_ITO["blue"], marker="o", linewidth=1.2)
        _style_axis(ax)
        _panel_label(ax, label)
        ax.set_xlabel("Dataset scale")
        ax.set_ylabel(ylabel)
    fig.tight_layout(pad=0.4, w_pad=0.8, h_pad=0.8)
    return fig, {}


def _build_appendix_b1(frames: dict[str, pd.DataFrame], _: dict[str, Any]) -> tuple[plt.Figure, dict[str, Any]]:
    seed = frames["ddpg_seed_diagnostics.csv"].copy()
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_IN, 10.8 * CM_TO_IN))
    panels = [
        ("reward_best", "Best reward", "a"),
        ("reward_final", "Final reward", "b"),
        ("plateau_episode", "Plateau episode", "c"),
        ("best_final_gap_ratio", "Best-to-final gap ratio", "d"),
    ]
    scenarios = ["Balanced_Performance", "Energy_Saving_Focus", "Energy_Generation_Focus"]
    colors = [OKABE_ITO["blue"], OKABE_ITO["green"], OKABE_ITO["red"]]
    for ax, (column, ylabel, label) in zip(axes.flatten(), panels, strict=True):
        values = [seed.loc[seed["scenario"] == scenario, column].to_numpy(dtype=float) for scenario in scenarios]
        ax.boxplot(values, tick_labels=["Balanced", "Saving", "Generation"], patch_artist=True, boxprops={"facecolor": "white", "linewidth": 0.9}, medianprops={"color": OKABE_ITO["black"]})
        for patch, color in zip(ax.artists, colors, strict=False):
            patch.set_facecolor(color)
        _style_axis(ax)
        _panel_label(ax, label)
        ax.set_ylabel(ylabel)
    fig.tight_layout(pad=0.4, w_pad=0.8, h_pad=0.8)
    return fig, {}


def _build_appendix_b2(frames: dict[str, Any], _: dict[str, Any]) -> tuple[plt.Figure, dict[str, Any]]:
    signatures = frames["morphology_signatures.csv"].copy()
    pivot = signatures.pivot(index="group_label", columns="feature", values="median")
    normalized = pivot.copy()
    for column in normalized.columns:
        values = normalized[column]
        normalized[column] = 2.0 * ((values - values.min()) / max(values.max() - values.min(), 1e-8)) - 1.0
    fig, ax = plt.subplots(figsize=(DOUBLE_COL_IN, 4.8 * CM_TO_IN))
    im = ax.imshow(normalized.to_numpy(dtype=float), cmap="RdBu_r", aspect="auto", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(normalized.columns)))
    ax.set_xticklabels([FEATURE_LABELS[column] for column in normalized.columns], rotation=45, ha="right")
    ax.set_yticks(np.arange(len(normalized.index)))
    ax.set_yticklabels(normalized.index)
    _panel_label(ax, "a")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=6.5)
    cbar.set_label("Relative median", fontsize=6.5)
    fig.tight_layout(pad=0.4)
    return fig, {}


def _build_appendix_b3(frames: dict[str, pd.DataFrame], manifest: dict[str, Any]) -> tuple[plt.Figure, dict[str, Any]]:
    ceiling = frames["benchmark_hv_ceiling.csv"].copy()
    reference_hash = _package_entry(manifest, "benchmark_hv_ceiling.csv")["reference_hash"]
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_IN, 10.8 * CM_TO_IN))
    metrics = [
        ("fraction_of_theoretical_max", "HV / ceiling", "a"),
        ("clipped_utopia_fraction", "Clipped-utopia fraction", "b"),
        ("unique_objective_tuples", "Unique objective tuples", "c"),
        ("unique_non_dominated_tuples", "Unique non-dominated tuples", "d"),
    ]
    labels = [item.replace("::", "\n").replace("_", " ") for item in ceiling["group"]]
    x = np.arange(len(ceiling))
    for ax, (column, ylabel, label) in zip(axes.flatten(), metrics, strict=True):
        ax.bar(x, ceiling[column], color=OKABE_ITO["blue"], alpha=0.85, edgecolor="black", linewidth=0.3)
        _style_axis(ax)
        _panel_label(ax, label)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.0)
        ax.set_ylabel(ylabel)
        if column == "fraction_of_theoretical_max":
            ax.text(0.02, 0.02, f"reference hash: {reference_hash[:12]}…", transform=ax.transAxes, ha="left", va="bottom", fontsize=6.2)
    fig.tight_layout(pad=0.4, w_pad=0.8, h_pad=0.8)
    return fig, {"reference_hash": reference_hash, "reference_protocol": BENCHMARK_REFERENCE_PROTOCOL}


def _build_appendix_b4(frames: dict[str, pd.DataFrame], _: dict[str, Any]) -> tuple[plt.Figure, dict[str, Any]]:
    gaps = frames["optimizer_linked_physical_gaps.csv"].copy()
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_IN, 5.2 * CM_TO_IN), sharey=False)
    targets = [("EUIt", "a"), ("EG", "b"), ("H", "c")]
    labels = [f"{row.optimizer_source}\n{row.scenario.split('_')[0]}" for row in gaps.itertuples()]
    x = np.arange(len(gaps))
    for ax, (target, label) in zip(axes, targets, strict=True):
        projection = gaps[f"projection_gap_{target}"].to_numpy(dtype=float)
        cross_model = gaps[f"analytic_to_physical_gap_{target}"].to_numpy(dtype=float)
        ax.bar(x, projection, color=OKABE_ITO["blue"], alpha=0.85, label="projection gap")
        ax.bar(x, cross_model, bottom=projection, color=OKABE_ITO["red"], alpha=0.55, label="cross-model gap")
        _style_axis(ax)
        _panel_label(ax, label)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=6.0)
        ax.set_ylabel(f"{target} gap")
    axes[0].legend(frameon=False, fontsize=6.5)
    fig.tight_layout(pad=0.4, w_pad=0.8)
    return fig, {}


def _build_appendix_b5(frames: dict[str, pd.DataFrame], _: dict[str, Any]) -> tuple[plt.Figure, dict[str, Any]]:
    profiles = frames["scale_study.csv"]
    # The actual nonlinear profiles are regenerated from the canonical surrogate and dataset.
    raise RuntimeError("placeholder should be replaced in builder pipeline")


def _build_appendix_b6(frames: dict[str, pd.DataFrame], _: dict[str, Any]) -> tuple[plt.Figure, dict[str, Any]]:
    cases = frames["climate_case_results.csv"].copy()
    targets = [
        ("delta_EUIt_vs_baseline", "ΔEUIt (kWh/m²/y)", "a"),
        ("delta_EG_vs_baseline", "ΔEG (10⁶ kWh/y)", "b"),
        ("delta_H_vs_baseline", "ΔH (h)", "c"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(DOUBLE_COL_IN, 5.2 * CM_TO_IN))
    pivot_index = sorted(cases["matched_sample_id"].unique())
    pivot_columns = ["Beijing", "Guangzhou", "Harbin"]
    for ax, (column, title, label) in zip(axes, targets, strict=True):
        pivot = cases.pivot(index="matched_sample_id", columns="station", values=column).reindex(index=pivot_index, columns=pivot_columns)
        im = ax.imshow(pivot.to_numpy(dtype=float), cmap="coolwarm", aspect="auto")
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                ax.text(j, i, f"{pivot.iloc[i, j]:.2f}", ha="center", va="center", fontsize=6.2)
        _panel_label(ax, label)
        ax.set_xticks(np.arange(len(pivot_columns)))
        ax.set_xticklabels(pivot_columns)
        ax.set_yticks(np.arange(len(pivot_index)))
        ax.set_yticklabels([str(item) for item in pivot_index])
        ax.set_xlabel("Climate station")
        ax.set_ylabel("Block ID")
        ax.set_title(title, fontsize=7.5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.tight_layout(pad=0.4, w_pad=0.8)
    return fig, {}


def _build_nonlinear_response_figure(repo_root: Path, roots: dict[str, Path]) -> tuple[plt.Figure, dict[str, Any]]:
    bundle = load_surrogate(roots["compare_root"] / "models" / "surrogate.pt")
    dataset = pd.read_csv(roots["compare_root"] / "data" / "simulated_samples.csv")
    base_point = dataset[MORPHOLOGY_FEATURES].median().to_numpy(dtype=float)
    pairs = [("OSR", "EUIt"), ("FAR", "EG"), ("SVF", "H"), ("theta", "H")]
    fig, axes = plt.subplots(2, 2, figsize=(DOUBLE_COL_IN, 10.4 * CM_TO_IN))
    for ax, (feature, target), label in zip(axes.flatten(), pairs, ["a", "b", "c", "d"], strict=True):
        values = np.linspace(float(dataset[feature].quantile(0.05)), float(dataset[feature].quantile(0.95)), 120)
        preds = []
        feature_index = MORPHOLOGY_FEATURES.index(feature)
        for value in values:
            action = base_point.copy()
            action[feature_index] = value
            preds.append(float(bundle.predict_action(action)[PERFORMANCE_TARGETS.index(target)]))
        ax.plot(values, preds, color=OKABE_ITO["blue"], linewidth=1.3)
        _style_axis(ax)
        _panel_label(ax, label)
        ax.set_xlabel(FEATURE_LABELS[feature])
        ax.set_ylabel(TARGET_LABELS[target])
    fig.tight_layout(pad=0.4, w_pad=0.8, h_pad=0.8)
    return fig, {"profile_pairs": pairs}


def _figure_specs(repo_root: Path, roots: dict[str, Path]) -> tuple[FigureSpec, ...]:
    return (
        FigureSpec(
            "M1",
            "data_and_surrogate_validation",
            "main",
            "Main Fig. 4 candidate",
            ("descriptor_coverage.csv", "surrogate_parity_mean_predictions.csv"),
            (
                "PCA cumulative explained variance for the 12 morphology descriptors.",
                "EUIt repeated-CV parity using sample-level mean out-of-fold predictions.",
                "EG repeated-CV parity using sample-level mean out-of-fold predictions.",
                "H repeated-CV parity using sample-level mean out-of-fold predictions.",
            ),
            "Supports descriptor-space coverage and analytic-target surrogate fidelity only.",
            _build_main_m1,
        ),
        FigureSpec(
            "M2",
            "surrogate_robustness",
            "main",
            "Main Fig. 5 candidate",
            ("surrogate_validation_regimes.csv",),
            (
                "nMAE heatmap across repeated CV, leave-one-OSLI-out, outer-shell, and feature-tail regimes.",
                "Spearman heatmap across the same surrogate-validation regimes.",
            ),
            "All panels remain analytic-target surrogate-validation evidence, not physical validation.",
            _build_main_m2,
        ),
        FigureSpec(
            "M3",
            "ddpg_training_dynamics",
            "main",
            "Main Fig. 6 candidate",
            ("ddpg_training_curves_summary.csv", "ddpg_seed_diagnostics.csv"),
            (
                "Episode cumulative reward across 20 seeds.",
                "Episode-end EUIt across 20 seeds.",
                "Episode-end EG across 20 seeds.",
                "Episode-end H across 20 seeds.",
            ),
            "Training dynamics describe serialized surrogate-query search only.",
            _build_main_m3,
        ),
        FigureSpec(
            "M4",
            "benchmark_fairness",
            "main",
            "Main Fig. 7 candidate",
            ("benchmark_utility.csv", "benchmark_equal_size_20.csv", "benchmark_output_contract_counts.csv"),
            (
                "Fixed-domain post-hoc utility for DDPG and NSGA-II across the three scalarization scenarios.",
                "Equal-size-20 HV with 5–95% intervals under benchmark-reference-v2.",
                "Equal-size-20 IGD with 5–95% intervals under benchmark-reference-v2.",
                "Output-contract asymmetry across retained rows, unique objective tuples, and unique feasible blocks.",
            ),
            "Equal-size metrics are canonical only under benchmark-reference-v2 and must stay separate from asymmetric full-archive diagnostics.",
            _build_main_m4,
        ),
        FigureSpec(
            "M5",
            "feasible_projection",
            "main",
            "Main Fig. 8 candidate",
            ("feasible_projection_summary.csv", "feasible_projection_metrics.csv"),
            (
                "Projection-distance distribution from descriptor candidates to feasible blocks.",
                "Duplicate-collapse rate with unique feasible-block counts.",
                "HV before and after projection under benchmark-reference-v2.",
                "IGD before and after projection under benchmark-reference-v2.",
            ),
            "Projection panels are representation-sensitivity diagnostics rather than physical validation.",
            _build_main_m5,
        ),
        FigureSpec(
            "M6",
            "physical_cross_model_stress_test",
            "main",
            "Main Fig. 9 candidate",
            ("physical_direct_cases.csv", "physical_stress_metrics.csv"),
            (
                "EUIt parity for the 18 direct feasible cases.",
                "Simplified rooftop-PV proxy parity for the 18 direct feasible cases.",
                "January 20 windowsill direct-sun-hours parity for the 18 direct feasible cases.",
            ),
            "This figure is limited to the direct-case physics-based cross-model stress test and does not support optimizer-superiority claims.",
            _build_main_m6,
        ),
        FigureSpec(
            "M7",
            "cross_climate_sensitivity",
            "main",
            "Main Fig. 10 candidate",
            ("climate_case_results.csv", "climate_summary.csv", "climate_rank_stability.csv"),
            (
                "Mean ΔEUIt relative to Dongtai with four-block spread.",
                "Mean ΔEG relative to Dongtai with four-block spread.",
                "Mean ΔH relative to Dongtai with four-block spread.",
                "Rank-stability heatmap across Beijing, Guangzhou, and Harbin.",
            ),
            "This figure is a limited four-block cross-climate physical sensitivity analysis only.",
            _build_main_m7,
        ),
        FigureSpec(
            "A1",
            "A1_descriptor_distributions",
            "appendix",
            "Appendix Fig. A1 candidate",
            ("descriptor_coverage.csv",),
            (
                "Descriptor interquartile summaries for the 12 morphology descriptors.",
                "OSLI frequency distribution.",
                "Normalized nearest-neighbor distance distribution.",
            ),
            "Descriptive coverage diagnostics only.",
            _build_appendix_a1,
        ),
        FigureSpec(
            "A2",
            "A2_residual_diagnostics",
            "appendix",
            "Appendix Fig. A2 candidate",
            ("surrogate_parity_mean_predictions.csv",),
            (
                "EUIt residual distribution.",
                "EG residual distribution.",
                "H residual distribution.",
            ),
            "Residual diagnostics describe analytic-target surrogate error only.",
            _build_appendix_a2,
        ),
        FigureSpec(
            "A3",
            "A3_scale_study",
            "appendix",
            "Appendix Fig. A3 candidate",
            ("scale_study.csv",),
            (
                "Mean target nMAE across dataset scales.",
                "Mean tail nMAE across dataset scales.",
                "Mean R² across dataset scales.",
                "Selection objective across dataset scales.",
            ),
            "Scale-study rows support the surrogate-selection rationale only.",
            _build_appendix_a3,
        ),
        FigureSpec(
            "B1",
            "B1_seed_diagnostics",
            "appendix",
            "Appendix Fig. B1 candidate",
            ("ddpg_seed_diagnostics.csv",),
            (
                "Best reward by scenario.",
                "Final reward by scenario.",
                "Plateau episode by scenario.",
                "Best-to-final regression ratio by scenario.",
            ),
            "Seed diagnostics are appendix-only training evidence.",
            _build_appendix_b1,
        ),
        FigureSpec(
            "B2",
            "B2_morphology_signatures",
            "appendix",
            "Appendix Fig. B2 candidate",
            ("morphology_signatures.csv",),
            ("Median morphology descriptor signatures for representative retained-output groups.",),
            "Descriptor signatures are descriptive summaries, not stable design rules.",
            _build_appendix_b2,
        ),
        FigureSpec(
            "B3",
            "B3_hv_ceiling_diagnostics",
            "appendix",
            "Appendix Fig. B3 candidate",
            ("benchmark_hv_ceiling.csv",),
            (
                "HV fraction of the theoretical ceiling.",
                "Clipped-utopia fraction.",
                "Unique objective tuple count.",
                "Unique non-dominated tuple count.",
            ),
            "HV ceiling panels explain saturation and duplicate collapse only.",
            _build_appendix_b3,
        ),
        FigureSpec(
            "B4",
            "B4_optimizer_linked_gap_decomposition",
            "appendix",
            "Appendix Fig. B4 candidate",
            ("optimizer_linked_physical_gaps.csv",),
            (
                "EUIt projection and cross-model gap decomposition for optimizer-linked cases.",
                "EG projection and cross-model gap decomposition for optimizer-linked cases.",
                "H projection and cross-model gap decomposition for optimizer-linked cases.",
            ),
            "Optimizer-linked cases remain appendix-only bridge diagnostics.",
            _build_appendix_b4,
        ),
        FigureSpec(
            "B5",
            "B5_nonlinear_response_profiles",
            "appendix",
            "Appendix Fig. B5 candidate",
            ("scale_study.csv",),
            (
                "OSR → EUIt surrogate response profile.",
                "FAR → EG surrogate response profile.",
                "SVF → H surrogate response profile.",
                "θ → H surrogate response profile.",
            ),
            "Selected surrogate response profiles illustrate local trends only.",
            lambda frames, manifest: _build_nonlinear_response_figure(repo_root, roots),
        ),
        FigureSpec(
            "B6",
            "B6_climate_case_detail",
            "appendix",
            "Appendix Fig. B6 candidate",
            ("climate_case_results.csv",),
            (
                "Per-block ΔEUIt heatmap across Beijing, Guangzhou, and Harbin.",
                "Per-block ΔEG heatmap across Beijing, Guangzhou, and Harbin.",
                "Per-block ΔH heatmap across Beijing, Guangzhou, and Harbin.",
            ),
            "Case-level climate details remain limited to four blocks and three additional climates.",
            _build_appendix_b6,
        ),
    )


def _units_from_labels(*labels: str) -> list[str]:
    return [label for label in labels if "(" in label and ")" in label]


def _metadata_path(base_path: Path) -> Path:
    return base_path.with_suffix(".metadata.json")


def _run_command(command: list[str], cwd: Path) -> str:
    completed = subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return completed.stdout


def _render_pdf_png(pdf_path: Path, render_dir: Path) -> Path:
    render_dir.mkdir(parents=True, exist_ok=True)
    prefix = render_dir / pdf_path.stem
    png_path = prefix.with_suffix(".png")
    if png_path.exists():
        png_path.unlink()
    subprocess.run(
        ["pdftoppm", "-singlefile", "-png", "-r", "180", str(pdf_path), str(prefix)],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return png_path


def _extract_pdf_text(pdf_path: Path, render_dir: Path) -> str:
    txt_path = render_dir / f"{pdf_path.stem}.txt"
    subprocess.run(["pdftotext", str(pdf_path), str(txt_path)], check=True, capture_output=True, text=True, encoding="utf-8", errors="replace")
    return txt_path.read_text(encoding="utf-8", errors="replace")


def _visual_qa(figures: list[dict[str, Any]], *, repo_root: Path, render_dir: Path, strict: bool = False) -> dict[str, Any]:
    qa_rows = []
    for item in figures:
        pdf_path = Path(item["outputs"]["pdf"])
        png_path = _render_pdf_png(pdf_path, render_dir)
        pdfinfo = _run_command(["pdfinfo", str(pdf_path)], repo_root)
        fonts = _run_command(["pdffonts", str(pdf_path)], repo_root)
        text = _extract_pdf_text(pdf_path, render_dir)
        image = Image.open(png_path)
        type3 = "Type 3" in fonts
        forbidden_hits = [token for token in FORBIDDEN_TEXT if token in text]
        qa = {
            "figure_id": item["figure_id"],
            "pdf_path": _relative_path(pdf_path, repo_root),
            "render_png": _relative_path(png_path, repo_root),
            "pdfinfo": pdfinfo,
            "pdffonts": fonts,
            "width_px": image.width,
            "height_px": image.height,
            "type3_fonts": type3,
            "forbidden_text_hits": forbidden_hits,
            "empty_render": image.width == 0 or image.height == 0,
            "unresolved_visual_concerns": [],
        }
        if type3:
            qa["unresolved_visual_concerns"].append("Type 3 font detected")
        if forbidden_hits:
            qa["unresolved_visual_concerns"].append("Forbidden text detected in rendered PDF text")
        qa_rows.append(qa)
        metadata_path = Path(item["metadata_path"])
        payload = _read_json(metadata_path)
        payload["visual_qa"] = qa
        _json_dump(payload, metadata_path)
    if strict:
        failures = [row for row in qa_rows if row["type3_fonts"] or row["forbidden_text_hits"]]
        if failures:
            raise RuntimeError("visual QA failed for " + ", ".join(row["figure_id"] for row in failures))
    return {"generated_at": _utc_now(), "figures": qa_rows}


def _write_gallery(figures: list[dict[str, Any]], qa: dict[str, Any], *, repo_root: Path, output_dir: Path) -> dict[str, str]:
    snapshot_dir = repo_root / "paper" / "snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    qa_map = {entry["figure_id"]: entry for entry in qa["figures"]}
    md_lines = ["# Round 2 Figure Gallery", ""]
    pdf_path = snapshot_dir / "round2-figure-gallery.pdf"
    with PdfPages(pdf_path) as pdf:
        for figure in figures:
            qa_entry = qa_map[figure["figure_id"]]
            metadata = _read_json(Path(figure["metadata_path"]))
            source_lines = ", ".join(f"`{source['path']}` ({source['sha256']})" for source in metadata["source_files"])
            md_lines.extend(
                [
                    f"## {figure['figure_id']} {figure['semantic_name']}",
                    f"- Planned manuscript location: {metadata['planned_location']}",
                    f"- Source files: {source_lines}",
                    f"- Panel descriptions: {'; '.join(metadata['panel_descriptions'])}",
                    f"- Claim boundary: {metadata['claim_boundary']}",
                    f"- Unresolved visual concerns: {', '.join(qa_entry['unresolved_visual_concerns']) if qa_entry['unresolved_visual_concerns'] else 'None'}",
                    "",
                ]
            )
            gallery_fig = plt.figure(figsize=GALLERY_PAGE_IN)
            grid = gallery_fig.add_gridspec(2, 1, height_ratios=[4.2, 1.6])
            image_ax = gallery_fig.add_subplot(grid[0, 0])
            image_ax.axis("off")
            rendered = Image.open(repo_root / qa_entry["render_png"])
            image_ax.imshow(rendered)
            text_ax = gallery_fig.add_subplot(grid[1, 0])
            text_ax.axis("off")
            source_paths = ", ".join(source["path"] for source in metadata["source_files"])
            source_hashes = ", ".join(source["sha256"][:12] + "…" for source in metadata["source_files"])
            text = "\n".join(
                [
                    f"{figure['figure_id']} — {figure['semantic_name']}",
                    f"Planned location: {metadata['planned_location']}",
                    f"Source files: {source_paths}",
                    f"Source SHA: {source_hashes}",
                    f"Claim boundary: {metadata['claim_boundary']}",
                    f"Unresolved concerns: {', '.join(qa_entry['unresolved_visual_concerns']) if qa_entry['unresolved_visual_concerns'] else 'None'}",
                ]
            )
            text_ax.text(0.01, 0.98, text, va="top", ha="left", fontsize=8, family="serif")
            pdf.savefig(gallery_fig)
            plt.close(gallery_fig)
    md_path = snapshot_dir / "round2-figure-gallery.md"
    md_path.write_text("\n".join(md_lines), encoding="utf-8")
    return {"pdf": str(pdf_path), "md": str(md_path)}


def _write_visio_docs(repo_root: Path) -> dict[str, str]:
    research_root = repo_root / "research" / "reviewer-round-02"
    research_root.mkdir(parents=True, exist_ok=True)
    fig1_text = _read_text_extract(research_root / "fig1_text_extract.txt")
    fig2_text = _read_text_extract(research_root / "fig2_text_extract.txt")
    fig3_text = _read_text_extract(research_root / "fig3_text_extract.txt")
    spec_lines = [
        "# Visio Figure 1–3 Revision Spec",
        "",
        "## Fig. 1",
        "- Restructure the figure into six blocks: feasible morphology generation, analytic response generation, DNN surrogate, descriptor-space optimization, feasible morphology projection, and physics-based stress testing.",
        "- Replace any direct EnergyPlus/Radiance-to-2000-sample implication with `analytic response generator`.",
        "- Replace the reward formula block with:",
        "  z_i = (y_i - y_i^{min}) / (y_i^{max} - y_i^{min})",
        "  u = (0,1,1)",
        "  d_w = || w ⊙ (z-u) ||_2 / ||w||_2",
        "  R = 1 - d_w",
        "- Label `w` as `axis-scaling coefficients`.",
        f"- Current OCR extract still contains: `{', '.join(token for token in ['street loactions', 'Acotr netword', 'Critic netword'] if token in fig1_text)}`.",
        "",
        "## Fig. 2",
        "- Reframe the title and flow as `serialized static black-box search`.",
        "- State explicitly: one episode = 40 sequential surrogate queries, reset = random action at episode start, termination = fixed 40-step horizon, and no physical time evolution.",
        "- Action must be drawn as a 12-dimensional absolute normalized descriptor vector, not as an incremental perturbation of the previous morphology.",
        f"- Existing text extract availability: {'present' if fig2_text else 'missing'}; use the task prompt as the authoritative rewrite spec.",
        "",
        "## Fig. 3",
        "- Replace network labels with: Actor network, Critic network, Target actor network, Target critic network.",
        "- Define actor input/output, critic input, replay tuple, and soft update exactly as in the task brief.",
        "- Add formula symbol definitions for Q, μ, γ, τ, N, θ, and θ′.",
        f"- Current OCR extract still contains legacy wording such as: `{('Target actor network' if 'Target actor network' in fig3_text else 'manual label check required')}`.",
    ]
    spec_path = research_root / "visio-figure-1-3-spec.md"
    spec_path.write_text("\n".join(spec_lines), encoding="utf-8")

    labels_csv = pd.DataFrame(
        [
            ["Fig1", "text", "street loactions", "street locations", "Correct typo or remove the phrase entirely."],
            ["Fig1", "text", "Acotr netword", "Actor network", "Actor label must be corrected."],
            ["Fig1", "text", "Critic netword", "Critic network", "Critic label must be corrected."],
            ["Fig1", "formula", "R = 10^6 - dweighted", "R = 1 - d_w", "Replace the obsolete reward offset."],
            ["Fig2", "flow", "implicit physical-time control", "serialized static black-box search", "Clarify no physical time evolution."],
            ["Fig3", "text", "Target actor network", "Target actor network", "Keep exact wording and define color/arrow semantics."],
            ["Fig3", "equation", "soft update legacy form", "θ′ ← τθ + (1-τ)θ′", "Use the canonical soft-update expression."],
        ],
        columns=["figure_id", "element_type", "original_text", "replacement_text", "note"],
    )
    labels_path = research_root / "visio-figure-1-3-labels.csv"
    labels_csv.to_csv(labels_path, index=False)
    return {"spec_md": str(spec_path), "labels_csv": str(labels_path)}


def _write_plan_docs(repo_root: Path, figures: list[dict[str, Any]]) -> dict[str, str]:
    research_root = repo_root / "research" / "reviewer-round-02"
    plan_lines = ["# Round 2 Figure Plan", ""]
    caption_lines = ["# Round 2 Caption Drafts", ""]
    table_lines = [
        "# Round 2 Table Plan",
        "",
        "1. Morphology descriptor definitions",
        "2. Evaluation-mode comparison",
        "3. Surrogate robustness summary",
        "4. Optimizer budget and output contract",
        "5. Canonical equal-size benchmark",
        "6. Physical cross-model stress metrics",
        "7. Climate sensitivity summary",
    ]
    for figure in figures:
        metadata = _read_json(Path(figure["metadata_path"]))
        source_paths = ", ".join(source["path"] for source in metadata["source_files"])
        plan_lines.extend(
            [
                f"## {figure['figure_id']} {figure['semantic_name']}",
                f"- Planned manuscript location: {metadata['planned_location']}",
                f"- Source files: {source_paths}",
                f"- Claim boundary: {metadata['claim_boundary']}",
                "",
            ]
        )
        caption_lines.extend(
            [
                f"## {figure['figure_id']} {figure['semantic_name']}",
                f"This candidate figure summarizes {'; '.join(metadata['panel_descriptions'])}. It uses {source_paths} and should be interpreted within the following boundary: {metadata['claim_boundary']}.",
                "",
            ]
        )
    plan_path = research_root / "round2-figure-plan.md"
    caption_path = research_root / "round2-caption-drafts.md"
    table_path = research_root / "round2-table-plan.md"
    plan_path.write_text("\n".join(plan_lines), encoding="utf-8")
    caption_path.write_text("\n".join(caption_lines), encoding="utf-8")
    table_path.write_text("\n".join(table_lines), encoding="utf-8")
    return {"plan": str(plan_path), "captions": str(caption_path), "tables": str(table_path)}


def build_round2_revision_figures(
    data_root: str | Path,
    output_dir: str | Path,
    *,
    build_gallery: bool = False,
    check_only: bool = False,
    formats: tuple[str, ...] = ("pdf", "png"),
    dpi: int = 600,
    overwrite: bool = False,
    strict: bool = False,
    repo_root: str | Path | None = None,
) -> dict[str, Any]:
    root = _repo_root(repo_root)
    data_root_path = (root / Path(data_root)).resolve()
    output_root = (root / Path(output_dir)).resolve()
    manifest = build_round2_figure_data_package(data_root_path, repo_root=root, strict=strict)
    manifest_path = data_root_path / "manifest.json"
    package_manifest = _load_package_manifest(data_root_path)
    frames = _load_package_frames(data_root_path, package_manifest, strict=strict)
    roots = _resolve_source_roots(root)
    specs = _figure_specs(root, roots)

    if check_only:
        return {
            "status": "checked",
            "manifest": str(manifest_path),
            "canonical_reference_hash": package_manifest["canonical_reference_hash"],
            "data_root": str(data_root_path),
        }

    _set_publication_style()
    built_figures = []
    for spec in specs:
        category_dir = output_root / spec.category
        category_dir.mkdir(parents=True, exist_ok=True)
        base_path = category_dir / spec.semantic_name
        if not overwrite and base_path.with_suffix(".pdf").exists():
            pass
        figure, extra = spec.builder(frames, package_manifest)
        outputs = _save_figure_outputs(figure, base_path, formats=formats, dpi=dpi)
        source_entries = [_package_entry(package_manifest, name) for name in spec.source_files]
        metadata = {
            "figure_id": spec.figure_id,
            "semantic_name": spec.semantic_name,
            "planned_location": spec.planned_location,
            "source_files": [
                {"path": entry["file_name"], "sha256": entry["sha256"], "reference_protocol": entry["reference_protocol"], "reference_hash": entry["reference_hash"]}
                for entry in source_entries
            ],
            "filters": {},
            "reference_protocol": next((entry["reference_protocol"] for entry in source_entries if entry["reference_protocol"]), ""),
            "reference_hash": next((entry["reference_hash"] for entry in source_entries if entry["reference_hash"]), ""),
            "script_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip(),
            "timestamp": _utc_now(),
            "units": [label for label in TARGET_LABELS.values()],
            "panel_descriptions": list(spec.panel_descriptions),
            "claim_boundary": spec.claim_boundary,
            "extra": extra,
        }
        metadata_path = _metadata_path(base_path)
        _json_dump(metadata, metadata_path)
        built_figures.append(
            {
                "figure_id": spec.figure_id,
                "semantic_name": spec.semantic_name,
                "category": spec.category,
                "outputs": outputs,
                "metadata_path": str(metadata_path),
            }
        )

    render_dir = root / "paper" / "manuscript" / "build" / "round2_candidate_render"
    qa = _visual_qa(built_figures, repo_root=root, render_dir=render_dir, strict=strict)
    qa_path = output_root / "visual_qa_summary.json"
    _json_dump(qa, qa_path)
    gallery = _write_gallery(built_figures, qa, repo_root=root, output_dir=output_root) if build_gallery else {}
    visio_docs = _write_visio_docs(root)
    plan_docs = _write_plan_docs(root, built_figures)
    return {
        "status": "built",
        "manifest": str(manifest_path),
        "manifest_sha256": _sha256_path(manifest_path),
        "canonical_reference_hash": package_manifest["canonical_reference_hash"],
        "figures": built_figures,
        "visual_qa": str(qa_path),
        "gallery": gallery,
        "visio_docs": visio_docs,
        "plan_docs": plan_docs,
    }

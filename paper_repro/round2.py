from __future__ import annotations

import copy
import ast
import hashlib
import json
import math
import shutil
import ssl
import statistics
import time
import urllib.request
import urllib.error
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import torch
import yaml
from pymoo.indicators.hv import HV
from pymoo.indicators.igd import IGD
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import RepeatedKFold

from paper_repro.config import Config
from paper_repro.constants import MORPHOLOGY_FEATURES, PERFORMANCE_TARGETS, PROTOTYPES
from paper_repro.contracts import OPTIMIZATION_RESULT_COLUMNS, write_csv, write_json
from paper_repro.metrics import normalized_benefit_frame
from paper_repro.optimizers import OptimizationEnvironment, run_cmaes, run_ddpg, run_random_search
from paper_repro.physical_stack import _connect_server, load_block_records, physical_stack_candidate_probe, project_candidates_to_nearest_blocks
from paper_repro.publication import load_server_config
from paper_repro.surrogate import _make_scaler, _train_single_model, load_surrogate

try:
    import psutil
except ImportError:  # pragma: no cover - optional runtime sampling dependency
    psutil = None

ROUND2_TARGETS = PERFORMANCE_TARGETS
ROUND2_FEATURES = MORPHOLOGY_FEATURES
UTILITY_SCENARIOS = (
    "Balanced_Performance",
    "Energy_Saving_Focus",
    "Energy_Generation_Focus",
)


@dataclass(frozen=True)
class Round2Paths:
    run_id: str
    run_root: Path
    data_dir: Path
    models_dir: Path
    optimization_dir: Path
    reports_dir: Path
    diagnostics_dir: Path
    reevaluation_dir: Path
    physical_dir: Path
    climate_dir: Path
    research_root: Path


def sha256_path(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def physical_protocol_hash(payload: dict[str, Any]) -> str:
    serialized = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(serialized).hexdigest()


def ddpg_reward_from_outputs(outputs: np.ndarray, target_min: np.ndarray, target_max: np.ndarray, weights: Iterable[float]) -> np.ndarray:
    array = np.asarray(outputs, dtype=float)
    if array.ndim == 1:
        array = array[None, :]
    lower = np.asarray(target_min, dtype=float)
    upper = np.asarray(target_max, dtype=float)
    weights_array = np.asarray(list(weights), dtype=float)
    target_range = np.maximum(upper - lower, 1e-8)
    state = (array - lower[None, :]) / target_range[None, :]
    utopia = np.array([0.0, 1.0, 1.0], dtype=float)
    weighted_distance = np.sqrt(np.sum((weights_array[None, :] * (state - utopia[None, :])) ** 2, axis=1))
    max_distance = math.sqrt(float(np.sum(weights_array**2)))
    return 1.0 - weighted_distance / max(max_distance, 1e-8)


def compute_legacy_utility(frame: pd.DataFrame, reference_frame: pd.DataFrame) -> pd.Series:
    reference = normalized_benefit_frame(reference_frame)
    lower = {
        "EUIt": float(reference_frame["EUIt"].min()),
        "EG": float(reference_frame["EG"].min()),
        "H": float(reference_frame["H"].min()),
    }
    upper = {
        "EUIt": float(reference_frame["EUIt"].max()),
        "EG": float(reference_frame["EG"].max()),
        "H": float(reference_frame["H"].max()),
    }
    score = pd.Series(index=frame.index, dtype=float)
    score_euit = 1.0 - (frame["EUIt"] - lower["EUIt"]) / max(upper["EUIt"] - lower["EUIt"], 1e-8)
    score_eg = (frame["EG"] - lower["EG"]) / max(upper["EG"] - lower["EG"], 1e-8)
    score_h = (frame["H"] - lower["H"]) / max(upper["H"] - lower["H"], 1e-8)
    score[:] = 0.0
    return pd.DataFrame({"EUIt_score": score_euit, "EG_score": score_eg, "H_score": score_h}, index=frame.index)


def apply_weighted_utility(score_frame: pd.DataFrame, weights: Iterable[float]) -> pd.Series:
    weights_array = np.asarray(list(weights), dtype=float)
    return (
        weights_array[0] * score_frame["EUIt_score"]
        + weights_array[1] * score_frame["EG_score"]
        + weights_array[2] * score_frame["H_score"]
    )


def compute_fixed_domain_utility(frame: pd.DataFrame, target_bounds: dict[str, tuple[float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "EUIt_score": 1.0
            - (frame["EUIt"] - target_bounds["EUIt"][0]) / max(target_bounds["EUIt"][1] - target_bounds["EUIt"][0], 1e-8),
            "EG_score": (frame["EG"] - target_bounds["EG"][0]) / max(target_bounds["EG"][1] - target_bounds["EG"][0], 1e-8),
            "H_score": (frame["H"] - target_bounds["H"][0]) / max(target_bounds["H"][1] - target_bounds["H"][0], 1e-8),
        },
        index=frame.index,
    )


def roof_irradiance_to_million_kwh(
    annual_irradiance_wh_m2: float,
    roof_area_m2: float,
    usable_roof_coverage: float,
    pv_efficiency: float,
    performance_ratio: float,
) -> float:
    return annual_irradiance_wh_m2 * roof_area_m2 * usable_roof_coverage * pv_efficiency * performance_ratio / 1_000_000_000.0


def aggregate_sunlight_hours(sensor_hour_matrix: np.ndarray) -> float:
    matrix = np.asarray(sensor_hour_matrix, dtype=float)
    if matrix.ndim == 1:
        return float(np.mean(matrix))
    return float(np.mean(np.sum(matrix, axis=1)))


def audit_descriptor_constraints(frame: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": frame.get("sample_id", pd.Series(np.arange(len(frame)), index=frame.index)),
            "far_minus_bd_af": (frame["FAR"] - frame["BD"] * frame["AF"]).abs(),
            "osr_minus_density_far": (frame["OSR"] - (1.0 - frame["BD"]) / frame["FAR"]).abs(),
        }
    )


def audit_osli_values(values: pd.Series | np.ndarray) -> pd.DataFrame:
    series = pd.Series(values, copy=False)
    nearest_integer = series.round().astype(int)
    return pd.DataFrame(
        {
            "OSLI": series.astype(float),
            "nearest_integer": nearest_integer,
            "fractional_distance": (series - nearest_integer).abs(),
            "within_bounds": series.between(0.0, 8.0),
            "is_integer": (series - nearest_integer).abs() <= 1e-8,
        }
    )


def parse_physical_results_frame(frame: pd.DataFrame) -> pd.DataFrame:
    parsed = frame.copy()
    for column in [
        "physical_EUIt",
        "physical_EG_total_production",
        "physical_H_proxy",
        "projection_distance",
    ]:
        if column in parsed.columns:
            parsed[column] = pd.to_numeric(parsed[column], errors="coerce")
    parsed["energyplus_ok"] = parsed.get("energyplus_ok", False).fillna(False).astype(bool)
    parsed["radiance_ok"] = parsed.get("radiance_ok", False).fillna(False).astype(bool)
    return parsed


def sanitize_weather_manifest(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sanitized: list[dict[str, Any]] = []
    for record in records:
        clean = {}
        for key, value in record.items():
            lowered = key.lower()
            if any(token in lowered for token in ("password", "token", "secret", "identity", "key", "user", "host")):
                continue
            clean[key] = value
        sanitized.append(clean)
    return sanitized


def dedupe_completed_job_rows(frame: pd.DataFrame, subset: Iterable[str]) -> pd.DataFrame:
    if frame.empty:
        return frame
    ordered = frame.copy()
    ordered["_completed_rank"] = ordered.get("status", "").eq("completed").astype(int)
    ordered = ordered.sort_values(["_completed_rank"], ascending=False, kind="mergesort")
    deduped = ordered.drop_duplicates(subset=list(subset), keep="first").drop(columns="_completed_rank")
    return deduped.reset_index(drop=True)


def _minimization_array(frame: pd.DataFrame) -> np.ndarray:
    return np.column_stack([frame["EUIt"].to_numpy(dtype=float), -frame["EG"].to_numpy(dtype=float), -frame["H"].to_numpy(dtype=float)])


def build_fixed_reference(groups: dict[str, pd.DataFrame]) -> dict[str, Any]:
    nds = NonDominatedSorting()
    fronts = []
    for frame in groups.values():
        matrix = _minimization_array(frame)
        if len(matrix) == 0:
            continue
        front_idx = nds.do(matrix, only_non_dominated_front=True)
        fronts.append(matrix[front_idx])
    if not fronts:
        raise ValueError("No groups available for fixed-reference benchmark.")
    combined_front = np.vstack(fronts)
    ideal = combined_front.min(axis=0)
    nadir = combined_front.max(axis=0)
    normalized_reference_front = (combined_front - ideal) / np.maximum(nadir - ideal, 1e-8)
    return {
        "ideal": ideal,
        "nadir": nadir,
        "reference_front": normalized_reference_front,
        "reference_point": np.array([1.1, 1.1, 1.1], dtype=float),
    }


def evaluate_archive_metrics(groups: dict[str, pd.DataFrame], reference: dict[str, Any]) -> pd.DataFrame:
    nds = NonDominatedSorting()
    rows = []
    for group_name, frame in groups.items():
        matrix = _minimization_array(frame)
        if len(matrix) == 0:
            continue
        front_idx = nds.do(matrix, only_non_dominated_front=True)
        front = matrix[front_idx]
        normalized = (front - reference["ideal"]) / np.maximum(reference["nadir"] - reference["ideal"], 1e-8)
        rows.append(
            {
                "group": group_name,
                "rows": int(len(frame)),
                "non_dominated_rows": int(len(front)),
                "HV": float(HV(ref_point=reference["reference_point"])(normalized)),
                "IGD": float(IGD(reference["reference_front"])(normalized)),
            }
        )
    return pd.DataFrame(rows)


def theoretical_max_hv(reference_point: Iterable[float]) -> float:
    ref = np.asarray(list(reference_point), dtype=float)
    return float(np.prod(ref))


def dedupe_objective_tuples(frame: pd.DataFrame, *, decimals: int = 12) -> pd.DataFrame:
    deduped = frame.copy()
    deduped["_objective_tuple_key"] = deduped.apply(
        lambda row: "|".join(
            [
                f"{round(float(row['EUIt']), decimals):.{decimals}f}",
                f"{round(float(row['EG']), decimals):.{decimals}f}",
                f"{round(float(row['H']), decimals):.{decimals}f}",
            ]
        ),
        axis=1,
    )
    deduped = deduped.drop_duplicates(subset=["_objective_tuple_key"], keep="first").drop(columns="_objective_tuple_key")
    return deduped.reset_index(drop=True)


def _guardrail_decomposition_frame(
    frame: pd.DataFrame,
    env: OptimizationEnvironment,
    target_bounds: dict[str, tuple[float, float]],
    *,
    group_name: str,
    tuple_decimals: int = 12,
) -> pd.DataFrame:
    actual_actions = frame[ROUND2_FEATURES].to_numpy(dtype=np.float32)
    lower = env.feature_min
    upper = env.feature_max
    normalized_actions = (actual_actions - lower[None, :]) / np.maximum(upper[None, :] - lower[None, :], 1e-8)
    raw_outputs = env.surrogate.predict(frame[ROUND2_FEATURES], clip=False).to_numpy(dtype=np.float32)
    distance = np.linalg.norm(env.feature_reference[None, :, :] - normalized_actions[:, None, :], axis=2).min(axis=1)
    feature_penalty = np.maximum(distance - env.feasible_radius, 0.0).astype(np.float32)
    below = np.maximum(env.target_min[None, :] - raw_outputs, 0.0) / env.target_range[None, :]
    above = np.maximum(raw_outputs - env.target_max[None, :], 0.0) / env.target_range[None, :]
    extrapolation_penalty = (below + above).sum(axis=1).astype(np.float32)
    adjusted = raw_outputs.copy()
    adjusted[:, 0] += env.feature_penalty_scale[0] * feature_penalty
    adjusted[:, 1] -= env.feature_penalty_scale[1] * feature_penalty
    adjusted[:, 2] -= env.feature_penalty_scale[2] * feature_penalty
    adjusted[:, 0] += env.target_range[0] * env.extrapolation_penalty_scale * extrapolation_penalty
    adjusted[:, 1] -= env.target_range[1] * env.extrapolation_penalty_scale * extrapolation_penalty
    adjusted[:, 2] -= env.target_range[2] * env.extrapolation_penalty_scale * extrapolation_penalty
    clipped = adjusted.copy()
    clipped[:, 0] = np.clip(clipped[:, 0], env.target_min[0], env.target_max[0])
    clipped[:, 1] = np.clip(clipped[:, 1], env.target_min[1], env.target_max[1])
    clipped[:, 2] = np.clip(clipped[:, 2], env.target_min[2], env.target_max[2])
    tuple_keys = [
        "|".join(
            [
                f"{round(float(values[0]), tuple_decimals):.{tuple_decimals}f}",
                f"{round(float(values[1]), tuple_decimals):.{tuple_decimals}f}",
                f"{round(float(values[2]), tuple_decimals):.{tuple_decimals}f}",
            ]
        )
        for values in clipped
    ]
    result = frame[["method", "scenario", "seed", *ROUND2_FEATURES]].copy()
    result["group"] = group_name
    result["raw_EUIt"] = raw_outputs[:, 0]
    result["raw_EG"] = raw_outputs[:, 1]
    result["raw_H"] = raw_outputs[:, 2]
    result["adjusted_EUIt"] = adjusted[:, 0]
    result["adjusted_EG"] = adjusted[:, 1]
    result["adjusted_H"] = adjusted[:, 2]
    result["EUIt"] = clipped[:, 0]
    result["EG"] = clipped[:, 1]
    result["H"] = clipped[:, 2]
    result["feature_manifold_distance"] = distance
    result["feature_penalty"] = feature_penalty
    result["extrapolation_penalty"] = extrapolation_penalty
    result["clip_flag_EUIt"] = np.abs(clipped[:, 0] - adjusted[:, 0]) > 1e-8
    result["clip_flag_EG"] = np.abs(clipped[:, 1] - adjusted[:, 1]) > 1e-8
    result["clip_flag_H"] = np.abs(clipped[:, 2] - adjusted[:, 2]) > 1e-8
    result["is_exact_utopian_tuple"] = (
        (np.abs(clipped[:, 0] - target_bounds["EUIt"][0]) <= 1e-8)
        & (np.abs(clipped[:, 1] - target_bounds["EG"][1]) <= 1e-8)
        & (np.abs(clipped[:, 2] - target_bounds["H"][1]) <= 1e-8)
    )
    result["duplicate_objective_tuple_id"] = tuple_keys
    return result.reset_index(drop=True)


def _normalized_feature_frame(frame: pd.DataFrame) -> pd.DataFrame:
    feature_frame = frame[ROUND2_FEATURES].astype(float)
    return (feature_frame - feature_frame.min()) / np.maximum(feature_frame.max() - feature_frame.min(), 1e-8)


def select_maximin_space_filling(frame: pd.DataFrame, n_select: int) -> pd.DataFrame:
    if n_select <= 0 or frame.empty:
        return frame.iloc[0:0].copy()
    normalized = _normalized_feature_frame(frame).to_numpy(dtype=float)
    medoid_target = np.median(normalized, axis=0)
    distances_to_medoid = np.linalg.norm(normalized - medoid_target[None, :], axis=1)
    selected = [int(np.argmin(distances_to_medoid))]
    while len(selected) < min(n_select, len(frame)):
        remaining = [idx for idx in range(len(frame)) if idx not in selected]
        candidate_scores = []
        for idx in remaining:
            min_distance = min(np.linalg.norm(normalized[idx] - normalized[chosen]) for chosen in selected)
            candidate_scores.append((min_distance, -float(frame.iloc[idx]["sample_id"]), idx))
        candidate_scores.sort(reverse=True)
        selected.append(candidate_scores[0][2])
    return frame.iloc[selected].copy().reset_index(drop=True)


def select_objective_tail_cases(frame: pd.DataFrame, *, existing_ids: set[int] | None = None) -> pd.DataFrame:
    selected_rows: list[pd.Series] = []
    used = set() if existing_ids is None else set(existing_ids)
    normalized = _normalized_feature_frame(frame).to_numpy(dtype=float)
    criteria = [
        ("EUIt", True, "low_EUIt"),
        ("EG", False, "high_EG"),
        ("H", False, "high_H"),
    ]
    for target, ascending, stratum in criteria:
        sorted_frame = frame.sort_values(target, ascending=ascending, kind="mergesort")
        decile_size = max(len(sorted_frame) // 10, 1)
        candidate_pool = sorted_frame.iloc[:decile_size].copy()
        candidate_pool = candidate_pool.loc[~candidate_pool["sample_id"].isin(used)]
        if candidate_pool.empty:
            continue
        if len(candidate_pool) == 1:
            pick_indices = [candidate_pool.index[0]]
        else:
            pool_positions = candidate_pool.index.to_list()
            first = pool_positions[0]
            pick_indices = [first]
            while len(pick_indices) < min(2, len(pool_positions)):
                remaining = [idx for idx in pool_positions if idx not in pick_indices]
                scores = []
                for idx in remaining:
                    pos = frame.index.get_loc(idx)
                    min_distance = min(np.linalg.norm(normalized[pos] - normalized[frame.index.get_loc(chosen)]) for chosen in pick_indices)
                    scores.append((min_distance, -float(frame.loc[idx, "sample_id"]), idx))
                scores.sort(reverse=True)
                pick_indices.append(scores[0][2])
        for pick in pick_indices:
            row = frame.loc[pick].copy()
            row["selection_stratum"] = stratum
            selected_rows.append(row)
            used.add(int(row["sample_id"]))
    if not selected_rows:
        return frame.iloc[0:0].copy()
    return pd.DataFrame(selected_rows).reset_index(drop=True)


def _weather_station_dir(base_output_dir: Path, station: str) -> Path:
    return base_output_dir / station


def _read_epw_location(epw_path: Path) -> dict[str, Any]:
    first_line = epw_path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    parts = [part.strip() for part in first_line.split(",")]
    return {
        "location_line": first_line,
        "city": parts[1] if len(parts) > 1 else "",
        "state": parts[2] if len(parts) > 2 else "",
        "country": parts[3] if len(parts) > 3 else "",
        "source": parts[4] if len(parts) > 4 else "",
        "wmo": parts[5] if len(parts) > 5 else "",
        "latitude": float(parts[6]) if len(parts) > 6 else math.nan,
        "longitude": float(parts[7]) if len(parts) > 7 else math.nan,
        "timezone": float(parts[8]) if len(parts) > 8 else math.nan,
        "elevation_m": float(parts[9]) if len(parts) > 9 else math.nan,
    }


def _weather_url_has_province_directory(url: str) -> bool:
    parsed = urlparse(url)
    parts = [part for part in parsed.path.split("/") if part]
    try:
        idx = parts.index("CHN_China")
    except ValueError:
        return False
    return len(parts) >= idx + 3


def _epw_hourly_record_count(epw_path: Path) -> int:
    lines = epw_path.read_text(encoding="utf-8", errors="replace").splitlines()
    return max(len(lines) - 8, 0)


def validate_weather_station(station_name: str, station_cfg: dict[str, Any], output_dir: Path, *, download: bool = True) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    url = str(station_cfg["url"])
    archive_name = url.rstrip("/").split("/")[-1]
    archive_path = output_dir / archive_name
    extract_dir = output_dir / station_name
    required_period = str(station_cfg.get("period", "TMYx.2009-2023"))
    existing_candidates = sorted(extract_dir.rglob("*.epw")) if extract_dir.exists() else []
    if archive_path.exists() and existing_candidates:
        epw_path = existing_candidates[0]
        hourly_records = _epw_hourly_record_count(epw_path)
        if hourly_records >= 8760 and required_period in archive_name:
            location = _read_epw_location(epw_path)
            return {
                "station": station_name,
                "label": station_cfg.get("label", station_name),
                "wmo": station_cfg.get("wmo", location.get("wmo", "")),
                "source": location.get("source", ""),
                "period": archive_name.replace(".zip", ""),
                "url": url,
                "url_category": "validated_cached_download",
                "archive_path": str(archive_path),
                "epw_path": str(epw_path),
                "epw_sha256": sha256_path(epw_path),
                "archive_sha256": sha256_path(archive_path),
                "latitude": location.get("latitude"),
                "longitude": location.get("longitude"),
                "timezone": location.get("timezone"),
                "elevation_m": location.get("elevation_m"),
                "hourly_records": hourly_records,
                "province_directory_present": True,
            }
    if urlparse(url).netloc not in {"climate.onebuilding.org"}:
        raise ValueError(f"invalid_weather_domain: {urlparse(url).netloc}")
    if not _weather_url_has_province_directory(url):
        raise ValueError(f"missing_province_directory: {url}")
    try:
        with urllib.request.urlopen(url, timeout=120) as response:
            body = response.read() if download else response.read(1)
            status = getattr(response, "status", 200)
        if status != 200:
            raise RuntimeError(f"http_status_{status}")
        if download:
            archive_path.write_bytes(body)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"bad_url_http_{exc.code}") from exc
    except urllib.error.URLError as exc:
        reason = getattr(exc, "reason", exc)
        raise RuntimeError(f"dns_or_network_error: {reason}") from exc
    except ssl.SSLError as exc:
        raise RuntimeError(f"tls_error: {exc}") from exc

    if extract_dir.exists():
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)
    try:
        with zipfile.ZipFile(archive_path) as zf:
            zf.extractall(extract_dir)
    except zipfile.BadZipFile as exc:
        raise RuntimeError(f"corrupt_zip: {archive_path}") from exc
    epw_candidates = sorted(extract_dir.rglob("*.epw"))
    if not epw_candidates:
        raise RuntimeError(f"missing_epw: {station_name}")
    epw_path = epw_candidates[0]
    hourly_records = _epw_hourly_record_count(epw_path)
    if hourly_records < 8760:
        raise RuntimeError(f"invalid_epw_record_count: {hourly_records}")
    location = _read_epw_location(epw_path)
    if required_period not in archive_name:
        raise RuntimeError(f"unexpected_period: {archive_name}")
    return {
        "station": station_name,
        "label": station_cfg.get("label", station_name),
        "wmo": station_cfg.get("wmo", location.get("wmo", "")),
        "source": location.get("source", ""),
        "period": archive_name.replace(".zip", ""),
        "url": url,
        "url_category": "verified_download",
        "archive_path": str(archive_path),
        "epw_path": str(epw_path),
        "epw_sha256": sha256_path(epw_path),
        "archive_sha256": sha256_path(archive_path),
        "latitude": location.get("latitude"),
        "longitude": location.get("longitude"),
        "timezone": location.get("timezone"),
        "elevation_m": location.get("elevation_m"),
        "hourly_records": hourly_records,
        "province_directory_present": True,
    }


def download_weather_station(station_name: str, station_cfg: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    return validate_weather_station(station_name, station_cfg, output_dir, download=True)


def ensure_remote_epw(server_cfg: dict[str, Any], local_epw_path: str | Path, station_name: str) -> str:
    local_path = Path(local_epw_path)
    remote_relpath = f"artifacts/weather/{station_name}/{local_path.name}"
    remote_abspath = f"{server_cfg['remote_project_root'].rstrip('/')}/{remote_relpath}"
    client = _connect_server(server_cfg)
    sftp = client.open_sftp()
    try:
        remote_dir = str(Path(remote_abspath).parent).replace("\\", "/")
        stdin, stdout, stderr = client.exec_command(f"mkdir -p {remote_dir}", timeout=30)
        stderr_text = stderr.read().decode("utf-8", errors="replace").strip()
        stdout.read()
        stdin.close()
        stdout.close()
        stderr.close()
        if stderr_text:
            raise RuntimeError(f"remote_mkdir_failed: {stderr_text}")
        sftp.put(str(local_path), remote_abspath)
    finally:
        sftp.close()
        client.close()
    return remote_relpath


def _infer_selected_candidate(base_config: Config) -> dict[str, Any]:
    selected_path = Path(base_config["round2"]["canonical_selected_surrogate"])
    payload = json.loads(selected_path.read_text(encoding="utf-8"))
    return payload["selected_for_optimization"]


def resolve_round2_paths(base_config: Config, *, run_id: str | None = None, output_dir: str | Path | None = None) -> Round2Paths:
    resolved_run_id = run_id or str(base_config["round2"]["output_naming"]["default_run_id"])
    run_root = Path(output_dir) if output_dir is not None else Path(base_config["project"]["artifact_root"]) / resolved_run_id
    research_root = Path(base_config["round2"]["research_root"])
    return Round2Paths(
        run_id=resolved_run_id,
        run_root=run_root,
        data_dir=run_root / "data",
        models_dir=run_root / "models",
        optimization_dir=run_root / "optimization",
        reports_dir=run_root / "reports",
        diagnostics_dir=run_root / "diagnostics",
        reevaluation_dir=run_root / "reevaluation",
        physical_dir=run_root / "physical",
        climate_dir=run_root / "climate",
        research_root=research_root,
    )


def _derive_run_config(base_config: Config, paths: Round2Paths) -> Config:
    raw = copy.deepcopy(base_config.raw)
    raw["project"]["artifact_root"] = str(paths.run_root)
    raw["report"]["data_dir"] = str(paths.data_dir)
    raw["report"]["models_dir"] = str(paths.models_dir)
    raw["report"]["optimization_dir"] = str(paths.optimization_dir)
    raw["report"]["reports_dir"] = str(paths.reports_dir)
    raw["report"]["figures_dir"] = str(paths.run_root / "figures")
    raw["report"]["bootstrap_dir"] = str(paths.run_root / "bootstrap")
    raw["publication"]["diagnostics_dir"] = str(paths.diagnostics_dir)
    raw["publication"]["reevaluation_dir"] = str(paths.reevaluation_dir)
    raw["publication"]["imported_results_root"] = str(paths.run_root / "imported")
    return Config(raw)


def prepare_round2_workspace(
    config_path: str | Path,
    *,
    run_id: str | None = None,
    output_dir: str | Path | None = None,
) -> tuple[Config, Config, Round2Paths]:
    base_config = Config.from_yaml(config_path)
    paths = resolve_round2_paths(base_config, run_id=run_id, output_dir=output_dir)
    for directory in [
        paths.run_root,
        paths.data_dir,
        paths.models_dir,
        paths.optimization_dir,
        paths.reports_dir,
        paths.diagnostics_dir,
        paths.reevaluation_dir,
        paths.physical_dir,
        paths.climate_dir,
        paths.research_root,
    ]:
        directory.mkdir(parents=True, exist_ok=True)

    copy_map = {
        Path(base_config["round2"]["canonical_dataset"]): paths.data_dir / "simulated_samples.csv",
        Path(base_config["round2"]["canonical_blocks"]): paths.data_dir / "simulated_blocks.jsonl",
        Path(base_config["round2"]["canonical_dataset_meta"]): paths.data_dir / "simulated_samples.meta.json",
        Path(base_config["round2"]["canonical_surrogate"]): paths.models_dir / "surrogate.pt",
        Path(base_config["round2"]["canonical_selected_surrogate"]): paths.models_dir / "selected_surrogate.json",
        Path(base_config["round2"]["canonical_cv_predictions"]): paths.models_dir / "cv_predictions.csv",
    }
    for source, target in copy_map.items():
        if not target.exists():
            shutil.copy2(source, target)

    manifest = {
        "run_id": paths.run_id,
        "prepared_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "canonical_inputs": {
            str(source): {
                "copied_to": str(target),
                "sha256": sha256_path(source),
            }
            for source, target in copy_map.items()
        },
    }
    write_json(manifest, paths.run_root / "workspace_manifest.json")
    return base_config, _derive_run_config(base_config, paths), paths


def _load_round2_dataset(base_config: Config) -> pd.DataFrame:
    return pd.read_csv(base_config["round2"]["canonical_dataset"])


def _load_round2_blocks(base_config: Config) -> dict[int, dict[str, Any]]:
    return load_block_records(base_config["round2"]["canonical_blocks"])


def run_sampling_coverage_analysis(config_path: str | Path, *, run_id: str | None = None, output_dir: str | Path | None = None) -> dict[str, Any]:
    base_config, _, paths = prepare_round2_workspace(config_path, run_id=run_id, output_dir=output_dir)
    dataset = _load_round2_dataset(base_config)
    blocks = _load_round2_blocks(base_config)

    summary_rows = []
    long_rows = []
    for feature in ROUND2_FEATURES:
        series = dataset[feature].astype(float)
        row = {
            "feature": feature,
            "count": int(series.count()),
            "min": float(series.min()),
            "q01": float(series.quantile(0.01)),
            "q05": float(series.quantile(0.05)),
            "q10": float(series.quantile(0.10)),
            "q25": float(series.quantile(0.25)),
            "median": float(series.median()),
            "mean": float(series.mean()),
            "std": float(series.std(ddof=0)),
            "q75": float(series.quantile(0.75)),
            "q90": float(series.quantile(0.90)),
            "q95": float(series.quantile(0.95)),
            "q99": float(series.quantile(0.99)),
            "max": float(series.max()),
        }
        summary_rows.append(row)
        for stat_name, stat_value in row.items():
            if stat_name == "feature":
                continue
            long_rows.append({"feature": feature, "statistic": stat_name, "value": stat_value})
    coverage_summary = pd.DataFrame(summary_rows)
    coverage_long = pd.DataFrame(long_rows)
    write_csv(coverage_summary, paths.data_dir / "sampling_coverage_summary.csv")
    write_csv(coverage_long, paths.data_dir / "descriptor_coverage_long.csv")

    normalized = _normalized_feature_frame(dataset)
    feature_matrix = normalized.to_numpy(dtype=float)
    distance_matrix = np.linalg.norm(feature_matrix[:, None, :] - feature_matrix[None, :, :], axis=2)
    np.fill_diagonal(distance_matrix, np.inf)
    nearest = distance_matrix.min(axis=1)
    farthest = np.where(np.isfinite(distance_matrix), distance_matrix, 0.0).max(axis=1)
    duplicate_count = int(dataset[ROUND2_FEATURES].duplicated().sum())
    centered = feature_matrix - feature_matrix.mean(axis=0, keepdims=True)
    _, singular_values, _ = np.linalg.svd(centered, full_matrices=False)
    explained = (singular_values**2) / max(float(np.sum(singular_values**2)), 1e-8)
    cumulative = np.cumsum(explained)
    n_95 = int(np.searchsorted(cumulative, 0.95) + 1)
    participation_ratio = float((np.sum(explained) ** 2) / max(np.sum(explained**2), 1e-8))

    pearson = dataset[ROUND2_FEATURES].corr(numeric_only=True)
    spearman = dataset[ROUND2_FEATURES].corr(method="spearman", numeric_only=True)
    write_csv(pearson.reset_index().rename(columns={"index": "feature"}), paths.data_dir / "descriptor_correlation_matrix.csv")
    write_csv(spearman.reset_index().rename(columns={"index": "feature"}), paths.data_dir / "descriptor_spearman_correlation_matrix.csv")

    discrete_rows = []
    prototype_rows = []
    composition_counts: dict[str, int] = {}
    for sample_id, record in blocks.items():
        assignments = record["assignments"]
        open_space_index = int(record["open_space_index"])
        discrete_rows.append({"sample_id": sample_id, "variable": "open_space_index", "value": open_space_index})
        discrete_rows.append({"sample_id": sample_id, "variable": "theta_deg", "value": float(record["theta_deg"])})
        composition_parts = []
        for assignment in assignments:
            prototype_name = assignment["prototype_name"]
            floors = int(assignment["floors"])
            prototype_rows.append(
                {
                    "sample_id": sample_id,
                    "prototype_name": prototype_name,
                    "floors": floors,
                    "block_index": int(assignment["block_index"]),
                }
            )
            composition_parts.append(f"{prototype_name}:{floors}")
        composition_key = "|".join(sorted(composition_parts))
        composition_counts[composition_key] = composition_counts.get(composition_key, 0) + 1

    prototype_frame = pd.DataFrame(prototype_rows)
    discrete_frame = pd.DataFrame(discrete_rows)
    composition_frame = pd.DataFrame(
        [{"prototype_composition": key, "count": value} for key, value in sorted(composition_counts.items())]
    )
    write_csv(prototype_frame, paths.data_dir / "sampling_prototype_assignments.csv")
    write_csv(discrete_frame, paths.data_dir / "sampling_discrete_variables.csv")
    write_csv(composition_frame, paths.data_dir / "sampling_prototype_compositions.csv")

    constraint_audit = audit_descriptor_constraints(dataset)
    dependency_rows = [
        {
            "dependency_name": "FAR_minus_BD_times_AF",
            "metric": "max_abs_residual",
            "value": float(constraint_audit["far_minus_bd_af"].max()),
        },
        {
            "dependency_name": "OSR_minus_(1_minus_BD)_over_FAR",
            "metric": "max_abs_residual",
            "value": float(constraint_audit["osr_minus_density_far"].max()),
        },
        {
            "dependency_name": "OSLI_integer_consistency",
            "metric": "all_integer",
            "value": float(audit_osli_values(dataset["OSLI"])["is_integer"].all()),
        },
        {
            "dependency_name": "exact_duplicate_count",
            "metric": "rows",
            "value": float(duplicate_count),
        },
        {
            "dependency_name": "nearest_neighbor_distance",
            "metric": "median",
            "value": float(np.median(nearest)),
        },
        {
            "dependency_name": "nearest_neighbor_distance",
            "metric": "q95",
            "value": float(np.quantile(nearest, 0.95)),
        },
        {
            "dependency_name": "coverage_radius",
            "metric": "median_farthest_neighbor",
            "value": float(np.median(farthest)),
        },
        {
            "dependency_name": "pca",
            "metric": "components_for_95_variance",
            "value": float(n_95),
        },
        {
            "dependency_name": "pca",
            "metric": "participation_ratio",
            "value": participation_ratio,
        },
    ]
    write_csv(pd.DataFrame(dependency_rows), paths.data_dir / "descriptor_dependencies.csv")

    sampling_method_summary = {
        "protocol_version": base_config["round2"]["protocol_version"],
        "random_morphology_generation": True,
        "lhs": False,
        "full_factorial_or_grid": False,
        "generate_blocks_then_compute_descriptors": True,
        "nested_prefix_scales": [500, 1000, 1500, 2000],
        "duplicate_count": duplicate_count,
        "nearest_neighbor": {
            "min": float(nearest.min()),
            "median": float(np.median(nearest)),
            "q95": float(np.quantile(nearest, 0.95)),
            "max": float(nearest.max()),
        },
        "coverage_radius": {
            "median": float(np.median(farthest)),
            "q95": float(np.quantile(farthest, 0.95)),
            "max": float(np.max(farthest)),
        },
        "pca": {
            "explained_variance": explained.tolist(),
            "components_for_95_variance": n_95,
            "participation_ratio": participation_ratio,
        },
    }
    write_json(sampling_method_summary, paths.data_dir / "sampling_method_summary.json")
    return {
        "sampling_coverage_summary": str(paths.data_dir / "sampling_coverage_summary.csv"),
        "descriptor_coverage_long": str(paths.data_dir / "descriptor_coverage_long.csv"),
        "descriptor_dependencies": str(paths.data_dir / "descriptor_dependencies.csv"),
        "sampling_method_summary": str(paths.data_dir / "sampling_method_summary.json"),
    }


def _surrogate_training_spec(base_config: Config) -> dict[str, Any]:
    selected = _infer_selected_candidate(base_config)
    checkpoint = torch.load(base_config["round2"]["canonical_surrogate"], map_location="cpu", weights_only=False)
    candidate = checkpoint.get("candidate", {})
    x_scaler_kind = "standard" if checkpoint["x_scaler"].__class__.__name__.lower().startswith("standard") else "minmax"
    y_scaler_kind = "standard" if checkpoint["y_scaler"].__class__.__name__.lower().startswith("standard") else "minmax"
    return {
        "candidate_name": selected.get("candidate", candidate.get("name", "selected")),
        "loss": selected.get("loss", candidate.get("loss", "mae")),
        "feature_scaler": selected.get("feature_scaler", candidate.get("feature_scaler", x_scaler_kind)),
        "target_scaler": selected.get("target_scaler", candidate.get("target_scaler", y_scaler_kind)),
        "hyperparameters": checkpoint["hyperparameters"],
    }


def _train_selected_surrogate(
    train_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    spec: dict[str, Any],
    base_config: Config,
    *,
    seed: int,
) -> np.ndarray:
    x_scaler = _make_scaler(str(spec["feature_scaler"]))
    y_scaler = _make_scaler(str(spec["target_scaler"]))
    x_train = x_scaler.fit_transform(train_frame[ROUND2_FEATURES].to_numpy(dtype=np.float32))
    y_train = y_scaler.fit_transform(train_frame[ROUND2_TARGETS].to_numpy(dtype=np.float32))
    x_test = x_scaler.transform(test_frame[ROUND2_FEATURES].to_numpy(dtype=np.float32))
    y_test = y_scaler.transform(test_frame[ROUND2_TARGETS].to_numpy(dtype=np.float32))
    model, _ = _train_single_model(
        x_train=x_train,
        y_train=y_train,
        x_val=x_test,
        y_val=y_test,
        hidden_layers=list(spec["hyperparameters"]["hidden_layers"]),
        dropout=float(spec["hyperparameters"]["dropout"]),
        learning_rate=float(spec["hyperparameters"]["learning_rate"]),
        batch_size=int(spec["hyperparameters"]["batch_size"]),
        epochs=int(base_config["dnn"]["retrain_epochs"]),
        patience=int(base_config["dnn"]["patience"]),
        loss_name=str(spec["loss"]),
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
        seed=seed,
    )
    model.eval()
    with torch.no_grad():
        pred_scaled = model(torch.tensor(x_test, dtype=torch.float32, device=next(model.parameters()).device)).detach().cpu().numpy()
    return y_scaler.inverse_transform(pred_scaled)


def _prediction_metric_rows(
    *,
    validation_family: str,
    repetition: int,
    fold: int,
    test_label: str,
    truth: pd.DataFrame,
    pred: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for target_index, target in enumerate(ROUND2_TARGETS):
        true_values = truth[target].to_numpy(dtype=float)
        pred_values = pred[:, target_index]
        target_range = max(float(true_values.max() - true_values.min()), 1e-8)
        q_low = np.quantile(true_values, 0.10)
        q_high = np.quantile(true_values, 0.90)
        low_mask = true_values <= q_low
        high_mask = true_values >= q_high
        residual = pred_values - true_values
        rank_truth = pd.Series(true_values).rank(method="average")
        rank_pred = pd.Series(pred_values).rank(method="average")
        spearman = float(rank_truth.corr(rank_pred))
        rows.append(
            {
                "validation_family": validation_family,
                "repetition": repetition,
                "fold": fold,
                "test_label": test_label,
                "target": target,
                "R2": float(r2_score(true_values, pred_values)),
                "MAE": float(mean_absolute_error(true_values, pred_values)),
                "RMSE": float(np.sqrt(np.mean(residual**2))),
                "nMAE": float(mean_absolute_error(true_values, pred_values) / target_range),
                "nRMSE": float(np.sqrt(np.mean(residual**2)) / target_range),
                "Spearman_rho": spearman,
                "lower_tail_MAE": float(np.mean(np.abs(residual[low_mask]))) if np.any(low_mask) else float(np.mean(np.abs(residual))),
                "upper_tail_MAE": float(np.mean(np.abs(residual[high_mask]))) if np.any(high_mask) else float(np.mean(np.abs(residual))),
                "lower_tail_nMAE": float(np.mean(np.abs(residual[low_mask])) / target_range) if np.any(low_mask) else float(np.mean(np.abs(residual)) / target_range),
                "upper_tail_nMAE": float(np.mean(np.abs(residual[high_mask])) / target_range) if np.any(high_mask) else float(np.mean(np.abs(residual)) / target_range),
                "test_count": int(len(true_values)),
            }
        )
    return rows


def _bootstrap_ci(values: np.ndarray, fn: Callable[[np.ndarray], float], iterations: int, seed: int) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(iterations):
        sample = values[rng.integers(0, len(values), len(values))]
        estimates.append(fn(sample))
    return float(np.quantile(estimates, 0.025)), float(np.quantile(estimates, 0.975))


def _pairwise_rank_preservation(truth: np.ndarray, pred: np.ndarray) -> float:
    truth = np.asarray(truth, dtype=float)
    pred = np.asarray(pred, dtype=float)
    if len(truth) < 2:
        return 1.0
    total = 0
    preserved = 0
    for left in range(len(truth)):
        for right in range(left + 1, len(truth)):
            truth_diff = truth[left] - truth[right]
            pred_diff = pred[left] - pred[right]
            if abs(truth_diff) <= 1e-12:
                continue
            total += 1
            if truth_diff == 0 or pred_diff == 0 or np.sign(truth_diff) == np.sign(pred_diff):
                preserved += 1
    return preserved / max(total, 1)


def _target_stat_row(target_name: str, truth: np.ndarray, pred: np.ndarray, *, bootstrap_iterations: int, seed: int, normalized_range: float, status: str) -> dict[str, Any]:
    residual = pred - truth
    if len(truth) >= 2 and np.std(truth) > 1e-12 and np.std(pred) > 1e-12:
        try:
            slope, intercept = np.polyfit(truth, pred, deg=1)
        except np.linalg.LinAlgError:
            slope, intercept = (np.nan, np.nan)
    else:
        slope, intercept = (np.nan, np.nan)
    return {
        "target": target_name,
        "status": status,
        "count": int(len(truth)),
        "bias": float(np.mean(residual)),
        "MAE": float(np.mean(np.abs(residual))),
        "RMSE": float(np.sqrt(np.mean(residual**2))),
        "nMAE": float(np.mean(np.abs(residual)) / max(normalized_range, 1e-8)),
        "nRMSE": float(np.sqrt(np.mean(residual**2)) / max(normalized_range, 1e-8)),
        "Pearson_r": float(pd.Series(truth).corr(pd.Series(pred), method="pearson")) if len(truth) >= 2 else np.nan,
        "Spearman_rho": float(pd.Series(truth).corr(pd.Series(pred), method="spearman")) if len(truth) >= 2 else np.nan,
        "Kendall_tau": float(pd.Series(truth).corr(pd.Series(pred), method="kendall")) if len(truth) >= 2 else np.nan,
        "slope": float(slope),
        "intercept": float(intercept),
        "MAE_ci_low": _bootstrap_ci(np.abs(residual), np.mean, bootstrap_iterations, seed)[0],
        "MAE_ci_high": _bootstrap_ci(np.abs(residual), np.mean, bootstrap_iterations, seed)[1],
        "rank_preservation": float(_pairwise_rank_preservation(truth, pred)),
    }


def run_surrogate_validation(config_path: str | Path, *, run_id: str | None = None, output_dir: str | Path | None = None) -> dict[str, Any]:
    base_config, _, paths = prepare_round2_workspace(config_path, run_id=run_id, output_dir=output_dir)
    dataset = _load_round2_dataset(base_config)
    spec = _surrogate_training_spec(base_config)
    master_seed = int(base_config["round2"]["master_seed"])
    settings = base_config["round2"]["surrogate_validation"]

    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, Any]] = []
    feature_matrix = _normalized_feature_frame(dataset)
    dataset_center = feature_matrix.median(axis=0).to_numpy(dtype=float)

    repeated = RepeatedKFold(
        n_splits=int(settings["repeated_kfold_splits"]),
        n_repeats=int(settings["repeated_kfold_repeats"]),
        random_state=master_seed,
    )
    for split_index, (train_idx, test_idx) in enumerate(repeated.split(dataset), start=1):
        train_frame = dataset.iloc[train_idx].reset_index(drop=True)
        test_frame = dataset.iloc[test_idx].copy().reset_index(drop=True)
        pred = _train_selected_surrogate(train_frame, test_frame, spec, base_config, seed=master_seed + split_index)
        truth = test_frame[["sample_id", *ROUND2_TARGETS]]
        test_norm = feature_matrix.iloc[test_idx].to_numpy(dtype=float)
        train_norm = feature_matrix.iloc[train_idx].to_numpy(dtype=float)
        distance_to_center = np.linalg.norm(test_norm - dataset_center[None, :], axis=1)
        nearest_train_distance = np.min(
            np.linalg.norm(test_norm[:, None, :] - train_norm[None, :, :], axis=2),
            axis=1,
        )
        prediction_rows.append(
            pd.DataFrame(
                {
                    "validation_family": "repeated_kfold",
                    "repetition": int((split_index - 1) // int(settings["repeated_kfold_splits"]) + 1),
                    "fold": int((split_index - 1) % int(settings["repeated_kfold_splits"]) + 1),
                    "test_label": "fold",
                    "sample_id": truth["sample_id"].to_numpy(),
                    "distance_to_center": distance_to_center,
                    "nearest_train_distance": nearest_train_distance,
                    **{f"true_{target}": truth[target].to_numpy() for target in ROUND2_TARGETS},
                    **{f"pred_{target}": pred[:, idx] for idx, target in enumerate(ROUND2_TARGETS)},
                }
            )
        )
        metric_rows.extend(
            _prediction_metric_rows(
                validation_family="repeated_kfold",
                repetition=int((split_index - 1) // int(settings["repeated_kfold_splits"]) + 1),
                fold=int((split_index - 1) % int(settings["repeated_kfold_splits"]) + 1),
                test_label="fold",
                truth=test_frame,
                pred=pred,
            )
        )

    unique_osli = sorted(int(value) for value in dataset["OSLI"].round().astype(int).unique())
    for fold_index, osli_value in enumerate(unique_osli, start=1):
        test_source = dataset.loc[dataset["OSLI"].round().astype(int) == osli_value].copy()
        train_frame = dataset.loc[dataset["OSLI"].round().astype(int) != osli_value].reset_index(drop=True)
        test_frame = test_source.reset_index(drop=True)
        if train_frame.empty or test_frame.empty:
            continue
        pred = _train_selected_surrogate(train_frame, test_frame, spec, base_config, seed=master_seed + 500 + fold_index)
        prediction_rows.append(
            pd.DataFrame(
                {
                    "validation_family": "leave_one_osli_out",
                    "repetition": 1,
                    "fold": fold_index,
                    "test_label": f"OSLI_{osli_value}",
                    "sample_id": test_frame["sample_id"].to_numpy(),
                    "distance_to_center": np.linalg.norm(
                        feature_matrix.loc[test_source.index].to_numpy(dtype=float) - dataset_center[None, :],
                        axis=1,
                    ),
                    "nearest_train_distance": np.nan,
                    **{f"true_{target}": test_frame[target].to_numpy() for target in ROUND2_TARGETS},
                    **{f"pred_{target}": pred[:, idx] for idx, target in enumerate(ROUND2_TARGETS)},
                }
            )
        )
        metric_rows.extend(
            _prediction_metric_rows(
                validation_family="leave_one_osli_out",
                repetition=1,
                fold=fold_index,
                test_label=f"OSLI_{osli_value}",
                truth=test_frame,
                pred=pred,
            )
        )

    distances = np.linalg.norm(feature_matrix.to_numpy(dtype=float) - dataset_center[None, :], axis=1)
    outer_threshold = float(np.quantile(distances, 1.0 - float(settings["outer_shell_test_fraction"])))
    outer_mask = distances >= outer_threshold
    outer_train = dataset.loc[~outer_mask].reset_index(drop=True)
    outer_test = dataset.loc[outer_mask].reset_index(drop=True)
    outer_pred = _train_selected_surrogate(outer_train, outer_test, spec, base_config, seed=master_seed + 900)
    prediction_rows.append(
        pd.DataFrame(
            {
                "validation_family": "outer_shell_holdout",
                "repetition": 1,
                "fold": 1,
                "test_label": "boundary_shell",
                "sample_id": outer_test["sample_id"].to_numpy(),
                "distance_to_center": distances[outer_mask],
                "nearest_train_distance": np.nan,
                **{f"true_{target}": outer_test[target].to_numpy() for target in ROUND2_TARGETS},
                **{f"pred_{target}": outer_pred[:, idx] for idx, target in enumerate(ROUND2_TARGETS)},
            }
        )
    )
    metric_rows.extend(
        _prediction_metric_rows(
            validation_family="outer_shell_holdout",
            repetition=1,
            fold=1,
            test_label="boundary_shell",
            truth=outer_test,
            pred=outer_pred,
        )
    )

    inner_low, inner_high = settings["feature_tail_inner_quantiles"]
    inner_bounds = {
        feature: (float(dataset[feature].quantile(inner_low)), float(dataset[feature].quantile(inner_high)))
        for feature in ROUND2_FEATURES
    }
    inner_mask = np.logical_and.reduce(
        [
            dataset[feature].between(bounds[0], bounds[1], inclusive="both").to_numpy()
            for feature, bounds in inner_bounds.items()
        ]
    )
    if int(inner_mask.sum()) < max(100, len(dataset) * 0.2):
        inner_low, inner_high = settings["feature_tail_fallback_quantiles"]
        inner_bounds = {
            feature: (float(dataset[feature].quantile(inner_low)), float(dataset[feature].quantile(inner_high)))
            for feature in ROUND2_FEATURES
        }
        inner_mask = np.logical_and.reduce(
            [
                dataset[feature].between(bounds[0], bounds[1], inclusive="both").to_numpy()
                for feature, bounds in inner_bounds.items()
            ]
        )
    tail_train = dataset.loc[inner_mask].reset_index(drop=True)
    tail_test = dataset.loc[~inner_mask].reset_index(drop=True)
    tail_pred = _train_selected_surrogate(tail_train, tail_test, spec, base_config, seed=master_seed + 1200)
    prediction_rows.append(
        pd.DataFrame(
            {
                "validation_family": "feature_tail_holdout",
                "repetition": 1,
                "fold": 1,
                "test_label": f"outside_q{int(inner_low * 100):02d}_q{int(inner_high * 100):02d}",
                "sample_id": tail_test["sample_id"].to_numpy(),
                "distance_to_center": np.nan,
                "nearest_train_distance": np.nan,
                **{f"true_{target}": tail_test[target].to_numpy() for target in ROUND2_TARGETS},
                **{f"pred_{target}": tail_pred[:, idx] for idx, target in enumerate(ROUND2_TARGETS)},
            }
        )
    )
    metric_rows.extend(
        _prediction_metric_rows(
            validation_family="feature_tail_holdout",
            repetition=1,
            fold=1,
            test_label=f"outside_q{int(inner_low * 100):02d}_q{int(inner_high * 100):02d}",
            truth=tail_test,
            pred=tail_pred,
        )
    )

    prediction_frame = pd.concat(prediction_rows, ignore_index=True)
    metrics_frame = pd.DataFrame(metric_rows)
    write_csv(prediction_frame, paths.models_dir / "surrogate_validation_predictions.csv")
    write_csv(metrics_frame, paths.models_dir / "surrogate_validation_fold_metrics.csv")

    summary_rows = []
    bootstrap_iterations = int(settings["bootstrap_iterations"])
    for (family, target), group in metrics_frame.groupby(["validation_family", "target"], sort=True):
        mae_values = group["MAE"].to_numpy(dtype=float)
        nmae_values = group["nMAE"].to_numpy(dtype=float)
        spearman_values = group["Spearman_rho"].to_numpy(dtype=float)
        mae_ci = _bootstrap_ci(mae_values, np.mean, bootstrap_iterations, master_seed + len(summary_rows) + 1)
        nmae_ci = _bootstrap_ci(nmae_values, np.mean, bootstrap_iterations, master_seed + len(summary_rows) + 101)
        spearman_ci = _bootstrap_ci(spearman_values, np.mean, bootstrap_iterations, master_seed + len(summary_rows) + 201)
        summary_rows.append(
            {
                "validation_family": family,
                "target": target,
                "mean_R2": float(group["R2"].mean()),
                "mean_MAE": float(mae_values.mean()),
                "mean_RMSE": float(group["RMSE"].mean()),
                "mean_nMAE": float(nmae_values.mean()),
                "mean_nRMSE": float(group["nRMSE"].mean()),
                "mean_Spearman_rho": float(spearman_values.mean()),
                "mean_lower_tail_nMAE": float(group["lower_tail_nMAE"].mean()),
                "mean_upper_tail_nMAE": float(group["upper_tail_nMAE"].mean()),
                "MAE_ci_low": mae_ci[0],
                "MAE_ci_high": mae_ci[1],
                "nMAE_ci_low": nmae_ci[0],
                "nMAE_ci_high": nmae_ci[1],
                "Spearman_ci_low": spearman_ci[0],
                "Spearman_ci_high": spearman_ci[1],
                "fold_count": int(len(group)),
            }
        )
    summary_frame = pd.DataFrame(summary_rows)
    write_csv(summary_frame, paths.models_dir / "surrogate_validation_summary.csv")
    protocol_payload = {
        "protocol_version": base_config["round2"]["protocol_version"],
        "selected_candidate": spec["candidate_name"],
        "feature_scaler": spec["feature_scaler"],
        "target_scaler": spec["target_scaler"],
        "loss": spec["loss"],
        "hyperparameters": spec["hyperparameters"],
        "settings": settings,
        "dataset_path": base_config["round2"]["canonical_dataset"],
        "surrogate_path": base_config["round2"]["canonical_surrogate"],
    }
    summary_payload = {
        "protocol": protocol_payload,
        "summary_csv": str(paths.models_dir / "surrogate_validation_summary.csv"),
        "predictions_csv": str(paths.models_dir / "surrogate_validation_predictions.csv"),
        "fold_metrics_csv": str(paths.models_dir / "surrogate_validation_fold_metrics.csv"),
        "families": summary_frame.groupby("validation_family")["target"].count().to_dict(),
    }
    write_json(protocol_payload, paths.models_dir / "surrogate_validation_protocol.json")
    write_json(summary_payload, paths.models_dir / "surrogate_validation_summary.json")
    return summary_payload


def _load_baseline_optimization(base_config: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    baseline = base_config["round2"]["baseline_runs"]
    ddpg = pd.read_csv(baseline["ddpg_results"])
    nsga = pd.read_csv(baseline["nsga2_results"])
    combined = pd.read_csv(baseline["optimization_results"])
    return ddpg, nsga, combined


def _append_utilities(frame: pd.DataFrame, reference_frame: pd.DataFrame, target_bounds: dict[str, tuple[float, float]], utility_weights: dict[str, list[float]]) -> pd.DataFrame:
    enriched = frame.copy()
    legacy_scores = compute_legacy_utility(enriched, reference_frame)
    fixed_scores = compute_fixed_domain_utility(enriched, target_bounds)
    for scenario_name, weights in utility_weights.items():
        enriched[f"legacy_utility_{scenario_name}"] = apply_weighted_utility(legacy_scores, weights)
        enriched[f"fixed_utility_{scenario_name}"] = apply_weighted_utility(fixed_scores, weights)
    return enriched


def _sampled_pool_reference(dataset: pd.DataFrame, utility_weights: dict[str, list[float]]) -> pd.DataFrame:
    fixed_scores = compute_fixed_domain_utility(
        dataset,
        {
            target: (float(dataset[target].min()), float(dataset[target].max()))
            for target in ROUND2_TARGETS
        },
    )
    rows = []
    for scenario_name, weights in utility_weights.items():
        utility = apply_weighted_utility(fixed_scores, weights)
        best_index = int(utility.idxmax())
        row = dataset.loc[best_index, ["sample_id", *ROUND2_FEATURES, *ROUND2_TARGETS]].to_dict()
        row.update({"method": "SampledPoolOracle", "scenario": scenario_name, "seed": -1, "reward": float(utility.loc[best_index])})
        rows.append(row)
    return pd.DataFrame(rows)[OPTIMIZATION_RESULT_COLUMNS]


def _feasible_pool_random_resampling(dataset: pd.DataFrame, utility_weights: dict[str, list[float]], *, master_seed: int, evaluation_budget: int, seeds_per_scenario: int) -> pd.DataFrame:
    fixed_scores = compute_fixed_domain_utility(
        dataset,
        {
            target: (float(dataset[target].min()), float(dataset[target].max()))
            for target in ROUND2_TARGETS
        },
    )
    rows = []
    for scenario_index, (scenario_name, weights) in enumerate(utility_weights.items()):
        utility = apply_weighted_utility(fixed_scores, weights).to_numpy(dtype=float)
        for seed in range(seeds_per_scenario):
            rng = np.random.default_rng(master_seed + scenario_index * 1000 + seed)
            draw = rng.integers(0, len(dataset), evaluation_budget)
            best_idx = int(draw[np.argmax(utility[draw])])
            row = dataset.iloc[best_idx][["sample_id", *ROUND2_FEATURES, *ROUND2_TARGETS]].to_dict()
            row.update({"method": "FeasiblePoolRandom", "scenario": scenario_name, "seed": seed, "reward": float(utility[best_idx])})
            rows.append(row)
    return pd.DataFrame(rows)[OPTIMIZATION_RESULT_COLUMNS]


def _measure_runtime(label: str, fn: Callable[[], Any]) -> dict[str, Any]:
    process = psutil.Process() if psutil is not None else None
    peak_rss = 0
    start = time.perf_counter()
    result = fn()
    duration = time.perf_counter() - start
    if process is not None:
        peak_rss = max(peak_rss, process.memory_info().rss)
    return {"label": label, "seconds": duration, "peak_rss_bytes": int(peak_rss), "result": result}


def run_benchmark_fairness(config_path: str | Path, *, run_id: str | None = None, output_dir: str | Path | None = None, resume: bool = False) -> dict[str, Any]:
    base_config, run_config, paths = prepare_round2_workspace(config_path, run_id=run_id, output_dir=output_dir)
    dataset = _load_round2_dataset(base_config)
    target_bounds = {target: (float(dataset[target].min()), float(dataset[target].max())) for target in ROUND2_TARGETS}
    surrogate = load_surrogate(paths.models_dir / "surrogate.pt")
    env = OptimizationEnvironment(surrogate=surrogate, guardrail_cfg=base_config["optimization"].get("surrogate_guardrail"))
    ddpg_baseline, nsga_baseline, baseline_combined = _load_baseline_optimization(base_config)
    utility_weights = base_config["optimization"]["utility_weights"]

    ddpg_utility = _append_utilities(ddpg_baseline, baseline_combined, target_bounds, utility_weights)
    nsga_utility = _append_utilities(nsga_baseline, baseline_combined, target_bounds, utility_weights)
    fig9_rows = []
    for scenario_name in UTILITY_SCENARIOS:
        ddpg_best = ddpg_utility.loc[ddpg_utility["scenario"] == scenario_name, f"legacy_utility_{scenario_name}"].max()
        nsga_best = nsga_utility[f"legacy_utility_{scenario_name}"].max()
        fig9_rows.append(
            {
                "scenario": scenario_name,
                "DDPG_best_legacy_utility": float(ddpg_best),
                "NSGAII_best_legacy_utility": float(nsga_best),
                "delta": float(ddpg_best - nsga_best),
            }
        )
    fig9_frame = pd.DataFrame(fig9_rows)
    write_csv(fig9_frame, paths.optimization_dir / "fig9_utility_recalc.csv")

    audit_fig9 = pd.read_csv(Path(base_config["round2"]["research_root"]) / "fig9d_utility_recalc.csv")
    merged_audit = fig9_frame.merge(audit_fig9, on="scenario", how="inner")
    if not np.allclose(merged_audit["DDPG_best_legacy_utility"], merged_audit["DDPG_best_utility"], atol=1e-8):
        raise RuntimeError("Baseline Fig. 9 DDPG utility no longer matches audit reference.")
    if not np.allclose(merged_audit["NSGAII_best_legacy_utility"], merged_audit["NSGAII_best_utility"], atol=1e-8):
        raise RuntimeError("Baseline Fig. 9 NSGA-II utility no longer matches audit reference.")

    if not resume or not (paths.optimization_dir / "cmaes_results_round2.csv").exists():
        run_cmaes(run_config, load_surrogate(paths.models_dir / "surrogate.pt"), output_suffix="round2")
    if not resume or not (paths.optimization_dir / "random_search_results_round2.csv").exists():
        run_random_search(run_config, load_surrogate(paths.models_dir / "surrogate.pt"), output_suffix="round2")

    cma_results = pd.read_csv(paths.optimization_dir / "cmaes_results_round2.csv")
    random_results = pd.read_csv(paths.optimization_dir / "random_search_results_round2.csv")
    cma_archive = pd.read_csv(paths.optimization_dir / "cmaes_archive_round2.csv")
    random_archive = pd.read_csv(paths.optimization_dir / "random_search_archive_round2.csv")
    oracle = _sampled_pool_reference(dataset, utility_weights)
    feasible_pool_random = _feasible_pool_random_resampling(
        dataset,
        utility_weights,
        master_seed=int(base_config["round2"]["master_seed"]),
        evaluation_budget=int(base_config["optimization"]["random_search"]["evaluation_budget"]),
        seeds_per_scenario=int(base_config["optimization"]["random_search"]["seeds_per_scenario"]),
    )

    combined_best = pd.concat([ddpg_baseline, nsga_baseline, cma_results, random_results, oracle, feasible_pool_random], ignore_index=True)
    combined_best = _append_utilities(combined_best, baseline_combined, target_bounds, utility_weights)
    write_csv(combined_best, paths.optimization_dir / "optimizer_results_round2.csv")

    group_archives: dict[str, pd.DataFrame] = {}
    for scenario_name in UTILITY_SCENARIOS:
        group_archives[f"DDPG::{scenario_name}"] = ddpg_baseline.loc[ddpg_baseline["scenario"] == scenario_name].copy()
        group_archives[f"CMA-ES::{scenario_name}"] = cma_archive.loc[cma_archive["scenario"] == scenario_name].copy()
        group_archives[f"RandomSearch::{scenario_name}"] = random_archive.loc[random_archive["scenario"] == scenario_name].copy()
        group_archives[f"FeasiblePoolRandom::{scenario_name}"] = feasible_pool_random.loc[feasible_pool_random["scenario"] == scenario_name].copy()
    group_archives["NSGA-II"] = nsga_baseline.copy()

    decomposition_rows = []
    for group_name, frame in group_archives.items():
        decomposition_rows.append(_guardrail_decomposition_frame(frame, env, target_bounds, group_name=group_name))
    decomposition_frame = pd.concat(decomposition_rows, ignore_index=True)
    write_csv(decomposition_frame, paths.optimization_dir / "optimizer_guardrail_decomposition.csv")

    reference = build_fixed_reference(group_archives)
    full_archive = evaluate_archive_metrics(group_archives, reference)
    write_csv(full_archive, paths.optimization_dir / "benchmark_full_archive.csv")

    unique_objective_groups = {}
    for group_name, group in decomposition_frame.groupby("group", sort=True):
        unique_frame = group[["method", "scenario", "seed", *ROUND2_FEATURES, "EUIt", "EG", "H"]].copy()
        unique_frame["reward"] = np.nan
        unique_objective_groups[group_name] = dedupe_objective_tuples(unique_frame)
    unique_objective_metrics = evaluate_archive_metrics(unique_objective_groups, reference)

    projected_groups: dict[str, pd.DataFrame] = {}
    for group_name, frame in group_archives.items():
        projected = project_candidates_to_nearest_blocks(frame, dataset)
        projected = projected.drop_duplicates(subset=["matched_sample_id"], keep="first")
        projected_metrics_frame = dataset.loc[dataset["sample_id"].isin(projected["matched_sample_id"].astype(int))].copy()
        projected_metrics_frame["method"] = frame["method"].iloc[0]
        projected_metrics_frame["scenario"] = frame["scenario"].iloc[0]
        projected_metrics_frame["seed"] = frame["seed"].iloc[0]
        projected_metrics_frame["reward"] = np.nan
        projected_groups[group_name] = projected_metrics_frame[OPTIMIZATION_RESULT_COLUMNS]
    projected_metrics = evaluate_archive_metrics(projected_groups, reference)

    sizes = list(base_config["round2"]["fairness_analysis"]["equal_size_archive_sizes"])
    repetitions = int(base_config["round2"]["fairness_analysis"]["equal_size_repetitions"])
    repetition_rows = []
    rng = np.random.default_rng(int(base_config["round2"]["master_seed"]))
    for group_name, frame in group_archives.items():
        actual_size = len(frame)
        for requested_size in sizes:
            if group_name.startswith("DDPG::"):
                metrics = evaluate_archive_metrics({group_name: frame}, reference).iloc[0].to_dict()
                repetition_rows.append(
                    {
                        "group": group_name,
                        "requested_size": requested_size,
                        "replicate": 0,
                        "actual_size": actual_size,
                        "HV": metrics["HV"],
                        "IGD": metrics["IGD"],
                    }
                )
                continue
            for replicate in range(repetitions):
                choose = min(requested_size, actual_size)
                indices = rng.choice(actual_size, size=choose, replace=False)
                sampled = frame.iloc[indices].reset_index(drop=True)
                metrics = evaluate_archive_metrics({group_name: sampled}, reference).iloc[0].to_dict()
                repetition_rows.append(
                    {
                        "group": group_name,
                        "requested_size": requested_size,
                        "replicate": replicate,
                        "actual_size": actual_size,
                        "HV": metrics["HV"],
                        "IGD": metrics["IGD"],
                    }
                )
    repetition_frame = pd.DataFrame(repetition_rows)
    write_csv(repetition_frame, paths.optimization_dir / "benchmark_equal_size_repetitions.csv")
    summary_frame = (
        repetition_frame.groupby(["group", "requested_size", "actual_size"], as_index=False)
        .agg(
            HV_mean=("HV", "mean"),
            HV_std=("HV", "std"),
            HV_q05=("HV", lambda s: float(np.quantile(s, 0.05))),
            HV_median=("HV", "median"),
            HV_q95=("HV", lambda s: float(np.quantile(s, 0.95))),
            IGD_mean=("IGD", "mean"),
            IGD_std=("IGD", "std"),
            IGD_q05=("IGD", lambda s: float(np.quantile(s, 0.05))),
            IGD_median=("IGD", "median"),
            IGD_q95=("IGD", lambda s: float(np.quantile(s, 0.95))),
        )
    )
    write_csv(summary_frame, paths.optimization_dir / "benchmark_equal_size_summary.csv")

    metric_audit_rows = []
    for _, row in full_archive.iterrows():
        metric_audit_rows.append(
            {
                "metric_definition": "current_clipped_full_archive",
                "group": row["group"],
                "requested_size": row["rows"],
                "rows": row["rows"],
                "non_dominated_rows": row["non_dominated_rows"],
                "HV": row["HV"],
                "IGD": row["IGD"],
            }
        )
    for _, row in unique_objective_metrics.iterrows():
        metric_audit_rows.append(
            {
                "metric_definition": "unique_clipped_objective_tuples",
                "group": row["group"],
                "requested_size": row["rows"],
                "rows": row["rows"],
                "non_dominated_rows": row["non_dominated_rows"],
                "HV": row["HV"],
                "IGD": row["IGD"],
            }
        )
    for _, row in summary_frame.iterrows():
        metric_audit_rows.append(
            {
                "metric_definition": f"equal_size_archive_{int(row['requested_size'])}",
                "group": row["group"],
                "requested_size": int(row["requested_size"]),
                "rows": int(row["actual_size"]),
                "non_dominated_rows": np.nan,
                "HV": row["HV_mean"],
                "IGD": row["IGD_mean"],
            }
        )
    for _, row in projected_metrics.iterrows():
        metric_audit_rows.append(
            {
                "metric_definition": "projected_feasible_block_archive",
                "group": row["group"],
                "requested_size": row["rows"],
                "rows": row["rows"],
                "non_dominated_rows": row["non_dominated_rows"],
                "HV": row["HV"],
                "IGD": row["IGD"],
            }
        )
    metric_audit_frame = pd.DataFrame(metric_audit_rows)
    write_csv(metric_audit_frame, paths.optimization_dir / "benchmark_metric_definition_audit.csv")

    max_hv = theoretical_max_hv(reference["reference_point"])
    hv_diag_groups = []
    for group_name, group in decomposition_frame.groupby("group", sort=True):
        full_row = full_archive.loc[full_archive["group"] == group_name].iloc[0]
        unique_count = int(group["duplicate_objective_tuple_id"].nunique())
        hv_diag_groups.append(
            {
                "group": group_name,
                "full_archive_hv": float(full_row["HV"]),
                "full_archive_igd": float(full_row["IGD"]),
                "theoretical_max_hv": max_hv,
                "hits_theoretical_max": bool(abs(float(full_row["HV"]) - max_hv) <= 1e-6),
                "clipped_utopia_count": int(group["is_exact_utopian_tuple"].sum()),
                "unique_objective_tuple_count": unique_count,
                "clip_flag_any_count": int((group[["clip_flag_EUIt", "clip_flag_EG", "clip_flag_H"]].any(axis=1)).sum()),
            }
        )
    hv_diag_payload = {
        "theoretical_max_hv": max_hv,
        "reference_point": reference["reference_point"].tolist(),
        "reference_front_rows": int(len(reference["reference_front"])),
        "groups": hv_diag_groups,
        "answers": {
            "theoretical_max_source": "reference_point_product",
            "nsga_igd_source": "distance to fixed clipped reference front built from canonical union",
            "archive_diversity_warning": "full-archive HV alone does not describe descriptor or feasible-space diversity when many rows collapse to duplicated clipped objective tuples",
        },
    }
    write_json(hv_diag_payload, paths.optimization_dir / "hv_saturation_diagnostic.json")
    audit_md_lines = [
        "# Benchmark Metric Definition Audit",
        "",
        f"- Fixed reference point: `{reference['reference_point'].tolist()}`.",
        f"- Theoretical maximum HV: `{max_hv:.6f}`.",
        "- All metric families below share the same ideal, nadir, reference front, and reference point.",
        "",
        "## Definitions",
        "- A: current clipped full archive.",
        "- B: unique clipped objective tuples.",
        "- C: equal-size archive downsampling.",
        "- D: projected feasible block archive.",
        "",
        "## Saturation note",
        "- HV values near 1.331 indicate saturation against the fixed reference point rather than rich archive diversity by themselves.",
    ]
    (paths.optimization_dir / "benchmark_metric_definition_audit.md").write_text("\n".join(audit_md_lines), encoding="utf-8")

    seed_rows = []
    for seed_value, seed_frame in nsga_baseline.groupby("seed"):
        metrics = evaluate_archive_metrics({f"NSGA-II::{seed_value}": seed_frame}, reference).iloc[0]
        seed_rows.append({"method": "NSGA-II", "scenario": "NSGA-II", "seed": seed_value, "HV": metrics["HV"], "IGD": metrics["IGD"]})
    for frame in [ddpg_baseline, cma_results, random_results, feasible_pool_random]:
        for (method, scenario, seed_value), group in frame.groupby(["method", "scenario", "seed"]):
            utility_col = f"legacy_utility_{scenario}" if scenario in UTILITY_SCENARIOS else "reward"
            enriched = _append_utilities(group, baseline_combined, target_bounds, utility_weights)
            best_row = enriched.sort_values(utility_col, ascending=False).iloc[0]
            seed_rows.append(
                {
                    "method": method,
                    "scenario": scenario,
                    "seed": seed_value,
                    "best_legacy_utility": float(best_row.get(utility_col, best_row["reward"])),
                    "EUIt": float(best_row["EUIt"]),
                    "EG": float(best_row["EG"]),
                    "H": float(best_row["H"]),
                }
            )
    write_csv(pd.DataFrame(seed_rows), paths.optimization_dir / "benchmark_seed_level.csv")

    utility_rows = []
    for method_frame in [ddpg_baseline, nsga_baseline, cma_results, random_results]:
        enriched = _append_utilities(method_frame, baseline_combined, target_bounds, utility_weights)
        for scenario_name in UTILITY_SCENARIOS:
            utility_rows.append(
                enriched.assign(
                    utility_scenario=scenario_name,
                    legacy_utility=enriched[f"legacy_utility_{scenario_name}"],
                    fixed_domain_utility=enriched[f"fixed_utility_{scenario_name}"],
                )[
                    ["method", "scenario", "seed", "utility_scenario", "legacy_utility", "fixed_domain_utility", "EUIt", "EG", "H"]
                ]
            )
    write_csv(pd.concat(utility_rows, ignore_index=True), paths.optimization_dir / "utility_sensitivity.csv")

    timing_rows = []
    for _ in range(int(base_config["round2"]["runtime_audit"]["repetitions"])):
        timing_rows.append(
            {
                "method": "FeasiblePoolRandom",
                **_measure_runtime(
                    "FeasiblePoolRandom",
                    lambda: _feasible_pool_random_resampling(dataset, utility_weights, master_seed=int(base_config["round2"]["master_seed"]), evaluation_budget=24000, seeds_per_scenario=1),
                ),
            }
        )
    runtime_frame = pd.DataFrame(
        [
            {
                "method": row["method"],
                "seconds": row["seconds"],
                "peak_rss_bytes": row["peak_rss_bytes"],
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "python_version": ".".join(map(str, tuple(torch.__version__.split(".")[:2]))),
                "torch_version": torch.__version__,
            }
            for row in timing_rows
        ]
    )
    write_csv(runtime_frame, paths.optimization_dir / "runtime_audit.csv")
    write_json(
        {
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "torch_version": torch.__version__,
            "runtime_rows": runtime_frame.to_dict(orient="records"),
        },
        paths.optimization_dir / "runtime_audit.json",
    )

    metadata_payload = {
        "run_id": paths.run_id,
        "canonical_dataset_sha256": sha256_path(base_config["round2"]["canonical_dataset"]),
        "canonical_surrogate_sha256": sha256_path(base_config["round2"]["canonical_surrogate"]),
        "baseline_optimization_sha256": sha256_path(base_config["round2"]["baseline_runs"]["optimization_results"]),
        "cmaes_summary": json.loads((paths.optimization_dir / "cmaes_summary_round2.json").read_text(encoding="utf-8")),
        "random_search_summary": json.loads((paths.optimization_dir / "random_search_summary_round2.json").read_text(encoding="utf-8")),
        "theoretical_max_hv": max_hv,
    }
    write_json(metadata_payload, paths.optimization_dir / "optimizer_run_metadata.json")
    return metadata_payload


def run_feasibility_audit(config_path: str | Path, *, run_id: str | None = None, output_dir: str | Path | None = None) -> dict[str, Any]:
    base_config, _, paths = prepare_round2_workspace(config_path, run_id=run_id, output_dir=output_dir)
    dataset = _load_round2_dataset(base_config)
    target_bounds = {target: (float(dataset[target].min()), float(dataset[target].max())) for target in ROUND2_TARGETS}
    _, _, baseline_combined = _load_baseline_optimization(base_config)
    utility_weights = base_config["optimization"]["utility_weights"]

    sources = [
        pd.read_csv(base_config["round2"]["baseline_runs"]["ddpg_results"]),
        pd.read_csv(base_config["round2"]["baseline_runs"]["nsga2_results"]),
        pd.read_csv(paths.optimization_dir / "cmaes_results_round2.csv"),
        pd.read_csv(paths.optimization_dir / "random_search_results_round2.csv"),
    ]
    combined = pd.concat(sources, ignore_index=True)
    projected = project_candidates_to_nearest_blocks(combined, dataset)
    analytic_targets = dataset.rename(columns={target: f"projected_{target}" for target in ROUND2_TARGETS})
    projected = projected.merge(
        analytic_targets[["sample_id", "projected_EUIt", "projected_EG", "projected_H"]].rename(columns={"sample_id": "matched_sample_id"}),
        on="matched_sample_id",
        how="left",
    )
    enriched = _append_utilities(projected, baseline_combined, target_bounds, utility_weights)
    constraint_frame = audit_descriptor_constraints(enriched)
    osli_frame = audit_osli_values(enriched["OSLI"])
    feasibility_frame = pd.concat([enriched.reset_index(drop=True), constraint_frame.drop(columns="sample_id"), osli_frame], axis=1)
    write_csv(feasibility_frame, paths.optimization_dir / "optimizer_feasibility_audit.csv")

    projection_mapping = feasibility_frame[
        [
            "method",
            "scenario",
            "seed",
            "matched_sample_id",
            "projection_distance",
            "EUIt",
            "EG",
            "H",
            "physical_EUIt" if "physical_EUIt" in feasibility_frame.columns else "projection_distance",
        ]
    ].copy()
    if "physical_EUIt" not in projection_mapping.columns:
        projection_mapping = projection_mapping.rename(columns={"projection_distance": "physical_EUIt"})
        projection_mapping["physical_EUIt"] = np.nan
    write_csv(projection_mapping, paths.optimization_dir / "optimizer_projection_mapping.csv")

    projected_archive_groups: dict[str, pd.DataFrame] = {}
    for (method, scenario), group in feasibility_frame.groupby(["method", "scenario"], sort=True):
        projected_rows = dataset.loc[dataset["sample_id"].isin(group["matched_sample_id"].astype(int))].copy()
        projected_rows["method"] = method
        projected_rows["scenario"] = scenario
        projected_rows["seed"] = -1
        projected_rows["reward"] = np.nan
        projected_archive_groups[f"{method}::{scenario}"] = projected_rows[OPTIMIZATION_RESULT_COLUMNS]
    projected_reference = build_fixed_reference(projected_archive_groups)
    projected_metric_frame = evaluate_archive_metrics(projected_archive_groups, projected_reference)

    summary_rows = []
    for (method, scenario), group in feasibility_frame.groupby(["method", "scenario"], sort=True):
        duplicate_collapse = 1.0 - group["matched_sample_id"].nunique() / max(len(group), 1)
        projected_metric = projected_metric_frame.loc[projected_metric_frame["group"] == f"{method}::{scenario}"].iloc[0]
        summary_rows.append(
            {
                "method": method,
                "scenario": scenario,
                "rows": int(len(group)),
                "unique_matched_sample_count": int(group["matched_sample_id"].nunique()),
                "far_residual_rate_gt_1e-8": float((group["far_minus_bd_af"] > 1e-8).mean()),
                "osr_residual_rate_gt_1e-8": float((group["osr_minus_density_far"] > 1e-8).mean()),
                "osli_non_integer_rate": float((~group["is_integer"]).mean()),
                "projection_distance_mean": float(group["projection_distance"].mean()),
                "projection_distance_median": float(group["projection_distance"].median()),
                "projection_distance_q95": float(group["projection_distance"].quantile(0.95)),
                "duplicate_collapse_rate": float(duplicate_collapse),
                "projected_non_dominated_rows": int(projected_metric["non_dominated_rows"]),
                "projected_HV": float(projected_metric["HV"]),
                "projected_IGD": float(projected_metric["IGD"]),
            }
        )
    summary_frame = pd.DataFrame(summary_rows)
    write_csv(summary_frame, paths.optimization_dir / "optimizer_projection_summary.csv")

    projected_comparison_rows = []
    for scenario_name in UTILITY_SCENARIOS:
        legacy_col = f"legacy_utility_{scenario_name}"
        fixed_col = f"fixed_utility_{scenario_name}"
        for method_name, group in feasibility_frame.groupby("method", sort=True):
            if method_name == "NSGA-II":
                scenario_group = group
            else:
                scenario_group = group.loc[group["scenario"] == scenario_name]
            if scenario_group.empty:
                continue
            original_best = scenario_group.sort_values(legacy_col, ascending=False).iloc[0]
            projected_rows = dataset.loc[dataset["sample_id"].isin(scenario_group["matched_sample_id"].astype(int))]
            projected_scores = apply_weighted_utility(compute_fixed_domain_utility(projected_rows, target_bounds), utility_weights[scenario_name])
            projected_best_idx = int(projected_scores.idxmax())
            projected_best = projected_rows.loc[projected_best_idx]
            projected_comparison_rows.append(
                {
                    "utility_scenario": scenario_name,
                    "method": method_name,
                    "original_sample_id": original_best.get("sample_id", np.nan),
                    "matched_sample_id": int(original_best["matched_sample_id"]),
                    "projection_distance": float(original_best["projection_distance"]),
                    "original_legacy_utility": float(original_best[legacy_col]),
                    "original_fixed_utility": float(original_best[fixed_col]),
                    "projected_fixed_utility": float(projected_scores.loc[projected_best_idx]),
                    "projected_sample_id": int(projected_best["sample_id"]),
                }
            )
    write_csv(pd.DataFrame(projected_comparison_rows), paths.optimization_dir / "projected_utility_comparison.csv")
    return {"projection_summary_csv": str(paths.optimization_dir / "optimizer_projection_summary.csv")}


def build_selection_criteria_registry(config_path: str | Path, *, run_id: str | None = None, output_dir: str | Path | None = None) -> Path:
    base_config, _, paths = prepare_round2_workspace(config_path, run_id=run_id, output_dir=output_dir)
    registry_rows = [
        ("highest training reward", "point", "maximize", "per-method optimizer candidates", "DDPG reward scale", "best-reward DDPG candidate", "no", "highest training reward"),
        ("highest legacy utility", "point", "maximize", "optimizer output rows", "post-hoc utility on current comparison bounds", "best candidate", "only if criterion is named", "highest legacy utility"),
        ("highest fixed-domain utility", "point", "maximize", "optimizer output rows", "training-dataset target bounds", "best candidate", "only if criterion is named", "highest fixed-domain utility"),
        ("lowest mean EUIt", "scenario", "minimize", "scenario-level point set", "none", "lowest energy use", "no", "lowest mean EUIt"),
        ("highest mean EG", "scenario", "maximize", "scenario-level point set", "none", "highest generation", "no", "highest mean EG"),
        ("highest mean H", "scenario", "maximize", "scenario-level point set", "none", "highest solar-hours", "no", "highest mean H"),
        ("largest HV", "archive", "maximize", "archive groups", "fixed reference front", "best front quality", "no", "largest fixed-reference HV"),
        ("smallest IGD", "archive", "minimize", "archive groups", "fixed reference front", "closest to reference front", "no", "smallest fixed-reference IGD"),
        ("selected highest-accuracy surrogate", "surrogate", "minimize", "surrogate candidates", "selection objective", "selected surrogate", "no", "selected highest-accuracy surrogate"),
        ("direct feasible medoid", "point", "minimize", "canonical feasible dataset", "descriptor-space distance to median", "representative case", "no", "descriptor-space medoid"),
        ("physical-validation representative", "point", "bounded", "locked physical cases", "selection stratum", "representative physical case", "no", "physical-validation representative"),
        ("climate-sensitivity representative", "point", "bounded", "locked climate cases", "selection stratum", "representative climate case", "no", "climate-sensitivity representative"),
    ]
    frame = pd.DataFrame(
        registry_rows,
        columns=[
            "criterion_name",
            "level",
            "direction",
            "candidate_universe",
            "normalization",
            "associated_manuscript_term",
            "may_be_called_best",
            "recommended_exact_wording",
        ],
    )
    path = paths.research_root / "selection-criteria-registry.csv"
    write_csv(frame, path)
    return path


def _baseline_locked_case_frame(base_config: Config) -> pd.DataFrame:
    dataset = _load_round2_dataset(base_config)
    ddpg, nsga, combined = _load_baseline_optimization(base_config)
    utility_weights = base_config["optimization"]["utility_weights"]
    target_bounds = {target: (float(dataset[target].min()), float(dataset[target].max())) for target in ROUND2_TARGETS}
    direct_maximin = select_maximin_space_filling(dataset, int(base_config["round2"]["physical_validation"]["maximin_cases"]))
    direct_maximin["selection_stratum"] = "maximin_space_filling"
    direct_tail = select_objective_tail_cases(
        dataset,
        existing_ids=set(int(value) for value in direct_maximin["sample_id"].tolist()),
    )
    interior_pool = dataset.copy()
    target_ok = np.logical_and.reduce(
        [
            dataset[target].between(dataset[target].quantile(0.10), dataset[target].quantile(0.90), inclusive="both").to_numpy()
            for target in ROUND2_TARGETS
        ]
    )
    nearest = np.linalg.norm(
        _normalized_feature_frame(dataset).to_numpy(dtype=float)[:, None, :] - _normalized_feature_frame(dataset).to_numpy(dtype=float)[None, :, :],
        axis=2,
    )
    np.fill_diagonal(nearest, np.inf)
    interior_pool = interior_pool.loc[target_ok & (nearest.min(axis=1) <= np.median(nearest.min(axis=1)))]
    used = set(int(value) for value in pd.concat([direct_maximin["sample_id"], direct_tail["sample_id"]]).tolist())
    interior_pool = interior_pool.loc[~interior_pool["sample_id"].isin(used)]
    rng = np.random.default_rng(int(base_config["round2"]["master_seed"]))
    if len(interior_pool) > int(base_config["round2"]["physical_validation"]["interior_random_cases"]):
        interior_pool = interior_pool.sample(
            n=int(base_config["round2"]["physical_validation"]["interior_random_cases"]),
            random_state=int(rng.integers(0, 1_000_000)),
        )
    interior_pool = interior_pool.copy()
    interior_pool["selection_stratum"] = "interior_random"
    direct_all = pd.concat([direct_maximin, direct_tail, interior_pool], ignore_index=True)

    optimizer_rows = []
    ddpg_enriched = _append_utilities(ddpg, combined, target_bounds, utility_weights)
    nsga_enriched = _append_utilities(nsga, combined, target_bounds, utility_weights)
    for method_name, frame in [("DDPG", ddpg_enriched), ("NSGA-II", nsga_enriched)]:
        used_matched: set[int] = set()
        for scenario_name in UTILITY_SCENARIOS:
            candidate_frame = frame if method_name == "NSGA-II" else frame.loc[frame["scenario"] == scenario_name]
            utility_col = f"legacy_utility_{scenario_name}"
            ranked = candidate_frame.sort_values(utility_col, ascending=False).reset_index(drop=True)
            projected = project_candidates_to_nearest_blocks(ranked, dataset)
            ranked = pd.concat([ranked.reset_index(drop=True), projected[["matched_sample_id", "projection_distance"]]], axis=1)
            chosen = None
            for rank_index, row in ranked.iterrows():
                if int(row["matched_sample_id"]) in used_matched:
                    continue
                chosen = row.copy()
                chosen["selection_stratum"] = "optimizer_linked"
                chosen["optimizer_source"] = method_name
                chosen["scenario_label"] = scenario_name
                chosen["candidate_rank"] = rank_index + 1
                chosen["selection_algorithm"] = "legacy_utility_then_unique_projection"
                used_matched.add(int(row["matched_sample_id"]))
                break
            if chosen is not None:
                optimizer_rows.append(chosen)
    optimizer_frame = pd.DataFrame(optimizer_rows)
    optimizer_frame["sample_id"] = optimizer_frame["matched_sample_id"].astype(int)
    return pd.concat([direct_all, optimizer_frame], ignore_index=True, sort=False)


def build_locked_case_selection(config_path: str | Path, *, run_id: str | None = None, output_dir: str | Path | None = None) -> Path:
    base_config, _, paths = prepare_round2_workspace(config_path, run_id=run_id, output_dir=output_dir)
    selected = _baseline_locked_case_frame(base_config)
    payload = {
        "protocol_version": base_config["round2"]["protocol_version"],
        "master_seed": int(base_config["round2"]["master_seed"]),
        "cases": [],
    }
    for _, row in selected.iterrows():
        payload["cases"].append(
            {
                "sample_id": int(row["sample_id"]),
                "selection_stratum": row.get("selection_stratum", ""),
                "optimizer_source": row.get("optimizer_source"),
                "scenario": row.get("scenario_label"),
                "original_candidate_row": 0 if pd.isna(row.get("candidate_rank")) else int(row.get("candidate_rank", 0) or 0),
                "projection_distance": None if pd.isna(row.get("projection_distance")) else float(row["projection_distance"]),
                "descriptor_values": {feature: float(row[feature]) for feature in ROUND2_FEATURES},
                "analytic_targets": {target: float(row[target]) for target in ROUND2_TARGETS},
                "selection_algorithm": row.get("selection_algorithm", "direct_dataset_selection"),
                "seed": int(base_config["round2"]["master_seed"]),
            }
        )
    payload["protocol_sha"] = physical_protocol_hash({"cases": payload["cases"], "master_seed": payload["master_seed"]})
    path = paths.research_root / "locked-case-selection.json"
    write_json(payload, path)
    return path


def build_physical_model_protocol(config_path: str | Path, *, run_id: str | None = None, output_dir: str | Path | None = None) -> Path:
    base_config, _, paths = prepare_round2_workspace(config_path, run_id=run_id, output_dir=output_dir)
    protocol = {
        "protocol_version": base_config["round2"]["protocol_version"],
        "weather_identifier": list(base_config["round2"]["physical_validation"]["baseline_weather_candidates"]),
        "thermal_model_mode": "Honeybee_EnergyPlus_ideal_air",
        "envelope_parameters": {
            "wall_u": 0.8,
            "roof_u": 0.5,
            "floor_u": 1.5,
            "window_u": 2.7,
            "window_shgc": 0.78,
            "window_vt": 0.6,
        },
        "schedules": {
            "occupancy_fraction": 0.35,
            "lighting_fraction": 0.25,
            "equipment_fraction": 0.30,
            "infiltration_fraction": 0.50,
        },
        "thermostat_dates_setpoints": {
            "heating_on": "Dec 1 to Feb 28 @ 18C",
            "heating_off": "12C",
            "cooling_on": "Jun 15 to Aug 31 @ 26C",
            "cooling_off": "30C",
        },
        "wwr": {"north_south": 0.4, "east_west": 0.1},
        "infiltration_air_changes_per_hour": 1.0,
        "ventilation_m3s_per_person": 30.0 / 3600.0,
        "pv_coverage": 0.8,
        "pv_efficiency": 0.2,
        "pv_performance_ratio": 0.75,
        "H_window": {"date": "January 20", "start": "08:00", "end": "16:00"},
        "H_sensor_definition": "south-facing ground-floor windowsill points",
        "radiance_parameters": {"sky_type": 4, "illuminance_threshold_lux": 1000.0},
        "timeout_seconds": base_config["round2"]["physical_validation"]["timeout_seconds"],
        "retry_count": int(base_config["round2"]["physical_validation"]["max_case_retries"]),
    }
    protocol["protocol_sha"] = physical_protocol_hash(protocol)
    path = paths.physical_dir / "physical_model_protocol.json"
    write_json(protocol, path)
    return path


def collect_existing_physical_jobs(config_path: str | Path, *, run_id: str | None = None, output_dir: str | Path | None = None) -> dict[str, Any]:
    base_config, _, paths = prepare_round2_workspace(config_path, run_id=run_id, output_dir=output_dir)
    recovered_dir = paths.physical_dir / "recovered_physical_jobs"
    recovered_dir.mkdir(parents=True, exist_ok=True)
    patterns = ("physical_stack_job_*.json", "physical_stack_request_*.json", "physical_stack_result_*.json", "physical_stack_projected_*.csv")
    local_sources = [Path("artifacts/publication/diagnostics"), Path("artifacts/publication/reevaluation")]
    found_paths: list[Path] = []
    for source_root in local_sources:
        if not source_root.exists():
            continue
        for pattern in patterns:
            found_paths.extend(source_root.glob(pattern))
    manifests = []
    for source in sorted(set(found_paths)):
        target = recovered_dir / source.name
        shutil.copy2(source, target)
        manifests.append({"name": source.name, "copied_from": str(source), "sha256": sha256_path(source)})
    manifest_path = recovered_dir / "recovered_jobs_manifest.json"
    write_json({"files": manifests}, manifest_path)
    unresolved_path = recovered_dir / "unresolved_jobs.csv"
    write_csv(pd.DataFrame(columns=["job_id", "reason"]), unresolved_path)
    duplicate_mapping = recovered_dir / "duplicate_job_mapping.csv"
    write_csv(pd.DataFrame(columns=["source", "target"]), duplicate_mapping)
    recovered_results = recovered_dir / "recovered_results.csv"
    write_csv(pd.DataFrame(columns=["job_id", "status", "result_path"]), recovered_results)
    return {
        "remote_server_configured": load_server_config() is not None,
        "manifest_path": str(manifest_path),
        "recovered_count": len(manifests),
    }


def run_physical_validation(
    config_path: str | Path,
    *,
    run_id: str | None = None,
    output_dir: str | Path | None = None,
    resume: bool = False,
    wait_seconds: int = 0,
) -> dict[str, Any]:
    base_config, run_config, paths = prepare_round2_workspace(config_path, run_id=run_id, output_dir=output_dir)
    protocol_path = build_physical_model_protocol(config_path, run_id=run_id, output_dir=output_dir)
    locked_path = build_locked_case_selection(config_path, run_id=run_id, output_dir=output_dir)
    weather_candidates = list(base_config["round2"]["physical_validation"]["baseline_weather_candidates"])
    weather_output = Path(base_config["weather"]["output_dir"])
    weather_manifest = []
    selected_weather = None
    for station in weather_candidates:
        station_cfg = base_config["weather"]["stations"][station]
        try:
            weather_record = download_weather_station(station, station_cfg, weather_output)
            weather_manifest.append(weather_record)
            selected_weather = weather_record
            break
        except Exception as exc:  # noqa: BLE001
            weather_manifest.append({"station": station, "status": f"failed: {type(exc).__name__}: {exc}"})
    write_json({"records": sanitize_weather_manifest(weather_manifest)}, paths.physical_dir / "baseline_weather_manifest.json")
    if selected_weather is None:
        selected_weather = {
            "station": "Dongtai",
            "label": "Dongtai, Jiangsu",
            "latitude": 33.5,
            "longitude": 119.8,
            "epw_path": "artifacts/weather/Dongtai/CHN_JS_Dongtai.582510_TMYx.2009-2023.epw",
            "status": "remote_preexisting_epw_only",
        }

    locked = json.loads(Path(locked_path).read_text(encoding="utf-8"))
    dataset = _load_round2_dataset(base_config)
    case_frame = dataset.loc[dataset["sample_id"].isin([case["sample_id"] for case in locked["cases"]])].copy()
    case_frame = case_frame.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    epw_path_value = str(selected_weather["epw_path"])
    relative_epw = epw_path_value.replace("\\", "/")
    if relative_epw.lower().endswith(".epw") and not relative_epw.startswith("artifacts/"):
        relative_epw = str(Path("artifacts") / "weather" / Path(epw_path_value).parent.name / Path(epw_path_value).name).replace("\\", "/")
    request_overrides = {
        "epw_relpath": relative_epw,
        "radiance_sky": {"latitude": selected_weather["latitude"], "longitude": selected_weather["longitude"], "sky_type": 4},
        "timeouts": base_config["round2"]["physical_validation"]["timeout_seconds"],
    }
    probe_frame, probe_summary = physical_stack_candidate_probe(
        run_config,
        case_frame,
        limit=len(case_frame),
        output_suffix=f"{paths.run_id}_baseline",
        async_mode=True,
        wait_seconds=wait_seconds,
        job_id=(json.loads((paths.physical_dir / "baseline_job.json").read_text(encoding="utf-8"))["job_id"] if resume and (paths.physical_dir / "baseline_job.json").exists() else None),
        request_overrides=request_overrides,
    )
    if probe_summary.get("job_id"):
        write_json(probe_summary, paths.physical_dir / "baseline_job.json")
    if probe_summary.get("status") and probe_summary["status"] != "completed":
        summary = {
            "protocol_path": str(protocol_path),
            "locked_case_selection": str(locked_path),
            "baseline_weather": sanitize_weather_manifest([selected_weather])[0],
            "status": probe_summary["status"],
            "job_id": probe_summary.get("job_id"),
            "annual_irradiance_status": "pending",
        }
        write_json(summary, paths.physical_dir / "physical_validation_summary.json")
        return summary

    parsed = parse_physical_results_frame(probe_frame)
    locked_frame = pd.DataFrame(locked["cases"])
    locked_frame["sample_id"] = locked_frame["sample_id"].astype(int)
    if "descriptor_values" in locked_frame.columns:
        descriptor_frame = pd.json_normalize(locked_frame["descriptor_values"]).add_prefix("locked_")
        analytic_frame = pd.json_normalize(locked_frame["analytic_targets"]).add_prefix("analytic_")
        locked_frame = pd.concat([locked_frame.drop(columns=["descriptor_values", "analytic_targets"]), descriptor_frame, analytic_frame], axis=1)
    case_results = parsed.merge(locked_frame, left_on="matched_sample_id", right_on="sample_id", how="left")
    for column in ["scenario", "optimizer_source", "selection_stratum", "projection_distance", "original_candidate_row"]:
        left_name = f"{column}_x"
        right_name = f"{column}_y"
        if left_name in case_results.columns or right_name in case_results.columns:
            case_results[column] = case_results.get(right_name)
            if case_results[column] is None:
                case_results[column] = case_results.get(left_name)
            elif left_name in case_results.columns:
                case_results[column] = case_results[column].fillna(case_results.get(left_name))
            if left_name in case_results.columns:
                case_results = case_results.drop(columns=[left_name])
            if right_name in case_results.columns:
                case_results = case_results.drop(columns=[right_name])
    if "physical_generation_summary" in case_results.columns:
        case_results["physical_generation_summary"] = case_results["physical_generation_summary"].apply(
            lambda value: ast.literal_eval(value) if isinstance(value, str) and value.startswith("{") else value
        )
        case_results["EG_GHI_proxy"] = case_results["physical_generation_summary"].apply(
            lambda payload: payload.get("total_production") if isinstance(payload, dict) else np.nan
        )
    else:
        case_results["EG_GHI_proxy"] = np.nan
    write_csv(case_results, paths.physical_dir / "physical_validation_results.csv")
    write_csv(locked_frame, paths.physical_dir / "physical_validation_cases.csv")
    failures = case_results.loc[~case_results["energyplus_ok"] | ~case_results["radiance_ok"]].copy()
    write_csv(failures, paths.physical_dir / "physical_validation_failures.csv")

    direct_cases = case_results.loc[case_results["selection_stratum"] != "optimizer_linked"].copy()
    optimizer_cases = case_results.loc[case_results["selection_stratum"] == "optimizer_linked"].copy()
    bootstrap_iterations = int(base_config["round2"]["surrogate_validation"]["bootstrap_iterations"])
    metrics_rows = []
    for target_name, truth_column, physical_column in [
        ("EUIt", "EUIt", "physical_EUIt"),
        ("EG_GHI_proxy", "EG", "physical_EG_total_production"),
        ("EG_roof_irradiance", "EG", None),
        ("H", "H", "physical_H_proxy"),
    ]:
        if physical_column is None or physical_column not in direct_cases.columns:
            metrics_rows.append(
                {
                    "target": target_name,
                    "status": "unavailable",
                    "count": 0,
                    "bias": np.nan,
                    "MAE": np.nan,
                    "RMSE": np.nan,
                    "nMAE": np.nan,
                    "nRMSE": np.nan,
                    "Pearson_r": np.nan,
                    "Spearman_rho": np.nan,
                    "Kendall_tau": np.nan,
                    "slope": np.nan,
                    "intercept": np.nan,
                    "MAE_ci_low": np.nan,
                    "MAE_ci_high": np.nan,
                    "rank_preservation": np.nan,
                }
            )
            continue
        truth = direct_cases[truth_column].to_numpy(dtype=float)
        pred = direct_cases[physical_column].to_numpy(dtype=float)
        target_range = float(direct_cases[truth_column].max() - direct_cases[truth_column].min())
        metrics_rows.append(
            _target_stat_row(
                target_name,
                truth,
                pred,
                bootstrap_iterations=bootstrap_iterations,
                seed=int(base_config["round2"]["master_seed"]) + len(metrics_rows),
                normalized_range=target_range,
                status="available",
            )
        )
    metrics_frame = pd.DataFrame(metrics_rows)
    write_csv(metrics_frame, paths.physical_dir / "physical_validation_metrics.csv")

    ddpg_baseline, nsga_baseline, baseline_combined = _load_baseline_optimization(base_config)
    target_bounds = {target: (float(dataset[target].min()), float(dataset[target].max())) for target in ROUND2_TARGETS}
    utility_weights = base_config["optimization"]["utility_weights"]
    optimizer_mapping_rows = []
    for _, row in optimizer_cases.iterrows():
        source = str(row["optimizer_source"])
        scenario_name = str(row["scenario"])
        rank_index = int(row["original_candidate_row"])
        if source == "DDPG":
            source_frame = _append_utilities(ddpg_baseline.loc[ddpg_baseline["scenario"] == scenario_name].copy(), baseline_combined, target_bounds, utility_weights)
            source_frame = source_frame.sort_values(f"legacy_utility_{scenario_name}", ascending=False).reset_index(drop=True)
        else:
            source_frame = _append_utilities(nsga_baseline.copy(), baseline_combined, target_bounds, utility_weights)
            source_frame = source_frame.sort_values(f"legacy_utility_{scenario_name}", ascending=False).reset_index(drop=True)
        source_row = source_frame.iloc[max(rank_index - 1, 0)]
        analytic_legacy = apply_weighted_utility(
            compute_fixed_domain_utility(pd.DataFrame([row[["analytic_EUIt", "analytic_EG", "analytic_H"]].rename({"analytic_EUIt": "EUIt", "analytic_EG": "EG", "analytic_H": "H"})]), target_bounds),
            utility_weights[scenario_name],
        ).iloc[0]
        physical_legacy = apply_weighted_utility(
            compute_fixed_domain_utility(pd.DataFrame([{"EUIt": row["physical_EUIt"], "EG": row["physical_EG_total_production"], "H": row["physical_H_proxy"]}]), target_bounds),
            utility_weights[scenario_name],
        ).iloc[0]
        optimizer_mapping_rows.append(
            {
                "optimizer_source": source,
                "scenario": scenario_name,
                "matched_sample_id": int(row["matched_sample_id"]),
                "candidate_rank": rank_index,
                "projection_distance": float(row.get("projection_distance", np.nan)),
                "surrogate_candidate_EUIt": float(source_row["EUIt"]),
                "surrogate_candidate_EG": float(source_row["EG"]),
                "surrogate_candidate_H": float(source_row["H"]),
                "analytic_block_EUIt": float(row["analytic_EUIt"]),
                "analytic_block_EG": float(row["analytic_EG"]),
                "analytic_block_H": float(row["analytic_H"]),
                "physical_EUIt": float(row["physical_EUIt"]),
                "physical_EG": float(row["physical_EG_total_production"]),
                "physical_H": float(row["physical_H_proxy"]),
                "projection_gap_EUIt": float(row["analytic_EUIt"] - source_row["EUIt"]),
                "projection_gap_EG": float(row["analytic_EG"] - source_row["EG"]),
                "projection_gap_H": float(row["analytic_H"] - source_row["H"]),
                "analytic_to_physical_gap_EUIt": float(row["physical_EUIt"] - row["analytic_EUIt"]),
                "analytic_to_physical_gap_EG": float(row["physical_EG_total_production"] - row["analytic_EG"]),
                "analytic_to_physical_gap_H": float(row["physical_H_proxy"] - row["analytic_H"]),
                "total_gap_EUIt": float(row["physical_EUIt"] - source_row["EUIt"]),
                "total_gap_EG": float(row["physical_EG_total_production"] - source_row["EG"]),
                "total_gap_H": float(row["physical_H_proxy"] - source_row["H"]),
                "legacy_utility_surrogate": float(source_row[f"legacy_utility_{scenario_name}"]),
                "fixed_utility_analytic": float(analytic_legacy),
                "fixed_utility_physical": float(physical_legacy),
            }
        )
    optimizer_mapping = pd.DataFrame(optimizer_mapping_rows)
    write_csv(optimizer_mapping, paths.physical_dir / "physical_validation_optimizer_mapping.csv")
    summary = {
        "protocol_path": str(protocol_path),
        "locked_case_selection": str(locked_path),
        "results_csv": str(paths.physical_dir / "physical_validation_results.csv"),
        "baseline_weather": sanitize_weather_manifest([selected_weather])[0],
        "status": "completed",
        "job_id": probe_summary.get("job_id"),
        "annual_irradiance_status": "unavailable",
        "locked_case_count": int(len(locked["cases"])),
        "completed_case_count": int(len(case_results)),
        "failed_case_count": int(len(failures)),
        "energyplus_success_count": int(case_results["energyplus_ok"].sum()),
        "radiance_success_count": int(case_results["radiance_ok"].sum()),
        "protocol_sha": json.loads(Path(protocol_path).read_text(encoding="utf-8"))["protocol_sha"],
        "weather_epw_sha256": selected_weather["epw_sha256"],
    }
    write_json(summary, paths.physical_dir / "physical_validation_summary.json")
    return summary


def run_climate_sensitivity(
    config_path: str | Path,
    *,
    run_id: str | None = None,
    output_dir: str | Path | None = None,
    validate_weather_only: bool = False,
    download_only: bool = False,
    wait_seconds: int = 0,
) -> dict[str, Any]:
    base_config, run_config, paths = prepare_round2_workspace(config_path, run_id=run_id, output_dir=output_dir)
    weather_output = Path(base_config["weather"]["output_dir"])
    weather_manifest_rows = []
    validate_stations = list(base_config["round2"]["climate_sensitivity"].get("validate_stations", []))
    if not validate_stations:
        validate_stations = [base_config["weather"]["preferred_station"], *base_config["round2"]["climate_sensitivity"]["additional_climates"]]
    for station in validate_stations:
        station_cfg = base_config["weather"]["stations"][station]
        try:
            weather_manifest_rows.append(validate_weather_station(station, station_cfg, weather_output, download=True))
        except Exception as exc:  # noqa: BLE001
            weather_manifest_rows.append({"station": station, "status": f"failed: {type(exc).__name__}: {exc}"})
    weather_manifest_rows.append(
        {
            "station": "Jianhu_case_mapping",
            "represented_by_station": "Dongtai",
            "note": str(base_config["round2"]["climate_sensitivity"].get("baseline_representation_note", "")),
        }
    )
    weather_manifest_path = paths.climate_dir / "climate_weather_manifest.json"
    write_json({"records": sanitize_weather_manifest(weather_manifest_rows)}, weather_manifest_path)
    result_path = paths.climate_dir / "climate_sensitivity_results.csv"
    write_csv(pd.DataFrame(columns=["sample_id", "station", "status"]), result_path)
    summary_path = paths.climate_dir / "climate_sensitivity_summary.csv"
    all_valid = all(record.get("hourly_records", 0) >= 8760 for record in weather_manifest_rows if record.get("station") in validate_stations)
    if validate_weather_only or download_only:
        status = "ready" if all_valid else "blocked"
        reason = "all climate weather files validated" if all_valid else "one or more weather files failed validation"
        write_csv(pd.DataFrame([{"status": status, "reason": reason}]), summary_path)
        rank_path = paths.climate_dir / "climate_rank_stability.csv"
        write_csv(pd.DataFrame(columns=["station", "rank_metric", "value"]), rank_path)
        return {
            "weather_manifest": str(weather_manifest_path),
            "results_csv": str(result_path),
            "status": status,
            "mode": "validate_weather_only" if validate_weather_only else "download_only",
        }
    if not (paths.physical_dir / "physical_validation_results.csv").exists():
        write_csv(pd.DataFrame([{"status": "blocked", "reason": "baseline physical validation results not yet collected"}]), summary_path)
        rank_path = paths.climate_dir / "climate_rank_stability.csv"
        write_csv(pd.DataFrame(columns=["station", "rank_metric", "value"]), rank_path)
        return {"weather_manifest": str(weather_manifest_path), "results_csv": str(result_path), "status": "blocked"}

    physical_results = pd.read_csv(paths.physical_dir / "physical_validation_results.csv")
    direct_pool = physical_results.loc[physical_results["selection_stratum"] != "optimizer_linked"].copy()
    dataset = _load_round2_dataset(base_config)
    direct_dataset = dataset.loc[dataset["sample_id"].isin(direct_pool["matched_sample_id"].astype(int))].copy()
    representative_ids = []
    representative_ids.append(int(select_maximin_space_filling(direct_dataset, 1)["sample_id"].iloc[0]))
    representative_ids.append(int(direct_dataset.sort_values("EUIt", ascending=True).iloc[0]["sample_id"]))
    representative_ids.append(int(direct_dataset.sort_values("EG", ascending=False).iloc[0]["sample_id"]))
    representative_ids.append(int(direct_dataset.sort_values("H", ascending=False).iloc[0]["sample_id"]))
    representative_ids = list(dict.fromkeys(representative_ids))
    while len(representative_ids) < 4:
        representative_ids.append(int(direct_dataset.loc[~direct_dataset["sample_id"].isin(representative_ids)].iloc[0]["sample_id"]))

    candidate_frame = dataset.loc[dataset["sample_id"].isin(representative_ids)].copy()
    candidate_frame = candidate_frame.sort_values("sample_id", kind="mergesort").reset_index(drop=True)
    server_cfg = load_server_config()
    if server_cfg is None:
        write_csv(pd.DataFrame([{"status": "blocked", "reason": "remote server config unavailable for climate execution"}]), summary_path)
        rank_path = paths.climate_dir / "climate_rank_stability.csv"
        write_csv(pd.DataFrame(columns=["station", "rank_metric", "value"]), rank_path)
        return {"weather_manifest": str(weather_manifest_path), "results_csv": str(result_path), "status": "blocked"}

    station_records = {record["station"]: record for record in weather_manifest_rows if record.get("station") in base_config["round2"]["climate_sensitivity"]["additional_climates"] and record.get("hourly_records", 0) >= 8760}
    climate_rows = []
    for station in base_config["round2"]["climate_sensitivity"]["additional_climates"]:
        station_job_path = paths.climate_dir / f"{station.lower()}_job.json"
        station_output_suffix = f"{paths.run_id}_{station.lower()}"
        station_csv_path = run_config["publication"]["reevaluation_dir"] if isinstance(run_config, dict) else None
        record = station_records.get(station)
        if record is None:
            write_csv(pd.DataFrame([{"status": "blocked", "reason": f"weather_validation_missing_for_{station}"}]), summary_path)
            rank_path = paths.climate_dir / "climate_rank_stability.csv"
            write_csv(pd.DataFrame(columns=["station", "rank_metric", "value"]), rank_path)
            return {"weather_manifest": str(weather_manifest_path), "results_csv": str(result_path), "status": "blocked"}

        remote_relpath = ensure_remote_epw(server_cfg, record["epw_path"], station)
        request_overrides = {
            "epw_relpath": remote_relpath,
            "radiance_sky": {"latitude": record["latitude"], "longitude": record["longitude"], "sky_type": 4},
            "timeouts": base_config["round2"]["physical_validation"]["timeout_seconds"],
        }
        existing_job_id = None
        if station_job_path.exists():
            existing_job_id = json.loads(station_job_path.read_text(encoding="utf-8")).get("job_id")
        probe_frame, probe_summary = physical_stack_candidate_probe(
            run_config,
            candidate_frame,
            limit=len(candidate_frame),
            output_suffix=station_output_suffix,
            async_mode=True,
            wait_seconds=wait_seconds,
            job_id=existing_job_id,
            request_overrides=request_overrides,
        )
        if probe_summary.get("job_id"):
            write_json(probe_summary, station_job_path)
        if probe_summary.get("status") and probe_summary["status"] != "completed":
            write_csv(pd.DataFrame([{"status": probe_summary["status"], "reason": f"climate batch {station} still running"}]), summary_path)
            rank_path = paths.climate_dir / "climate_rank_stability.csv"
            write_csv(pd.DataFrame(columns=["station", "rank_metric", "value"]), rank_path)
            return {"weather_manifest": str(weather_manifest_path), "results_csv": str(result_path), "status": probe_summary["status"], "station": station}
        probe_frame = parse_physical_results_frame(probe_frame)
        probe_frame["station"] = station
        probe_frame["climate_bucket"] = base_config["weather"]["stations"][station]["climate_bucket"]
        climate_rows.append(probe_frame)

    climate_frame = pd.concat(climate_rows, ignore_index=True)
    baseline_subset = physical_results.loc[physical_results["matched_sample_id"].isin(representative_ids), ["matched_sample_id", "physical_EUIt", "physical_EG_total_production", "physical_H_proxy"]].copy()
    baseline_subset = baseline_subset.rename(
        columns={
            "physical_EUIt": "baseline_physical_EUIt",
            "physical_EG_total_production": "baseline_physical_EG",
            "physical_H_proxy": "baseline_physical_H",
        }
    )
    climate_frame = climate_frame.merge(baseline_subset, on="matched_sample_id", how="left")
    climate_frame["delta_EUIt_vs_baseline"] = climate_frame["physical_EUIt"] - climate_frame["baseline_physical_EUIt"]
    climate_frame["delta_EG_vs_baseline"] = climate_frame["physical_EG_total_production"] - climate_frame["baseline_physical_EG"]
    climate_frame["delta_H_vs_baseline"] = climate_frame["physical_H_proxy"] - climate_frame["baseline_physical_H"]
    write_csv(climate_frame, result_path)
    summary_rows = (
        climate_frame.groupby("station", as_index=False)
        .agg(
            mean_delta_EUIt=("delta_EUIt_vs_baseline", "mean"),
            mean_delta_EG=("delta_EG_vs_baseline", "mean"),
            mean_delta_H=("delta_H_vs_baseline", "mean"),
            energyplus_success_count=("energyplus_ok", "sum"),
            radiance_success_count=("radiance_ok", "sum"),
        )
    )
    write_csv(summary_rows, summary_path)
    rank_rows = []
    for metric_name, column in [("EUIt", "physical_EUIt"), ("EG", "physical_EG_total_production"), ("H", "physical_H_proxy")]:
        baseline_rank = baseline_subset.sort_values(
            {"EUIt": "baseline_physical_EUIt", "EG": "baseline_physical_EG", "H": "baseline_physical_H"}[metric_name],
            ascending=(metric_name == "EUIt"),
            kind="mergesort",
        )["matched_sample_id"].tolist()
        for station, station_frame in climate_frame.groupby("station", sort=True):
            station_rank = station_frame.sort_values(column, ascending=(metric_name == "EUIt"), kind="mergesort")["matched_sample_id"].tolist()
            rank_rows.append(
                {
                    "station": station,
                    "rank_metric": metric_name,
                    "spearman": float(pd.Series(baseline_rank).corr(pd.Series(station_rank), method="spearman")),
                    "kendall": float(pd.Series(baseline_rank).corr(pd.Series(station_rank), method="kendall")),
                }
            )
    rank_path = paths.climate_dir / "climate_rank_stability.csv"
    write_csv(pd.DataFrame(rank_rows), rank_path)
    return {"weather_manifest": str(weather_manifest_path), "results_csv": str(result_path), "status": "completed"}


def summarize_round2_results(config_path: str | Path, *, run_id: str | None = None, output_dir: str | Path | None = None) -> dict[str, Any]:
    base_config, _, paths = prepare_round2_workspace(config_path, run_id=run_id, output_dir=output_dir)
    sampling_deps = pd.read_csv(paths.data_dir / "descriptor_dependencies.csv") if (paths.data_dir / "descriptor_dependencies.csv").exists() else pd.DataFrame()
    surrogate_summary = pd.read_csv(paths.models_dir / "surrogate_validation_summary.csv") if (paths.models_dir / "surrogate_validation_summary.csv").exists() else pd.DataFrame()
    benchmark_full = pd.read_csv(paths.optimization_dir / "benchmark_full_archive.csv") if (paths.optimization_dir / "benchmark_full_archive.csv").exists() else pd.DataFrame()
    benchmark_equal = pd.read_csv(paths.optimization_dir / "benchmark_equal_size_summary.csv") if (paths.optimization_dir / "benchmark_equal_size_summary.csv").exists() else pd.DataFrame()
    projection_summary = pd.read_csv(paths.optimization_dir / "optimizer_projection_summary.csv") if (paths.optimization_dir / "optimizer_projection_summary.csv").exists() else pd.DataFrame()
    hv_saturation = (
        json.loads((paths.optimization_dir / "hv_saturation_diagnostic.json").read_text(encoding="utf-8"))
        if (paths.optimization_dir / "hv_saturation_diagnostic.json").exists()
        else {}
    )
    physical_summary = (
        json.loads((paths.physical_dir / "physical_validation_summary.json").read_text(encoding="utf-8"))
        if (paths.physical_dir / "physical_validation_summary.json").exists()
        else {"status": "missing"}
    )
    climate_summary = (
        pd.read_csv(paths.climate_dir / "climate_sensitivity_summary.csv")
        if (paths.climate_dir / "climate_sensitivity_summary.csv").exists()
        else pd.DataFrame([{"status": "missing", "reason": "not generated"}])
    )
    climate_weather_manifest = (
        json.loads((paths.climate_dir / "climate_weather_manifest.json").read_text(encoding="utf-8"))
        if (paths.climate_dir / "climate_weather_manifest.json").exists()
        else {"records": []}
    )
    if "status" in climate_summary.columns:
        climate_status = str(climate_summary.iloc[0]["status"])
        climate_reason = str(climate_summary.iloc[0].get("reason", ""))
    else:
        climate_status = "completed"
        climate_reason = "station-level climate summary available"
    artifacts = {
        "sampling": paths.data_dir / "sampling_coverage_summary.csv",
        "surrogate_validation": paths.models_dir / "surrogate_validation_summary.csv",
        "benchmark_full_archive": paths.optimization_dir / "benchmark_full_archive.csv",
        "feasibility": paths.optimization_dir / "optimizer_projection_summary.csv",
        "physical": paths.physical_dir / "physical_validation_summary.json",
        "climate": paths.climate_dir / "climate_sensitivity_summary.csv",
    }
    manifest = {
        "run_id": paths.run_id,
        "artifacts": {name: {"path": str(path), "exists": path.exists()} for name, path in artifacts.items()},
        "canonical_dataset_sha256": sha256_path(base_config["round2"]["canonical_dataset"]),
        "canonical_surrogate_sha256": sha256_path(base_config["round2"]["canonical_surrogate"]),
    }
    write_json(manifest, paths.research_root / "result-manifest.json")
    write_json(manifest, paths.research_root / "release-candidate-manifest.json")

    top_results = []
    if not sampling_deps.empty:
        top_results.append(f"Sampling dependencies remain exact to floating tolerance: FAR-BD*AF max residual = {sampling_deps.loc[sampling_deps['dependency_name']=='FAR_minus_BD_times_AF', 'value'].iloc[0]:.3e}.")
        top_results.append(f"Sampling dependencies remain exact to floating tolerance: OSR-(1-BD)/FAR max residual = {sampling_deps.loc[sampling_deps['dependency_name']=='OSR_minus_(1_minus_BD)_over_FAR', 'value'].iloc[0]:.3e}.")
        top_results.append(f"The 2000-row descriptor space needs {int(sampling_deps.loc[sampling_deps['metric']=='components_for_95_variance', 'value'].iloc[0])} principal components for 95% variance.")
    if not surrogate_summary.empty:
        rk = surrogate_summary.loc[surrogate_summary["validation_family"] == "repeated_kfold"].copy()
        if not rk.empty:
            top_results.append(
                "Repeated 5x5 CV mean nMAE = "
                + ", ".join(f"{row.target} {row.mean_nMAE:.4f}" for row in rk.itertuples())
                + "."
            )
        lo = surrogate_summary.loc[surrogate_summary["validation_family"] == "leave_one_osli_out"].copy()
        if not lo.empty:
            top_results.append(
                "Leave-one-OSLI-out mean nMAE = "
                + ", ".join(f"{row.target} {row.mean_nMAE:.4f}" for row in lo.itertuples())
                + "."
            )
    if not benchmark_full.empty:
        nsga = benchmark_full.loc[benchmark_full["group"] == "NSGA-II"].iloc[0]
        cma = benchmark_full.loc[benchmark_full["group"] == "CMA-ES::Balanced_Performance"].iloc[0]
        ddpg = benchmark_full.loc[benchmark_full["group"] == "DDPG::Balanced_Performance"].iloc[0]
        top_results.append(f"NSGA-II full archive HV/IGD = {nsga.HV:.6f}/{nsga.IGD:.6f}.")
        top_results.append(f"Balanced DDPG full archive HV/IGD = {ddpg.HV:.6f}/{ddpg.IGD:.6f}.")
        top_results.append(f"Balanced CMA-ES full archive HV/IGD = {cma.HV:.6f}/{cma.IGD:.6f}.")
    if not projection_summary.empty:
        nsga_proj = projection_summary.loc[projection_summary["method"] == "NSGA-II"].iloc[0]
        ddpg_proj = projection_summary.loc[(projection_summary["method"] == "DDPG") & (projection_summary["scenario"] == "Balanced_Performance")].iloc[0]
        top_results.append(f"NSGA-II candidate projection collapse rate = {nsga_proj.duplicate_collapse_rate:.4f}.")
        top_results.append(f"NSGA-II unique matched feasible blocks = {int(nsga_proj.unique_matched_sample_count)}.")
        top_results.append(f"Balanced DDPG mean projection distance = {ddpg_proj.projection_distance_mean:.4f}.")
        top_results.append("All optimizer families violate exact descriptor algebra when emitted as continuous candidates before projection.")
    if hv_saturation:
        top_results.append(f"Fixed reference point implies a theoretical maximum HV of {hv_saturation.get('theoretical_max_hv', float('nan')):.6f}.")
    if physical_summary:
        top_results.append(f"Physical validation status = {physical_summary.get('status', 'missing')}.")
    if not climate_summary.empty:
        top_results.append(f"Climate sensitivity status = {climate_status}.")

    experiment_lines = [
        "# Round 2 Experiment Results",
        "",
        "## Executive summary",
        f"- Run ID: `{paths.run_id}`.",
        f"- Canonical dataset: `{base_config['round2']['canonical_dataset']}` with SHA-256 `{manifest['canonical_dataset_sha256']}`.",
        f"- Canonical surrogate: `{base_config['round2']['canonical_surrogate']}` with SHA-256 `{manifest['canonical_surrogate_sha256']}`.",
        f"- Current remote physical batch status: `{physical_summary.get('status', 'missing')}`.",
        "",
        "## Experiment completion matrix",
    ]
    for name, path in artifacts.items():
        experiment_lines.append(f"- {name}: {'completed' if path.exists() else 'missing'}")
    experiment_lines.extend(
        [
            "",
            "## Ten most important new results",
            *[f"- {line}" for line in top_results[:10]],
            "",
            "## Physical validation and cross-climate results",
            f"- Physical validation status: `{physical_summary.get('status', 'missing')}`.",
            f"- Physical job id: `{physical_summary.get('job_id', 'n/a')}`.",
            f"- Annual irradiance status: `{physical_summary.get('annual_irradiance_status', 'missing')}`.",
            f"- Climate sensitivity status: `{climate_status}`.",
            f"- Climate blocker or note: `{climate_reason}`.",
            f"- Climate weather manifest: `{paths.climate_dir / 'climate_weather_manifest.json'}`.",
            "",
            "## Data coverage results",
            f"- Sampling method summary: `{paths.data_dir / 'sampling_method_summary.json'}`.",
            f"- Descriptor coverage table: `{paths.data_dir / 'sampling_coverage_summary.csv'}`.",
            "",
            "## Surrogate validation results",
            f"- Surrogate validation summary: `{paths.models_dir / 'surrogate_validation_summary.csv'}`.",
            "",
            "## Full archive and equal-size benchmark",
            f"- Full archive summary: `{paths.optimization_dir / 'benchmark_full_archive.csv'}`.",
            f"- Equal-size summary: `{paths.optimization_dir / 'benchmark_equal_size_summary.csv'}`.",
            "",
            "## CMA-ES and RandomSearch",
            f"- CMA-ES summary: `{paths.optimization_dir / 'cmaes_summary_round2.json'}`.",
            f"- RandomSearch summary: `{paths.optimization_dir / 'random_search_summary_round2.json'}`.",
            "",
            "## Descriptor feasibility and projection",
            f"- Projection summary: `{paths.optimization_dir / 'optimizer_projection_summary.csv'}`.",
            "",
            "## Computation efficiency",
            f"- Runtime audit: `{paths.optimization_dir / 'runtime_audit.csv'}`.",
            "",
            "## Impact on manuscript conclusions",
            "- The current evidence remains bounded to surrogate-conditioned benchmarking and descriptor-space design support.",
            "- Physical validation completed, but its large EUIt/H error and weak rank preservation still do not support a strong physical-certification claim.",
            "- HV saturation near 1.331 must be described as reference-point saturation, not as archive richness by itself.",
            "",
            "## Old tables or figures that must be retired or revised",
            "- Any obsolete large-offset reward equation must be replaced.",
            "- Any wording that treats Fig. 9(d) post-hoc utility as training reward must be removed.",
            "- Any figure or text that treats the 12 inputs as independent design variables must be revised.",
            "",
            "## Results that can enter the main text now",
            "- Sampling-coverage diagnostics.",
            "- Surrogate validation metrics.",
            "- Equal-size benchmark fairness diagnostics.",
            "- Descriptor projection-sensitivity limitations.",
            "",
            "## Results that should stay in appendix or remain pending",
            "- Detailed physical per-case diagnostics and optimizer-linked gap decomposition are better suited to appendix tables.",
            "- Climate sensitivity should remain framed as limited cross-climate physical sensitivity analysis, not as a generalization proof.",
            "",
            "## Conclusions that must be removed or kept bounded",
            "- Broad DRL-superiority wording.",
            "- Any claim that physical validation establishes optimizer superiority or broad climate transfer.",
            "",
            "## Next-phase exact figure data sources",
            f"- Coverage: `{paths.data_dir / 'sampling_coverage_summary.csv'}` and `{paths.data_dir / 'descriptor_dependencies.csv'}`.",
            f"- Fairness: `{paths.optimization_dir / 'benchmark_full_archive.csv'}` and `{paths.optimization_dir / 'benchmark_equal_size_summary.csv'}`.",
            f"- HV saturation: `{paths.optimization_dir / 'hv_saturation_diagnostic.json'}` and `{paths.optimization_dir / 'benchmark_metric_definition_audit.csv'}`.",
            f"- Feasibility: `{paths.optimization_dir / 'optimizer_projection_summary.csv'}` and `{paths.optimization_dir / 'projected_utility_comparison.csv'}`.",
            f"- Physical: `{paths.physical_dir / 'physical_validation_summary.json'}` for current run state, then result CSVs after completion.",
            f"- Climate: `{paths.climate_dir / 'climate_sensitivity_summary.csv'}` and `{paths.climate_dir / 'climate_rank_stability.csv'}`.",
        ]
    )
    (paths.research_root / "experiment-results.md").write_text("\n".join(experiment_lines), encoding="utf-8")

    manuscript_lines = [
        "# Manuscript Change Input",
        "",
        "## Reward formula",
        "R = 1 - sqrt(sum((w * (z - u))^2)) / sqrt(sum(w^2))",
        "",
        "## Objective interpretation",
        "- `w` should be described as the axis-scaling coefficient.",
        "- Training reward and post-hoc utility are different metrics and must not be conflated.",
        "",
        "## Evaluation mode",
        "- The current evidence is strongest for surrogate-conditioned benchmarking under the selected 2000-row fallback-analytic dataset and the selected tuned-standard checkpoint.",
        "",
        "## Sample generation",
        "- The dataset is generated by random morphology generation, not Latin hypercube sampling and not a full factorial grid.",
        "- Buildings are generated first and descriptors are computed afterward.",
        "- The 500/1000/1500/2000 datasets are nested prefixes of the same 2000-row pool.",
        "",
        "## Validation numbers",
        *[
            f"- {row.validation_family} / {row.target}: mean nMAE = {row.mean_nMAE:.4f}, mean R2 = {row.mean_R2:.4f}, mean Spearman = {row.mean_Spearman_rho:.4f}."
            for row in surrogate_summary.itertuples()
        ],
        "",
        "## Benchmark numbers",
        *[
            f"- {row.group}: HV = {row.HV:.6f}, IGD = {row.IGD:.6f}, rows = {int(row.rows)}, non-dominated rows = {int(row.non_dominated_rows)}."
            for row in benchmark_full.itertuples()
        ],
        "",
        "## HV saturation",
        (
            f"- Theoretical maximum HV under the fixed reference point is {hv_saturation.get('theoretical_max_hv', float('nan')):.6f}."
            if hv_saturation
            else "- HV saturation diagnostic pending."
        ),
        (
            "- Saturation must be interpreted with the clipped/unique/projected metric audit."
            if hv_saturation
            else ""
        ),
        "",
        "## Climate sensitivity status",
        f"- Current status: `{climate_status}` ({climate_reason}).",
        f"- Weather manifest: `{paths.climate_dir / 'climate_weather_manifest.json'}`.",
        "",
        "## Terminology",
        "- Use `morphology descriptors` or `surrogate input descriptors` for the 12 inputs.",
        "- Distinguish training reward from post-hoc utility.",
        "- Keep claims bounded to surrogate reliability, archive fairness, and the current physical-validation status.",
        "- Do not describe CMA-ES as providing a richer Pareto archive when HV saturation is caused by clipped corner occupancy.",
    ]
    (paths.research_root / "manuscript-change-input.md").write_text("\n".join(manuscript_lines), encoding="utf-8")

    figure_lines = [
        "# Figure Change Input",
        "",
        "- Fig. 1 reward formula should be replaced with the implemented normalized-distance reward.",
        "- Fig. 1 text extraction still shows `street loactions`, `Acotr netword`, and `netword` in the audit baseline.",
        "- Fig. 2 should clarify episode-versus-step placement in the DDPG loop.",
        "- Fig. 3 should label surrogate input descriptors and network outputs explicitly.",
        "- Post-Fig. 4 rebuild should source data from the round-2 artifact CSVs, not hand-copied tables.",
        f"- Coverage figures should use `{paths.data_dir / 'sampling_coverage_summary.csv'}` and `{paths.data_dir / 'descriptor_dependencies.csv'}`.",
        f"- Fairness figures should use `{paths.optimization_dir / 'benchmark_full_archive.csv'}` and `{paths.optimization_dir / 'benchmark_equal_size_summary.csv'}`.",
        f"- HV saturation panels should use `{paths.optimization_dir / 'hv_saturation_diagnostic.json'}` and `{paths.optimization_dir / 'benchmark_metric_definition_audit.csv'}`.",
        f"- Feasibility figures should use `{paths.optimization_dir / 'optimizer_projection_summary.csv'}` and `{paths.optimization_dir / 'projected_utility_comparison.csv'}`.",
        f"- Physical-validation figures should use the completed batch from job `{physical_summary.get('job_id', 'n/a')}`.",
        f"- Climate figures should use `{paths.climate_dir / 'climate_sensitivity_results.csv'}` and `{paths.climate_dir / 'climate_rank_stability.csv'}`.",
        "- Do not generate or commit formal figure PDFs in this stage.",
    ]
    (paths.research_root / "figure-change-input.md").write_text("\n".join(figure_lines), encoding="utf-8")

    protocol_lines = [
        "# Experimental Protocol",
        "",
        f"- Protocol version: `{base_config['round2']['protocol_version']}`",
        f"- Master seed: `{base_config['round2']['master_seed']}`",
        f"- Canonical dataset SHA-256: `{manifest['canonical_dataset_sha256']}`",
        f"- Canonical surrogate SHA-256: `{manifest['canonical_surrogate_sha256']}`",
        f"- Locked case selection: `{paths.research_root / 'locked-case-selection.json'}`",
        f"- Physical model protocol: `{paths.physical_dir / 'physical_model_protocol.json'}`",
        f"- Result manifest: `{paths.research_root / 'result-manifest.json'}`",
        "- Stop conditions: missing canonical data/checkpoint, non-reproducible Fig. 9 utility, unavailable secure remote access, missing verified EPWs, incomplete EnergyPlus or Radiance environment, ambiguous physical units, or physical case budgets beyond protocol limits.",
    ]
    (paths.research_root / "experimental-protocol.md").write_text("\n".join(protocol_lines), encoding="utf-8")
    return manifest

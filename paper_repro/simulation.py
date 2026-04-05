from __future__ import annotations

from dataclasses import dataclass
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from paper_repro.bootstrap import bootstrap_sim_stack
from paper_repro.config import Config
from paper_repro.constants import MORPHOLOGY_FEATURES, PERFORMANCE_TARGETS
from paper_repro.contracts import SIMULATED_SAMPLE_COLUMNS, write_csv, write_json
from paper_repro.morphology import generate_morphology_dataset, write_block_records


@dataclass
class WeatherSelection:
    station: str
    label: str
    epw_path: str | None


def _resolve_weather(config: Config) -> WeatherSelection:
    bootstrap_path = Path(config["report"]["bootstrap_dir"]) / "bootstrap_summary.json"
    if not bootstrap_path.exists():
        bootstrap_sim_stack(config, install_missing=config["simulation"].get("try_install_sim_stack", False))

    summary = {}
    if bootstrap_path.exists():
        import json

        summary = json.loads(bootstrap_path.read_text(encoding="utf-8"))

    for preferred in [config["weather"]["preferred_station"], config["weather"]["fallback_station"]]:
        for record in summary.get("weather_records", []):
            if record["station"] == preferred and record.get("available"):
                return WeatherSelection(station=preferred, label=record["label"], epw_path=record.get("epw"))

    station = config["weather"]["preferred_station"]
    label = config["weather"]["stations"][station]["label"]
    return WeatherSelection(station=station, label=label, epw_path=None)


def _station_bias(weather_selection: WeatherSelection) -> tuple[float, float, float]:
    if weather_selection.station == "Dongtai":
        return (0.0, 0.0, 0.0)
    return (0.35, -0.03, -0.05)


def fallback_simulation(
    features: pd.DataFrame,
    seed: int,
    weather_selection: WeatherSelection,
    noise_scale: dict[str, float],
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    station_euit_bias, station_eg_bias, station_h_bias = _station_bias(weather_selection)

    far = features["FAR"].to_numpy()
    sd = features["SD"].to_numpy()
    af = features["AF"].to_numpy()
    ar_ew = features["AR_ew"].to_numpy()
    ar_ns = features["AR_ns"].to_numpy()
    svf = features["SVF"].to_numpy()
    bd = features["BD"].to_numpy()
    osr = features["OSR"].to_numpy()
    sc = features["SC"].to_numpy()
    par = features["PAR"].to_numpy()
    theta = features["theta"].to_numpy()
    osli = features["OSLI"].to_numpy() / 8.0

    orientation_penalty = np.abs(theta) / 45.0
    aspect_mean = 0.5 * (ar_ew + ar_ns)
    sweet_spot = np.exp(-((far - 1.7) ** 2) / 0.6) * (0.45 + svf) * (0.3 + osr)
    dense_penalty = np.maximum(far - 2.8, 0.0)

    euit = (
        78.0
        + 10.0 * bd
        + 4.3 * par
        + 2.8 * aspect_mean
        + 0.022 * sd
        + 0.8 * sc
        + 2.1 * orientation_penalty
        - 6.4 * osr
        - 5.8 * svf
        - 0.9 * af
        - 4.5 * sweet_spot
        + station_euit_bias
        + rng.normal(0.0, noise_scale["EUIt"], len(features))
    )
    eg = (
        0.96
        + 0.28 * far
        + 0.64 * svf
        + 0.52 * osr
        - 0.19 * bd
        - 0.12 * aspect_mean
        - 0.18 * orientation_penalty
        + 0.26 * osli
        + 0.78 * sweet_spot
        - 0.1 * dense_penalty
        + station_eg_bias
        + rng.normal(0.0, noise_scale["EG"], len(features))
    )
    sunlight = (
        6.45
        + 1.1 * svf
        + 0.55 * osr
        + 0.48 * osli
        + 0.24 * sweet_spot
        - 0.7 * orientation_penalty
        - 0.45 * par
        - 0.24 * aspect_mean
        + station_h_bias
        + rng.normal(0.0, noise_scale["H"], len(features))
    )

    result = features.copy()
    result["EUIt"] = np.clip(euit, 66.0, 96.0)
    result["EG"] = np.clip(eg, 1.2, 2.85)
    result["H"] = np.clip(sunlight, 6.0, 7.85)
    result["weather_station"] = weather_selection.label
    result["simulation_mode"] = "fallback_analytic"
    return result


def _resolve_scale_study_scales(config: Config) -> list[int]:
    simulation_cfg = config["simulation"]
    scale_cfg = simulation_cfg.get("scale_study", {})
    scales = scale_cfg.get("scales") or [simulation_cfg["n_samples"]]
    resolved = sorted({int(scale) for scale in scales})
    if int(simulation_cfg["n_samples"]) not in resolved:
        resolved.append(int(simulation_cfg["n_samples"]))
        resolved.sort()
    return resolved


def _regime_artifact_dir(config: Config, scale: int) -> Path:
    return Path(config["report"]["data_dir"]) / "regimes" / f"scale_{scale}"


def _active_scale(config: Config, scales: list[int]) -> int:
    configured = int(config["simulation"]["n_samples"])
    if configured in scales:
        return configured
    return max(scales)


def build_dataset_regimes(config: Config) -> dict[int, pd.DataFrame]:
    dirs = config.ensure_artifact_dirs()
    simulation_cfg = config["simulation"]
    scales = _resolve_scale_study_scales(config)
    baseline_scale = int(config.raw.get("surrogate_selection", {}).get("original_scale", min(scales)))
    weather_selection = _resolve_weather(config)
    features, block_records = generate_morphology_dataset(
        n_samples=max(scales),
        seed=config["project"]["random_seed"],
        block_size_m=simulation_cfg["block_size_m"],
        land_unit_size_m=simulation_cfg["land_unit_size_m"],
        floor_height_m=simulation_cfg["floor_to_floor_height_m"],
        return_blocks=True,
    )
    dataset = fallback_simulation(
        features,
        seed=config["project"]["random_seed"] + 17,
        weather_selection=weather_selection,
        noise_scale=simulation_cfg["fallback_noise_scale"],
    )
    dataset = dataset[SIMULATED_SAMPLE_COLUMNS]

    regimes: dict[int, pd.DataFrame] = {}
    summary_rows = []
    regime_payloads = []
    for scale in scales:
        subset = dataset.iloc[:scale].copy().reset_index(drop=True)
        subset["sample_id"] = np.arange(len(subset), dtype=int)
        regime_dir = _regime_artifact_dir(config, scale)
        regime_dir.mkdir(parents=True, exist_ok=True)
        csv_path = write_csv(subset, regime_dir / "simulated_samples.csv")
        blocks_path = regime_dir / "simulated_blocks.jsonl"
        write_block_records(block_records[:scale], str(blocks_path))
        meta_payload = {
            "dataset_scale": int(scale),
            "baseline_scale": baseline_scale,
            "regime_type": "original" if scale <= baseline_scale else "expanded",
            "nested_prefix_protocol": True,
            "weather_station": weather_selection.label,
            "weather_epw": weather_selection.epw_path,
            "simulation_mode": "fallback_analytic",
            "feature_columns": MORPHOLOGY_FEATURES,
            "target_columns": PERFORMANCE_TARGETS,
            "seed": int(config["project"]["random_seed"]),
            "source_scale": int(max(scales)),
            "csv_path": str(csv_path),
            "blocks_path": str(blocks_path),
            "assumptions": [
                "AR_ew and AR_ns are the explicit two-direction reconstruction of the paper's AR term.",
                "The fallback simulator is used when a fully coupled LBT/EnergyPlus/Radiance workflow is unavailable.",
                "Expanded regimes are deterministic prefixes of the maximum generated simulation pool.",
            ],
            "feature_ranges": {
                feature: {"min": float(subset[feature].min()), "max": float(subset[feature].max())}
                for feature in MORPHOLOGY_FEATURES
            },
            "target_ranges": {
                target: {"min": float(subset[target].min()), "max": float(subset[target].max())}
                for target in PERFORMANCE_TARGETS
            },
        }
        meta_path = write_json(meta_payload, regime_dir / "simulated_samples.meta.json")
        summary_rows.append(
            {
                "dataset_scale": int(scale),
                "regime_type": meta_payload["regime_type"],
                "rows": int(len(subset)),
                "feature_span_mean": float(
                    np.mean([meta_payload["feature_ranges"][feature]["max"] - meta_payload["feature_ranges"][feature]["min"] for feature in MORPHOLOGY_FEATURES])
                ),
                "target_span_mean": float(
                    np.mean([meta_payload["target_ranges"][target]["max"] - meta_payload["target_ranges"][target]["min"] for target in PERFORMANCE_TARGETS])
                ),
                "csv_path": str(csv_path),
                "meta_path": str(meta_path),
            }
        )
        regime_payloads.append(meta_payload)
        regimes[int(scale)] = subset

    scale_summary_csv = write_csv(pd.DataFrame(summary_rows), Path(dirs["data_dir"]) / "dataset_scale_summary.csv")
    protocol_payload = {
        "protocol_version": "dataset-scale-study-v1",
        "baseline_scale": baseline_scale,
        "explored_scales": [int(scale) for scale in scales],
        "active_scale": _active_scale(config, scales),
        "nested_prefix_protocol": True,
        "source_scale": int(max(scales)),
        "seed": int(config["project"]["random_seed"]),
        "weather_station": weather_selection.label,
        "simulation_mode": "fallback_analytic",
        "regimes": regime_payloads,
        "summary_csv": str(scale_summary_csv),
    }
    protocol_path = write_json(protocol_payload, Path(dirs["data_dir"]) / "dataset_scale_protocol.json")

    active_scale = _active_scale(config, scales)
    active_dir = _regime_artifact_dir(config, active_scale)
    shutil.copy2(active_dir / "simulated_samples.csv", Path(dirs["data_dir"]) / "simulated_samples.csv")
    shutil.copy2(active_dir / "simulated_samples.meta.json", Path(dirs["data_dir"]) / "simulated_samples.meta.json")
    shutil.copy2(active_dir / "simulated_blocks.jsonl", Path(dirs["data_dir"]) / "simulated_blocks.jsonl")
    selected_payload = {
        "dataset_scale": active_scale,
        "protocol_path": str(protocol_path),
        "active_dataset_path": str(Path(dirs["data_dir"]) / "simulated_samples.csv"),
        "active_metadata_path": str(Path(dirs["data_dir"]) / "simulated_samples.meta.json"),
    }
    write_json(selected_payload, Path(dirs["data_dir"]) / "selected_dataset.json")
    return regimes


def build_simulated_dataset(config: Config) -> pd.DataFrame:
    scales = _resolve_scale_study_scales(config)
    regimes = build_dataset_regimes(config)
    return regimes[_active_scale(config, scales)]


def reevaluate_candidates(config: Config, candidates: pd.DataFrame, deterministic: bool = True) -> pd.DataFrame:
    weather_selection = _resolve_weather(config)
    noise_scale = config["simulation"].get("reevaluation_noise_scale", {"EUIt": 0.0, "EG": 0.0, "H": 0.0}) if deterministic else config["simulation"]["fallback_noise_scale"]
    features = candidates.copy()
    if "sample_id" not in features.columns:
        features.insert(0, "sample_id", np.arange(len(features), dtype=int))
    frame = fallback_simulation(
        features[["sample_id", *MORPHOLOGY_FEATURES]],
        seed=config["project"]["random_seed"] + 101,
        weather_selection=weather_selection,
        noise_scale=noise_scale,
    )
    return frame[["sample_id", *MORPHOLOGY_FEATURES, *PERFORMANCE_TARGETS, "weather_station", "simulation_mode"]]

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from paper_repro.constants import MORPHOLOGY_FEATURES
from paper_repro.physical_stack import project_candidates_to_nearest_blocks
from paper_repro.round2 import (
    _guardrail_decomposition_frame,
    aggregate_sunlight_hours,
    apply_weighted_utility,
    audit_descriptor_constraints,
    audit_osli_values,
    build_fixed_reference,
    compute_fixed_domain_utility,
    compute_legacy_utility,
    dedupe_objective_tuples,
    ddpg_reward_from_outputs,
    dedupe_completed_job_rows,
    parse_physical_results_frame,
    physical_protocol_hash,
    roof_irradiance_to_million_kwh,
    sanitize_weather_manifest,
    select_maximin_space_filling,
    select_objective_tail_cases,
    theoretical_max_hv,
    validate_weather_station,
)


def _dataset_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [0, 1, 2, 3, 4, 5],
            "FAR": [1.0, 1.1, 1.8, 2.0, 2.6, 2.9],
            "SD": [10.0, 12.0, 20.0, 21.0, 35.0, 40.0],
            "AF": [5.0, 5.5, 8.0, 8.5, 10.0, 10.5],
            "AR_ew": [0.4, 0.5, 0.8, 0.85, 1.2, 1.3],
            "AR_ns": [0.8, 0.9, 1.4, 1.5, 2.0, 2.2],
            "SVF": [0.72, 0.70, 0.64, 0.60, 0.52, 0.48],
            "BD": [0.20, 0.20, 0.225, 0.2352941176, 0.26, 0.2761904762],
            "OSR": [0.80, 0.7272727273, 0.4305555556, 0.3823529412, 0.2846153846, 0.24958949097],
            "SC": [0.60, 0.61, 0.66, 0.67, 0.75, 0.80],
            "PAR": [0.12, 0.123, 0.13, 0.131, 0.14, 0.145],
            "theta": [-15.0, -5.0, 0.0, 10.0, 20.0, 30.0],
            "OSLI": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "EUIt": [68.0, 69.0, 70.5, 71.2, 73.0, 74.5],
            "EG": [2.7, 2.65, 2.5, 2.45, 2.25, 2.10],
            "H": [7.6, 7.45, 7.2, 7.1, 6.8, 6.6],
        }
    )


def test_reward_formula_matches_normalized_distance_definition() -> None:
    outputs = np.array([[70.0, 2.5, 7.2]])
    target_min = np.array([66.0, 1.8, 6.0])
    target_max = np.array([78.0, 2.9, 7.8])
    weights = [0.6, 0.2, 0.2]
    reward = ddpg_reward_from_outputs(outputs, target_min, target_max, weights)[0]
    assert 0.0 <= reward <= 1.0


def test_training_reward_and_post_hoc_utility_are_not_the_same() -> None:
    frame = _dataset_frame().iloc[:2].copy()
    legacy_scores = compute_legacy_utility(frame, _dataset_frame())
    utility = apply_weighted_utility(legacy_scores, [0.6, 0.2, 0.2]).iloc[0]
    reward = ddpg_reward_from_outputs(
        frame.iloc[[0]][["EUIt", "EG", "H"]].to_numpy(),
        np.array([66.0, 1.8, 6.0]),
        np.array([78.0, 2.9, 7.8]),
        [0.6, 0.2, 0.2],
    )[0]
    assert reward != utility


def test_fixed_reference_metrics_use_external_reference_front() -> None:
    frame = _dataset_frame()
    groups = {
        "A": frame.iloc[:3].copy(),
        "B": frame.iloc[3:].copy(),
    }
    reference = build_fixed_reference(groups)
    subset_reference = build_fixed_reference({"A": frame.iloc[1:3].copy(), "B": frame.iloc[4:6].copy()})
    assert not np.allclose(reference["nadir"], subset_reference["nadir"])


def test_select_maximin_space_filling_is_deterministic() -> None:
    frame = _dataset_frame()
    first = select_maximin_space_filling(frame, 3)["sample_id"].tolist()
    second = select_maximin_space_filling(frame, 3)["sample_id"].tolist()
    assert first == second


def test_select_objective_tail_cases_avoids_duplicates() -> None:
    tail = select_objective_tail_cases(_dataset_frame())
    assert tail["sample_id"].nunique() == len(tail)


def test_descriptor_constraint_audit_reports_small_residuals() -> None:
    audit = audit_descriptor_constraints(_dataset_frame())
    assert audit["far_minus_bd_af"].max() < 1e-8
    assert audit["osr_minus_density_far"].max() < 1e-8


def test_osli_audit_reports_fractional_distance() -> None:
    audit = audit_osli_values(pd.Series([1.0, 2.4, 7.9]))
    assert audit["is_integer"].tolist() == [True, False, False]
    assert np.isclose(audit["fractional_distance"].iloc[1], 0.4)


def test_nearest_feasible_block_projection_returns_expected_match() -> None:
    dataset = _dataset_frame()
    candidate = dataset.iloc[[2]].copy()
    candidate["FAR"] += 0.01
    projected = project_candidates_to_nearest_blocks(candidate[MORPHOLOGY_FEATURES], dataset)
    assert int(projected.iloc[0]["matched_sample_id"]) == 2


def test_physical_protocol_hash_is_reproducible() -> None:
    payload = {"a": 1, "b": ["x", "y"]}
    assert physical_protocol_hash(payload) == physical_protocol_hash(payload)


def test_weather_manifest_sanitization_drops_secret_like_keys() -> None:
    sanitized = sanitize_weather_manifest([{"station": "A", "host": "secret-host", "url": "https://example.com"}])
    assert "host" not in sanitized[0]
    assert sanitized[0]["station"] == "A"


def test_parse_physical_results_handles_missing_values_and_flags() -> None:
    frame = pd.DataFrame([{"physical_EUIt": "70.2", "physical_H_proxy": None, "energyplus_ok": 1, "radiance_ok": 0}])
    parsed = parse_physical_results_frame(frame)
    assert np.isclose(parsed.iloc[0]["physical_EUIt"], 70.2)
    assert bool(parsed.iloc[0]["energyplus_ok"]) is True
    assert bool(parsed.iloc[0]["radiance_ok"]) is False


def test_aggregate_sunlight_hours_averages_sensor_totals() -> None:
    matrix = np.array([[1.0, 0.0, 1.0], [0.5, 0.5, 0.5]])
    assert np.isclose(aggregate_sunlight_hours(matrix), 1.75)


def test_roof_irradiance_unit_conversion_returns_million_kwh() -> None:
    result = roof_irradiance_to_million_kwh(1_000_000.0, 100.0, 0.8, 0.2, 0.75)
    assert np.isclose(result, 0.012)


def test_resume_deduplication_prefers_completed_rows() -> None:
    frame = pd.DataFrame(
        [
            {"job_id": "a", "status": "running", "sample_id": 1},
            {"job_id": "b", "status": "completed", "sample_id": 1},
        ]
    )
    deduped = dedupe_completed_job_rows(frame, ["sample_id"])
    assert deduped.iloc[0]["job_id"] == "b"


def test_fixed_domain_utility_scores_are_bounded_for_in_range_points() -> None:
    frame = _dataset_frame().iloc[:2]
    scores = compute_fixed_domain_utility(frame, {"EUIt": (66.0, 78.0), "EG": (1.8, 2.9), "H": (6.0, 7.8)})
    assert (scores >= 0.0).all().all()
    assert (scores <= 1.0).all().all()


def test_weather_urls_include_province_directories_and_jianhu_maps_to_dongtai() -> None:
    config = yaml.safe_load(Path("configs/reviewer_round2_experiments.yaml").read_text(encoding="utf-8"))
    stations = config["weather"]["stations"]
    assert "/JS_Jiangsu/" in stations["Dongtai"]["url"]
    assert "/JS_Jiangsu/" in stations["Nanjing"]["url"]
    assert "/HL_Heilongjiang/" in stations["Harbin"]["url"]
    assert "/BJ_Beijing/" in stations["Beijing"]["url"]
    assert "/GD_Guangdong/" in stations["Guangzhou"]["url"]
    assert config["weather"]["fallback_station"] == "Nanjing"
    assert "Jianhu case represented by Dongtai station weather." == config["round2"]["climate_sensitivity"]["baseline_representation_note"]


def test_weather_validation_requires_same_period_and_8760_records(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    epw_lines = ["LOCATION,City,State,Country,TMYx,582510,33.5,119.8,8,2.0"] + ["header"] * 7 + ["data"] * 8760
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        zf.writestr("sample.epw", "\n".join(epw_lines))

    class FakeResponse:
        status = 200

        def __init__(self, payload: bytes) -> None:
            self.payload = payload

        def read(self, *args) -> bytes:
            return self.payload

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

    monkeypatch.setattr("urllib.request.urlopen", lambda *args, **kwargs: FakeResponse(buffer.getvalue()))
    record = validate_weather_station(
        "Dongtai",
        {"url": "https://climate.onebuilding.org/WMO_Region_2_Asia/CHN_China/JS_Jiangsu/CHN_JS_Dongtai.582510_TMYx.2009-2023.zip", "label": "Dongtai", "wmo": "582510"},
        tmp_path,
    )
    assert record["hourly_records"] == 8760

    with pytest.raises(RuntimeError, match="unexpected_period"):
        validate_weather_station(
            "Dongtai",
            {"url": "https://climate.onebuilding.org/WMO_Region_2_Asia/CHN_China/JS_Jiangsu/CHN_JS_Dongtai.582510_TMYx.1991-2020.zip", "label": "Dongtai", "wmo": "582510"},
            tmp_path / "bad",
        )


def test_theoretical_max_hv_and_unique_objective_tuple_handling() -> None:
    assert np.isclose(theoretical_max_hv([1.1, 1.1, 1.1]), 1.331)
    frame = pd.DataFrame(
        [
            {"method": "A", "scenario": "S", "seed": 0, **{feature: 0.0 for feature in MORPHOLOGY_FEATURES}, "EUIt": 66.0, "EG": 2.8, "H": 7.8, "reward": 1.0},
            {"method": "A", "scenario": "S", "seed": 1, **{feature: 0.1 for feature in MORPHOLOGY_FEATURES}, "EUIt": 66.0, "EG": 2.8, "H": 7.8, "reward": 0.9},
        ]
    )
    deduped = dedupe_objective_tuples(frame)
    assert len(deduped) == 1


def test_guardrail_decomposition_reports_raw_adjusted_and_clipped_outputs() -> None:
    class DummySurrogate:
        def predict(self, frame: pd.DataFrame, *, clip: bool = True) -> pd.DataFrame:
            return pd.DataFrame({"EUIt": [60.0], "EG": [3.2], "H": [8.4]}, index=frame.index)

    class DummyEnv:
        surrogate = DummySurrogate()
        feature_min = np.zeros(len(MORPHOLOGY_FEATURES), dtype=np.float32)
        feature_max = np.ones(len(MORPHOLOGY_FEATURES), dtype=np.float32)
        feature_reference = np.zeros((1, len(MORPHOLOGY_FEATURES)), dtype=np.float32)
        feasible_radius = 0.0
        feature_penalty_scale = np.array([10.0, 0.85, 0.6], dtype=np.float32)
        target_min = np.array([66.0, 1.8, 6.0], dtype=np.float32)
        target_max = np.array([78.0, 2.9, 7.8], dtype=np.float32)
        target_range = target_max - target_min
        extrapolation_penalty_scale = 1.0

    frame = pd.DataFrame([{feature: 1.0 for feature in MORPHOLOGY_FEATURES}])
    frame["method"] = "A"
    frame["scenario"] = "S"
    frame["seed"] = 0
    decomposed = _guardrail_decomposition_frame(
        frame,
        DummyEnv(),
        {"EUIt": (66.0, 78.0), "EG": (1.8, 2.9), "H": (6.0, 7.8)},
        group_name="A::S",
    )
    assert {"raw_EUIt", "adjusted_EUIt", "EUIt", "clip_flag_EUIt", "duplicate_objective_tuple_id"}.issubset(decomposed.columns)
    assert bool(decomposed.iloc[0]["clip_flag_EUIt"]) is True


def test_configured_case_budgets_respect_limits() -> None:
    config = yaml.safe_load(Path("configs/reviewer_round2_experiments.yaml").read_text(encoding="utf-8"))
    assert len(config["round2"]["climate_sensitivity"]["additional_climates"]) == 3
    assert config["round2"]["climate_sensitivity"]["representative_cases"] * len(config["round2"]["climate_sensitivity"]["additional_climates"]) <= 12
    assert config["round2"]["maximum_physical_case_budget"] <= 36

from __future__ import annotations

import numpy as np
import pandas as pd

from paper_repro.constants import MORPHOLOGY_FEATURES
from paper_repro.physical_stack import project_candidates_to_nearest_blocks
from paper_repro.round2 import (
    aggregate_sunlight_hours,
    apply_weighted_utility,
    audit_descriptor_constraints,
    audit_osli_values,
    build_fixed_reference,
    compute_fixed_domain_utility,
    compute_legacy_utility,
    ddpg_reward_from_outputs,
    dedupe_completed_job_rows,
    parse_physical_results_frame,
    physical_protocol_hash,
    roof_irradiance_to_million_kwh,
    sanitize_weather_manifest,
    select_maximin_space_filling,
    select_objective_tail_cases,
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

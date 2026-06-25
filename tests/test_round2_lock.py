from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from paper_repro.constants import MORPHOLOGY_FEATURES
from paper_repro.round2 import build_fixed_reference, evaluate_archive_metrics, theoretical_max_hv
from paper_repro.round2_lock import (
    _build_canonical_metric_rows,
    build_equal_size_tables,
    build_projected_metric_rows,
    canonical_reference_hash,
    validate_result_registry,
)


def _dataset_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "sample_id": [0, 1, 2, 3, 4, 5],
            "FAR": [1.00, 1.10, 1.80, 2.00, 2.60, 2.90],
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


def _archive_frame(method: str, scenario: str, rows: pd.DataFrame, *, seed: int = 0) -> pd.DataFrame:
    frame = rows.copy()
    frame["method"] = method
    frame["scenario"] = scenario
    frame["seed"] = seed
    frame["reward"] = np.linspace(1.0, 0.1, len(frame))
    return frame[["method", "scenario", "seed", *MORPHOLOGY_FEATURES, "EUIt", "EG", "H", "reward"]]


def test_equal_size_tables_skip_oversized_requests_and_keep_effective_size() -> None:
    dataset = _dataset_frame()
    groups = {
        "DDPG::Balanced_Performance": _archive_frame("DDPG", "Balanced_Performance", dataset.iloc[:2]),
        "NSGA-II": _archive_frame("NSGA-II", "NSGA-II", dataset.iloc[2:6]),
    }
    reference = build_fixed_reference(groups)
    repetitions, summary = build_equal_size_tables(groups, reference, sizes=[2, 4], repetitions=4, master_seed=11)
    ddpg_valid = repetitions.loc[repetitions["group"] == "DDPG::Balanced_Performance"]
    assert set(ddpg_valid["requested_sample_size"]) == {2}
    assert set(ddpg_valid["effective_sample_size"]) == {2}
    ddpg_invalid = summary.loc[
        (summary["group"] == "DDPG::Balanced_Performance") & (summary["requested_sample_size"] == 4)
    ].iloc[0]
    assert ddpg_invalid["status"] == "not_applicable"
    assert pd.isna(ddpg_invalid["HV_mean"])
    nsga_valid = summary.loc[(summary["group"] == "NSGA-II") & (summary["requested_sample_size"] == 4)].iloc[0]
    assert nsga_valid["effective_sample_size"] == 4


def test_projected_metric_rows_use_supplied_fixed_reference() -> None:
    dataset = _dataset_frame()
    ddpg_source = dataset.iloc[[0, 1]].assign(
        FAR=[1.02, 1.08],
        SD=[10.2, 11.8],
        EUIt=[66.2, 66.4],
        EG=[2.88, 2.86],
        H=[7.78, 7.76],
    )
    nsga_source = dataset.iloc[[2, 3, 4, 5]].assign(
        FAR=[1.78, 2.03, 2.62, 2.88],
        EUIt=[67.0, 67.2, 67.4, 67.6],
        EG=[2.82, 2.80, 2.78, 2.76],
        H=[7.74, 7.72, 7.70, 7.68],
    )
    ddpg_rows = _archive_frame("DDPG", "Balanced_Performance", ddpg_source)
    nsga_rows = _archive_frame("NSGA-II", "NSGA-II", nsga_source)
    groups = {
        "DDPG::Balanced_Performance": ddpg_rows,
        "NSGA-II": nsga_rows,
    }
    fixed_reference = build_fixed_reference(groups)
    projected_groups, _, projected_metrics = build_projected_metric_rows(dataset, groups, fixed_reference)
    local_reference = build_fixed_reference(projected_groups)
    local_metrics = evaluate_archive_metrics(projected_groups, local_reference)
    fixed_value = float(projected_metrics.loc[projected_metrics["group"] == "DDPG::Balanced_Performance", "HV"].iloc[0])
    local_value = float(local_metrics.loc[local_metrics["group"] == "DDPG::Balanced_Performance", "HV"].iloc[0])
    assert not np.isclose(fixed_value, local_value)


def test_canonical_metric_rows_share_one_reference_hash() -> None:
    dataset = _dataset_frame()
    groups = {
        "DDPG::Balanced_Performance": _archive_frame("DDPG", "Balanced_Performance", dataset.iloc[:2]),
        "NSGA-II": _archive_frame("NSGA-II", "NSGA-II", dataset.iloc[2:6]),
    }
    reference = build_fixed_reference(groups)
    reference_hash = canonical_reference_hash(reference["reference_front"])
    full_archive = evaluate_archive_metrics(groups, reference)
    unique_groups = {group: frame.drop_duplicates(subset=["EUIt", "EG", "H"]).reset_index(drop=True) for group, frame in groups.items()}
    unique_metrics = evaluate_archive_metrics(unique_groups, reference)
    projected_groups, projected_metadata, projected_metrics = build_projected_metric_rows(dataset, groups, reference)
    _, equal_size_summary = build_equal_size_tables(groups, reference, sizes=[2], repetitions=3, master_seed=3)
    canonical_rows = _build_canonical_metric_rows(
        groups,
        reference_hash,
        full_archive,
        unique_metrics,
        projected_groups,
        projected_metadata,
        projected_metrics,
        equal_size_summary,
    )
    assert canonical_rows.loc[canonical_rows["status"] == "valid", "reference_hash"].nunique() == 1


def test_validate_result_registry_checks_source_sha_and_boundaries(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("canonical", encoding="utf-8")
    good_entry = {
        "result_id": "sample.coverage",
        "source_file": str(source),
        "source_sha256": __import__("hashlib").sha256(source.read_bytes()).hexdigest(),
        "supersedes": [],
        "valid_for_main_text": True,
        "value": "weak",
    }
    bad_sha_entry = dict(good_entry, result_id="bad.sha", source_sha256="x" * 64)
    bad_superseded_entry = dict(good_entry, result_id="bad.superseded", value_status="superseded")
    bad_physical_entry = dict(
        good_entry,
        result_id="physical_evidence.metric_agreement",
        value="strong validation",
    )
    issues = validate_result_registry([good_entry, bad_sha_entry, bad_superseded_entry, bad_physical_entry])
    assert any("sha mismatch for bad.sha" in issue for issue in issues)
    assert any("superseded result cannot be valid_for_main_text: bad.superseded" in issue for issue in issues)
    assert any("forbidden physical wording in registry: physical_evidence.metric_agreement" in issue for issue in issues)


def test_reference_hash_is_stable_and_theoretical_hv_matches_reference_point() -> None:
    reference_front = np.asarray([[0.0, 0.1, 0.2], [0.3, 0.4, 0.5], [0.0, 0.1, 0.2]], dtype=float)
    assert canonical_reference_hash(reference_front) == canonical_reference_hash(reference_front.copy())
    assert np.isclose(theoretical_max_hv([1.1, 1.1, 1.1]), 1.331)

from __future__ import annotations

import pandas as pd

from paper_repro.surrogate import select_best_surrogate_record, select_final_scale_record


def test_select_best_surrogate_record_prefers_lower_error_then_higher_r2() -> None:
    frame = pd.DataFrame(
        [
            {
                "candidate": "a",
                "mean_target_nmae": 0.10,
                "mean_tail_nmae": 0.08,
                "mean_r2": 0.92,
                "selection_objective": 0.12,
            },
            {
                "candidate": "b",
                "mean_target_nmae": 0.10,
                "mean_tail_nmae": 0.07,
                "mean_r2": 0.90,
                "selection_objective": 0.13,
            },
            {
                "candidate": "c",
                "mean_target_nmae": 0.11,
                "mean_tail_nmae": 0.05,
                "mean_r2": 0.99,
                "selection_objective": 0.12,
            },
        ]
    )
    selected = select_best_surrogate_record(frame)
    assert selected["candidate"] == "b"


def test_select_final_scale_record_prefers_smallest_eligible_expanded_scale() -> None:
    regime_best = pd.DataFrame(
        [
            {
                "dataset_scale": 500,
                "candidate": "baseline",
                "mean_target_nmae": 0.20,
                "mean_tail_nmae": 0.18,
                "mean_r2": 0.70,
                "selection_objective": 0.24,
            },
            {
                "dataset_scale": 1000,
                "candidate": "expanded_small",
                "mean_target_nmae": 0.11,
                "mean_tail_nmae": 0.10,
                "mean_r2": 0.88,
                "selection_objective": 0.135,
            },
            {
                "dataset_scale": 2000,
                "candidate": "expanded_large",
                "mean_target_nmae": 0.10,
                "mean_tail_nmae": 0.09,
                "mean_r2": 0.89,
                "selection_objective": 0.122,
            },
        ]
    )
    selected, details = select_final_scale_record(regime_best, original_scale=500, tolerance_pct=0.10)
    assert selected["dataset_scale"] == 1000
    assert details["mode"] == "smallest_within_primary_tolerance"

from __future__ import annotations

import pandas as pd

from paper_repro.metrics import compute_hv_igd_by_method, normalized_benefit_frame, summarize_surrogate_predictions


def test_metric_helpers_return_expected_columns() -> None:
    frame = pd.DataFrame(
        {
            "method": ["A", "A", "B", "B"],
            "scenario": ["A", "A", "B", "B"],
            "EUIt": [70.0, 71.0, 72.0, 73.0],
            "EG": [2.4, 2.3, 2.2, 2.1],
            "H": [7.3, 7.2, 7.1, 7.0],
        }
    )
    hv_igd = compute_hv_igd_by_method(frame)
    assert {"method", "HV", "IGD"} == set(hv_igd.columns)
    normalized = normalized_benefit_frame(frame)
    assert {"EUIt_score", "EG_score", "H_score"}.issubset(normalized.columns)


def test_surrogate_prediction_summary_reports_target_and_tail_metrics() -> None:
    cv_predictions = pd.DataFrame(
        {
            "sample_id": [0, 1, 2, 3],
            "true_EUIt": [70.0, 74.0, 78.0, 82.0],
            "pred_EUIt": [71.0, 73.0, 79.0, 81.0],
            "true_EG": [1.3, 1.6, 1.9, 2.2],
            "pred_EG": [1.35, 1.55, 1.95, 2.1],
            "true_H": [6.2, 6.6, 7.0, 7.4],
            "pred_H": [6.1, 6.7, 6.95, 7.45],
        }
    )
    summary = summarize_surrogate_predictions(cv_predictions)
    assert {"aggregate", "per_target", "quantiles"} == set(summary)
    assert len(summary["per_target"]) == 3
    assert summary["aggregate"]["mean_target_nmae"] >= 0.0
    assert summary["aggregate"]["mean_tail_nmae"] >= 0.0

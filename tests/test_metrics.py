from __future__ import annotations

import pandas as pd

from paper_repro.metrics import compute_hv_igd_by_method, normalized_benefit_frame


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

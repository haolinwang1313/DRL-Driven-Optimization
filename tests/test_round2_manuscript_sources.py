from pathlib import Path

from tools.build_round2_manuscript_sources import check_sources


def test_round2_manuscript_sources_pass_quality_checks() -> None:
    root = Path(__file__).resolve().parents[1]
    assert check_sources(root) == []

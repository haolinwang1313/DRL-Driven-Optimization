from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pandas as pd
import pytest

from paper_repro.publication_figures_round2 import (
    _load_package_frames,
    _load_package_manifest,
    _sha256_path,
    build_round2_figure_data_package,
    build_round2_revision_figures,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "paper" / "manuscript" / "figure_data" / "round2"
OUTPUT_ROOT = REPO_ROOT / "paper" / "manuscript" / "figures" / "round2_candidate"


def _pdf_text(path: Path) -> str:
    completed = subprocess.run(
        ["pdftotext", str(path), "-"],
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout


@pytest.fixture(scope="session")
def built_round2_assets() -> dict:
    build_round2_figure_data_package(DATA_ROOT, repo_root=REPO_ROOT, strict=True)
    return build_round2_revision_figures(
        DATA_ROOT,
        OUTPUT_ROOT,
        build_gallery=True,
        strict=True,
        overwrite=True,
        repo_root=REPO_ROOT,
    )


def test_figure_builder_rejects_superseded_source_files(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    csv_path = data_root / "dummy.csv"
    pd.DataFrame([{"value": 1}]).to_csv(csv_path, index=False)
    manifest = {
        "files": [
            {
                "file_name": "dummy.csv",
                "sha256": _sha256_path(csv_path),
                "status": "superseded",
            }
        ]
    }
    (data_root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="status != valid"):
        _load_package_frames(data_root, manifest, strict=True)


def test_figure_builder_verifies_source_sha(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    csv_path = data_root / "dummy.csv"
    pd.DataFrame([{"value": 1}]).to_csv(csv_path, index=False)
    manifest = {
        "files": [
            {
                "file_name": "dummy.csv",
                "sha256": "0" * 64,
                "status": "valid",
            }
        ]
    }
    with pytest.raises(RuntimeError, match="sha mismatch"):
        _load_package_frames(data_root, manifest, strict=True)


def test_figure_data_package_contains_no_absolute_paths_or_secret_like_fields() -> None:
    manifest = build_round2_figure_data_package(DATA_ROOT, repo_root=REPO_ROOT, strict=True)
    manifest_json = _load_package_manifest(DATA_ROOT)
    assert manifest["canonical_reference_hash"] == manifest_json["canonical_reference_hash"]
    for entry in manifest_json["files"]:
        assert not Path(entry["file_name"]).is_absolute()
        for source in entry["source_files"]:
            assert not Path(source["path"]).is_absolute()
        frame = pd.read_csv(DATA_ROOT / entry["file_name"])
        lowered_columns = [column.lower() for column in frame.columns]
        assert all(token not in column for column in lowered_columns for token in ("password", "token", "secret", "host", "user", "pid"))


def test_all_target_axes_contain_units(built_round2_assets: dict) -> None:
    required_terms = {
        "data_and_surrogate_validation.pdf": ["kWh", "h"],
        "ddpg_training_dynamics.pdf": ["kWh", "h"],
        "physical_cross_model_stress_test.pdf": ["kWh", "h"],
        "cross_climate_sensitivity.pdf": ["kWh", "h"],
    }
    for relative_name, expected_terms in required_terms.items():
        if relative_name.startswith("A") or relative_name.startswith("B"):
            pdf_path = OUTPUT_ROOT / "appendix" / relative_name
        else:
            pdf_path = OUTPUT_ROOT / "main" / relative_name
        text = _pdf_text(pdf_path)
        for term in expected_terms:
            assert term in text


def test_parity_axes_use_identical_limits(built_round2_assets: dict) -> None:
    for figure_name in [
        OUTPUT_ROOT / "main" / "data_and_surrogate_validation.metadata.json",
        OUTPUT_ROOT / "main" / "physical_cross_model_stress_test.metadata.json",
    ]:
        metadata = json.loads(figure_name.read_text(encoding="utf-8"))
        for _, limits in metadata["extra"]["parity_axes"].items():
            assert limits["xlim"] == pytest.approx(limits["ylim"])


def test_all_hv_igd_figures_include_reference_hash(built_round2_assets: dict) -> None:
    for figure_name in [
        OUTPUT_ROOT / "main" / "benchmark_fairness.metadata.json",
        OUTPUT_ROOT / "main" / "feasible_projection.metadata.json",
        OUTPUT_ROOT / "appendix" / "B3_hv_ceiling_diagnostics.metadata.json",
    ]:
        metadata = json.loads(figure_name.read_text(encoding="utf-8"))
        assert metadata["reference_hash"]
        assert metadata["reference_protocol"] == "benchmark-reference-v2"


def test_no_oversized_equal_size_rows_or_fake_ddpg_sizes_enter_package() -> None:
    frame = pd.read_csv(DATA_ROOT / "benchmark_equal_size_20.csv")
    assert frame["requested_sample_size"].eq(20).all()
    assert not ((frame["method"] == "DDPG") & frame["effective_sample_size"].gt(20)).any()
    assert not ((frame["method"] == "FeasiblePoolRandom") & frame["effective_sample_size"].gt(20)).any()


def test_physical_and_climate_case_counts_are_canonical() -> None:
    physical = pd.read_csv(DATA_ROOT / "physical_direct_cases.csv")
    assert len(physical) == 18
    assert set(physical["case_family"]) == {"direct_feasible"}
    assert "optimizer_linked" not in set(physical["selection_stratum"])
    linked = pd.read_csv(DATA_ROOT / "optimizer_linked_physical_gaps.csv")
    assert len(linked) == 6
    climate = pd.read_csv(DATA_ROOT / "climate_case_results.csv")
    assert len(climate) == 12
    assert climate["matched_sample_id"].nunique() == 4
    assert climate["station"].nunique() == 3


def test_figure_metadata_contains_source_sha(built_round2_assets: dict) -> None:
    for metadata_path in OUTPUT_ROOT.rglob("*.metadata.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["source_files"]
        assert all(item["sha256"] for item in metadata["source_files"])


def test_visual_qa_rejects_type3_and_forbidden_phrases(built_round2_assets: dict) -> None:
    qa = json.loads((OUTPUT_ROOT / "visual_qa_summary.json").read_text(encoding="utf-8"))
    assert all(not item["type3_fonts"] for item in qa["figures"])
    assert all(not item["forbidden_text_hits"] for item in qa["figures"])

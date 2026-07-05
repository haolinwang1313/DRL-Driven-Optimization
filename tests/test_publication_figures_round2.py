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


def _managed_candidate_paths(pattern: str) -> list[Path]:
    return [*(OUTPUT_ROOT / "main").glob(pattern), *(OUTPUT_ROOT / "appendix").glob(pattern)]


def _all_candidate_paths(pattern: str) -> list[Path]:
    return [*(OUTPUT_ROOT / "manual").glob(pattern), *_managed_candidate_paths(pattern)]


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


def _head_bytes(path: str) -> bytes:
    completed = subprocess.run(
        ["git", "show", f"HEAD:{path}"],
        check=True,
        capture_output=True,
        cwd=REPO_ROOT,
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
    for metadata_path in _all_candidate_paths("*.metadata.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        assert metadata["source_files"]
        assert all(item["sha256"] for item in metadata["source_files"])


def test_visual_qa_rejects_type3_and_forbidden_phrases(built_round2_assets: dict) -> None:
    qa = json.loads((OUTPUT_ROOT / "visual_qa_summary.json").read_text(encoding="utf-8"))
    assert all(not item["type3_fonts"] for item in qa["figures"])
    assert all(not item["forbidden_text_hits"] for item in qa["figures"])
    assert all(item["arial_font_hits"] for item in qa["figures"])
    assert all(not item["forbidden_font_hits"] for item in qa["figures"] if item["font_policy"] == "strict_arial")


def _metadata(relative_path: str) -> dict:
    return json.loads((OUTPUT_ROOT / relative_path).read_text(encoding="utf-8"))


def test_m4_only_contains_ddpg_and_nsga_main_comparison(built_round2_assets: dict) -> None:
    metadata = _metadata("main/benchmark_fairness.metadata.json")
    assert metadata["extra"]["plotted_methods"] == ["DDPG", "NSGA-II"]
    assert metadata["extra"]["excluded_methods"] == ["CMA-ES", "RandomSearch", "FeasiblePoolRandom"]


def test_m4_removes_output_contract_panel_and_baseline_labels(built_round2_assets: dict) -> None:
    metadata = _metadata("main/benchmark_fairness.metadata.json")
    text = _pdf_text(OUTPUT_ROOT / "main" / "benchmark_fairness.pdf")
    assert metadata["extra"]["output_contract_panel_removed"] is True
    assert len(metadata["panel_descriptions"]) == 3
    assert "CMA-ES" not in text
    assert "RandomSearch" not in text
    assert "FeasiblePoolRandom" not in text


def test_m5_keeps_only_main_groups_and_no_secondary_axis(built_round2_assets: dict) -> None:
    metadata = _metadata("main/feasible_projection.metadata.json")
    assert metadata["extra"]["plotted_methods"] == ["DDPG", "NSGA-II"]
    assert metadata["extra"]["uses_secondary_y_axis"] is False
    assert metadata["extra"]["nsga_projection_compression"] == {"descriptor_rows": 2000, "unique_projected_blocks": 51}


def test_m5_removes_projected_hv_igd_panels_and_diagnostic_labels(built_round2_assets: dict) -> None:
    metadata = _metadata("main/feasible_projection.metadata.json")
    text = _pdf_text(OUTPUT_ROOT / "main" / "feasible_projection.pdf")
    assert len(metadata["panel_descriptions"]) == 2
    assert metadata["extra"]["projected_hv_igd_moved_to_supplementary"] is True
    assert "CMA-ES" not in text
    assert "RandomSearch" not in text
    assert "FeasiblePoolRandom" not in text


def test_m6_statistics_font_stays_below_axis_label_font(built_round2_assets: dict) -> None:
    metadata = _metadata("main/physical_cross_model_stress_test.metadata.json")
    assert metadata["extra"]["statistics_fontsize"] <= metadata["extra"]["axis_label_fontsize"]
    assert metadata["extra"]["direct_case_count"] == 18


def test_m7_palette_is_muted_and_heatmap_zero_centered(built_round2_assets: dict) -> None:
    metadata = _metadata("main/cross_climate_sensitivity.metadata.json")
    saturation = metadata["extra"]["palette_saturation"]
    assert metadata["extra"]["climate_palette"] == {
        "Beijing": "#539F97",
        "Guangzhou": "#6C7AAD",
        "Harbin": "#BE7A7A",
    }
    assert metadata["extra"]["heatmap_anchor_colors"] == {
        "negative": "#BE7A7A",
        "center": "#F5F4F0",
        "positive": "#6C7AAD",
    }
    assert max(saturation.values()) <= 0.36
    assert metadata["extra"]["heatmap_center"] == 0.0


def test_m7_axis_ranges_and_source_sha_remain_unchanged(built_round2_assets: dict) -> None:
    metadata = _metadata("main/cross_climate_sensitivity.metadata.json")
    head_metadata = json.loads(_head_bytes("paper/manuscript/figures/round2_candidate/main/cross_climate_sensitivity.metadata.json").decode("utf-8"))
    assert metadata["source_files"] == head_metadata["source_files"]
    expected_limits = {
        "a": {"xlim": [-0.54, 2.5400000000000005], "ylim": [-8.66269999999999, 61.158699999999996]},
        "b": {"xlim": [-0.54, 2.5400000000000005], "ylim": [-0.08581965839999979, 0.18471756239999998]},
        "c": {"xlim": [-0.54, 2.5400000000000005], "ylim": [-1.0899999999999996, 0.3400000000000003]},
        "d": {"xlim": [-0.5, 2.5], "ylim": [2.5, -0.5]},
    }
    for panel, expected in expected_limits.items():
        assert metadata["extra"]["axis_limits"][panel]["xlim"] == pytest.approx(expected["xlim"])
        assert metadata["extra"]["axis_limits"][panel]["ylim"] == pytest.approx(expected["ylim"])


def test_all_auto_candidate_pdfs_use_arial_without_times_or_type3(built_round2_assets: dict) -> None:
    for pdf_path in _managed_candidate_paths("*.pdf"):
        fonts = subprocess.run(
            ["pdffonts", str(pdf_path)],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        ).stdout
        assert "Arial" in fonts, pdf_path
        assert "Type 3" not in fonts, pdf_path
        assert "Times" not in fonts, pdf_path
        assert "NewTX" not in fonts, pdf_path


def test_manual_method_figure_hashes_are_preserved_and_record_only(built_round2_assets: dict) -> None:
    metadata = _metadata("manual/fig1.metadata.json")
    assert metadata["figure_id"] == "Fig1"
    assert metadata["font_policy"] == "manual_preserve_with_embedded_symbol_fonts"
    assert metadata["candidate_status"] == "preferred_manual_candidate"
    assert metadata["outputs"]["pdf"]["sha256"] == _sha256_path(OUTPUT_ROOT / "manual" / "fig1.pdf")

    for figure_name in ("fig2", "fig3"):
        metadata = _metadata(f"manual/{figure_name}.metadata.json")
        assert metadata["candidate_status"] == "preferred_manual_candidate"
        assert metadata["extra"]["manual_asset"] is True
        assert metadata["extra"]["final_preferred"] is True
        assert metadata["extra"]["automatic_editing_allowed"] is False
        assert metadata["outputs"]["pdf"]["sha256"] == _sha256_path(OUTPUT_ROOT / "manual" / f"{figure_name}.pdf")


def test_final_manual_candidates_replace_tex_round2_round3_candidates(built_round2_assets: dict) -> None:
    fig2 = _metadata("manual/fig2.metadata.json")
    fig3 = _metadata("manual/fig3.metadata.json")

    assert fig2["semantic_name"] == "manual_fig2_workflow"
    assert fig2["source_files"][0]["path"].endswith("manual/fig2.pdf")
    assert fig2["font_policy"] == "strict_arial"
    assert fig3["semantic_name"] == "manual_fig3_ddpg_architecture"
    assert fig3["source_files"][0]["path"].endswith("manual/fig3.pdf")
    assert fig3["font_policy"] == "manual_preserve_with_embedded_symbol_fonts"
    for obsolete in (
        "fig2_serialized_search_round2.metadata.json",
        "fig2_workflow_round3.metadata.json",
        "fig3_actor_critic_round2.metadata.json",
        "fig3_ddpg_architecture_round3.metadata.json",
    ):
        assert not (OUTPUT_ROOT / "manual" / obsolete).exists()


def test_s5_uses_short_labels_and_log_scale(built_round2_assets: dict) -> None:
    metadata = _metadata("appendix/B3_hv_ceiling_diagnostics.metadata.json")
    text = _pdf_text(OUTPUT_ROOT / "appendix" / "B3_hv_ceiling_diagnostics.pdf")
    assert metadata["figure_id"] == "S5"
    assert metadata["extra"]["panel_b_scale"] == "log"
    assert all(len(label) <= 7 for label in metadata["extra"]["short_labels"])
    assert "Balanced_Performance" not in text
    assert "Energy_Saving_Focus" not in text
    assert "Energy_Generation_Focus" not in text


def test_s8_legend_stays_outside_axes(built_round2_assets: dict) -> None:
    metadata = _metadata("appendix/B4_optimizer_linked_gap_decomposition.metadata.json")
    assert metadata["figure_id"] == "S8"
    assert metadata["extra"]["legend_outside_axes"] is True
    assert len(metadata["extra"]["case_labels"]) == 6


def test_visual_qa_summary_uses_current_si_numbering(built_round2_assets: dict) -> None:
    summary_text = (OUTPUT_ROOT / "visual_qa_summary.json").read_text(encoding="utf-8")
    assert "S5 uses a log scale for tuple counts" in summary_text
    assert "S8 keeps the legend outside the plotting axes" in summary_text
    assert "S6 uses a log scale for tuple counts" not in summary_text
    assert "S7 keeps the legend outside the plotting axes" not in summary_text


def test_gallery_separates_main_and_supplementary_sections(built_round2_assets: dict) -> None:
    gallery_text = (REPO_ROOT / "paper" / "snapshots" / "round2-figure-gallery.md").read_text(encoding="utf-8")
    assert "## Part I - Main manuscript candidates" in gallery_text
    assert "## Part II - Supplementary Information candidates" in gallery_text
    assert "Manual Fig. 1 candidate" in gallery_text
    assert "Fig. 2 manual workflow candidate" in gallery_text
    assert "Fig. 3 manual DDPG architecture candidate" in gallery_text
    assert "Old TeX-based Fig. 2/Fig. 3 candidates are intentionally not regenerated." in gallery_text


def test_supplementary_figure_ids_are_unique(built_round2_assets: dict) -> None:
    supplementary_ids = []
    for metadata_path in (OUTPUT_ROOT / "appendix").glob("*.metadata.json"):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        supplementary_ids.append(metadata["figure_id"])
    assert len(supplementary_ids) == len(set(supplementary_ids))
    assert set(supplementary_ids) == {"S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8", "S9"}


def test_main_vs_supplement_map_covers_all_19_candidate_figures(built_round2_assets: dict) -> None:
    text = (REPO_ROOT / "research" / "reviewer-round-02" / "main-vs-supplement-map.md").read_text(encoding="utf-8")
    for label in ["Manual Fig. 1", "Fig. 2", "Fig. 3"]:
        assert label in text
    for label in ["Fig. 4", "Fig. 5", "Fig. 6", "Fig. 7", "Fig. 8", "Fig. 9", "Fig. 10"]:
        assert label in text
    for label in ["Fig. S1", "Fig. S2", "Fig. S3", "Fig. S4", "Fig. S5", "Fig. S6", "Fig. S7", "Fig. S8", "Fig. S9"]:
        assert label in text


def test_manual_finalization_docs_are_written(built_round2_assets: dict) -> None:
    root = REPO_ROOT / "research" / "reviewer-round-02"
    finalization = (root / "manual-figures-finalization.md").read_text(encoding="utf-8")
    closure = (root / "visualization-closure.md").read_text(encoding="utf-8")
    assert "manual/fig1.pdf" in finalization
    assert "manual/fig2.pdf" in finalization
    assert "manual/fig3.pdf" in finalization
    assert "Manuscript, appendix, and response-letter sources are unchanged" in closure

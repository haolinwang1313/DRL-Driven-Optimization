from __future__ import annotations

import json
import re
import subprocess
from hashlib import sha256
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_TEX = REPO_ROOT / "paper" / "manuscript" / "figures" / "source" / "fig2_serialized_search_round2.tex"
STYLE_TEX = REPO_ROOT / "paper" / "manuscript" / "figures" / "source" / "round2_figure_style.tex"
OUTPUT_ROOT = REPO_ROOT / "paper" / "manuscript" / "figures" / "round2_candidate" / "manual"
OUTPUT_PDF = OUTPUT_ROOT / "fig2_serialized_search_round2.pdf"
OUTPUT_PNG = OUTPUT_ROOT / "fig2_serialized_search_round2.png"
METADATA = OUTPUT_ROOT / "fig2_serialized_search_round2.metadata.json"


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str]) -> str:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return completed.stdout


def test_fig2_tikz_source_matches_simplified_query_contract() -> None:
    source = SOURCE_TEX.read_text(encoding="utf-8")
    assert r"\input{round2_figure_style.tex}" in source
    assert "newtx" not in source
    assert "\\usepackage[T1]{fontenc}" not in source

    for token in (
        "Current state",
        "Actor query",
        "Guarded surrogate",
        "Next state and reward",
        "absolute descriptor query",
        "40 sequential surrogate queries",
        "Static black-box search; no physical-time evolution.",
    ):
        assert token in source

    assert source.count("\\node[") >= 9
    for forbidden in (
        "Replay buffer",
        "Actor--critic update",
        "Critic loss",
        "Target actor",
        r"\bm w^B",
        "(0.6,0.2,0.2)",
        r"\|\bm w\odot",
        "EnergyPlus",
        "Radiance",
    ):
        assert forbidden not in source


def test_fig2_outputs_metadata_and_hashes_are_consistent() -> None:
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    assert metadata["figure_id"] == "Fig2"
    assert metadata["semantic_name"] == "serialized_surrogate_query_process"
    assert metadata["font_policy"] == "strict_arial"
    assert metadata["source_files"][0]["sha256"] == _sha256(SOURCE_TEX)
    assert metadata["source_files"][1]["sha256"] == _sha256(STYLE_TEX)
    assert metadata["outputs"]["pdf"]["sha256"] == _sha256(OUTPUT_PDF)
    assert metadata["outputs"]["png"]["sha256"] == _sha256(OUTPUT_PNG)
    assert metadata["extra"]["main_nodes"] == ["Current state", "Actor query", "Guarded surrogate", "Next state and reward"]
    assert metadata["extra"]["episode_length"] == 40
    assert metadata["extra"]["episodes_per_seed"] == 600
    assert metadata["extra"]["seeds_per_scenario"] == 20


def test_fig2_pdf_uses_arial_no_type3_or_times_and_expected_size() -> None:
    fonts = _run(["pdffonts", str(OUTPUT_PDF)])
    assert "Arial" in fonts
    assert "Type 3" not in fonts
    assert "Times" not in fonts
    assert "NewTX" not in fonts

    text = _run(["pdftotext", str(OUTPUT_PDF), "-"])
    for pattern in (
        r"Current\s+state",
        r"Actor\s+query",
        r"Guarded\s+surrogate",
        r"Next\s+state\s+and\s+reward",
        r"40\s+sequential\s+surrogate\s+queries",
        r"no\s+physical-time\s+evolution",
    ):
        assert re.search(pattern, text)
    for forbidden in ("Replay buffer", "Actor--critic update", "Critic loss", "Target actor", "EnergyPlus", "Radiance"):
        assert forbidden not in text

    pdfinfo = _run(["pdfinfo", str(OUTPUT_PDF)])
    match = re.search(r"Page size:\s+([0-9.]+) x ([0-9.]+) pts", pdfinfo)
    assert match is not None
    width_pt, height_pt = map(float, match.groups())
    width_cm = width_pt * 2.54 / 72.0
    height_cm = height_pt * 2.54 / 72.0
    assert 17.45 <= width_cm <= 17.70
    assert 5.80 <= height_cm <= 6.35

from __future__ import annotations

import json
import re
import subprocess
from hashlib import sha256
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_TEX = REPO_ROOT / "paper" / "manuscript" / "figures" / "source" / "fig3_actor_critic_round2.tex"
STYLE_TEX = REPO_ROOT / "paper" / "manuscript" / "figures" / "source" / "round2_figure_style.tex"
OUTPUT_ROOT = REPO_ROOT / "paper" / "manuscript" / "figures" / "round2_candidate" / "manual"
OUTPUT_PDF = OUTPUT_ROOT / "fig3_actor_critic_round2.pdf"
OUTPUT_PNG = OUTPUT_ROOT / "fig3_actor_critic_round2.png"
METADATA = OUTPUT_ROOT / "fig3_actor_critic_round2.metadata.json"


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


def test_fig3_tikz_source_matches_learning_equation_contract() -> None:
    source = SOURCE_TEX.read_text(encoding="utf-8")
    assert r"\input{round2_figure_style.tex}" in source
    assert "newtx" not in source
    assert "\\usepackage[T1]{fontenc}" not in source

    for token in (
        "Network architecture",
        "Learning equations",
        "Online actor",
        "Online critic",
        "Target actor",
        "Target critic",
        "1. TD target",
        "2. Critic loss",
        "3. Actor objective",
        "4. Soft updates",
        r"\theta^{\mu'}",
        r"\theta^{Q'}",
    ):
        assert token in source

    for forbidden in (
        "Guarded surrogate",
        "episode",
        "Query 40",
        "Gaussian exploration",
        "reward distance",
        r"\alpha_\mu",
        r"\alpha_Q",
        "0.002",
        "0.001$\\par\n  \\end{minipage}\n};\n\n\\node[updatebox",
    ):
        assert forbidden not in source

    assert source.count("\\draw[dataflow]") == 8
    assert source.count("\\draw[softflow]") == 2


def test_fig3_outputs_metadata_and_hashes_are_consistent() -> None:
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    assert metadata["figure_id"] == "Fig3"
    assert metadata["semantic_name"] == "actor_critic_architecture_and_learning"
    assert metadata["font_policy"] == "strict_arial"
    assert metadata["source_files"][0]["sha256"] == _sha256(SOURCE_TEX)
    assert metadata["source_files"][1]["sha256"] == _sha256(STYLE_TEX)
    assert metadata["outputs"]["pdf"]["sha256"] == _sha256(OUTPUT_PDF)
    assert metadata["outputs"]["png"]["sha256"] == _sha256(OUTPUT_PNG)
    assert metadata["extra"]["equation_groups"] == ["TD target", "Critic loss", "Actor objective", "Soft updates"]
    assert metadata["extra"]["arrow_line_styles"] == ["solid", "dashed"]
    assert metadata["extra"]["actor_architecture"] == {
        "input_dim": 3,
        "hidden_layers": [64, 32],
        "output_dim": 12,
        "output_activation": "Sigmoid",
    }
    assert metadata["extra"]["critic_architecture"] == {
        "input_dim": 15,
        "hidden_layers": [64, 32],
        "output_dim": 1,
    }
    assert metadata["extra"]["batch_size"] == 128
    assert metadata["extra"]["gamma"] == 0.999
    assert metadata["extra"]["tau"] == 0.001


def test_fig3_pdf_uses_arial_no_type3_or_times_and_expected_size() -> None:
    fonts = _run(["pdffonts", str(OUTPUT_PDF)])
    assert "Arial" in fonts
    assert "Type 3" not in fonts
    assert "Times" not in fonts
    assert "NewTX" not in fonts

    text = _run(["pdftotext", str(OUTPUT_PDF), "-"])
    for token in ("Network architecture", "Learning equations", "Online actor", "Online critic", "TD target", "Critic loss", "Actor objective", "Soft updates"):
        assert token in text
    for forbidden in ("Guarded surrogate", "episode", "Query 40", "Gaussian exploration", "learning rate"):
        assert forbidden not in text

    pdfinfo = _run(["pdfinfo", str(OUTPUT_PDF)])
    match = re.search(r"Page size:\s+([0-9.]+) x ([0-9.]+) pts", pdfinfo)
    assert match is not None
    width_pt, height_pt = map(float, match.groups())
    width_cm = width_pt * 2.54 / 72.0
    height_cm = height_pt * 2.54 / 72.0
    assert 17.45 <= width_cm <= 17.70
    assert 6.15 <= height_cm <= 6.85

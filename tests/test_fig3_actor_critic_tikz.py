from __future__ import annotations

import json
import re
import subprocess
from hashlib import sha256
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_TEX = REPO_ROOT / "paper" / "manuscript" / "figures" / "source" / "fig3_ddpg_architecture_round3.tex"
STYLE_TEX = REPO_ROOT / "paper" / "manuscript" / "figures" / "source" / "round3_figure_style.tex"
OUTPUT_ROOT = REPO_ROOT / "paper" / "manuscript" / "figures" / "round2_candidate" / "manual"
OUTPUT_PDF = OUTPUT_ROOT / "fig3_ddpg_architecture_round3.pdf"
OUTPUT_PNG = OUTPUT_ROOT / "fig3_ddpg_architecture_round3.png"
METADATA = OUTPUT_ROOT / "fig3_ddpg_architecture_round3.metadata.json"
SUPERSEDED_METADATA = OUTPUT_ROOT / "fig3_actor_critic_round2.metadata.json"
SUPERSEDED_SOURCE_TEX = REPO_ROOT / "paper" / "manuscript" / "figures" / "source" / "fig3_actor_critic_round2.tex"
SUPERSEDED_OUTPUT_PDF = OUTPUT_ROOT / "fig3_actor_critic_round2.pdf"


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str]) -> str:
    candidates = [command]
    if command and command[0] == "pdfinfo":
        fallbacks = [
            Path("C:/texlive/2024/bin/windows/pdfinfo.exe"),
            Path("C:/CTEX/MiKTeX/miktex/bin/x64/pdfinfo.exe"),
        ]
        candidates = [[str(path), *command[1:]] for path in fallbacks if path.exists()] + candidates
    last_error: subprocess.CalledProcessError | FileNotFoundError | None = None
    for candidate in candidates:
        try:
            completed = subprocess.run(
                candidate,
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            return completed.stdout
        except (subprocess.CalledProcessError, FileNotFoundError) as error:
            last_error = error
    if last_error is not None:
        raise last_error
    raise RuntimeError("empty command")


def test_fig3_tikz_source_matches_learning_equation_contract() -> None:
    source = SOURCE_TEX.read_text(encoding="utf-8")
    assert r"\input{round3_figure_style.tex}" in source
    assert "newtx" not in source
    assert "\\usepackage[T1]{fontenc}" not in source

    for token in (
        "Interaction and storage",
        "DDPG learning core",
        "Environment",
        "Experience replay",
        "Online actor",
        "Online critic",
        "Target actor",
        "Target critic",
        "TD target",
        "Critic loss",
        "Actor update",
        "Soft update",
        r"\theta'",
    ):
        assert token in source

    for forbidden in (
        "Query horizon",
        "All episodes",
        "Query 40",
        r"\alpha_\mu",
        r"\alpha_Q",
    ):
        assert forbidden not in source

    assert source.count("\\draw[mainarrow]") == 10
    assert source.count("\\draw[softarrow]") == 2


def test_fig3_outputs_metadata_and_hashes_are_consistent() -> None:
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    assert metadata["figure_id"] == "Fig3"
    assert metadata["semantic_name"] == "ddpg_architecture_round3"
    assert metadata["font_policy"] == "strict_arial"
    assert metadata["candidate_status"] == "preferred_candidate"
    assert metadata["source_files"][0]["sha256"] == _sha256(SOURCE_TEX)
    assert metadata["source_files"][1]["sha256"] == _sha256(STYLE_TEX)
    assert metadata["outputs"]["pdf"]["sha256"] == _sha256(OUTPUT_PDF)
    assert metadata["outputs"]["png"]["sha256"] == _sha256(OUTPUT_PNG)
    for block in ("Environment", "Experience replay buffer", "Online actor", "Online critic", "Target actor", "Target critic", "TD target", "Critic loss", "Actor update", "Soft update"):
        assert block in metadata["extra"]["core_blocks"]
    assert metadata["extra"]["equation_groups"] == ["TD target", "Critic loss", "Actor objective", "Soft update"]
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

    superseded = json.loads(SUPERSEDED_METADATA.read_text(encoding="utf-8"))
    assert superseded["candidate_status"] == "superseded_candidate"
    assert superseded["extra"]["superseded_by"] == "fig3_ddpg_architecture_round3"
    assert superseded["source_files"][0]["sha256"] == _sha256(SUPERSEDED_SOURCE_TEX)
    assert SUPERSEDED_OUTPUT_PDF.exists()


def test_fig3_pdf_uses_arial_no_type3_or_times_and_expected_size() -> None:
    fonts = _run(["pdffonts", str(OUTPUT_PDF)])
    assert "Arial" in fonts
    assert "Type 3" not in fonts
    assert "Times" not in fonts
    assert "NewTX" not in fonts

    text = _run(["pdftotext", str(OUTPUT_PDF), "-"])
    for token in ("Interaction and storage", "DDPG learning core", "Environment", "Experience replay", "Online actor", "Online critic", "Target actor", "Target critic", "TD target", "Critic loss", "Actor update", "Soft update"):
        assert token in text
    for forbidden in ("Query horizon", "All episodes", "Query 40", "learning rate"):
        assert forbidden not in text
    assert re.search(r"\bStart\b", text) is None
    assert re.search(r"\bEnd\b", text) is None

    pdfinfo = _run(["pdfinfo", str(OUTPUT_PDF)])
    match = re.search(r"Page size:\s+([0-9.]+) x ([0-9.]+) pts", pdfinfo)
    assert match is not None
    width_pt, height_pt = map(float, match.groups())
    width_cm = width_pt * 2.54 / 72.0
    height_cm = height_pt * 2.54 / 72.0
    assert 17.45 <= width_cm <= 17.70
    assert 7.65 <= height_cm <= 7.90

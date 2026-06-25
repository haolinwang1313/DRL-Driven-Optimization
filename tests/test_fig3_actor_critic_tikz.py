from __future__ import annotations

import json
import re
import subprocess
from hashlib import sha256
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_TEX = REPO_ROOT / "paper" / "manuscript" / "figures" / "source" / "fig3_actor_critic_round2.tex"
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


def test_fig3_tikz_source_matches_ddpg_update_contract() -> None:
    source = SOURCE_TEX.read_text(encoding="utf-8")
    required = [
        r"\mathcal B",
        r"\mu",
        r"Q",
        r"1-d_i",
        r"L_\mu",
        r"L_Q",
        r"\theta^{\mu'}\leftarrow\tau\theta^\mu+(1-\tau)\theta^{\mu'}",
        r"\theta^{Q'}\leftarrow\tau\theta^Q+(1-\tau)\theta^{Q'}",
    ]
    for token in required:
        assert token in source
    for forbidden in ("10^6", "Acotr", "netword", r"\Delta", "incremental action", "physical time evolution"):
        assert forbidden not in source


def test_fig3_compiled_outputs_and_metadata_are_consistent() -> None:
    assert SOURCE_TEX.exists()
    assert OUTPUT_PDF.exists()
    assert OUTPUT_PNG.exists()
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    assert metadata["figure_id"] == "Fig3"
    assert metadata["semantic_name"] == "actor_critic_architecture"
    assert metadata["source_tex_sha256"] == _sha256(SOURCE_TEX)
    assert metadata["output_pdf_sha256"] == _sha256(OUTPUT_PDF)
    assert metadata["output_png_sha256"] == _sha256(OUTPUT_PNG)
    assert metadata["actor_architecture"] == {
        "input_dim": 3,
        "hidden_layers": [64, 32],
        "output_dim": 12,
        "output_activation": "Sigmoid",
    }
    assert metadata["critic_architecture"] == {
        "input_dim": 15,
        "hidden_layers": [64, 32],
        "output_dim": 1,
    }
    assert metadata["dimensions"]["state_dim"] == 3
    assert metadata["dimensions"]["action_dim"] == 12
    assert metadata["batch_size"] == 128
    assert metadata["gamma"] == 0.999
    assert metadata["tau"] == 0.001


def test_fig3_pdf_has_extractable_text_no_type3_fonts_and_expected_size() -> None:
    fonts = _run(["pdffonts", str(OUTPUT_PDF)])
    assert "Type 3" not in fonts
    text = _run(["pdftotext", str(OUTPUT_PDF), "-"])
    for token in ("Replay mini-batch", "Online actor", "Online critic", "Temporal-difference target"):
        assert token in text
    for forbidden in ("10^6", "Acotr", "netword", "Delta action", "physical time evolution"):
        assert forbidden not in text
    pdfinfo = _run(["pdfinfo", str(OUTPUT_PDF)])
    match = re.search(r"Page size:\s+([0-9.]+) x ([0-9.]+) pts", pdfinfo)
    assert match is not None
    width_pt, height_pt = map(float, match.groups())
    width_cm = width_pt * 2.54 / 72.0
    height_cm = height_pt * 2.54 / 72.0
    assert 17.45 <= width_cm <= 17.70
    assert height_cm <= 8.70

from __future__ import annotations

import json
import re
import subprocess
from hashlib import sha256
from pathlib import Path

from paper_repro.config import Config


REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_TEX = REPO_ROOT / "paper" / "manuscript" / "figures" / "source" / "fig2_serialized_search_round2.tex"
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


def test_fig2_tikz_source_matches_serialized_search_contract() -> None:
    source = SOURCE_TEX.read_text(encoding="utf-8")
    required = [
        r"\bm s_t",
        r"\bm a_t",
        r"[0,1]^{12}",
        r"\operatorname{clip}",
        r"\mathcal N",
        r"(\bm s_t,\bm a_t,r_t,\bm s_{t+1},d_t)",
        "40 sequential surrogate queries",
        "600 episodes",
        "20 seeds",
        r"\bm w^B=(\tfrac13,\tfrac13,\tfrac13)",
        r"\bm w^S=",
        "(0.6,0.2,0.2)",
        r"\bm w^G=",
        "(0.2,0.6,0.2)",
    ]
    for token in required:
        assert token in source
    for forbidden in (
        "Ladybug",
        "Honeybee",
        "EnergyPlus",
        "Radiance",
        "morphology evolution",
        "incremental action",
        r"\Delta\bm a",
    ):
        assert forbidden not in source
    assert "R=10^6" not in source.replace(" ", "")
    assert "10^6-d" not in source.replace(" ", "")


def test_fig2_outputs_metadata_and_config_values_are_consistent() -> None:
    assert SOURCE_TEX.exists()
    assert OUTPUT_PDF.exists()
    assert OUTPUT_PNG.exists()
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    cfg = Config.from_yaml(REPO_ROOT / "configs" / "revision.yaml")
    ddpg = cfg["optimization"]["ddpg"]
    assert metadata["figure_id"] == "Fig2"
    assert metadata["semantic_name"] == "serialized_static_black_box_search"
    assert metadata["source_tex_sha256"] == _sha256(SOURCE_TEX)
    assert metadata["output_pdf_sha256"] == _sha256(OUTPUT_PDF)
    assert metadata["output_png_sha256"] == _sha256(OUTPUT_PNG)
    assert metadata["state_dimension"] == 3
    assert metadata["action_dimension"] == 12
    assert metadata["absolute_action"] is True
    assert metadata["episode_length"] == ddpg["max_steps_per_episode"] == 40
    assert metadata["episodes_per_seed"] == ddpg["max_episodes"] == 600
    assert metadata["seeds_per_scenario"] == ddpg["seeds_per_scenario"] == 20
    assert metadata["initial_noise_std"] == ddpg["initial_noise_std"] == 1.0
    assert metadata["noise_decay"] == ddpg["noise_decay"] == 0.9998
    assert metadata["batch_size"] == ddpg["batch_size"] == 128
    assert metadata["replay_capacity"] == ddpg["replay_buffer_size"] == 1_000_000
    assert metadata["scenario_weights"]["Balanced"] == [1 / 3, 1 / 3, 1 / 3]
    assert metadata["scenario_weights"]["Saving"] == [0.6, 0.2, 0.2]
    assert metadata["scenario_weights"]["Generation"] == [0.2, 0.6, 0.2]


def test_fig2_pdf_has_extractable_text_no_type3_fonts_and_expected_size() -> None:
    fonts = _run(["pdffonts", str(OUTPUT_PDF)])
    assert "Type 3" not in fonts
    text = _run(["pdftotext", str(OUTPUT_PDF), "-"])
    for pattern in (
        r"Per-step\s+surrogate-query\s+loop",
        r"Absolute\s+descriptor\s+action",
        r"Guarded\s+surrogate\s+evaluator",
        r"40\s+sequential\s+surrogate\s+queries",
        r"Reward\s+scenarios",
    ):
        assert re.search(pattern, text)
    for forbidden in ("Ladybug", "Honeybee", "EnergyPlus", "Radiance", "10^6"):
        assert forbidden not in text
    pdfinfo = _run(["pdfinfo", str(OUTPUT_PDF)])
    match = re.search(r"Page size:\s+([0-9.]+) x ([0-9.]+) pts", pdfinfo)
    assert match is not None
    width_pt, height_pt = map(float, match.groups())
    width_cm = width_pt * 2.54 / 72.0
    height_cm = height_pt * 2.54 / 72.0
    assert 17.45 <= width_cm <= 17.70
    assert height_cm <= 8.40

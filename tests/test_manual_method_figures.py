from __future__ import annotations

import json
import re
import subprocess
from hashlib import sha256
from pathlib import Path

import pytest

from paper_repro.publication_figures_round2 import build_round2_figure_data_package, build_round2_revision_figures


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT / "paper" / "manuscript" / "figure_data" / "round2"
OUTPUT_ROOT = REPO_ROOT / "paper" / "manuscript" / "figures" / "round2_candidate"
MANUAL_ROOT = OUTPUT_ROOT / "manual"
TYPO_BLACKLIST = ("loactions", "Acotr", "netword", "Strat", "R = 10^6")


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _run(command: list[str]) -> str:
    candidates = [command]
    if command and command[0] in {"pdfinfo", "pdffonts", "pdftotext"}:
        fallbacks = [
            Path(f"C:/texlive/2024/bin/windows/{command[0]}.exe"),
            Path(f"C:/CTEX/MiKTeX/miktex/bin/x64/{command[0]}.exe"),
        ]
        candidates = [[str(path), *command[1:]] for path in fallbacks if path.exists()] + candidates
    last_error: subprocess.CalledProcessError | FileNotFoundError | None = None
    for candidate in candidates:
        try:
            return subprocess.run(
                candidate,
                cwd=REPO_ROOT,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            ).stdout
        except (subprocess.CalledProcessError, FileNotFoundError) as error:
            last_error = error
    if last_error is not None:
        raise last_error
    raise RuntimeError("empty command")


@pytest.fixture(scope="session")
def built_manual_figures() -> None:
    build_round2_figure_data_package(DATA_ROOT, repo_root=REPO_ROOT, strict=True)
    build_round2_revision_figures(DATA_ROOT, OUTPUT_ROOT, build_gallery=True, strict=True, repo_root=REPO_ROOT)


def _metadata(stem: str) -> dict:
    return json.loads((MANUAL_ROOT / f"{stem}.metadata.json").read_text(encoding="utf-8"))


def test_final_manual_pdf_files_are_the_preferred_method_figures(built_manual_figures: None) -> None:
    for stem in ("fig1", "fig2", "fig3"):
        pdf_path = MANUAL_ROOT / f"{stem}.pdf"
        metadata = _metadata(stem)
        assert pdf_path.exists()
        assert metadata["candidate_status"] == "preferred_manual_candidate"
        assert metadata["source_files"][0]["path"].endswith(f"manual/{stem}.pdf")
        assert metadata["outputs"]["pdf"]["sha256"] == _sha256(pdf_path)
        assert metadata["extra"]["manual_asset"] is True
        assert metadata["extra"]["automatic_editing_allowed"] is False

    for obsolete in (
        "fig2_serialized_search_round2",
        "fig2_workflow_round3",
        "fig3_actor_critic_round2",
        "fig3_ddpg_architecture_round3",
    ):
        assert not (MANUAL_ROOT / f"{obsolete}.metadata.json").exists()
        assert not (MANUAL_ROOT / f"{obsolete}.pdf").exists()
        assert not (MANUAL_ROOT / f"{obsolete}.png").exists()


def test_final_manual_pdf_text_contract_and_typo_blacklist(built_manual_figures: None) -> None:
    expected = {
        "fig1": ("Feasible morphology generation", "Analytic response", "Surrogate-assisted optimization", "Feasible design assessment"),
        "fig2": ("Start", "Actor query", "Guarded surrogate evaluation", "Query horizon reached", "End"),
        "fig3": ("Surrogate environment", "Online actor", "Online critic", "Target actor", "Target critic", "Critic loss"),
    }
    for stem, required_terms in expected.items():
        text = _run(["pdftotext", str(MANUAL_ROOT / f"{stem}.pdf"), "-"])
        for term in required_terms:
            assert term in text
        for typo in TYPO_BLACKLIST:
            assert typo not in text

    fig2_text = _run(["pdftotext", str(MANUAL_ROOT / "fig2.pdf"), "-"])
    assert not any(term in fig2_text for term in ("Target actor", "Target critic", "TD target", "Critic loss"))
    fig3_text = _run(["pdftotext", str(MANUAL_ROOT / "fig3.pdf"), "-"])
    assert re.search(r"\bStart\b", fig3_text) is None
    assert re.search(r"\bEnd\b", fig3_text) is None


def test_final_manual_pdf_font_and_page_checks(built_manual_figures: None) -> None:
    for stem in ("fig1", "fig2", "fig3"):
        pdf_path = MANUAL_ROOT / f"{stem}.pdf"
        fonts = _run(["pdffonts", str(pdf_path)])
        assert "Arial" in fonts
        assert "Type 3" not in fonts
        info = _run(["pdfinfo", str(pdf_path)])
        assert re.search(r"^Pages:\s+1$", info, flags=re.MULTILINE)
        assert re.search(r"^Encrypted:\s+no$", info, flags=re.MULTILINE)

    fig2_fonts = _run(["pdffonts", str(MANUAL_ROOT / "fig2.pdf")])
    assert "Times" not in fig2_fonts
    assert "NewTX" not in fig2_fonts

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re


SOURCE_PATHS = (
    Path("paper/manuscript/manuscript_body.tex"),
    Path("paper/manuscript/manuscript_clean.tex"),
    Path("paper/manuscript/manuscript_highlighted.tex"),
    Path("paper/manuscript/references.tex"),
    Path("paper/supplementary/supplementary_information.tex"),
    Path("paper/highlights/highlights.tex"),
)


@dataclass(frozen=True)
class Check:
    name: str
    pattern: re.Pattern[str]


CHECKS = (
    Check("ASCII double quote", re.compile(r'"')),
    Check(
        "forbidden wording",
        re.compile(
            r"\b(diagnostic|diagnostics|evidence|chain|proof|confirmation|closure|"
            r"physical support|external support|publication-grade)\b",
            re.IGNORECASE,
        ),
    ),
    Check(
        "old reward formula",
        re.compile(r"10\^6|10\\textsuperscript|R\s*=\s*10|dweighted|d_\{weighted\}"),
    ),
    Check(
        "old data-source wording",
        re.compile(
            r"EnergyPlus-generated|Radiance-generated|Grasshopper process|"
            r"simulated morphologies|performance simulation",
            re.IGNORECASE,
        ),
    ),
    Check("old appendix input", re.compile(r"\\input\{appendix\}|Appendix~[AB]|Appendix [AB]")),
    Check("response-letter residue", re.compile(r"response letter|Response\.", re.IGNORECASE)),
)

CITE_GROUP = re.compile(r"\\cite\{([^}]*)\}")
REV_COMMAND = re.compile(r"\\rev\{")


REQUIRED_MAIN_FIGURES = (
    "fig1.pdf",
    "fig2.pdf",
    "fig3.pdf",
    "data_and_surrogate_validation.pdf",
    "surrogate_robustness.pdf",
    "ddpg_training_dynamics.pdf",
    "benchmark_fairness.pdf",
    "feasible_projection.pdf",
    "physical_cross_model_stress_test.pdf",
    "cross_climate_sensitivity.pdf",
)

REQUIRED_SI_FIGURE_MARKERS = (
    "A1_descriptor_distributions.pdf",
    "A2_residual_",
    "A3_scale_study.pdf",
    "B1_seed_",
    "B2_morphology_signatures.pdf",
    "B3_hv_ceiling_",
    "B4_optimizer_linked_gap_decomposition.pdf",
    "B5_nonlinear_response_profiles.pdf",
    "B6_climate_case_detail.pdf",
)


def _line_hits(path: Path, check: Check) -> list[str]:
    hits: list[str] = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if check.pattern.search(line):
            hits.append(f"{path}:{line_no}: {check.name}: {line.strip()}")
    return hits


def _matching_brace(text: str, open_index: int) -> int:
    depth = 0
    i = open_index
    while i < len(text):
        if text[i] == "\\":
            i += 2
            continue
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i
        i += 1
    raise ValueError("unmatched brace in revision markup")


def _revision_char_count(text: str) -> int:
    count = 0
    pos = 0
    while True:
        match = REV_COMMAND.search(text, pos)
        if not match:
            return count
        start = match.end() - 1
        end = _matching_brace(text, start)
        count += end - start - 1
        pos = end + 1


def _extra_source_checks(root: Path) -> list[str]:
    failures: list[str] = []
    body_path = root / "paper/manuscript/manuscript_body.tex"
    if not body_path.exists():
        return failures

    body = body_path.read_text(encoding="utf-8")
    if r"\begin{highlights}" in body:
        failures.append(f"{body_path}: main manuscript must not contain a highlights environment")

    for line_no, line in enumerate(body.splitlines(), start=1):
        for match in CITE_GROUP.finditer(line):
            keys = [key.strip() for key in match.group(1).split(",") if key.strip()]
            if len(keys) > 2:
                failures.append(
                    f"{body_path}:{line_no}: citation group has {len(keys)} keys: {match.group(0)}"
                )

    revision_chars = _revision_char_count(body)
    if revision_chars > int(len(body) * 0.35):
        failures.append(
            f"{body_path}: revision markup is {revision_chars / max(len(body), 1):.1%} of source; limit is 35%"
        )

    manuscript_entry = root / "paper/manuscript/manuscript.tex"
    if manuscript_entry.exists() and "manuscript_clean.tex" not in manuscript_entry.read_text(encoding="utf-8"):
        failures.append(f"{manuscript_entry}: manuscript entry must input manuscript_clean.tex")

    return failures


def check_sources(root: Path) -> list[str]:
    failures: list[str] = []
    resolved_paths = [root / path for path in SOURCE_PATHS]
    for path in resolved_paths:
        if not path.exists():
            failures.append(f"{path}: missing source file")
            continue
        for check in CHECKS:
            failures.extend(_line_hits(path, check))

    failures.extend(_extra_source_checks(root))

    body_path = root / "paper/manuscript/manuscript_body.tex"
    si_path = root / "paper/supplementary/supplementary_information.tex"
    if body_path.exists():
        body = body_path.read_text(encoding="utf-8")
        for figure in REQUIRED_MAIN_FIGURES:
            if figure not in body:
                failures.append(f"{body_path}: missing main figure reference {figure}")
    if si_path.exists():
        si = si_path.read_text(encoding="utf-8")
        for marker in REQUIRED_SI_FIGURE_MARKERS:
            if marker not in si:
                failures.append(f"{si_path}: missing SI figure reference {marker}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check round-2 manuscript source quality.")
    parser.add_argument("--check-only", action="store_true", help="Run source checks and exit.")
    args = parser.parse_args()

    failures = check_sources(Path.cwd())
    if failures:
        for failure in failures:
            print(failure)
        return 1
    print("round2 manuscript source checks passed")
    return 0 if args.check_only else 0


if __name__ == "__main__":
    raise SystemExit(main())

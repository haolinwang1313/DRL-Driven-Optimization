from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re


SOURCE_PATHS = (
    Path("paper/manuscript/manuscript_body.tex"),
    Path("paper/manuscript/manuscript_clean.tex"),
    Path("paper/manuscript/manuscript_highlighted.tex"),
    Path("paper/manuscript/references.bib"),
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
    Check(
        "round-2 over-strong wording",
        re.compile(
            r"unsupported|not support|not assumed|not be interpreted|"
            r"do not replace|not optimizer|not physical|superiority|certification|"
            r"validation proof|generalization proof|reported as|reported with|"
            r"reporting metric|for reproduction|weak ranking transfer",
            re.IGNORECASE,
        ),
    ),
    Check(
        "internal result-lock wording",
        re.compile(
            r"benchmark-reference-v2|reference hash|simulation(?:_|\\_)mode|"
            r"round-2 assessment|retained-output contract|SI baseline|not_applicable|"
            r"metadata_error|canonical registry|local-reference projected metrics|"
            r"source_archive_size|effective_sample_size",
            re.IGNORECASE,
        ),
    ),
)

CITE_GROUP = re.compile(r"\\cite\{([^}]*)\}")
REV_COMMAND = re.compile(r"\\rev\{")
BIB_KEY = re.compile(r"@\w+\s*\{\s*([^,\s]+)")
LABEL = re.compile(r"\\label\{((?:fig|tab):[^}]+)\}")
REF = re.compile(r"\\(?:[A-Za-z]*ref)\{([^}]+)\}")
CAPTION = re.compile(r"\\caption\{")
COMMAND = re.compile(r"\\[A-Za-z]+(?:\[[^\]]*\])?(?:\{([^{}]*)\})?")


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


def _command_args(text: str, command: re.Pattern[str]) -> list[tuple[int, str]]:
    args: list[tuple[int, str]] = []
    pos = 0
    while True:
        match = command.search(text, pos)
        if not match:
            return args
        start = match.end() - 1
        end = _matching_brace(text, start)
        args.append((match.start(), text[start + 1 : end]))
        pos = end + 1


def _latex_word_count(text: str) -> int:
    text = COMMAND.sub(lambda match: match.group(1) or " ", text)
    text = re.sub(r"[{}$\\]", " ", text)
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)?", text))


def _section_text(text: str, start_heading: str, end_heading: str) -> str:
    start = text.find(start_heading)
    if start == -1:
        return ""
    end = text.find(end_heading, start + len(start_heading))
    return text[start:] if end == -1 else text[start:end]


def _citation_keys(text: str) -> set[str]:
    keys: set[str] = set()
    for match in CITE_GROUP.finditer(text):
        keys.update(key.strip() for key in match.group(1).split(",") if key.strip())
    return keys


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

    if r"\begin{thebibliography}" in body or r"\input{references}" in body:
        failures.append(f"{body_path}: main manuscript must use BibTeX with elsarticle-num")
    if "elsarticle-num" not in body or r"\bibliography{references}" not in body:
        failures.append(f"{body_path}: missing elsarticle-num bibliography configuration")

    captions = _command_args(body, CAPTION)
    label_matches = list(LABEL.finditer(body))
    for index, (caption_pos, caption_text) in enumerate(captions):
        next_caption_pos = captions[index + 1][0] if index + 1 < len(captions) else len(body)
        label = next(
            (match.group(1) for match in label_matches if caption_pos < match.start() < next_caption_pos),
            "",
        )
        word_count = _latex_word_count(caption_text)
        if label.startswith("fig:"):
            limit = 65 if label == "fig:workflow" else 55
            if word_count > limit:
                failures.append(f"{body_path}: {label} caption has {word_count} words; limit is {limit}")
        elif label.startswith("tab:") and word_count > 20:
            failures.append(f"{body_path}: {label} caption has {word_count} words; limit is 20")
        if re.search(r"rather than|unsupported|validation proof|superiority|reported with", caption_text, re.I):
            failures.append(f"{body_path}: {label or 'caption'} contains forbidden caption wording")

    labels = {match.group(1) for match in label_matches}
    refs = {match.group(1) for match in REF.finditer(body)}
    for label in sorted(labels - refs):
        failures.append(f"{body_path}: {label} is not referenced in the main text")

    bib_path = root / "paper/manuscript/references.bib"
    if bib_path.exists():
        bib_keys = set(BIB_KEY.findall(bib_path.read_text(encoding="utf-8")))
        intro = _section_text(body, r"\section{Introduction}", r"\section{Methodology}")
        intro_keys = _citation_keys(intro)
        missing_from_intro = sorted(bib_keys - intro_keys)
        if missing_from_intro:
            failures.append(
                f"{body_path}: bibliography keys missing from Introduction citations: {', '.join(missing_from_intro)}"
            )
        unused = sorted(bib_keys - _citation_keys(body))
        if unused:
            failures.append(f"{bib_path}: unused bibliography keys: {', '.join(unused)}")

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

"""Local structural checks for the round-2 response letter.

The script checks only the response package source and an optional extracted
``response_letter.txt`` next to it. It does not inspect or modify manuscript,
SI, figure, experiment, or canonical result files.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


EXPECTED_IDS = [f"R1-{i}" for i in range(1, 16)]
EXPECTED_IDS += [f"R2-{i}" for i in range(1, 9)]
EXPECTED_IDS += [f"R3-{i}" for i in range(1, 6)]
EXPECTED_IDS += [f"R4-{i}" for i in range(1, 6)]

LITERATURE_IDS = {"R1-4", "R1-10", "R4-1", "R4-5"}
CORE_RESPONSE_IDS = {"R1-1", "R1-3", "R1-7", "R2-1", "R2-3", "R2-4", "R2-5", "R2-7", "R4-1", "R4-2", "R4-5"}
SIMPLE_RESPONSE_IDS = {"R3-1", "R3-4"}

BANNED_AUTHOR_PHRASES = [
    "The relevant revisions appear in",
    "The change appears in",
    "The changes appear in",
    "The revised figure reports",
    "The numerical values are shown",
    "The details are provided",
    "Please see Fig",
    "Please see Table",
    "The revision makes this point explicit",
    "Further details are provided in the Supplementary Information",
    "how the evidence should be read",
    "The evidence now",
    "This evidence",
    "evidence package",
    "comment-level evidence package",
    "Nomenclature-to-Table 1 cross-reference",
    "This point matters because",
    "The revised material now gives the reader",
    "how the revision should be interpreted",
    "This is why the response",
    "supporting blocks",
    "auditable before selected physical calculations are added",
]

BANNED_INTERNAL = [
    "trellis",
    "codex",
    "chatgpt",
    "pull request",
    "PR #",
    "server job",
    "job_id",
    "D:\\",
    "/mnt/data",
    "canonical-result",
    "canonical registry",
    "local path",
]

BANNED_STRONG = [
    "physical validation confirms",
    "proves the superiority",
    "generalization proof",
    "physical closure",
    "external confirmation",
    "successful validation",
    "publication-grade",
    "full physical stack",
    "12 independent design variables",
    "direct EnergyPlus/Radiance simulation for all 2000",
    "ranking transfer is confirmed",
    "EG_roof_irradiance was computed",
    "10^" + "6 - d",
    "10^" + "6-d",
]

APPLIED_ENERGY_DOI = "10.1016/j.apenergy.2026.128294"
APPLIED_ENERGY_KEY = "wang2026surrogateDistrictMorphology"
APPLIED_ENERGY_TITLE = "A surrogate-assisted framework for district-scale urban morphology optimization toward reduced building energy demand"
R2_8_REMOVED_SENTENCE = "The response PDF keeps implementation-tracking details outside the reviewer-facing reproducibility statement"


def strip_commentboxes(text: str) -> str:
    return re.sub(
        r"\\begin\{commentbox\}.*?\\end\{commentbox\}",
        "",
        text,
        flags=re.DOTALL,
    )


def split_comment_blocks(text: str) -> dict[str, str]:
    parts = re.split(r"(?=\\comment\{R[1-4]-\d+\})", text)
    blocks: dict[str, str] = {}
    for part in parts:
        match = re.match(r"\\comment\{(R[1-4]-\d+)\}", part)
        if match:
            blocks[match.group(1)] = part
    return blocks


def response_body(block: str) -> str:
    match = re.search(
        r"\\responseheading\{Response\.\}\s*\\checkresponse\{%(.*?)\n\}\s*\\responseheading\{Revisions made in the manuscript\.\}",
        block,
        flags=re.DOTALL,
    )
    return match.group(1) if match else ""


def plain_words(latex: str) -> list[str]:
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\[[^\]]*\])?(?:\{[^{}]*\})?", " ", latex)
    text = re.sub(r"[$^_{}\\]", " ", text)
    return re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", text)


def has_figure_table_or_equation(block: str) -> bool:
    markers = [
        r"\compactfig",
        r"\EvaluationModeTable",
        r"\CompactDescriptorTable",
        r"\PhysicalSummaryTable",
        r"\RewardEquations",
        r"\AnalyticEquations",
        r"\begin{tabular",
        r"\[",
    ]
    return any(marker in block for marker in markers)


def bib_entry(text: str, key: str) -> str:
    match = re.search(rf"@\w+\{{{re.escape(key)},(.*?)(?=\n@\w+\{{|\Z)", text, flags=re.DOTALL)
    return match.group(1) if match else ""


def table_r1_rows(text: str) -> int:
    match = re.search(
        r"\\caption\{Editor-facing overview of the main revision packages\.\}.*?\\midrule(.*?)\\bottomrule",
        text,
        flags=re.DOTALL,
    )
    if not match:
        return -1
    return len(re.findall(r"\\\\\s*$", match.group(1), flags=re.MULTILINE))


def check_text(label: str, text: str, errors: list[str], *, strong_claims: bool = True) -> None:
    lowered = text.lower()
    if "scope boundary." in lowered:
        errors.append(f"{label}: visible Scope boundary heading remains")
    for phrase in BANNED_AUTHOR_PHRASES:
        if re.search(re.escape(phrase), text, flags=re.IGNORECASE):
            errors.append(f"{label}: banned lazy phrase: {phrase}")
    for phrase in BANNED_INTERNAL:
        if phrase.lower() in lowered:
            errors.append(f"{label}: internal/local term: {phrase}")
    if strong_claims:
        for phrase in BANNED_STRONG:
            if phrase.lower() in lowered:
                errors.append(f"{label}: strong or forbidden claim: {phrase}")


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("response_letter.tex")
    path = path.resolve()
    text = path.read_text(encoding="utf-8")
    author_text = strip_commentboxes(text)
    blocks = split_comment_blocks(text)
    errors: list[str] = []

    found_ids = re.findall(r"\\comment\{(R[1-4]-\d+)\}", text)
    if found_ids != EXPECTED_IDS:
        errors.append(f"comment IDs mismatch: {found_ids}")

    if len(blocks) != 33:
        errors.append(f"expected 33 comment blocks, found {len(blocks)}")

    if not re.search(r"\\makeresponsetitle\s*\\clearpage\s*\\responsecontents", text):
        errors.append("Contents does not start after an explicit title-page clearpage")

    if "\\responsecontents\n\n\\section*{Summary of major revisions}" not in text:
        errors.append("Summary does not immediately follow the responsecontents block")

    if text.count(r"\reviewerpage{") != 4:
        errors.append("expected four reviewerpage sections")

    if r"\RenewDocumentCommand{\checkresponse}{+m}" not in text:
        errors.append("checkresponse is not locally redefined to remove visible check marks")

    print("manual visual check required: Contents links appear black; DOI links appear blue roman.")

    if r"\texttt{\textbackslash cite" in text:
        errors.append("response_letter.tex: raw citation command display remains")
    if R2_8_REMOVED_SENTENCE.lower() in text.lower():
        errors.append("R2-8: implementation-tracking sentence remains")
    if r"\section*{Closing Remark}" not in text:
        errors.append("Closing Remark heading is missing")
    if r"\section*{Closing}" in text:
        errors.append("old Closing section heading remains")
    if r"\addcontentsline{toc}{section}{Closing Remark}" not in text:
        errors.append("Closing Remark is missing from Contents")
    if r"\addcontentsline{toc}{section}{Closing}" in text:
        errors.append("old Closing contents entry remains")
    if not re.search(r"\\clearpage\s*\\section\*\{Closing Remark\}", text):
        errors.append("Closing Remark does not start after an explicit clearpage")

    rows = table_r1_rows(text)
    if rows != 6:
        errors.append(f"Table R1 should have 6 revision package rows, found {rows}")

    for cid in EXPECTED_IDS:
        block = blocks.get(cid, "")
        if block.count(r"\responseheading{Response.}") != 1:
            errors.append(f"{cid}: missing or repeated Response heading")
        if block.count(r"\responseheading{Revisions made in the manuscript.}") != 1:
            errors.append(f"{cid}: missing or repeated revisions heading")
        if block.count(r"\responseheading{Relevant revised manuscript and supporting material.}") != 1:
            errors.append(f"{cid}: missing or repeated supporting-material heading")
        if r"\begin{revisionbox}" not in block:
            errors.append(f"{cid}: missing revisionbox")
        commentbox = re.search(
            r"\\begin\{commentbox\}(.*?)\\end\{commentbox\}",
            block,
            flags=re.DOTALL,
        )
        if not commentbox or "The reviewer" in commentbox.group(1):
            errors.append(f"{cid}: commentbox does not look like original reviewer text")
        if cid in LITERATURE_IDS and r"\begin{responsereferences}" not in block:
            errors.append(f"{cid}: missing local references")

        words = len(plain_words(response_body(block)))
        required = 80 if cid in SIMPLE_RESPONSE_IDS else 180 if cid in CORE_RESPONSE_IDS else 140
        if cid.startswith("R2-"):
            required = max(required, 180)
        if cid == "R3-5" or cid in {f"R4-{i}" for i in range(1, 6)}:
            required = max(required, 220)
        if words < required:
            errors.append(f"{cid}: response too short ({words} words, need {required})")

    for cid in [f"R2-{i}" for i in range(1, 6)]:
        block = blocks.get(cid, "")
        if not has_figure_table_or_equation(block):
            errors.append(f"{cid}: missing figure, equation, or table support")

    if r"\AnalyticEquations" not in blocks.get("R2-7", ""):
        errors.append("R2-7: missing \\AnalyticEquations")

    r28_author = strip_commentboxes(blocks.get("R2-8", ""))
    if re.search(r"\b(branch|commit|PR\s*#|pull request|server job|job_id|D:\\|/mnt/data|local path)\b", r28_author, flags=re.IGNORECASE):
        errors.append("R2-8: internal branch/commit/PR/local/server wording remains")

    r110 = blocks.get("R1-10", "")
    if r"\texttt{\textbackslash cite" in r110:
        errors.append("R1-10: raw citation command display remains in supporting material")
    if "puterman1994mdp" not in r110 and "Puterman" not in r110:
        errors.append("R1-10: missing Puterman manuscript citation or local reference")
    if "Sutton and Barto" not in r110:
        errors.append("R1-10: missing Sutton and Barto local reference")

    r45 = blocks.get("R4-5", "")
    if APPLIED_ENERGY_KEY not in r45 and APPLIED_ENERGY_TITLE not in r45:
        errors.append("R4-5: missing accepted Applied Energy morphology reference")

    r12 = blocks.get("R1-2", "")
    if "Nomenclature" not in r12 or "Formula symbols" not in r12:
        errors.append("R1-2 supporting material does not include Nomenclature and Formula symbols")
    if "Abbreviations block excerpt" in r12:
        errors.append("R1-2 still uses the old Abbreviations block excerpt label")
    if "Nomenclature-to-Table 1 cross-reference" in r12:
        errors.append("R1-2 still includes the removed Nomenclature-to-Table 1 cross-reference")
    if re.search(r"Descriptor\s*&\s*Symbol|Symbol\s*&\s*Unit", r12):
        errors.append("R1-2 Table 1 excerpt still contains a Symbol column")
    if re.search(r"\\url\{https://doi\.org|\\url\{http", text):
        errors.append("local references still use \\url for DOI/URL links")

    manuscript_path = path.parents[2] / "manuscript" / "manuscript_body.tex"
    if manuscript_path.exists():
        manuscript = manuscript_path.read_text(encoding="utf-8")
        if re.search(r"descriptor symbols.*defined in Table", manuscript, flags=re.IGNORECASE):
            errors.append("manuscript Nomenclature still cross-references descriptor symbols to Table 1")
        if re.search(r"Descriptor\s*&\s*Symbol|Symbol\s*&\s*Unit", manuscript):
            errors.append("manuscript Table 1 still contains a Symbol column")
        if APPLIED_ENERGY_KEY not in manuscript:
            errors.append("manuscript_body.tex: missing accepted Applied Energy morphology citation")

    references_tex_path = path.parents[2] / "manuscript" / "references.tex"
    if references_tex_path.exists():
        references_tex = references_tex_path.read_text(encoding="utf-8")
        if APPLIED_ENERGY_DOI not in references_tex:
            errors.append(f"references.tex: missing DOI {APPLIED_ENERGY_DOI}")

    references_path = path.parents[2] / "manuscript" / "references.bib"
    if references_path.exists():
        references = references_path.read_text(encoding="utf-8")
        sutton = bib_entry(references, "sutton2018reinforcement")
        puterman = bib_entry(references, "puterman1994mdp")
        applied = bib_entry(references, APPLIED_ENERGY_KEY)
        if re.search(r"\bdoi\s*=", sutton, flags=re.IGNORECASE):
            errors.append("references.bib: Sutton and Barto entry must not contain a DOI")
        puterman_dois = re.findall(r"\bdoi\s*=\s*\{([^}]+)\}", puterman, flags=re.IGNORECASE)
        if puterman_dois and puterman_dois != ["10.1002/9780470316887"]:
            errors.append(f"references.bib: Puterman DOI mismatch: {puterman_dois}")
        if APPLIED_ENERGY_DOI not in applied:
            errors.append(f"references.bib: missing DOI {APPLIED_ENERGY_DOI}")

    check_text("author tex", author_text, errors)

    txt_path = path.with_suffix(".txt")
    if txt_path.exists():
        check_text("pdf text", txt_path.read_text(encoding="utf-8", errors="ignore"), errors, strong_claims=False)

    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1

    print("response QA passed")
    print("comments: 33/33")
    print("headings: Response/Revisions/Supporting material present for every comment")
    print("response expansion: word-count thresholds passed")
    print("layout checks: title/contents/reviewer pagination markers passed")
    print("Table R1: 6 revision package rows")
    print("R1-2: Nomenclature supporting material present")
    print("R1-10: MDP/RL citations and local references present")
    print("R2: expanded responses and supporting material checks passed")
    print("final targeted pass: R1-10/R2-8/R3-5/R4/Closing/Applied Energy checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Local structural checks for the round-2 response letter.

The script checks only the response package source. It does not inspect or
modify manuscript, SI, figure, experiment, or canonical result files.
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

BANNED_AUTHOR_PHRASES = [
    "The revised figure reports the results",
    "The numerical values are shown in the revised table",
    "The details are provided in the Supplementary Material",
    "We added relevant discussion in the manuscript",
    "The revision makes this point explicit",
    "Please see Fig",
    "Please see Table",
    "These quantitative results are reported in the revised figure and supporting tables",
    "Further details are provided in the Supplementary Information",
]

BANNED_INTERNAL = [
    "trellis",
    "codex",
    "chatgpt",
    "branch",
    "commit",
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
    "10^6 - d",
    "10^6-d",
]


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


def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("response_letter.tex")
    text = path.read_text(encoding="utf-8")
    author_text = strip_commentboxes(text)
    blocks = split_comment_blocks(text)
    errors: list[str] = []

    found_ids = re.findall(r"\\comment\{(R[1-4]-\d+)\}", text)
    if found_ids != EXPECTED_IDS:
        errors.append(f"comment IDs mismatch: {found_ids}")

    if len(blocks) != 33:
        errors.append(f"expected 33 comment blocks, found {len(blocks)}")

    for cid in EXPECTED_IDS:
        block = blocks.get(cid, "")
        if block.count(r"\responseheading{Response.}") != 1:
            errors.append(f"{cid}: missing or repeated Response heading")
        if block.count(r"\responseheading{Revisions made in the manuscript.}") != 1:
            errors.append(f"{cid}: missing or repeated revisions heading")
        if block.count(r"\responseheading{Relevant revised manuscript and supporting evidence.}") != 1:
            errors.append(f"{cid}: missing or repeated evidence heading")
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

    for phrase in BANNED_AUTHOR_PHRASES:
        if re.search(re.escape(phrase), author_text, flags=re.IGNORECASE):
            errors.append(f"banned lazy phrase in author text: {phrase}")

    lowered = author_text.lower()
    for phrase in BANNED_INTERNAL:
        if phrase.lower() in lowered:
            errors.append(f"internal/local term in author text: {phrase}")

    for phrase in BANNED_STRONG:
        if phrase.lower() in lowered:
            errors.append(f"strong or forbidden claim in author text: {phrase}")

    if "evidence store" in lowered:
        errors.append("summary still uses evidence store")

    if errors:
        for error in errors:
            print(f"FAIL: {error}")
        return 1

    print("response evidence QA passed")
    print("comments: 33/33")
    print("headings: Response/Revisions/Evidence present for every comment")
    print("literature reference blocks: present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

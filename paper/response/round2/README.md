# Round 2 Response Letter Editorial Revision

This directory contains the Applied Energy round-2 response package. The current version is the response editorial revision after human review of PR #5.

## Files

- `response_letter.tex`: main response letter source.
- `response_letter.pdf`: compiled response PDF.
- `response_matrix.csv`: 33-row comment-level evidence matrix with expansion and Nomenclature QA fields.
- `response_input_manifest.json`: input, source, asset, build, and QA manifest.
- `check_response_evidence.py`: local structural QA for the response letter.
- `template/response_slu.sty`: copied local response-letter style file.
- `template/RESPONSE_LETTER_CHECKLIST.md`: copied local response-letter checklist.

## Current Revision Scope

- This version performs a human-review pass focused on Reviewer 2 expansion and reference/link formatting.
- Contents now starts on a standalone page.
- Contents internal hyperlinks remain black.
- DOI/URL links in local references use blue roman styling through `\responsedoi{...}`.
- Each reviewer section starts on a new page.
- Table R1 is simplified to six editor-facing revision packages.
- Manuscript Nomenclature cross-reference sentence was removed.
- Manuscript Table 1 Symbol column was removed.
- Response heading changed from `supporting evidence` to `supporting material`.
- Repeated `evidence` wording was removed from author response prose.
- Visible `Scope boundary.` headings were removed from the PDF-facing supporting boxes.
- Response prose was expanded across all substantive comments.
- `Revisions made in the manuscript.` sections were rewritten from location-only summaries to action-specific descriptions.
- R1-1/R1-2/R1-3/R1-4/R1-7/R1-10/R1-11/R1-14/R1-15 were revised after human review.
- R1-10 MDP/RL references were updated: Puterman DOI was verified and Sutton/Barto uses ISBN metadata without an invented DOI.
- R1-15 broader-audience and AI-assisted urban-planning framing was expanded.
- R2-1 to R2-8 were expanded with detailed, conciliatory responses and enriched supporting material.
- R1-2 is synchronized with the manuscript front-matter Nomenclature update and four-column Table 1.
- `manuscript_clean.pdf`, `manuscript_highlighted.pdf`, and `response_letter.pdf` were rebuilt.

## Protected Scope

This update did not modify:

- `paper/supplementary/`
- manuscript or SI figure assets
- experiments
- `tools/`
- `paper_repro/`
- canonical result files
- `.trellis/tasks/`

## Build

Commands used:

```powershell
cd paper/manuscript
latexmk -xelatex -interaction=nonstopmode -halt-on-error manuscript_clean.tex
latexmk -g -xelatex -interaction=nonstopmode -halt-on-error manuscript_highlighted.tex

cd ../response/round2
latexmk -xelatex -interaction=nonstopmode -halt-on-error response_letter.tex
```

Latest local build result:

- Response PDF before the first editorial revision: 49 pages.
- Response PDF after the first editorial revision: 56 pages.
- Response PDF after the latest Reviewer 2/reference/link pass: 63 pages.
- Manuscript clean PDF: 17 A4 pages.
- Manuscript highlighted PDF: 17 A4 pages.
- Expected warnings: Perl locale warning, copied-template package-name warning, xeCJK monofont warning, underfull hbox warnings, and embedded-figure PDF-version warnings.
- Known inherited PDF-font note: existing protected figure PDFs still carry some non-embedded TrueType fonts. This task did not regenerate or alter those figures.

## QA

Commands used:

```powershell
uv run python check_response_evidence.py response_letter.tex
pdftotext -layout -enc UTF-8 response_letter.pdf response_letter.txt
pdftotext -layout -enc UTF-8 ../../manuscript/manuscript_clean.pdf ../../manuscript/manuscript_clean.txt
pdftotext -layout -enc UTF-8 ../../manuscript/manuscript_highlighted.pdf ../../manuscript/manuscript_highlighted.txt
C:\texlive\2024\bin\windows\pdfinfo.exe response_letter.pdf
C:\texlive\2024\bin\windows\pdfinfo.exe ..\..\manuscript\manuscript_clean.pdf
C:\texlive\2024\bin\windows\pdfinfo.exe ..\..\manuscript\manuscript_highlighted.pdf
pdffonts response_letter.pdf
```

Latest QA results:

- `check_response_evidence.py`: passed.
- Contents/link-color check: passed by rendered visual inspection; Contents links appear black, and DOI links appear blue roman.
- Puterman DOI verification: passed through DOI resolution and Crossref/Wiley metadata for `10.1002/9780470316887`.
- R1-10 reference sync: manuscript excerpt displays `\cite{puterman1994mdp,sutton2018reinforcement}`, local references include Puterman DOI and Sutton/Barto ISBN.
- R1-15 broader-audience framing: expanded.
- R2-1 to R2-8 expansion: passed word-count and supporting-material checks.
- `uv run pytest -q`: passed, 63 tests.
- `uv run python -m paper_repro.cli --help`: passed.
- `uv run python -c "from paper_repro.config import Config; Config.from_yaml('configs/revision.yaml'); print('config ok')"`: passed.
- `uv run python -m compileall paper_repro tools tests`: passed.
- `git diff --check`: passed.
- Comment IDs in source: 33/33.
- Every comment has exactly one `Response.`, one `Revisions made in the manuscript.`, and one `Relevant revised manuscript and supporting material.` heading.
- Response expansion thresholds passed.
- Contents/reviewer pagination markers passed.
- Table R1 has six revision package rows.
- R1-2 supporting material includes Nomenclature and Formula symbols.
- R1-2 no longer includes the removed Nomenclature-to-Table 1 cross-reference block.
- Manuscript and response Table 1 excerpts no longer include the redundant Symbol column.
- DOI/URL source scan: local references no longer use `\url{...}` for DOI links.
- `Scope boundary.` scan: no source or PDF-text hits.
- Generic location-only phrase scan: no source or PDF-text hits.
- Strong-claim scan: only reviewer-original quoted text contains `full physical stack` / `publication-grade physical closure`; author response and manuscript source are clear.

## Manual Checks Remaining

- Add final manuscript/SI line numbers if Applied Energy requires line-specific references.
- Human review of `response_letter.pdf`, especially Table R1 and expanded response wording.
- Author review of expanded wording before submission.

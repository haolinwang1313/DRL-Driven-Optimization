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

- Contents now starts on a standalone page.
- Each reviewer section starts on a new page.
- Table R1 is simplified to six editor-facing revision packages.
- Visible `Scope boundary.` headings were removed from the PDF-facing evidence boxes.
- Response prose was expanded across all substantive comments.
- `Revisions made in the manuscript.` sections were rewritten from location-only summaries to action-specific descriptions.
- R1-2 is synchronized with the manuscript front-matter Nomenclature update.
- `manuscript_clean.pdf` and `manuscript_highlighted.pdf` were rebuilt because the front-matter Nomenclature changed.

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

- Response PDF before this editorial revision: 49 pages.
- Response PDF after this editorial revision: 56 pages.
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
- Comment IDs in source: 33/33.
- Every comment has exactly one `Response.`, one `Revisions made in the manuscript.`, and one `Relevant revised manuscript and supporting evidence.` heading.
- Response expansion thresholds passed.
- Contents/reviewer pagination markers passed.
- Table R1 has six revision package rows.
- R1-2 evidence includes Nomenclature and Formula symbols.
- `Scope boundary.` scan: no source or PDF-text hits.
- Generic location-only phrase scan: no source or PDF-text hits.
- Strong-claim scan: only reviewer-original quoted text contains `full physical stack` / `publication-grade physical closure`; author response and manuscript source are clear.

## Manual Checks Remaining

- Add final manuscript/SI line numbers if Applied Energy requires line-specific references.
- Human visual review of `response_letter.pdf`, especially Contents pagination and Table R1.
- Author review of expanded wording before submission.

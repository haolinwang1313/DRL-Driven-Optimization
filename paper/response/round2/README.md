# Round 2 Comment-Level Response Letter Draft

This directory contains the Applied Energy round-2 response package. The current version converts the earlier detailed rebuttal draft into a comment-level evidence-package draft.

## Files

- `response_letter.tex`: main response letter source.
- `response_letter.pdf`: compiled response PDF.
- `response_matrix.csv`: 33-row comment-level evidence package matrix.
- `response_input_manifest.json`: input, source, asset, build, and QA manifest.
- `check_response_evidence.py`: local structural QA for the response letter.
- `template/response_slu.sty`: copied local response-letter style file.
- `template/RESPONSE_LETTER_CHECKLIST.md`: copied local response-letter checklist.

## Current Draft Scope

- The response letter covers 33/33 reviewer comments.
- Each comment includes the original reviewer comment, `Response.`, `Revisions made in the manuscript.`, and `Relevant revised manuscript and supporting evidence.` sections.
- Relevant manuscript excerpts, SI excerpts, figure assets, compact table excerpts, equations, and local reference blocks are placed under the corresponding comment rather than collected in one central dossier.
- Table R1 remains only an editor-facing overview.
- Manual line numbers still need to be checked after the final line-numbered manuscript and SI PDFs are exported.

## Protected Scope

This response-package update did not modify:

- `paper/manuscript/`
- `paper/supplementary/`
- manuscript or SI figure assets
- experiments
- `tools/`
- `paper_repro/`
- canonical result files

Figures are included from existing manuscript and SI figure assets. No figure was regenerated for this response package.

## Build

Build from this directory:

```powershell
latexmk -xelatex -interaction=nonstopmode -halt-on-error response_letter.tex
```

Latest local build result:

- Status: passed.
- PDF pages before this conversion: 25.
- PDF pages after this conversion: 49.
- Page size: A4.
- PDF file: `response_letter.pdf`.
- Expected warnings: locale warning, copied-template package-name warning, minor underfull hbox warnings, and embedded-figure PDF-version warnings.

## QA

Commands used:

```powershell
uv run python check_response_evidence.py response_letter.tex
latexmk -xelatex -interaction=nonstopmode -halt-on-error response_letter.tex
C:\texlive\2024\bin\windows\pdfinfo.exe .\response_letter.pdf
pdffonts .\response_letter.pdf
pdftotext -layout -enc UTF-8 .\response_letter.pdf - | Out-File -LiteralPath .\response_letter.txt -Encoding utf8
```

Latest QA results:

- `check_response_evidence.py`: passed.
- Comment IDs in source: 33/33.
- `Response.` headings in extracted PDF text: 33.
- `Revisions made in the manuscript.` headings in extracted PDF text: 33.
- `Relevant revised manuscript and supporting evidence.` headings in extracted PDF text: 33.
- `revisionbox` blocks in source: 33.
- Local reference blocks present for literature-oriented comments.
- `pdfinfo`: passed with 49 A4 pages.
- `pdffonts`: no Type 3 fonts found.
- `pdftotext`: text extraction passed.
- Forbidden lazy-phrase scan: passed.
- Forbidden internal-text scan: passed.
- Strong-claim scan: only hits reviewer-original quoted text in comment boxes, not author response text.

Known PDF-font note: included protected figure PDFs still carry some non-embedded TrueType fonts, inherited from the existing figure assets. This response task did not alter those figures.

## Manual Checks Remaining

- Add final manuscript/SI line numbers if Applied Energy requires line-specific references.
- Human visual review of `response_letter.pdf` is still needed before submission.
- Confirm final synchronization after any human edits to the manuscript, SI, or response letter.

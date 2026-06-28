# Round 2 Response Letter Draft

This directory contains the draft point-by-point response letter for the Applied Energy second-round revision.

## Files

- `response_letter.tex`: main response letter source.
- `response_letter.pdf`: compiled local PDF after the build step.
- `response_matrix.csv`: reviewer-comment coverage matrix.
- `response_input_manifest.json`: input, template, and build manifest.
- `template/response_slu.sty`: copied style file from the user-specified template directory.
- `template/RESPONSE_LETTER_CHECKLIST.md`: copied template checklist.

## Template

Template path used:

`D:\Code\Latex_Template4Writing\resopnse-slu`

The alternative spelling requested for checking, `D:\Code\Latex_Template4Writing\response-slu`, did not exist during drafting. The template repository was read only; only the necessary style/checklist files were copied into this response package.

## Inputs

Available input files:

- `D:\桌面\Comments02.txt`
- `D:\桌面\Paper02R101.md`
- `D:\桌面\Paper02R102.md`
- `paper\manuscript\manuscript_clean.tex`
- `paper\manuscript\manuscript_highlighted.tex`
- `paper\supplementary\supplementary_information.tex`
- `research\reviewer-round-02\canonical-result-lock.md`
- `research\reviewer-round-02\canonical-result-registry.json`
- `research\reviewer-round-02\canonical-benchmark-reference.json`
- `research\reviewer-round-02\manuscript-change-input.md`
- `research\reviewer-round-02\figure-change-input.md`
- `research\reviewer-round-02\round2-caption-drafts.md`
- `research\reviewer-round-02\round2-table-plan.md`

Missing exact-PDF inputs at draft time:

- `manuscript_clean.pdf`
- `manuscript_highlighted.pdf`
- `supplementary_information.pdf`

Those missing PDFs are recorded as manual-check items in `response_input_manifest.json`. The response draft relies on the current TeX sources and locked result files rather than guessing substitute PDF paths.

## Build

The template requires XeLaTeX or LuaLaTeX because it uses `fontspec`. Build from this directory:

```powershell
latexmk -xelatex -interaction=nonstopmode -halt-on-error response_letter.tex
```

Then extract text for QA:

```powershell
pdftotext response_letter.pdf response_letter.txt
pdffonts response_letter.pdf
pdfinfo response_letter.pdf
```

## Coverage

The response matrix covers all comments in `Comments02.txt`:

- Reviewer 1: 15 comments.
- Reviewer 2: 8 comments.
- Reviewer 3: 5 comments.
- Reviewer 4: 5 comments.
- Total: 33 comments.

## Manual Checks

- Add final line numbers only after the final line-numbered clean and highlighted PDFs are exported, if the journal requires line references.
- Confirm the exact PDF input paths and hashes when `manuscript_clean.pdf`, `manuscript_highlighted.pdf`, and `supplementary_information.pdf` are exported.
- Visually inspect `response_letter.pdf` before submission.
- Confirm that the response letter remains synchronized with the final manuscript/SI wording after any human edits.

# Manual Figures Finalization

## Final preferred artifacts
- Fig. 1: `paper/manuscript/figures/round2_candidate/manual/fig1.pdf`.
- Fig. 2: `paper/manuscript/figures/round2_candidate/manual/fig2.pdf`.
- Fig. 3: `paper/manuscript/figures/round2_candidate/manual/fig3.pdf`.

## QA policy
- All three PDFs must render to non-empty PNG previews.
- All three PDFs must avoid Type 3 fonts and forbidden reviewer-claim wording.
- Fig. 2 is checked as strict Arial; Fig. 1 and Fig. 3 preserve embedded symbol/equation fonts from the manual Visio export.
- The builder records SHA-256 hashes, `pdfinfo`, `pdffonts`, and extracted text in metadata and `visual_qa_summary.json`.

## Scope lock
- The automated workflow does not edit, crop, recolor, or regenerate these manual PDFs.
- Old TeX-based Fig. 2/Fig. 3 candidates are intentionally removed from the preferred candidate set.
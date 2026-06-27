# Reference Style Audit

Scope: PR #4 fifth human-review pass.

## Rendering Decision

- `paper/manuscript/references.tex` is the authoritative rendered reference source for the clean and highlighted manuscript.
- `paper/manuscript/references.bib` is retained only as source metadata and is not called by the submission TeX entrypoints.
- DOI URLs are emitted only when Crossref returned an exact normalized title match consistent with the local publication venue or when the user provided final confirmation for the specific entry.

## DOI Status

Verified DOI URLs were added in `references.tex` for:

- `ref1`
- `ref2`
- `ref4`
- `ref6`
- `ref7`
- `ref3`
- `ref5`
- `ref41`
- `wang2026unlocking`
- `wang2026climateResilient`
- `ref8`
- `ref18`
- `ref14`
- `ref15`
- `ref19`
- `ref20`
- `ref9`
- `ref10`
- `ref11`
- `ref12`
- `ref16`
- `ref17`
- `ref24`
- `ref13`
- `ref22`
- `ref23`
- `ref25`
- `ref26`
- `ref31`
- `ref27`
- `ref28`
- `ref29`
- `ref30`
- `ref32`
- `ref33`
- `ref34`
- `ref35`
- `ref36`
- `ref37`
- `ref38`
- `ref39`
- `ref40`

All user-confirmed DOI URLs have been added.

## Final user-confirmed DOI additions

The following DOI URLs were added in the final PR #4 cleanup:

- `ref4`: https://doi.org/10.1016/j.enbuild.2024.115224
- `ref11`: https://doi.org/10.1016/j.apenergy.2016.09.027
- `ref17`: https://doi.org/10.1016/j.jobe.2024.109304
- `ref23`: https://doi.org/10.1016/j.apenergy.2024.124003
- `ref32`: https://doi.org/10.48550/arXiv.1509.02971
- `ref33`: https://doi.org/10.1016/j.apenergy.2020.116117

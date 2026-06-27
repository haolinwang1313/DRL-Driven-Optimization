# Reference Style Audit

Scope: PR #4 fifth human-review pass.

## Rendering Decision

- `paper/manuscript/references.tex` is the authoritative rendered reference source for the clean and highlighted manuscript.
- `paper/manuscript/references.bib` is retained only as source metadata and is not called by the submission TeX entrypoints.
- DOI URLs are emitted only when Crossref returned an exact normalized title match consistent with the local publication venue. Mismatched first hits, such as SSRN or book-chapter records for journal entries, were not used.

## DOI Status

Verified DOI URLs were added in `references.tex` for:

- `ref1`
- `ref2`
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
- `ref12`
- `ref16`
- `ref24`
- `ref13`
- `ref22`
- `ref25`
- `ref26`
- `ref31`
- `ref27`
- `ref28`
- `ref29`
- `ref30`
- `ref34`
- `ref35`
- `ref36`
- `ref37`
- `ref38`
- `ref39`
- `ref40`

Entries still marked `doi_missing_or_unverified`:

- `ref4`: Crossref first hit was an SSRN record, while the local source metadata lists `Energy and Buildings`.
- `ref11`: Crossref title differed from the local source title.
- `ref17`: Crossref title differed from the local source title.
- `ref23`: Crossref first hit was an SSRN record, while the local source metadata lists `Applied Energy`.
- `ref32`: Crossref first hit did not match the local arXiv preprint.
- `ref33`: Crossref first hit was a book-chapter record, while the local source metadata lists `Applied Energy`.

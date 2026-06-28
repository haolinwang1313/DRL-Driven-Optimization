# Response Letter Checklist for `resopnse-slu`

Use this checklist before compiling a response letter.

## A. Global structure

- [ ] The response letter has an opening letter.
- [ ] The response letter has a summary table of major revisions.
- [ ] Reviewer 1 starts on a new page.
- [ ] Reviewer 2 starts on a new page.
- [ ] Reviewer 3 starts on a new page.
- [ ] Each reviewer comment is reproduced in `commentbox`.
- [ ] Each comment has exactly one `Response.` heading.
- [ ] Each comment has a `Revisions made in the manuscript.` section unless the comment is purely editorial.
- [ ] Each substantive comment has `Relevant revised manuscript and supporting evidence.`.
- [ ] No upfront `Revision evidence dossier` is used.

## B. Evidence placement

- [ ] Every cited main-text figure appears under the relevant comment.
- [ ] Every cited supplementary table has a compact excerpt or clear local evidence.
- [ ] Equation-related comments show the revised equations.
- [ ] Figure-only evidence blocks include the manuscript paragraph that interprets the figure.
- [ ] Table-only evidence blocks include the manuscript or supplementary paragraph that explains the table.
- [ ] No figure is duplicated both outside and inside `revisionbox` under the same comment.
- [ ] Mixed-location evidence uses `\revisionseparator`.

## C. Literature comments

- [ ] Each added literature reference appears under the relevant comment in `responsereferences`.
- [ ] The discussion paragraph containing the added citation is shown in `revisionbox`.
- [ ] Reference metadata includes title, journal, year, pages if available, and DOI if available.

## D. Model-scope comments

- [ ] Planning-scale spatial abstraction is distinguished from feeder-level validation.
- [ ] Chronological stress-period diagnostics are distinguished from full-year capacity-expansion validation.
- [ ] Storage lifecycle-cost approximation is distinguished from electrochemical degradation modelling.
- [ ] Site-conditioned numerical envelope is distinguished from transferable numerical results.
- [ ] Reproducibility materials are separated from manuscript-writing and internal revision artifacts.

## E. Writing quality

- [ ] No author prose uses double quotation marks to soften imprecise terminology.
- [ ] No author prose contains `(e.g., ...)`, `(i.e., ...)`, or `(such as ...)`.
- [ ] No author prose uses template phrases such as `The revision makes this point explicit`.
- [ ] Strong reviewer comments receive multi-paragraph responses.
- [ ] Responses are specific, evidence-based, and non-defensive.

## F. Compilation and review

- [ ] Compile with XeLaTeX or LuaLaTeX if required by the template.
- [ ] Check the PDF visually for commentbox and revisionbox colors.
- [ ] Check that page breaks occur before reviewer sections.
- [ ] Check the table of contents links.
- [ ] Render key pages for manual review if the response is long.

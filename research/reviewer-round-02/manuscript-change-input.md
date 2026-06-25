# Manuscript Change Input

## Canonical benchmark reference
- Use `benchmark-reference-v2` only, with reference hash `a972173040d6682fb41b794f65befc6efcc93a1616cb405262f3ab504ddeffcc`.
- Do not mix projected-local and fixed-reference HV/IGD in the same comparison table.

## Canonical benchmark numbers
- NSGA-II full archive: HV = 1.330999, IGD = 0.004947.
- NSGA-II projected feasible archive (fixed reference): HV = 1.229611, IGD = 0.156912.
- Balanced DDPG projected feasible archive (fixed reference): HV = 0.990990, IGD = 0.215410.

## Equal-size wording
- `source_archive_size` is the retained archive size before downsampling.
- `effective_sample_size` is the sampled row count actually used in each repetition.
- Oversized requests must be labelled `not_applicable`, not reported as valid equal-size results.

## Physical and climate terminology
- Use `limited physics-based cross-model stress test`.
- Use `limited four-block cross-climate physical sensitivity analysis`.
- Do not use `successful physical validation`, `physical closure`, `external confirmation`, or `physical support for optimizer ranking`.
- Keep the evidence-level labels: execution closure = complete, metric agreement = weak, ranking transfer = unsupported, optimizer superiority under physical evaluation = unsupported.
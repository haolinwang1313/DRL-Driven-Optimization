# Round 2 Experiment Results

## Canonical result lock summary
- This stage does not rerun DDPG, NSGA-II, CMA-ES, RandomSearch, physical validation, or climate sensitivity.
- The projected HV/IGD conflict came from two different reference definitions: projected-local reference in `optimizer-projection-summary.csv` versus fixed benchmark reference in `benchmark-metric-definition-audit.csv`.
- The canonical benchmark reference is `benchmark-reference-v2` with hash `a972173040d6682fb41b794f65befc6efcc93a1616cb405262f3ab504ddeffcc`.

## Canonical projected metrics
- NSGA-II projected feasible HV/IGD (fixed reference) = `1.229611` / `0.156912`.
- Balanced DDPG projected feasible HV/IGD (fixed reference) = `0.990990` / `0.215410`.

## Canonical fairness metrics
- NSGA-II full-archive HV/IGD = `1.330999` / `0.004947`.
- Balanced DDPG equal-size-20 HV/IGD mean = `0.661912` / `0.465522`.

## Physical and climate wording
- Physical evidence is now locked as `limited physics-based cross-model stress test`.
- Cross-climate evidence is now locked as `limited four-block cross-climate physical sensitivity analysis`.
- Physical evidence level: execution closure = `complete`, metric agreement = `weak`, ranking transfer = `unsupported`.

## Canonical files
- `research/reviewer-round-02/canonical-benchmark-reference.json`
- `research/reviewer-round-02/canonical_benchmark_metrics.csv`
- `research/reviewer-round-02/benchmark_equal_size_repetitions_v2.csv`
- `research/reviewer-round-02/benchmark_equal_size_summary_v2.csv`
- `research/reviewer-round-02/metric-lineage-audit.md`
- `research/reviewer-round-02/optimizer-output-contract.csv`
- `research/reviewer-round-02/hv-ceiling-interpretation.md`
- `research/reviewer-round-02/canonical-result-registry.json`
- `research/reviewer-round-02/canonical-result-lock.md`
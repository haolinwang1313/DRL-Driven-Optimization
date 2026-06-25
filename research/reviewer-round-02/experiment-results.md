# Round 2 Experiment Results

## Executive summary
- Run ID: `20260625_round2_closure`.
- Canonical dataset: `artifacts/server_runs/20260405_highest_precision_2000_compare/data/simulated_samples.csv` with SHA-256 `b8bc287ad3d9c8db9f7e090630fd3fe1f2276e0d3f8072f3fa13adcee8cba5cc`.
- Canonical surrogate: `artifacts/server_runs/20260405_highest_precision_2000_compare/models/surrogate.pt` with SHA-256 `85fdc5b36f69fb610e224a66d841021c188471c5874966cfeca50b9cd9cf3af4`.
- Current remote physical batch status: `completed`.

## Experiment completion matrix
- sampling: completed
- surrogate_validation: completed
- benchmark_full_archive: completed
- feasibility: completed
- physical: completed
- climate: completed

## Ten most important new results
- Sampling dependencies remain exact to floating tolerance: FAR-BD*AF max residual = 1.332e-15.
- Sampling dependencies remain exact to floating tolerance: OSR-(1-BD)/FAR max residual = 2.220e-16.
- The 2000-row descriptor space needs 6 principal components for 95% variance.
- Repeated 5x5 CV mean nMAE = EG 0.0265, EUIt 0.0176, H 0.0145.
- Leave-one-OSLI-out mean nMAE = EG 0.0342, EUIt 0.0185, H 0.0185.
- NSGA-II full archive HV/IGD = 1.330999/0.004947.
- Balanced DDPG full archive HV/IGD = 0.661912/0.465522.
- Balanced CMA-ES full archive HV/IGD = 1.331000/0.004946.
- NSGA-II candidate projection collapse rate = 0.9745.
- NSGA-II unique matched feasible blocks = 51.

## Physical validation and cross-climate results
- Physical validation status: `completed`.
- Physical job id: `981003a345cc`.
- Annual irradiance status: `unavailable`.
- Climate sensitivity status: `completed`.
- Climate blocker or note: `station-level climate summary available`.
- Climate weather manifest: `artifacts\reviewer_round_02\20260625_round2_closure\climate\climate_weather_manifest.json`.

## Data coverage results
- Sampling method summary: `artifacts\reviewer_round_02\20260625_round2_closure\data\sampling_method_summary.json`.
- Descriptor coverage table: `artifacts\reviewer_round_02\20260625_round2_closure\data\sampling_coverage_summary.csv`.

## Surrogate validation results
- Surrogate validation summary: `artifacts\reviewer_round_02\20260625_round2_closure\models\surrogate_validation_summary.csv`.

## Full archive and equal-size benchmark
- Full archive summary: `artifacts\reviewer_round_02\20260625_round2_closure\optimization\benchmark_full_archive.csv`.
- Equal-size summary: `artifacts\reviewer_round_02\20260625_round2_closure\optimization\benchmark_equal_size_summary.csv`.

## CMA-ES and RandomSearch
- CMA-ES summary: `artifacts\reviewer_round_02\20260625_round2_closure\optimization\cmaes_summary_round2.json`.
- RandomSearch summary: `artifacts\reviewer_round_02\20260625_round2_closure\optimization\random_search_summary_round2.json`.

## Descriptor feasibility and projection
- Projection summary: `artifacts\reviewer_round_02\20260625_round2_closure\optimization\optimizer_projection_summary.csv`.

## Computation efficiency
- Runtime audit: `artifacts\reviewer_round_02\20260625_round2_closure\optimization\runtime_audit.csv`.

## Impact on manuscript conclusions
- The current evidence remains bounded to surrogate-conditioned benchmarking and descriptor-space design support.
- Physical validation completed, but its large EUIt/H error and weak rank preservation still do not support a strong physical-certification claim.
- HV saturation near 1.331 must be described as reference-point saturation, not as archive richness by itself.

## Old tables or figures that must be retired or revised
- Any reward equation using `10^6 - d_weighted` must be replaced.
- Any wording that treats Fig. 9(d) post-hoc utility as training reward must be removed.
- Any figure or text that treats the 12 inputs as independent design variables must be revised.

## Results that can enter the main text now
- Sampling-coverage diagnostics.
- Surrogate validation metrics.
- Equal-size benchmark fairness diagnostics.
- Descriptor projection-sensitivity limitations.

## Results that should stay in appendix or remain pending
- Detailed physical per-case diagnostics and optimizer-linked gap decomposition are better suited to appendix tables.
- Climate sensitivity should remain framed as limited cross-climate physical sensitivity analysis, not as a generalization proof.

## Conclusions that must be removed or kept bounded
- Broad DRL-superiority wording.
- Any claim that physical validation establishes optimizer superiority or broad climate transfer.

## Next-phase exact figure data sources
- Coverage: `artifacts\reviewer_round_02\20260625_round2_closure\data\sampling_coverage_summary.csv` and `artifacts\reviewer_round_02\20260625_round2_closure\data\descriptor_dependencies.csv`.
- Fairness: `artifacts\reviewer_round_02\20260625_round2_closure\optimization\benchmark_full_archive.csv` and `artifacts\reviewer_round_02\20260625_round2_closure\optimization\benchmark_equal_size_summary.csv`.
- HV saturation: `artifacts\reviewer_round_02\20260625_round2_closure\optimization\hv_saturation_diagnostic.json` and `artifacts\reviewer_round_02\20260625_round2_closure\optimization\benchmark_metric_definition_audit.csv`.
- Feasibility: `artifacts\reviewer_round_02\20260625_round2_closure\optimization\optimizer_projection_summary.csv` and `artifacts\reviewer_round_02\20260625_round2_closure\optimization\projected_utility_comparison.csv`.
- Physical: `artifacts\reviewer_round_02\20260625_round2_closure\physical\physical_validation_summary.json` for current run state, then result CSVs after completion.
- Climate: `artifacts\reviewer_round_02\20260625_round2_closure\climate\climate_sensitivity_summary.csv` and `artifacts\reviewer_round_02\20260625_round2_closure\climate\climate_rank_stability.csv`.
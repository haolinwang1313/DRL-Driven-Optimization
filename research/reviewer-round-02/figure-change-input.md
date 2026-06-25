# Figure Change Input

- Fig. 1 reward formula should be replaced with the implemented normalized-distance reward.
- Fig. 1 text extraction still shows `street loactions`, `Acotr netword`, and `netword` in the audit baseline.
- Fig. 2 should clarify episode-versus-step placement in the DDPG loop.
- Fig. 3 should label surrogate input descriptors and network outputs explicitly.
- Post-Fig. 4 rebuild should source data from the round-2 artifact CSVs, not hand-copied tables.
- Coverage figures should use `artifacts\reviewer_round_02\20260625_round2_closure\data\sampling_coverage_summary.csv` and `artifacts\reviewer_round_02\20260625_round2_closure\data\descriptor_dependencies.csv`.
- Fairness figures should use `artifacts\reviewer_round_02\20260625_round2_closure\optimization\benchmark_full_archive.csv` and `artifacts\reviewer_round_02\20260625_round2_closure\optimization\benchmark_equal_size_summary.csv`.
- HV saturation panels should use `artifacts\reviewer_round_02\20260625_round2_closure\optimization\hv_saturation_diagnostic.json` and `artifacts\reviewer_round_02\20260625_round2_closure\optimization\benchmark_metric_definition_audit.csv`.
- Feasibility figures should use `artifacts\reviewer_round_02\20260625_round2_closure\optimization\optimizer_projection_summary.csv` and `artifacts\reviewer_round_02\20260625_round2_closure\optimization\projected_utility_comparison.csv`.
- Physical-validation figures should use the completed batch from job `981003a345cc`.
- Climate figures should use `artifacts\reviewer_round_02\20260625_round2_closure\climate\climate_sensitivity_results.csv` and `artifacts\reviewer_round_02\20260625_round2_closure\climate\climate_rank_stability.csv`.
- Do not generate or commit formal figure PDFs in this stage.
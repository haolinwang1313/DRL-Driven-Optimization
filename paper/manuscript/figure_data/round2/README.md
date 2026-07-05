# Round 2 Figure Data Package

- Generated at: `2026-07-05T04:02:36+00:00`
- Build commit: `b6dee545d094929de30b8381fdd770e8852c95ed`
- Canonical reference hash: `a972173040d6682fb41b794f65befc6efcc93a1616cb405262f3ab504ddeffcc`

## Files
### descriptor_coverage.csv
- Description: Coverage, PCA, OSLI frequency, and nearest-neighbor summaries for the 2000-row canonical descriptor dataset.
- Representation family: `sample_coverage`
- SHA-256: `c9c77a2738ec0cb359d6a6801b67689b8d2c4e46c632d7e21cfbe4fefbfe8897`
- Source files: `artifacts/reviewer_round_02/20260625_round2_closure/data/sampling_coverage_summary.csv` (b9ad291c515b8399cb5c108fd5ff0f8a5a9d7a2457c4b633fc4d25ddfd1535f4), `artifacts/reviewer_round_02/20260625_round2_closure/data/simulated_samples.csv` (b8bc287ad3d9c8db9f7e090630fd3fe1f2276e0d3f8072f3fa13adcee8cba5cc)
- Generation command: `uv run python tools/build_round2_revision_figures.py --data-root paper/manuscript/figure_data/round2 --output-dir paper/manuscript/figures/round2_candidate --build-gallery`
- Valid for main text: `True`
- Valid for appendix: `True`
- Claim boundary: Supports descriptor-space coverage and PCA breadth only; not an optimizer ranking.

### descriptor_dependencies.csv
- Description: Descriptor dependency diagnostics and algebraic residual summaries.
- Representation family: `sample_coverage`
- SHA-256: `d94f1ded512e1f1bd284a428c162e291724aa585739babe712d478a79aeeacf6`
- Source files: `artifacts/reviewer_round_02/20260625_round2_closure/data/descriptor_dependencies.csv` (0ed4aa36cbb9f71d163140afa54a44a229fcd4c7ee2710b939b8d825fdf59d2d)
- Generation command: `uv run python tools/build_round2_revision_figures.py --data-root paper/manuscript/figure_data/round2 --output-dir paper/manuscript/figures/round2_candidate --build-gallery`
- Valid for main text: `True`
- Valid for appendix: `True`
- Claim boundary: Records descriptor algebra and PCA threshold facts only.

### surrogate_parity_mean_predictions.csv
- Description: Repeated-kfold out-of-fold sample-level mean predictions for parity plots.
- Representation family: `repeated_cv`
- SHA-256: `5837b986d33d904cf641a57584e85718cbc64d64bffdb762665adf8886fe8c07`
- Source files: `artifacts/reviewer_round_02/20260625_round2_closure/models/surrogate_validation_predictions.csv` (224c76d6371c582aa049a2ef79b749f9f29f00808f4c1e141073e9d9f43fcb98)
- Generation command: `uv run python tools/build_round2_revision_figures.py --data-root paper/manuscript/figure_data/round2 --output-dir paper/manuscript/figures/round2_candidate --build-gallery`
- Valid for main text: `True`
- Valid for appendix: `True`
- Claim boundary: Cross-validated surrogate predictions remain analytic-target surrogates, not EnergyPlus truth.

### surrogate_validation_regimes.csv
- Description: Validation-family summary used for robustness heatmaps.
- Representation family: `surrogate_validation`
- SHA-256: `4a35f8c39487082357292a0d50ae4b7f4061b284a820e5ac298598e89528b016`
- Source files: `research/reviewer-round-02/surrogate-validation-summary.csv` (07541b44af7b5aeca0ffb48c07f27ae993d657c0c609cea8788f39590043d7c2)
- Generation command: `uv run python tools/build_round2_revision_figures.py --data-root paper/manuscript/figure_data/round2 --output-dir paper/manuscript/figures/round2_candidate --build-gallery`
- Valid for main text: `True`
- Valid for appendix: `True`
- Claim boundary: These rows assess surrogate accuracy against the analytic response generator only.

### ddpg_training_curves_summary.csv
- Description: Seed-aggregated DDPG training curves across scenarios and targets.
- Representation family: `training_dynamics`
- SHA-256: `36a74af74f7465ab3b48084815e31b36403507d14c93014468752fa8b4cd61ba`
- Source files: `artifacts/publication/optimization/ddpg_logs_all_guardrail_full.json` (5569b5389f0c1d0520bf9ee5c1c132e9810c9a14e14aab8f7644fec720139fd8)
- Generation command: `uv run python tools/build_round2_revision_figures.py --data-root paper/manuscript/figure_data/round2 --output-dir paper/manuscript/figures/round2_candidate --build-gallery`
- Valid for main text: `True`
- Valid for appendix: `True`
- Claim boundary: One episode equals 40 sequential surrogate queries; this is serialized black-box search, not physical time evolution.

### ddpg_seed_diagnostics.csv
- Description: Per-seed DDPG plateau, regression, and episode-return diagnostics.
- Representation family: `training_dynamics`
- SHA-256: `bf356c2b98596fc41923a72ad91bbebfa7066b35d3f42e3b6fc17369ea7c21b6`
- Source files: `artifacts/publication/optimization/ddpg_logs_all_guardrail_full.json` (5569b5389f0c1d0520bf9ee5c1c132e9810c9a14e14aab8f7644fec720139fd8)
- Generation command: `uv run python tools/build_round2_revision_figures.py --data-root paper/manuscript/figure_data/round2 --output-dir paper/manuscript/figures/round2_candidate --build-gallery`
- Valid for main text: `False`
- Valid for appendix: `True`
- Claim boundary: Seed-level DDPG diagnostics are appendix-only and do not imply physical validation.

### benchmark_utility.csv
- Description: Per-seed best utility summaries for fixed-domain and legacy post-hoc utility.
- Representation family: `post_hoc_utility`
- SHA-256: `c8dafd40dd57df9bdf21fbf5d5f8af3a798b1f82a03b414c9b1fe0e5c9be13de`
- Source files: `artifacts/reviewer_round_02/20260625_round2_closure/optimization/utility_sensitivity.csv` (2cca0fa0330e01fae41d0ec5d27ed8cc05ab4395f003806b0dac2de3d6c0b3bd)
- Generation command: `uv run python tools/build_round2_revision_figures.py --data-root paper/manuscript/figure_data/round2 --output-dir paper/manuscript/figures/round2_candidate --build-gallery`
- Valid for main text: `True`
- Valid for appendix: `True`
- Claim boundary: Post-hoc fixed-domain utility is a bounded analytic comparison, not the DDPG training reward.

### benchmark_equal_size_20.csv
- Description: All valid equal-size-20 benchmark repetitions with the canonical benchmark reference.
- Representation family: `descriptor_equal_size_archive`
- SHA-256: `f60b07256e7b64ea2f3197c7f428c999857666ae8cb04558971f8269092c7168`
- Source files: `research/reviewer-round-02/benchmark_equal_size_repetitions_v2.csv` (882d9060d168c441af81cb0f2de52e69898b8fb23bd703d18e77152bfa352e81), `research/reviewer-round-02/canonical-benchmark-reference.json` (9b1163ae8d52d0a554554b7aee7c0b5bd6acc85139c3deea0434737755779c55)
- Generation command: `uv run python tools/build_round2_revision_figures.py --data-root paper/manuscript/figure_data/round2 --output-dir paper/manuscript/figures/round2_candidate --build-gallery`
- Valid for main text: `True`
- Valid for appendix: `True`
- Claim boundary: Only truthful requested-size-20 rows are retained; oversized DDPG and FeasiblePoolRandom requests are excluded.

### benchmark_output_contract_counts.csv
- Description: Optimizer output contract counts and retained-object metadata.
- Representation family: `output_contract`
- SHA-256: `83c911df9c51878df891b3ab7ca635f3a6e1822fede8caab09e51013cd756c65`
- Source files: `research/reviewer-round-02/optimizer-output-contract.csv` (83c911df9c51878df891b3ab7ca635f3a6e1822fede8caab09e51013cd756c65)
- Generation command: `uv run python tools/build_round2_revision_figures.py --data-root paper/manuscript/figure_data/round2 --output-dir paper/manuscript/figures/round2_candidate --build-gallery`
- Valid for main text: `True`
- Valid for appendix: `True`
- Claim boundary: These counts describe retained output structure and comparability limits, not direct optimizer superiority.

### benchmark_hv_ceiling.csv
- Description: Canonical HV ceiling diagnostics parsed from the locked round-2 interpretation.
- Representation family: `hv_ceiling`
- SHA-256: `fa862502f1966a89702ac730992db9f642cf304b3a6949d879a9207b370c4f4a`
- Source files: `research/reviewer-round-02/hv-ceiling-interpretation.md` (f84083f902887fe006437a0d4f85d87ab003cd3d05e19e782ee9707207238968), `research/reviewer-round-02/canonical-benchmark-reference.json` (9b1163ae8d52d0a554554b7aee7c0b5bd6acc85139c3deea0434737755779c55)
- Generation command: `uv run python tools/build_round2_revision_figures.py --data-root paper/manuscript/figure_data/round2 --output-dir paper/manuscript/figures/round2_candidate --build-gallery`
- Valid for main text: `False`
- Valid for appendix: `True`
- Claim boundary: HV saturation indicates reference-volume coverage and must be interpreted with duplicate and tuple-count diagnostics.

### feasible_projection_summary.csv
- Description: Method-level projection collapse and canonical before/after HV/IGD summaries.
- Representation family: `projected_feasible_morphology_archive`
- SHA-256: `16f00a0bdf86e0a5e726af8826cbce84445959d8635b1a52e7ebf8b2a0b7e9bd`
- Source files: `research/reviewer-round-02/canonical_benchmark_metrics.csv` (3156f07790201af6e2cf5a213af73ff8042e71fdfa6d7c4ddbd17f7db2cfe523), `research/reviewer-round-02/optimizer-projection-summary.csv` (5180b52b1b5897843214dc79a80889efe62bc1097b642879e1e7c3f43fbd0073)
- Generation command: `uv run python tools/build_round2_revision_figures.py --data-root paper/manuscript/figure_data/round2 --output-dir paper/manuscript/figures/round2_candidate --build-gallery`
- Valid for main text: `True`
- Valid for appendix: `True`
- Claim boundary: Projection diagnostics measure descriptor-to-feasible representation sensitivity only; they are not physical validation.

### feasible_projection_metrics.csv
- Description: Candidate-level projection rows used for distance distributions.
- Representation family: `projected_feasible_morphology_archive`
- SHA-256: `1132a919d3db4aae0454a314b4a981f6045762acf5b8c48a111426ab4f99b75b`
- Source files: `artifacts/reviewer_round_02/20260625_round2_closure/optimization/optimizer_feasibility_audit.csv` (e00091f4fcacce8d91fd2210117242c9188543f7176aa87f93b4d81c9c372154)
- Generation command: `uv run python tools/build_round2_revision_figures.py --data-root paper/manuscript/figure_data/round2 --output-dir paper/manuscript/figures/round2_candidate --build-gallery`
- Valid for main text: `True`
- Valid for appendix: `True`
- Claim boundary: Candidate-level projection distances are descriptor-space diagnostics only.

### physical_direct_cases.csv
- Description: Direct feasible physical-evaluation cases only; optimizer-linked cases are excluded.
- Representation family: `physical_evaluated_subset`
- SHA-256: `0403f94ea9bd6d4d089b930723e08f860cdf3da726f99dcaaee038fa6b7f39d3`
- Source files: `artifacts/reviewer_round_02/20260625_round2_closure/physical/physical_validation_results.csv` (110ba9c0c4c725fd3cccf2b1332031cdde41339bc61ba2381c12245d8f1b515b)
- Generation command: `uv run python tools/build_round2_revision_figures.py --data-root paper/manuscript/figure_data/round2 --output-dir paper/manuscript/figures/round2_candidate --build-gallery`
- Valid for main text: `True`
- Valid for appendix: `True`
- Claim boundary: These 18 rows support only the limited physics-based cross-model stress test.

### physical_stress_metrics.csv
- Description: Summary metrics for the limited physics-based cross-model stress test.
- Representation family: `physical_evaluated_subset`
- SHA-256: `cf9ec8e791d55efe5edafe4a1f1cd779523801e0d6bce8bbbdcadd03bd7f4303`
- Source files: `research/reviewer-round-02/physical-validation-metrics.csv` (6dc6c48f9f4dfabff5a3ea4a54437a9201b852dbcc13e0ae704b0a03456abf0f)
- Generation command: `uv run python tools/build_round2_revision_figures.py --data-root paper/manuscript/figure_data/round2 --output-dir paper/manuscript/figures/round2_candidate --build-gallery`
- Valid for main text: `True`
- Valid for appendix: `True`
- Claim boundary: Metric agreement remains weak and ranking transfer unsupported.

### optimizer_linked_physical_gaps.csv
- Description: Gap decomposition for the six optimizer-linked physical cases.
- Representation family: `physical_evaluated_subset`
- SHA-256: `79d6291c83058eab43693346035444a30e2b97f3bac28ed993a11affe53e3383`
- Source files: `research/reviewer-round-02/physical-validation-optimizer-mapping.csv` (b612f84753d5e9c01e584ba3a032e739dd04576cad67ad9d5a0b1919961e0893)
- Generation command: `uv run python tools/build_round2_revision_figures.py --data-root paper/manuscript/figure_data/round2 --output-dir paper/manuscript/figures/round2_candidate --build-gallery`
- Valid for main text: `False`
- Valid for appendix: `True`
- Claim boundary: Optimizer-linked cases are appendix-only and must not be mixed into direct-case parity plots.

### climate_case_results.csv
- Description: Per-case cross-climate physical sensitivity rows for Beijing, Guangzhou, and Harbin.
- Representation family: `limited_four_block_cross_climate_physical_sensitivity_analysis`
- SHA-256: `f624253730cdf63ce60f4e774516b4854f0c3dc6a33c4b2580a02b5df4bc3536`
- Source files: `research/reviewer-round-02/climate-sensitivity-results.csv` (9c7bdfb2789d31e19c6ad96a023dcbdbd52cfe60232064aba3f659a6667083fa)
- Generation command: `uv run python tools/build_round2_revision_figures.py --data-root paper/manuscript/figure_data/round2 --output-dir paper/manuscript/figures/round2_candidate --build-gallery`
- Valid for main text: `True`
- Valid for appendix: `True`
- Claim boundary: These rows cover exactly four direct feasible blocks across three additional climates and do not prove surrogate generalization.

### climate_summary.csv
- Description: Station-level mean climate sensitivity summary.
- Representation family: `limited_four_block_cross_climate_physical_sensitivity_analysis`
- SHA-256: `7a03bcf3714f398a72ad99f172320caac79dc865bc8989fc65d53a8262feb1e6`
- Source files: `research/reviewer-round-02/climate-sensitivity-summary.csv` (ac5d093411c7514d0fad23384c102f5fd1d81d1a8ece9efab0d0989b82f68e1a)
- Generation command: `uv run python tools/build_round2_revision_figures.py --data-root paper/manuscript/figure_data/round2 --output-dir paper/manuscript/figures/round2_candidate --build-gallery`
- Valid for main text: `True`
- Valid for appendix: `True`
- Claim boundary: Mean climate deltas summarize only four direct-feasible cases.

### climate_rank_stability.csv
- Description: Target-wise climate rank stability for the four direct-feasible cases.
- Representation family: `limited_four_block_cross_climate_physical_sensitivity_analysis`
- SHA-256: `ebd3f1d2ec549a98ab0173e104910fbcdf88df91a899446f8466dc6de30680dc`
- Source files: `research/reviewer-round-02/climate-rank-stability.csv` (c501c172b5f3d6aaf84b692edc9f699876171207e6d2c8ecf814e47beb8c88e7)
- Generation command: `uv run python tools/build_round2_revision_figures.py --data-root paper/manuscript/figure_data/round2 --output-dir paper/manuscript/figures/round2_candidate --build-gallery`
- Valid for main text: `True`
- Valid for appendix: `True`
- Claim boundary: Rank stability remains target-dependent and is not a climate-generalization proof.

### scale_study.csv
- Description: Dataset-scale surrogate-selection comparison across 500/1000/1500/2000 rows.
- Representation family: `scale_study`
- SHA-256: `87120cdcbdb9cdd09560f4778f0351398933affa460de2884913fdb6554e9bf7`
- Source files: `artifacts/server_runs/20260405_surrogate_rebenchmark/data/dataset_scale_summary.csv` (613a1daef6ff009432502297aab888cde969c39289c2ff8283aa9c3565c17f87), `artifacts/server_runs/20260405_surrogate_rebenchmark/models/surrogate_comparison.csv` (f5679cda7cd057b3aac8cbfe0bdd7b6aca7881307fdfc0293c538da0bb9668e8), `artifacts/server_runs/20260405_surrogate_rebenchmark/models/surrogate_regime_winners.csv` (bf1fa5d78772ab5cc115e04f72cad76e40a51b87efc5b5e1f0cfb72e0a0108e6)
- Generation command: `uv run python tools/build_round2_revision_figures.py --data-root paper/manuscript/figure_data/round2 --output-dir paper/manuscript/figures/round2_candidate --build-gallery`
- Valid for main text: `False`
- Valid for appendix: `True`
- Claim boundary: Scale-study rows trace surrogate-selection evidence only and are not physical validation.

### morphology_signatures.csv
- Description: Median and interquartile morphology descriptor signatures for representative groups.
- Representation family: `morphology_descriptor_signatures`
- SHA-256: `aa6df073123507e16a287b9407fb657b2875118f2be839926c4e4b4009287acd`
- Source files: `artifacts/server_runs/20260405_highest_precision_2000_compare/optimization/nsga2_results.csv` (b83015b321f3392781a50755e3889fb82f960853c4b201ee7b68591a54de0143), `artifacts/server_runs/20260405_highest_precision_2000_compare/optimization/ddpg_results.csv` (93ceef4839e69b313c176acfecb417d4cd9a6115cdf2c82e3e12305bab3930c6), `artifacts/reviewer_round_02/20260625_round2_closure/optimization/random_search_results_round2.csv` (d106b551952effc766dd5aff31318dcd4add0ff40646d23a19d1ee2ce9604365)
- Generation command: `uv run python tools/build_round2_revision_figures.py --data-root paper/manuscript/figure_data/round2 --output-dir paper/manuscript/figures/round2_candidate --build-gallery`
- Valid for main text: `False`
- Valid for appendix: `True`
- Claim boundary: Median morphology signatures are descriptive summaries of retained outputs, not stable design rules.

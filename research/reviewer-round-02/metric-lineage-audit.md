# Metric Lineage Audit

## Root cause
- `optimizer_projection_summary.csv` was generated with a projected-only local reference front.
- `benchmark-metric-definition-audit.csv` evaluated projected archives against the fixed benchmark reference.
- `benchmark-equal-size-summary.csv` overloaded `actual_size` with source archive size and silently kept oversized requests.

## Lineage records
- `research/reviewer-round-02/optimizer-projection-summary.csv` / `projected_HV/projected_IGD`
  status: `valid_but_local_reference`
  replacement: `research/reviewer-round-02/canonical_benchmark_metrics.csv`
  note: run_feasibility_audit rebuilt a projected-only reference front before computing HV/IGD.
- `research/reviewer-round-02/benchmark-metric-definition-audit.csv` / `projected_feasible_block_archive`
  status: `valid_fixed_reference`
  replacement: `research/reviewer-round-02/canonical_benchmark_metrics.csv`
  note: run_benchmark_fairness evaluated projected archives against the fixed benchmark reference.
- `research/reviewer-round-02/benchmark-equal-size-summary.csv` / `actual_size column`
  status: `metadata_error`
  replacement: `research/reviewer-round-02/benchmark_equal_size_summary_v2.csv`
  note: The summary stored source archive size in a column that reads like sampled row count.
- `research/reviewer-round-02/benchmark-equal-size-summary.csv` / `DDPG::Balanced_Performance requested_size=40`
  status: `obsolete`
  replacement: `research/reviewer-round-02/benchmark_equal_size_summary_v2.csv`
  note: Older tables reported oversized equal-size requests as if they were valid metric rows.
- `research/reviewer-round-02/benchmark-equal-size-summary.csv` / `DDPG::Balanced_Performance requested_size=60`
  status: `obsolete`
  replacement: `research/reviewer-round-02/benchmark_equal_size_summary_v2.csv`
  note: Older tables reported oversized equal-size requests as if they were valid metric rows.
- `research/reviewer-round-02/benchmark-equal-size-summary.csv` / `DDPG::Balanced_Performance requested_size=100`
  status: `obsolete`
  replacement: `research/reviewer-round-02/benchmark_equal_size_summary_v2.csv`
  note: Older tables reported oversized equal-size requests as if they were valid metric rows.
- `research/reviewer-round-02/benchmark-equal-size-summary.csv` / `FeasiblePoolRandom::Balanced_Performance requested_size=40`
  status: `obsolete`
  replacement: `research/reviewer-round-02/benchmark_equal_size_summary_v2.csv`
  note: Older tables reported oversized equal-size requests as if they were valid metric rows.
- `research/reviewer-round-02/benchmark-equal-size-summary.csv` / `FeasiblePoolRandom::Balanced_Performance requested_size=60`
  status: `obsolete`
  replacement: `research/reviewer-round-02/benchmark_equal_size_summary_v2.csv`
  note: Older tables reported oversized equal-size requests as if they were valid metric rows.
- `research/reviewer-round-02/benchmark-equal-size-summary.csv` / `FeasiblePoolRandom::Balanced_Performance requested_size=100`
  status: `obsolete`
  replacement: `research/reviewer-round-02/benchmark_equal_size_summary_v2.csv`
  note: Older tables reported oversized equal-size requests as if they were valid metric rows.
- `research/reviewer-round-02/benchmark-equal-size-summary.csv` / `DDPG::Energy_Saving_Focus requested_size=40`
  status: `obsolete`
  replacement: `research/reviewer-round-02/benchmark_equal_size_summary_v2.csv`
  note: Older tables reported oversized equal-size requests as if they were valid metric rows.
- `research/reviewer-round-02/benchmark-equal-size-summary.csv` / `DDPG::Energy_Saving_Focus requested_size=60`
  status: `obsolete`
  replacement: `research/reviewer-round-02/benchmark_equal_size_summary_v2.csv`
  note: Older tables reported oversized equal-size requests as if they were valid metric rows.
- `research/reviewer-round-02/benchmark-equal-size-summary.csv` / `DDPG::Energy_Saving_Focus requested_size=100`
  status: `obsolete`
  replacement: `research/reviewer-round-02/benchmark_equal_size_summary_v2.csv`
  note: Older tables reported oversized equal-size requests as if they were valid metric rows.
- `research/reviewer-round-02/benchmark-equal-size-summary.csv` / `FeasiblePoolRandom::Energy_Saving_Focus requested_size=40`
  status: `obsolete`
  replacement: `research/reviewer-round-02/benchmark_equal_size_summary_v2.csv`
  note: Older tables reported oversized equal-size requests as if they were valid metric rows.
- `research/reviewer-round-02/benchmark-equal-size-summary.csv` / `FeasiblePoolRandom::Energy_Saving_Focus requested_size=60`
  status: `obsolete`
  replacement: `research/reviewer-round-02/benchmark_equal_size_summary_v2.csv`
  note: Older tables reported oversized equal-size requests as if they were valid metric rows.
- `research/reviewer-round-02/benchmark-equal-size-summary.csv` / `FeasiblePoolRandom::Energy_Saving_Focus requested_size=100`
  status: `obsolete`
  replacement: `research/reviewer-round-02/benchmark_equal_size_summary_v2.csv`
  note: Older tables reported oversized equal-size requests as if they were valid metric rows.
- `research/reviewer-round-02/benchmark-equal-size-summary.csv` / `DDPG::Energy_Generation_Focus requested_size=40`
  status: `obsolete`
  replacement: `research/reviewer-round-02/benchmark_equal_size_summary_v2.csv`
  note: Older tables reported oversized equal-size requests as if they were valid metric rows.
- `research/reviewer-round-02/benchmark-equal-size-summary.csv` / `DDPG::Energy_Generation_Focus requested_size=60`
  status: `obsolete`
  replacement: `research/reviewer-round-02/benchmark_equal_size_summary_v2.csv`
  note: Older tables reported oversized equal-size requests as if they were valid metric rows.
- `research/reviewer-round-02/benchmark-equal-size-summary.csv` / `DDPG::Energy_Generation_Focus requested_size=100`
  status: `obsolete`
  replacement: `research/reviewer-round-02/benchmark_equal_size_summary_v2.csv`
  note: Older tables reported oversized equal-size requests as if they were valid metric rows.
- `research/reviewer-round-02/benchmark-equal-size-summary.csv` / `FeasiblePoolRandom::Energy_Generation_Focus requested_size=40`
  status: `obsolete`
  replacement: `research/reviewer-round-02/benchmark_equal_size_summary_v2.csv`
  note: Older tables reported oversized equal-size requests as if they were valid metric rows.
- `research/reviewer-round-02/benchmark-equal-size-summary.csv` / `FeasiblePoolRandom::Energy_Generation_Focus requested_size=60`
  status: `obsolete`
  replacement: `research/reviewer-round-02/benchmark_equal_size_summary_v2.csv`
  note: Older tables reported oversized equal-size requests as if they were valid metric rows.
- `research/reviewer-round-02/benchmark-equal-size-summary.csv` / `FeasiblePoolRandom::Energy_Generation_Focus requested_size=100`
  status: `obsolete`
  replacement: `research/reviewer-round-02/benchmark_equal_size_summary_v2.csv`
  note: Older tables reported oversized equal-size requests as if they were valid metric rows.
# Data

This directory contains the public APEN data inventory and the canonical analytic-response dataset.

```text
data/
|-- catalog.yaml
|-- data_dictionary.md
|-- generated/
|   `-- canonical_2000/
|       |-- selected_dataset.public.json
|       |-- SHA256SUMS.txt
|       |-- simulated_blocks.jsonl
|       |-- simulated_samples.csv
|       `-- simulated_samples.meta.json
`-- README.md
```

`simulated_samples.csv` has 2000 rows and uses `fallback_analytic` target generation. `simulated_blocks.jsonl` has 2000 generated block records. See `catalog.yaml` for hashes, row counts, and release roles.

The external spreadsheet path `data/external/benchmark/dataset.xlsx` is recorded in the catalog as `not_included`; it is not required for the current public analytic-response benchmark package.

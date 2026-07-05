# Surrogate-Conditioned Benchmark Fragility in Block-Scale Urban Energy Design

[![Python](https://img.shields.io/badge/Python-%3E%3D3.10-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains a public Python package for surrogate-conditioned urban morphology analysis, including morphology generation, synthetic response construction, surrogate helpers, optimisation utilities, metrics, and figure helpers.

## Overview

The current archive includes:

- `paper_repro/`: public source modules for morphology generation, synthetic response construction, surrogate helpers, optimisation utilities, metrics, and figure helpers.
- `tests/`: smoke and regression tests for the public source modules.
- `data/`: public data inventory placeholders for release-time metadata.
- `docs/`: usage and reproducibility notes.

## Repository Structure

```text
.
|-- data/                        # Public data inventory, dictionaries, samples, or redistributable data
|   |-- catalog.yaml
|   |-- data_dictionary.md
|   `-- README.md
|-- docs/                        # Usage and reproducibility notes
|   |-- reproducibility.md
|   |-- usage.md
|-- paper_repro/                  # Public Python package
|   |-- config.py
|   |-- metrics.py
|   |-- morphology.py
|   |-- optimizers.py
|   |-- simulation.py
|   `-- surrogate.py
|-- tests/                       # Public tests or smoke checks
|   |-- conftest.py
|   |-- test_ddpg_reward_contract.py
|   |-- test_metrics.py
|   |-- test_morphology.py
|   |-- test_simulation_scale_study.py
|   `-- test_surrogate_selection.py
|-- pyproject.toml                # Python package metadata
|-- uv.lock                       # Locked dependency snapshot
|-- LICENSE                         # MIT License
`-- README.md                      # Project documentation
```

## Dependencies & Installation

Create a Python environment and install the public package. Install the project from `pyproject.toml`.

```bash
python -m venv .venv
python -m pip install -e ".[test]"
```

## Usage

Run all commands from the repository root. See `docs/usage.md` for public commands that match the files included in this release.

## Data Availability

See `data/README.md` and `data/catalog.yaml` for the public data inventory.

## License

This project is licensed under the MIT License. See `LICENSE`.

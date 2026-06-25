from __future__ import annotations

from pathlib import Path

from paper_repro.config import Config
from paper_repro.simulation import build_dataset_regimes


def _test_config(tmp_path: Path) -> Config:
    artifact_root = tmp_path / "artifacts"
    return Config(
        {
            "project": {
                "name": "paper02_repro",
                "random_seed": 20260310,
                "artifact_root": str(artifact_root),
                "benchmark_dataset": "missing.xlsx",
                "manuscript_pdf": "missing.pdf",
                "supplementary_pdf": "missing.pdf",
            },
            "weather": {
                "download": False,
                "output_dir": str(artifact_root / "weather"),
                "preferred_station": "Dongtai",
                "fallback_station": "Nanjing",
                "stations": {
                    "Dongtai": {"label": "Dongtai, Jiangsu", "url": "https://example.com/dongtai.zip"},
                    "Nanjing": {"label": "Nanjing, Jiangsu", "url": "https://example.com/nanjing.zip"},
                },
            },
            "simulation": {
                "n_samples": 500,
                "scale_study": {"scales": [500, 750]},
                "block_size_m": 240.0,
                "grid_size": 3,
                "land_unit_size_m": 80.0,
                "road_offset_m": 5.0,
                "setback_offset_m": 10.0,
                "floor_to_floor_height_m": 3.0,
                "try_install_sim_stack": False,
                "use_physical_stack_if_available": False,
                "fallback_noise_scale": {"EUIt": 0.22, "EG": 0.03, "H": 0.02},
            },
            "surrogate_selection": {"original_scale": 500},
            "report": {
                "figures_dir": str(artifact_root / "figures"),
                "models_dir": str(artifact_root / "models"),
                "data_dir": str(artifact_root / "data"),
                "optimization_dir": str(artifact_root / "optimization"),
                "bootstrap_dir": str(artifact_root / "bootstrap"),
                "reports_dir": str(artifact_root / "reports"),
            },
        }
    )


def test_build_dataset_regimes_writes_requested_scale_artifacts(tmp_path: Path) -> None:
    config = _test_config(tmp_path)
    regimes = build_dataset_regimes(config)
    assert set(regimes) == {500, 750}
    assert len(regimes[500]) == 500
    assert len(regimes[750]) == 750
    assert (tmp_path / "artifacts" / "data" / "dataset_scale_protocol.json").exists()
    assert (tmp_path / "artifacts" / "data" / "regimes" / "scale_750" / "simulated_samples.csv").exists()
    assert (tmp_path / "artifacts" / "data" / "simulated_samples.csv").exists()

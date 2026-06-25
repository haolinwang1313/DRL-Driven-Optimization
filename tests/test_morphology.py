from __future__ import annotations

import numpy as np

from paper_repro.morphology import block_to_features, generate_morphology_dataset, random_block


def test_generated_morphology_columns_and_ranges() -> None:
    frame = generate_morphology_dataset(
        n_samples=32,
        seed=7,
        block_size_m=240.0,
        land_unit_size_m=80.0,
        floor_height_m=3.0,
    )
    assert {"FAR", "SVF", "OSR", "theta", "OSLI"}.issubset(frame.columns)
    assert frame["OSLI"].between(0, 8).all()
    assert frame["theta"].between(-45, 45).all()
    assert frame["SVF"].between(0.05, 0.95).all()


def test_block_features_are_positive() -> None:
    rng = np.random.default_rng(11)
    block = random_block(rng, block_size_m=240.0, land_unit_size_m=80.0, floor_height_m=3.0)
    features = block_to_features(block)
    assert features["FAR"] > 0
    assert features["BD"] > 0
    assert features["AF"] >= 1
    assert features["AR_ew"] > 0
    assert features["AR_ns"] > 0


def test_generate_dataset_can_return_block_metadata() -> None:
    frame, blocks = generate_morphology_dataset(
        n_samples=5,
        seed=11,
        block_size_m=240.0,
        land_unit_size_m=80.0,
        floor_height_m=3.0,
        return_blocks=True,
    )
    assert len(frame) == 5
    assert len(blocks) == 5
    assert "assignments" in blocks[0]

from __future__ import annotations

import numpy as np
import pandas as pd

from paper_repro.constants import MORPHOLOGY_FEATURES
from paper_repro.optimizers import OptimizationEnvironment


SCENARIOS = {
    "Balanced_Performance": [1 / 3, 1 / 3, 1 / 3],
    "Energy_Saving_Focus": [0.6, 0.2, 0.2],
    "Energy_Generation_Focus": [0.2, 0.6, 0.2],
}


class FakeSurrogate:
    feature_bounds = {name: (0.0, 1.0) for name in MORPHOLOGY_FEATURES}
    target_bounds = {
        "EUIt": (66.0, 96.0),
        "EG": (1.2, 2.85),
        "H": (6.0, 7.85),
    }
    feature_reference = np.vstack(
        [
            np.zeros(len(MORPHOLOGY_FEATURES), dtype=np.float32),
            np.ones(len(MORPHOLOGY_FEATURES), dtype=np.float32),
        ]
    )

    def predict(self, frame: pd.DataFrame, *, clip: bool = False) -> pd.DataFrame:
        del clip
        return pd.DataFrame(
            {
                "EUIt": np.full(len(frame), 66.0),
                "EG": np.full(len(frame), 2.85),
                "H": np.full(len(frame), 7.85),
            }
        )


def _environment() -> OptimizationEnvironment:
    return OptimizationEnvironment(FakeSurrogate(), guardrail_cfg={"feature_radius_percentile": 100})


def test_ddpg_reward_contract_bounds_and_utopia_points() -> None:
    env = _environment()
    ideal = np.array([66.0, 2.85, 7.85], dtype=np.float32)
    anti_ideal = np.array([96.0, 1.2, 6.0], dtype=np.float32)
    interior = np.array(
        [
            [66.0, 2.85, 7.85],
            [76.0, 2.30, 7.20],
            [96.0, 1.20, 6.00],
        ],
        dtype=np.float32,
    )

    for weights in SCENARIOS.values():
        rewards = env.reward_batch(interior, weights)
        assert np.all((0.0 <= rewards) & (rewards <= 1.0))
        assert env.reward(ideal, weights) == np.float32(1.0)
        assert env.reward(anti_ideal, weights) == np.float32(0.0)
        assert np.isclose(env.reward(interior[1], weights), env.reward_batch(interior[[1]], weights)[0])


def test_ddpg_episode_return_stays_within_40_step_bound() -> None:
    env = _environment()
    outputs = np.tile(np.array([[76.0, 2.30, 7.20]], dtype=np.float32), (40, 1))
    for weights in SCENARIOS.values():
        episode_return = float(env.reward_batch(outputs, weights).sum())
        assert 0.0 <= episode_return <= 40.0

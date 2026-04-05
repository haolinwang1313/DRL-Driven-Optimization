from __future__ import annotations

import os

import torch

from paper_repro.config import Config


def resolve_device(config: Config | None = None) -> torch.device:
    env_override = os.environ.get("PAPER_REPRO_DEVICE")
    if env_override:
        return torch.device(env_override)
    if config is not None:
        requested = config.raw.get("runtime", {}).get("device", "auto")
        if requested != "auto":
            return torch.device(requested)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

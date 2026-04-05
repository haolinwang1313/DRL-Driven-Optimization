from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from paper_repro.surrogate import load_surrogate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    args = parser.parse_args()
    path = Path(args.model_path)
    bundle = load_surrogate(path)
    print("loaded", path)
    print(bundle.hyperparameters)
    print(bundle.model)


if __name__ == "__main__":
    try:
        main()
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        raise

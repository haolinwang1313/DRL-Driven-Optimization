from __future__ import annotations

import argparse
import json

from paper_repro.round2_lock import lock_canonical_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Lock Paper02 round-2 canonical benchmark and result registry outputs.")
    parser.add_argument("--config", default="configs/reviewer_round2_experiments.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    payload = lock_canonical_results(args.config, run_id=args.run_id, output_dir=args.output_dir)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

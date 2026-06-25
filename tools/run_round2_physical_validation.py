from __future__ import annotations

import argparse
import json

from paper_repro.round2 import run_physical_validation


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Paper02 round-2 physical validation.")
    parser.add_argument("--config", default="configs/reviewer_round2_experiments.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--wait-seconds", type=int, default=0)
    args = parser.parse_args()
    if args.dry_run:
        payload = {"status": "dry_run", "config": args.config, "run_id": args.run_id, "output_dir": args.output_dir}
    else:
        payload = run_physical_validation(
            args.config,
            run_id=args.run_id,
            output_dir=args.output_dir,
            resume=args.resume,
            wait_seconds=args.wait_seconds,
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

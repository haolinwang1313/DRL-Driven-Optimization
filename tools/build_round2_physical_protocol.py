from __future__ import annotations

import argparse
import json

from paper_repro.round2 import build_locked_case_selection, build_physical_model_protocol, collect_existing_physical_jobs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Paper02 round-2 physical-validation protocol artifacts.")
    parser.add_argument("--config", default="configs/reviewer_round2_experiments.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    if args.dry_run:
        payload = {"status": "dry_run", "config": args.config, "run_id": args.run_id, "output_dir": args.output_dir}
    else:
        payload = {
            "locked_case_selection": str(build_locked_case_selection(args.config, run_id=args.run_id, output_dir=args.output_dir)),
            "physical_model_protocol": str(build_physical_model_protocol(args.config, run_id=args.run_id, output_dir=args.output_dir)),
            "recovered_jobs": collect_existing_physical_jobs(args.config, run_id=args.run_id, output_dir=args.output_dir),
        }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json

from paper_repro.round2 import build_selection_criteria_registry, run_feasibility_audit, run_sampling_coverage_analysis


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Paper02 round-2 sampling and feasibility diagnostics.")
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
            "sampling": run_sampling_coverage_analysis(args.config, run_id=args.run_id, output_dir=args.output_dir),
            "feasibility": run_feasibility_audit(args.config, run_id=args.run_id, output_dir=args.output_dir),
            "selection_registry": str(build_selection_criteria_registry(args.config, run_id=args.run_id, output_dir=args.output_dir)),
        }
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json

from paper_repro.publication_figures_round2 import build_round2_revision_figures


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the canonical round-2 publication figure-data package and candidate figures.")
    parser.add_argument("--data-root", default="paper/manuscript/figure_data/round2")
    parser.add_argument("--output-dir", default="paper/manuscript/figures/round2_candidate")
    parser.add_argument("--build-gallery", action="store_true")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--formats", default="pdf,png")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    payload = build_round2_revision_figures(
        args.data_root,
        args.output_dir,
        build_gallery=args.build_gallery,
        check_only=args.check_only,
        formats=tuple(item.strip() for item in args.formats.split(",") if item.strip()),
        dpi=args.dpi,
        overwrite=args.overwrite,
        strict=args.strict,
    )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

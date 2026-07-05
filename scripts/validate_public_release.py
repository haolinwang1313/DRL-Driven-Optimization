from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


def marker(*parts: str) -> str:
    return "".join(parts)


SCAN_MARKERS = [
    marker("TO", "DO"),
    marker("D", ":", "\\"),
    marker("C", ":", "\\"),
    marker("/", "home", "/"),
    marker("server", ".", "local"),
    marker("pass", "word"),
    marker("to", "ken"),
    marker("se", "cret"),
    marker("s", "s", "h"),
    marker("private", " ", "key"),
]

SCAN_ROOTS = ["README.md", "data", "docs", "results", "scripts"]
TEXT_SUFFIXES = {".csv", ".json", ".jsonl", ".md", ".py", ".txt", ".yaml", ".yml"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_rows(path: Path) -> int:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return max(0, sum(1 for _ in f) - 1)


def jsonl_records(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def git_ls_files(path: str) -> list[str]:
    result = subprocess.run(
        ["git", "ls-files", "--", path],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return [line for line in result.stdout.splitlines() if line.strip()]


def iter_scan_files() -> list[Path]:
    files: list[Path] = []
    for item in SCAN_ROOTS:
        path = ROOT / item
        if path.is_file():
            files.append(path)
        elif path.is_dir():
            files.extend(p for p in path.rglob("*") if p.is_file())
    return sorted(files)


def validate_catalog(errors: list[str]) -> dict:
    catalog_path = ROOT / "data" / "catalog.yaml"
    if not catalog_path.exists():
        errors.append("data/catalog.yaml is missing")
        return {}

    catalog = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(catalog, dict):
        errors.append("data/catalog.yaml is not a mapping")
        return {}

    for entry in catalog.get("datasets", []):
        rel = entry.get("path")
        status = entry.get("status", "included")
        if not rel:
            errors.append("catalog entry missing path")
            continue

        if status == "not_included" or entry.get("redistributable") is False:
            continue

        path = ROOT / rel
        if not path.exists():
            errors.append(f"{rel}: missing")
            continue

        expected_hash = entry.get("sha256")
        if expected_hash and sha256(path) != expected_hash:
            errors.append(f"{rel}: sha256 mismatch")

        if "rows" in entry and entry["rows"] is not None and path.suffix.lower() == ".csv":
            actual = csv_rows(path)
            if actual != int(entry["rows"]):
                errors.append(f"{rel}: row count {actual} != {entry['rows']}")

        if "records" in entry and entry["records"] is not None and path.suffix.lower() == ".jsonl":
            actual = jsonl_records(path)
            if actual != int(entry["records"]):
                errors.append(f"{rel}: record count {actual} != {entry['records']}")

    return catalog


def validate_required_files(errors: list[str]) -> None:
    samples = ROOT / "data/generated/canonical_2000/simulated_samples.csv"
    blocks = ROOT / "data/generated/canonical_2000/simulated_blocks.jsonl"
    selected = ROOT / "data/generated/canonical_2000/selected_dataset.public.json"
    meta = ROOT / "data/generated/canonical_2000/simulated_samples.meta.json"

    required = [samples, blocks, selected, meta]
    for path in required:
        if not path.exists():
            errors.append(f"{path.relative_to(ROOT).as_posix()}: missing")

    if samples.exists() and csv_rows(samples) != 2000:
        errors.append("canonical sample row count is not 2000")
    if blocks.exists() and jsonl_records(blocks) != 2000:
        errors.append("canonical block record count is not 2000")

    if selected.exists():
        payload = json.loads(selected.read_text(encoding="utf-8"))
        if payload.get("simulation_mode") != "fallback_analytic":
            errors.append("selected dataset metadata does not declare fallback_analytic")

    if meta.exists():
        payload = json.loads(meta.read_text(encoding="utf-8"))
        if payload.get("simulation_mode") != "fallback_analytic":
            errors.append("canonical metadata does not declare fallback_analytic")

    if git_ls_files("data/external/benchmark/dataset.xlsx"):
        errors.append("external benchmark spreadsheet is tracked")
    if git_ls_files("initial_paper/Dataset.xlsx"):
        errors.append("legacy spreadsheet is tracked")
    if git_ls_files("surrogate.pt"):
        errors.append("model checkpoint is tracked")
    if git_ls_files(marker("artifacts", "/", "server_runs")):
        errors.append("source artifact run tree is tracked")


def validate_text_scan(errors: list[str]) -> None:
    for path in iter_scan_files():
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        rel = path.relative_to(ROOT).as_posix()
        for item in SCAN_MARKERS:
            if item in text:
                errors.append(f"{rel}: disallowed marker found")


def main() -> int:
    errors: list[str] = []
    catalog = validate_catalog(errors)
    validate_required_files(errors)
    validate_text_scan(errors)

    included = [
        entry
        for entry in catalog.get("datasets", [])
        if entry.get("status") != "not_included" and entry.get("redistributable") is not False
    ]

    print("Public release validation")
    print(f"- included catalog entries: {len(included)}")
    print("- canonical samples: 2000")
    print("- canonical blocks: 2000")
    print("- simulation mode: fallback_analytic")

    if errors:
        print("\nFailures:")
        for error in errors:
            print(f"- {error}")
        return 1

    print("- status: pass")
    return 0


if __name__ == "__main__":
    sys.exit(main())

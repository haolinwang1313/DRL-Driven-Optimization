from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

from paper_repro.config import Config
from paper_repro.constants import COMMON_USER_EXECUTABLE_HINTS, SIM_STACK_EXECUTABLES, SIM_STACK_PACKAGES
from paper_repro.contracts import write_json


def _detect_python_packages() -> dict[str, dict[str, str | bool]]:
    detected: dict[str, dict[str, str | bool]] = {}
    for package_name, module_name in SIM_STACK_PACKAGES.items():
        try:
            module = importlib.import_module(module_name)
            detected[package_name] = {"available": True, "module": module_name, "version": getattr(module, "__version__", "unknown")}
        except Exception as exc:
            detected[package_name] = {"available": False, "module": module_name, "error": f"{type(exc).__name__}: {exc}"}
    return detected


def _detect_executables() -> dict[str, str | None]:
    detected = {name: shutil.which(command) for name, command in SIM_STACK_EXECUTABLES.items()}
    for name, hints in COMMON_USER_EXECUTABLE_HINTS.items():
        if detected.get(name):
            continue
        for hint in hints:
            if hint.exists():
                detected[name] = str(hint)
                break
    return detected


def _pip_install(packages: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pip", "install", *packages],
        capture_output=True,
        text=True,
        check=False,
    )


def _download_weather_archive(url: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    file_name = url.rsplit("/", 1)[-1]
    archive_path = output_dir / file_name
    if not archive_path.exists():
        urllib.request.urlretrieve(url, archive_path)  # noqa: S310
    return archive_path


def _extract_epw(archive_path: Path, output_dir: Path) -> Path:
    with zipfile.ZipFile(archive_path, "r") as archive:
        archive.extractall(output_dir)
    epw_files = sorted(output_dir.glob("*.epw"))
    if not epw_files:
        raise FileNotFoundError(f"No EPW file found in {archive_path}")
    return epw_files[0]


def bootstrap_sim_stack(config: Config, install_missing: bool = False) -> dict:
    config.ensure_artifact_dirs()
    weather_cfg = config["weather"]
    python_packages = _detect_python_packages()
    executables = _detect_executables()
    pip_result = None

    missing_packages = [name for name, result in python_packages.items() if not result["available"]]
    if install_missing and missing_packages:
        pip_result = _pip_install(missing_packages)
        python_packages = _detect_python_packages()

    weather_records = []
    if weather_cfg.get("download", True):
        for station_name in [weather_cfg["preferred_station"], weather_cfg["fallback_station"]]:
            station_cfg = weather_cfg["stations"][station_name]
            try:
                archive_path = _download_weather_archive(station_cfg["url"], Path(weather_cfg["output_dir"]))
                extract_dir = Path(weather_cfg["output_dir"]) / station_name
                extract_dir.mkdir(parents=True, exist_ok=True)
                epw_path = _extract_epw(archive_path, extract_dir)
                weather_records.append(
                    {
                        "station": station_name,
                        "label": station_cfg["label"],
                        "archive": str(archive_path),
                        "epw": str(epw_path),
                        "available": True,
                    }
                )
            except Exception as exc:
                weather_records.append(
                    {
                        "station": station_name,
                        "label": station_cfg["label"],
                        "available": False,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )

    summary = {
        "python_executable": sys.executable,
        "python_packages": python_packages,
        "executables": executables,
        "pip_install_attempted": install_missing,
        "pip_result": {
            "returncode": None if pip_result is None else pip_result.returncode,
            "stdout_tail": None if pip_result is None else pip_result.stdout[-4000:],
            "stderr_tail": None if pip_result is None else pip_result.stderr[-4000:],
        },
        "weather_records": weather_records,
    }
    write_json(summary, Path(config["report"]["bootstrap_dir"]) / "bootstrap_summary.json")
    return summary

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import tarfile
import zipfile
from pathlib import Path
from urllib.request import urlretrieve

ENERGYPLUS_URLS = {
    "ubuntu22.04": "https://github.com/NatLabRockies/EnergyPlus/releases/download/v25.2.0/EnergyPlus-25.2.0-cf7368216c-Linux-Ubuntu22.04-x86_64.tar.gz",
    "ubuntu24.04": "https://github.com/NatLabRockies/EnergyPlus/releases/download/v26.1.0/EnergyPlus-26.1.0-6f2e40d102-Linux-Ubuntu24.04-x86_64.tar.gz",
}
RADIANCE_URL = "https://github.com/LBNL-ETA/Radiance/releases/download/rad6R0P2/Radiance_c1700d56_Linux.zip"


def _detect_linux_target() -> str:
    os_release = Path("/etc/os-release")
    if os_release.exists():
        text = os_release.read_text(encoding="utf-8", errors="replace").lower()
        if 'version_id="22.04"' in text or "ubuntu 22.04" in text:
            return "ubuntu22.04"
        if 'version_id="24.04"' in text or "ubuntu 24.04" in text:
            return "ubuntu24.04"
    return "ubuntu22.04"


def install_energyplus(root: Path, target_key: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    url = ENERGYPLUS_URLS[target_key]
    archive = root / Path(url).name
    if not archive.exists():
        urlretrieve(url, archive)  # noqa: S310
    target = root / archive.stem.replace(".tar", "")
    if not target.exists():
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(target)
    if target.is_dir() and len(list(target.iterdir())) == 1:
        child = next(target.iterdir())
        if child.is_dir():
            return child
    return target


def install_radiance(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "radiance.zip"
    if not archive.exists():
        urlretrieve(RADIANCE_URL, archive)  # noqa: S310
    target = root / "Radiance"
    if not target.exists():
        with zipfile.ZipFile(archive, "r") as zf:
            zf.extractall(target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path.home() / "opt"))
    parser.add_argument("--energyplus-target", choices=sorted(ENERGYPLUS_URLS), default="")
    args = parser.parse_args()
    root = Path(args.root)
    eplus_target = args.energyplus_target or _detect_linux_target()
    eplus_dir = install_energyplus(root, eplus_target)
    rad_dir = install_radiance(root)
    payload = {
        "energyplus_target": eplus_target,
        "energyplus_root": str(eplus_dir),
        "radiance_root": str(rad_dir),
        "energyplus_exists": any(eplus_dir.rglob("energyplus")),
        "radiance_rtrace_exists": any(rad_dir.rglob("rtrace")),
        "radiance_oconv_exists": any(rad_dir.rglob("oconv")),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

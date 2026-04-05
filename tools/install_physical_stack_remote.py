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


ENERGYPLUS_URL = "https://github.com/NatLabRockies/EnergyPlus/releases/download/v26.1.0/EnergyPlus-26.1.0-6f2e40d102-Linux-Ubuntu24.04-x86_64.tar.gz"
RADIANCE_URL = "https://github.com/LBNL-ETA/Radiance/releases/download/rad6R0P2/Radiance_c1700d56_Linux.zip"


def install_energyplus(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "energyplus.tar.gz"
    if not archive.exists():
        urlretrieve(ENERGYPLUS_URL, archive)  # noqa: S310
    target = root / "EnergyPlus"
    if not target.exists():
        with tarfile.open(archive, "r:gz") as tf:
            tf.extractall(target)
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
    args = parser.parse_args()
    root = Path(args.root)
    eplus_dir = install_energyplus(root)
    rad_dir = install_radiance(root)
    payload = {
        "energyplus_root": str(eplus_dir),
        "radiance_root": str(rad_dir),
        "energyplus_exists": any(eplus_dir.rglob("energyplus")),
        "radiance_rtrace_exists": any(rad_dir.rglob("rtrace")),
        "radiance_oconv_exists": any(rad_dir.rglob("oconv")),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()

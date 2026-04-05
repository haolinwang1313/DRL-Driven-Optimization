from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import paramiko
import yaml

from paper_repro.config import Config
from paper_repro.contracts import write_json

REQUIRED_PUBLICATION_FILES = {
    "models": ["cv_predictions.csv", "surrogate.pt", "surrogate_summary.json"],
    "optimization": ["ddpg_results.csv", "ddpg_logs.json", "nsga2_results.csv"],
    "data": ["simulated_samples.csv", "simulated_samples.meta.json"],
}


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_server_config(path: str | Path | None = None) -> dict[str, Any] | None:
    candidate = Path(path) if path is not None else (_repo_root() / "server.local.yaml")
    if not candidate.exists():
        return None
    with candidate.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _copy_local_tree(source_root: Path, target_root: Path) -> None:
    for section in REQUIRED_PUBLICATION_FILES:
        source_dir = source_root / section
        if not source_dir.exists():
            continue
        destination_dir = target_root / section
        destination_dir.mkdir(parents=True, exist_ok=True)
        for item in source_dir.iterdir():
            if item.is_dir():
                shutil.copytree(item, destination_dir / item.name, dirs_exist_ok=True)
            else:
                shutil.copy2(item, destination_dir / item.name)


def _sync_into_active_report_dirs(source_root: Path, config: Config) -> None:
    mapping = {
        "models": Path(config["report"]["models_dir"]),
        "optimization": Path(config["report"]["optimization_dir"]),
        "data": Path(config["report"]["data_dir"]),
        "bootstrap": Path(config["report"]["bootstrap_dir"]),
        "figures": Path(config["report"]["figures_dir"]),
        "reports": Path(config["report"]["reports_dir"]),
        "diagnostics": Path(config["publication"]["diagnostics_dir"]),
        "reevaluation": Path(config["publication"]["reevaluation_dir"]),
    }
    for section, destination_dir in mapping.items():
        source_dir = source_root / section
        if not source_dir.exists():
            continue
        destination_dir.mkdir(parents=True, exist_ok=True)
        for item in source_dir.iterdir():
            if item.is_dir():
                shutil.copytree(item, destination_dir / item.name, dirs_exist_ok=True)
            else:
                shutil.copy2(item, destination_dir / item.name)


def _rsync_remote_tree(server_cfg: dict[str, Any], target_root: Path) -> None:
    remote_root = server_cfg["remote_results_root"].rstrip("/")
    identity_file = server_cfg["identity_file"]
    port = str(server_cfg.get("port", 22))
    user = server_cfg["user"]
    host = server_cfg["host"]
    for section in REQUIRED_PUBLICATION_FILES:
        destination_dir = target_root / section
        destination_dir.mkdir(parents=True, exist_ok=True)
        remote = f"{user}@{host}:{remote_root}/{section}/"
        subprocess.run(
            [
                "rsync",
                "-az",
                "-e",
                f"ssh -i {identity_file} -p {port}",
                remote,
                str(destination_dir),
            ],
            check=True,
        )


def _sftp_copy_dir(sftp: paramiko.SFTPClient, remote_dir: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    for entry in sftp.listdir_attr(remote_dir):
        remote_path = f"{remote_dir}/{entry.filename}"
        local_path = local_dir / entry.filename
        if entry.st_mode & 0o40000:  # directory bit
            _sftp_copy_dir(sftp, remote_path, local_path)
        else:
            sftp.get(remote_path, str(local_path))


def _paramiko_remote_tree(server_cfg: dict[str, Any], target_root: Path) -> None:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    connect_kwargs: dict[str, Any] = {
        "hostname": server_cfg["host"],
        "port": int(server_cfg.get("port", 22)),
        "username": server_cfg["user"],
        "timeout": 20,
    }
    identity_file = server_cfg.get("identity_file")
    password = server_cfg.get("password")
    if identity_file:
        connect_kwargs["key_filename"] = identity_file
    if password:
        connect_kwargs["password"] = password
    client.connect(**connect_kwargs)
    sftp = client.open_sftp()
    try:
        remote_root = server_cfg["remote_results_root"].rstrip("/")
        for section in REQUIRED_PUBLICATION_FILES:
            _sftp_copy_dir(sftp, f"{remote_root}/{section}", target_root / section)
        for extra in ("figures", "reports", "bootstrap", "diagnostics", "reevaluation"):
            remote_section = f"{remote_root}/{extra}"
            try:
                sftp.stat(remote_section)
            except FileNotFoundError:
                continue
            _sftp_copy_dir(sftp, remote_section, target_root / extra)
    finally:
        sftp.close()
        client.close()


def sync_publication_results(config: Config, server_cfg_path: str | Path | None = None) -> dict[str, Any]:
    server_cfg = load_server_config(server_cfg_path)
    if server_cfg is None:
        raise FileNotFoundError("server.local.yaml not found; create it from server.local.example.yaml")

    target_root = Path(config["publication"]["imported_results_root"])
    target_root.mkdir(parents=True, exist_ok=True)
    local_results_root = server_cfg.get("local_results_root")
    if local_results_root:
        _copy_local_tree(Path(local_results_root), target_root)
        _sync_into_active_report_dirs(Path(local_results_root), config)
        mode = "local_copy"
    else:
        if shutil.which("rsync") and not server_cfg.get("force_paramiko", False):
            _rsync_remote_tree(server_cfg, target_root)
            mode = "rsync"
        else:
            _paramiko_remote_tree(server_cfg, target_root)
            mode = "paramiko_sftp"
        _sync_into_active_report_dirs(target_root, config)

    payload = {
        "mode": mode,
        "target_root": str(target_root),
        "required_sections": list(REQUIRED_PUBLICATION_FILES),
    }
    write_json(payload, Path(config["report"]["reports_dir"]) / "publication_sync_summary.json")
    return payload


def validate_publication_results(config: Config) -> dict[str, Any]:
    imported_root = Path(config["publication"]["imported_results_root"])
    missing: dict[str, list[str]] = {}
    for section, files in REQUIRED_PUBLICATION_FILES.items():
        for file_name in files:
            candidate = imported_root / section / file_name
            if not candidate.exists():
                missing.setdefault(section, []).append(file_name)

    if missing:
        raise FileNotFoundError(json.dumps(missing, indent=2, ensure_ascii=False))

    meta_path = imported_root / "data" / "simulated_samples.meta.json"
    metadata = {}
    if meta_path.exists():
        metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        if (
            config["publication"].get("require_physical_results", False)
            and not config["publication"].get("allow_fallback_override", False)
            and metadata.get("simulation_mode") == "fallback_analytic"
        ):
            raise RuntimeError("Publication mode refuses fallback_analytic data; import physical-stack results or enable override explicitly.")

    payload = {
        "imported_root": str(imported_root),
        "metadata": metadata,
        "status": "ok",
    }
    write_json(payload, Path(config["report"]["reports_dir"]) / "publication_validation_summary.json")
    return payload

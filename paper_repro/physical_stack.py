from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import paramiko

from paper_repro.config import Config
from paper_repro.constants import MORPHOLOGY_FEATURES, PROTOTYPES
from paper_repro.contracts import write_csv, write_json
from paper_repro.publication import load_server_config


def load_block_records(path: str | Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            records[int(payload["sample_id"])] = payload
    return records


def project_candidates_to_nearest_blocks(candidates: pd.DataFrame, dataset: pd.DataFrame) -> pd.DataFrame:
    feature_frame = dataset[MORPHOLOGY_FEATURES].to_numpy(dtype=float)
    lower = feature_frame.min(axis=0)
    upper = feature_frame.max(axis=0)
    scale = np.maximum(upper - lower, 1e-8)
    normalized_dataset = (feature_frame - lower) / scale
    rows: list[dict[str, Any]] = []
    for candidate_index, (_, row) in enumerate(candidates.iterrows()):
        candidate_vec = row[MORPHOLOGY_FEATURES].to_numpy(dtype=float)
        normalized_candidate = (candidate_vec - lower) / scale
        distances = np.linalg.norm(normalized_dataset - normalized_candidate[None, :], axis=1)
        best_index = int(np.argmin(distances))
        matched = dataset.iloc[best_index]
        record = {
            "candidate_index": candidate_index,
            "matched_sample_id": int(matched["sample_id"]),
            "projection_distance": float(distances[best_index]),
            **{feature: float(row[feature]) for feature in MORPHOLOGY_FEATURES},
        }
        for column in row.index:
            if column in record or column in MORPHOLOGY_FEATURES:
                continue
            record[column] = row[column]
        rows.append(record)
    return pd.DataFrame(rows)


def _remote_python_payload(request_remote_path: str, result_remote_path: str, status_remote_path: str) -> str:
    prototype_payload = {
        name: {
            "width_m": spec.width_m,
            "depth_m": spec.depth_m,
            "courtyard_fraction": spec.courtyard_fraction,
        }
        for name, spec in PROTOTYPES.items()
    }
    return f"""
import json
import os
import subprocess
from pathlib import Path
from statistics import mean

from honeybee.room import Room
from honeybee.model import Model
from honeybee_energy.lib.programtypes import office_program
from honeybee_energy.hvac.idealair import IdealAirSystem
from honeybee_energy.load.people import People
from honeybee_energy.load.lighting import Lighting
from honeybee_energy.load.equipment import ElectricEquipment
from honeybee_energy.load.infiltration import Infiltration
from honeybee_energy.load.setpoint import Setpoint
from honeybee_energy.load.ventilation import Ventilation
from honeybee_energy.schedule.ruleset import ScheduleRuleset
from ladybug.epw import EPW
from ladybug_geometry.geometry3d.pointvector import Point3D

REQUEST_PATH = Path({request_remote_path!r})
RESULT_PATH = Path({result_remote_path!r})
STATUS_PATH = Path({status_remote_path!r})
PROJECT_ROOT = REQUEST_PATH.parents[2]
PAYLOAD = json.loads(REQUEST_PATH.read_text(encoding='utf-8'))
STATUS_PATH.write_text(json.dumps({{'status': 'running', 'remote_request': str(REQUEST_PATH)}}, ensure_ascii=False), encoding='utf-8')
PROTOTYPES = {json.dumps(prototype_payload, ensure_ascii=False)}
BLOCK_COORD = {{
    0: (0, 0), 1: (1, 0), 2: (2, 0),
    3: (0, 1), 4: (1, 1), 5: (2, 1),
    6: (0, 2), 7: (1, 2), 8: (2, 2),
}}

def make_model(block_record):
    rooms = []
    land_unit = float(block_record.get('land_unit_size_m', 80.0))
    theta = float(block_record.get('theta_deg', 0.0)) % 360.0
    floor_height = float(block_record.get('floor_height_m', 3.0))
    window_ratio_ns = 0.4
    window_ratio_ew = 0.1
    people_per_area = 0.03
    lighting_w_per_area = 5.0
    equipment_w_per_area = 1.9
    ventilation_m3s_per_person = 30.0 / 3600.0
    heating_c = 18.0
    cooling_c = 26.0
    roof_area_total = 0.0
    for assignment in block_record['assignments']:
        proto = PROTOTYPES[assignment['prototype_name']]
        cell_x, cell_y = BLOCK_COORD[int(assignment['block_index'])]
        width = float(proto['width_m'])
        depth = float(proto['depth_m'])
        height = float(assignment['floors']) * floor_height
        origin_x = cell_x * land_unit + (land_unit - width) / 2.0
        origin_y = cell_y * land_unit + (land_unit - depth) / 2.0
        room = Room.from_box(
            f\"room_{{assignment['block_index']}}\",
            width=width,
            depth=depth,
            height=height,
            orientation_angle=theta,
            origin=Point3D(origin_x, origin_y, 0.0),
        )
        room.properties.energy.program_type = office_program
        room.properties.energy.reset_loads_to_program()
        room.properties.energy.hvac = IdealAirSystem(f\"ideal_{{assignment['block_index']}}\")
        room.properties.energy.people = People(f\"people_{{assignment['block_index']}}\", people_per_area)
        room.properties.energy.lighting = Lighting(f\"lights_{{assignment['block_index']}}\", lighting_w_per_area)
        room.properties.energy.electric_equipment = ElectricEquipment(f\"equip_{{assignment['block_index']}}\", equipment_w_per_area)
        exterior_area = sum(face.area for face in room.faces if str(face.boundary_condition) == 'Outdoors')
        infiltration_per_area = (1.0 * room.volume / 3600.0) / max(exterior_area, 1e-6)
        room.properties.energy.infiltration = Infiltration(f\"infil_{{assignment['block_index']}}\", infiltration_per_area)
        room.properties.energy.ventilation = Ventilation(f\"vent_{{assignment['block_index']}}\", flow_per_person=ventilation_m3s_per_person)
        room.properties.energy.setpoint = Setpoint(
            f\"setpoint_{{assignment['block_index']}}\",
            ScheduleRuleset.from_constant_value(f\"heat_{{assignment['block_index']}}\", heating_c),
            ScheduleRuleset.from_constant_value(f\"cool_{{assignment['block_index']}}\", cooling_c),
        )
        south_aperture_points = []
        for face in room.faces:
            if str(face.boundary_condition) != 'Outdoors' or str(face.type) != 'Wall':
                if str(face.boundary_condition) == 'Outdoors' and str(face.type) == 'RoofCeiling':
                    roof_area_total += face.area
                continue
            nx = getattr(face.normal, 'x', 0.0)
            ny = getattr(face.normal, 'y', 0.0)
            ratio = window_ratio_ns if abs(ny) >= abs(nx) else window_ratio_ew
            if ratio > 0:
                face.apertures_by_ratio_rectangle(ratio, 1.5, 0.8, 0.1)
            if abs(ny) >= abs(nx) and ny < 0:
                for aperture in face.apertures:
                    south_aperture_points.append((aperture.geometry.center, face.normal))
        rooms.append(room)
    return Model(f\"candidate_{{block_record['sample_id']}}\", rooms), south_aperture_points, roof_area_total

def run_case(case):
    sample_id = case['matched_sample_id']
    block_record = case['block_record']
    case_dir = PROJECT_ROOT / 'artifacts' / 'physical_stack_candidates' / f'sample_{{sample_id}}'
    base_env = os.environ.copy()
    base_env['PATH'] = str(PROJECT_ROOT / '.venv' / 'bin') + ':' + base_env.get('PATH', '')
    rad_env = os.environ.copy()
    rad_env['PATH'] = PAYLOAD['radiance_env']['PATH'].replace('$' + '{{PATH}}', base_env.get('PATH', ''))
    rad_env['RAYPATH'] = PAYLOAD['radiance_env']['RAYPATH']
    if case_dir.exists():
        subprocess.run(['rm', '-rf', str(case_dir)], check=False)
    case_dir.mkdir(parents=True, exist_ok=True)
    model, south_aperture_points, roof_area_total = make_model(block_record)
    hbjson_base = case_dir / 'model'
    model.to_hbjson(name=str(hbjson_base), folder='.')
    hbjson_path = case_dir / 'model.hbjson'

    energy_cmd = [
        'honeybee-energy', 'simulate', 'model',
        str(hbjson_path),
        str(PROJECT_ROOT / PAYLOAD['epw_relpath']),
        '-f', str(case_dir / 'sim_out'),
        '-log', str(case_dir / 'sim_log.json'),
    ]
    energy = subprocess.run(energy_cmd, capture_output=True, text=True, env=base_env)
    eui = None
    generation = None
    pv_generation_million_kwh = None
    if energy.returncode == 0:
        eui_cmd = [
            'honeybee-energy', 'result', 'energy-use-intensity',
            str(case_dir / 'sim_out' / 'run' / 'eplusout.sql'),
        ]
        eui_out = subprocess.run(eui_cmd, capture_output=True, text=True, env=base_env)
        if eui_out.returncode == 0:
            eui = json.loads(eui_out.stdout).get('eui')
        gen_cmd = [
            'honeybee-energy', 'result', 'generation-summary',
            str(case_dir / 'sim_out' / 'run' / 'eplusout.sql'),
        ]
        gen_out = subprocess.run(gen_cmd, capture_output=True, text=True, env=base_env)
        if gen_out.returncode == 0:
            generation = json.loads(gen_out.stdout)
    epw = EPW(str(PROJECT_ROOT / PAYLOAD['epw_relpath']))
    annual_ghi_wh_m2 = sum(epw.global_horizontal_radiation.values)
    pv_coverage = 0.8
    pv_efficiency = 0.2
    pv_performance_ratio = 0.75
    pv_generation_million_kwh = (
        annual_ghi_wh_m2 * roof_area_total * pv_coverage * pv_efficiency * pv_performance_ratio / 1_000_000_000.0
    )

    rad_dir = case_dir / 'rad'
    rad_translate = subprocess.run(
        ['honeybee-radiance', 'translate', 'model-to-rad-folder', str(hbjson_path), '--folder', str(rad_dir), '-cg', '--log-file', str(case_dir / 'rad_folder_log.json')],
        capture_output=True,
        text=True,
        env=base_env,
    )
    radiance_mean = None
    sunlight_hours = None
    if rad_translate.returncode == 0:
        sampled_hours = [9, 12, 15]
        hour_indices = [(19 * 24) + (hour - 1) for hour in sampled_hours]
        hourly_means = []
        hit_count = 0
        sensor_path = case_dir / 'south_window.pts'
        if south_aperture_points:
            with sensor_path.open('w', encoding='utf-8') as sensor_file:
                for center, normal in south_aperture_points:
                    sensor_file.write(f"{{center.x}} {{center.y}} {{center.z}} {{-normal.x}} {{-normal.y}} {{-normal.z}}\\n")
        for idx, epw_index in enumerate(hour_indices):
            if not south_aperture_points:
                break
            sky_dir = case_dir / f'sky_{{idx}}'
            sky_dir.mkdir(parents=True, exist_ok=True)
            sim_hour = sampled_hours[idx]
            sky_cmd = [
                'honeybee-radiance', 'sky', 'climate-based', '20', 'Jan', f\"{{sim_hour}}:00\",
                '-lat', str(PAYLOAD['radiance_sky']['latitude']),
                '-lon', str(PAYLOAD['radiance_sky']['longitude']),
                '-dni', str(epw.direct_normal_radiation.values[epw_index]),
                '-dhi', str(epw.diffuse_horizontal_radiation.values[epw_index]),
                '--folder', str(sky_dir),
                '--name', 'test_sky',
            ]
            sky = subprocess.run(sky_cmd, capture_output=True, text=True, env=base_env)
            if sky.returncode != 0:
                continue
            octree_cmd = [
                'honeybee-radiance', 'octree', 'from-folder-static',
                str(rad_dir),
                '--add-before', str(sky_dir / 'test_sky'),
                '-o', str(case_dir / 'test.oct'),
            ]
            octree = subprocess.run(octree_cmd, capture_output=True, text=True, env=rad_env)
            if octree.returncode != 0:
                continue
            pt_out = case_dir / f'pt_{{idx}}.res'
            point_in_time = subprocess.run(
                [
                    'honeybee-radiance', 'raytrace', 'point-in-time',
                    str(case_dir / 'test.oct'),
                    str(sensor_path),
                    '-m', 'illuminance',
                    '-o', str(pt_out),
                ],
                capture_output=True,
                text=True,
                env=rad_env,
            )
            if point_in_time.returncode != 0 or not pt_out.exists():
                continue
            values = []
            for raw_line in pt_out.read_text(encoding='utf-8', errors='replace').splitlines():
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    values.append(float(raw_line.split()[0]))
                except Exception:
                    continue
            if values:
                current_mean = mean(values)
                hourly_means.append(current_mean)
                if current_mean >= 1000.0:
                    hit_count += 1
        if hourly_means:
            radiance_mean = mean(hourly_means)
            sunlight_hours = float(hit_count) * (9.0 / max(len(sampled_hours), 1))

    return {{
        'candidate_index': case['candidate_index'],
        'matched_sample_id': sample_id,
        'projection_distance': case['projection_distance'],
        'method': case.get('method'),
        'scenario': case.get('scenario'),
        'seed': case.get('seed'),
        'physical_EUIt': eui,
        'physical_generation_summary': generation,
        'physical_EG_total_production': pv_generation_million_kwh if pv_generation_million_kwh is not None else (None if generation is None else generation.get('total_production')),
        'physical_EG_consumption_purchased': None if generation is None else generation.get('consumption_purchased'),
        'physical_radiance_mean_sensor_value': radiance_mean,
        'physical_H_proxy': sunlight_hours if sunlight_hours is not None else radiance_mean,
        'simulation_mode': 'physical_stack_probe',
        'energyplus_ok': energy.returncode == 0,
        'radiance_ok': radiance_mean is not None,
    }}

try:
    results = []
    total_cases = len(PAYLOAD['cases'])
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    for index, case in enumerate(PAYLOAD['cases'], start=1):
        STATUS_PATH.write_text(
            json.dumps(
                {{
                    'status': 'running',
                    'remote_request': str(REQUEST_PATH),
                    'current_case_index': index,
                    'total_cases': total_cases,
                    'completed_cases': len(results),
                }},
                ensure_ascii=False,
            ),
            encoding='utf-8',
        )
        result = run_case(case)
        results.append(result)
        RESULT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
        STATUS_PATH.write_text(
            json.dumps(
                {{
                    'status': 'running',
                    'remote_request': str(REQUEST_PATH),
                    'current_case_index': index,
                    'total_cases': total_cases,
                    'completed_cases': len(results),
                    'result_path': str(RESULT_PATH),
                }},
                ensure_ascii=False,
            ),
            encoding='utf-8',
        )
    STATUS_PATH.write_text(json.dumps({{'status': 'completed', 'count': len(results), 'result_path': str(RESULT_PATH)}}, ensure_ascii=False), encoding='utf-8')
    print(json.dumps({{'count': len(results), 'result_path': str(RESULT_PATH)}}, ensure_ascii=False))
except Exception as exc:
    STATUS_PATH.write_text(json.dumps({{'status': 'failed', 'error': f'{{type(exc).__name__}}: {{exc}}'}}, ensure_ascii=False), encoding='utf-8')
    raise
"""


def _connect_server(server_cfg: dict[str, Any]) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    connect_kwargs: dict[str, Any] = {
        "hostname": server_cfg["host"],
        "port": int(server_cfg.get("port", 22)),
        "username": server_cfg["user"],
        "timeout": 20,
    }
    if server_cfg.get("identity_file"):
        connect_kwargs["key_filename"] = server_cfg["identity_file"]
    if server_cfg.get("password"):
        connect_kwargs["password"] = server_cfg["password"]
    client.connect(**connect_kwargs)
    return client


def _prepare_probe_job(
    config: Config,
    candidates: pd.DataFrame,
    *,
    limit: int,
    server_cfg: dict[str, Any],
    output_suffix: str,
) -> tuple[pd.DataFrame, Path, Path, str, str, Path]:
    dataset_path = Path(config["report"]["data_dir"]) / "simulated_samples.csv"
    blocks_path = Path(config["report"]["data_dir"]) / "simulated_blocks.jsonl"
    dataset = pd.read_csv(dataset_path)
    block_records = load_block_records(blocks_path)
    projected = project_candidates_to_nearest_blocks(candidates.head(limit).reset_index(drop=True), dataset)
    projected["matched_sample_id"] = projected["matched_sample_id"].astype(int)

    request_id = uuid.uuid4().hex[:12]
    diagnostics_dir = Path(config["publication"]["diagnostics_dir"])
    local_request = diagnostics_dir / f"physical_stack_request_{request_id}.json"
    local_result = diagnostics_dir / f"physical_stack_result_{request_id}.json"
    local_projected = diagnostics_dir / f"physical_stack_projected_{request_id}.csv"
    remote_request = f"{server_cfg['remote_project_root'].rstrip('/')}/artifacts/physical_stack_batches/request_{request_id}.json"
    remote_result = f"{server_cfg['remote_project_root'].rstrip('/')}/artifacts/physical_stack_batches/result_{request_id}.json"

    radiance_root = "/home/ac/opt/Radiance/extracted/radiance-6.0.c1700d56cc-Linux/usr/local/radiance"
    payload = {
        "epw_relpath": "artifacts/weather/Dongtai/CHN_JS_Dongtai.582510_TMYx.2009-2023.epw",
        "radiance_sky": {"latitude": 33.5, "longitude": 119.8, "sky_type": 4},
        "radiance_env": {
            "PATH": f"{radiance_root}/bin:" + "${PATH}",
            "RAYPATH": f".:{radiance_root}/lib",
        },
        "cases": [],
    }
    for row in projected.to_dict(orient="records"):
        sample_id = int(row["matched_sample_id"])
        payload["cases"].append(
            {
                "candidate_index": int(row["candidate_index"]),
                "matched_sample_id": sample_id,
                "projection_distance": float(row["projection_distance"]),
                "method": row.get("method"),
                "scenario": row.get("scenario"),
                "seed": row.get("seed"),
                "block_record": block_records[sample_id],
            }
        )
    write_json(payload, local_request)
    write_csv(projected, local_projected)
    return projected, local_request, local_result, remote_request, remote_result, local_projected


def _submit_async_probe(
    config: Config,
    *,
    server_cfg: dict[str, Any],
    local_request: Path,
    local_result: Path,
    local_projected: Path,
    remote_request: str,
    remote_result: str,
    candidate_source: str,
    limit: int,
    output_suffix: str,
) -> dict[str, Any]:
    client = _connect_server(server_cfg)
    sftp = client.open_sftp()
    try:
        remote_request_parent = str(Path(remote_request).parent).replace("\\", "/")
        remote_log = f"{remote_request_parent}/probe_{Path(remote_request).stem.replace('request_', '')}.log"
        remote_pid = f"{remote_request_parent}/probe_{Path(remote_request).stem.replace('request_', '')}.pid"
        remote_script = f"{remote_request_parent}/probe_{Path(remote_request).stem.replace('request_', '')}.py"
        remote_status = f"{remote_request_parent}/probe_{Path(remote_request).stem.replace('request_', '')}.status.json"
        mkdir_cmd = f"mkdir -p {remote_request_parent}"
        stdin, stdout, stderr = client.exec_command(mkdir_cmd, timeout=120)
        stderr_text = stderr.read().decode("utf-8", errors="replace")
        if stderr_text.strip():
            raise RuntimeError(stderr_text)
        sftp.put(str(local_request), remote_request)
        python_code = _remote_python_payload(remote_request, remote_result, remote_status)
        with sftp.open(remote_script, "w") as handle:
            handle.write(python_code)
        with sftp.open(remote_status, "w") as handle:
            handle.write(json.dumps({"status": "submitted", "remote_request": remote_request}, ensure_ascii=False))
        remote_python = f"{server_cfg['remote_project_root'].rstrip('/')}/.venv/bin/python"
        launch_cmd = (
            f"sh -c 'cd {server_cfg['remote_project_root']} && "
            f"nohup {remote_python} {remote_script} > {remote_log} 2>&1 < /dev/null & "
            f"printf %s $! > {remote_pid}'"
        )
        stdin, stdout, stderr = client.exec_command(launch_cmd, timeout=10)
        time.sleep(1.0)
        stdout.close()
        stderr.close()
        stdin.close()
        try:
            with sftp.open(remote_pid, "r") as handle:
                pid_text = handle.read().decode("utf-8", errors="replace").strip()
        except FileNotFoundError as exc:
            raise RuntimeError(f"Remote PID file was not created for async probe: {remote_pid}") from exc
    finally:
        sftp.close()
        client.close()

    job_payload = {
        "job_id": Path(remote_request).stem.replace("request_", ""),
        "status": "submitted",
        "candidate_source": candidate_source,
        "limit": int(limit),
        "output_suffix": output_suffix,
        "local_request": str(local_request),
        "local_result": str(local_result),
        "local_projected": str(local_projected),
        "remote_request": remote_request,
        "remote_result": remote_result,
        "remote_status": remote_status,
        "remote_pid": pid_text,
    }
    job_path = Path(config["publication"]["diagnostics_dir"]) / f"physical_stack_job_{job_payload['job_id']}.json"
    write_json(job_payload, job_path)
    job_payload["job_path"] = str(job_path)
    return job_payload


def _collect_async_probe(config: Config, *, server_cfg: dict[str, Any], job_id: str, wait_seconds: int = 0) -> dict[str, Any]:
    job_path = Path(config["publication"]["diagnostics_dir"]) / f"physical_stack_job_{job_id}.json"
    if not job_path.exists():
        raise FileNotFoundError(f"Async physical-stack job not found: {job_id}")
    job_payload = json.loads(job_path.read_text(encoding="utf-8"))
    client = _connect_server(server_cfg)
    sftp = client.open_sftp()
    try:
        remote_result = job_payload["remote_result"]
        remote_status = job_payload.get("remote_status", "")
        remote_pid = str(job_payload.get("remote_pid", "")).strip()
        deadline = time.time() + max(wait_seconds, 0)
        while True:
            status_payload = None
            if remote_status:
                try:
                    with sftp.open(remote_status, "r") as handle:
                        status_payload = json.loads(handle.read().decode("utf-8", errors="replace"))
                except FileNotFoundError:
                    status_payload = None
            if status_payload and status_payload.get("status") == "failed":
                job_payload["status"] = "failed"
                job_payload["error"] = status_payload.get("error", "")
                write_json(job_payload, job_path)
                job_payload["job_path"] = str(job_path)
                return job_payload
            if status_payload and status_payload.get("status") == "completed":
                break
            if remote_pid:
                stdin, stdout, stderr = client.exec_command(f"ps -p {remote_pid} -o pid=", timeout=30)
                pid_live = bool(stdout.read().decode("utf-8", errors="replace").strip())
                stderr.read()
                if not pid_live and status_payload and status_payload.get("status") == "running":
                    job_payload["status"] = "failed"
                    job_payload["error"] = "stale_running_job_without_live_process"
                    job_payload["remote_status_payload"] = status_payload
                    write_json(job_payload, job_path)
                    job_payload["job_path"] = str(job_path)
                    return job_payload
            if time.time() >= deadline:
                job_payload["status"] = "running"
                if status_payload is not None:
                    job_payload["remote_status_payload"] = status_payload
                write_json(job_payload, job_path)
                job_payload["job_path"] = str(job_path)
                return job_payload
            time.sleep(5)
        local_result = Path(job_payload["local_result"])
        sftp.get(remote_result, str(local_result))
    finally:
        sftp.close()
        client.close()

    projected = pd.read_csv(job_payload["local_projected"])
    result_payload = json.loads(Path(job_payload["local_result"]).read_text(encoding="utf-8"))
    result_frame = pd.DataFrame(result_payload)
    merged = projected.merge(
        result_frame,
        on=["candidate_index", "matched_sample_id", "projection_distance", "method", "scenario", "seed"],
        how="left",
    )
    suffix = f"_{job_payload['output_suffix']}" if job_payload["output_suffix"] else ""
    reevaluation_dir = Path(config["publication"]["reevaluation_dir"])
    csv_path = write_csv(merged, reevaluation_dir / f"physical_stack_candidate_probe{suffix}.csv")
    summary = {
        "job_id": job_id,
        "status": "completed",
        "candidate_source": job_payload["candidate_source"],
        "rows": int(len(merged)),
        "csv_path": str(csv_path),
        "local_result": job_payload["local_result"],
        "job_path": str(job_path),
    }
    write_json(summary, reevaluation_dir / f"physical_stack_candidate_probe_summary{suffix}.json")
    write_json({**job_payload, **summary}, job_path)
    return summary


def physical_stack_candidate_probe(
    config: Config,
    candidates: pd.DataFrame,
    *,
    limit: int = 5,
    server_cfg_path: str | Path | None = None,
    output_suffix: str = "",
    async_mode: bool = False,
    wait_seconds: int = 0,
    job_id: str | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    server_cfg = load_server_config(server_cfg_path)
    if server_cfg is None:
        raise FileNotFoundError("server.local.yaml not found for physical-stack probe.")

    if job_id is not None:
        summary = _collect_async_probe(config, server_cfg=server_cfg, job_id=job_id, wait_seconds=wait_seconds)
        if summary.get("status") != "completed":
            return pd.DataFrame(), summary
        return pd.read_csv(summary["csv_path"]), summary

    if candidates.empty:
        raise ValueError("No candidate rows provided for physical-stack probing.")

    projected, local_request, local_result, remote_request, remote_result, local_projected = _prepare_probe_job(
        config,
        candidates,
        limit=limit,
        server_cfg=server_cfg,
        output_suffix=output_suffix,
    )

    if async_mode:
        summary = _submit_async_probe(
            config,
            server_cfg=server_cfg,
            local_request=local_request,
            local_result=local_result,
            local_projected=local_projected,
            remote_request=remote_request,
            remote_result=remote_result,
            candidate_source="inline_frame",
            limit=limit,
            output_suffix=output_suffix,
        )
        if wait_seconds > 0:
            collected = _collect_async_probe(config, server_cfg=server_cfg, job_id=summary["job_id"], wait_seconds=wait_seconds)
            if collected.get("status") == "completed":
                return pd.read_csv(collected["csv_path"]), collected
            return pd.DataFrame(), collected
        return pd.DataFrame(), summary

    client = _connect_server(server_cfg)
    sftp = client.open_sftp()
    try:
        remote_request_parent = str(Path(remote_request).parent).replace("\\", "/")
        mkdir_cmd = f"mkdir -p {remote_request_parent}"
        stdin, stdout, stderr = client.exec_command(mkdir_cmd, timeout=120)
        stderr_text = stderr.read().decode("utf-8", errors="replace")
        if stderr_text.strip():
            raise RuntimeError(stderr_text)
        sftp.put(str(local_request), remote_request)
        python_code = _remote_python_payload(remote_request, remote_result)
        cmd = f"cd {server_cfg['remote_project_root']} && . .venv/bin/activate && python - <<'PY'\n{python_code}\nPY"
        stdin, stdout, stderr = client.exec_command(cmd, timeout=7200)
        _stdout_text = stdout.read().decode("utf-8", errors="replace")
        stderr_text = stderr.read().decode("utf-8", errors="replace")
        if stderr_text.strip():
            raise RuntimeError(stderr_text)
        sftp.get(remote_result, str(local_result))
    finally:
        sftp.close()
        client.close()

    result_payload = json.loads(local_result.read_text(encoding="utf-8"))
    result_frame = pd.DataFrame(result_payload)
    merged = projected.merge(
        result_frame,
        on=["candidate_index", "matched_sample_id", "projection_distance", "method", "scenario", "seed"],
        how="left",
    )
    suffix = f"_{output_suffix}" if output_suffix else ""
    reevaluation_dir = Path(config["publication"]["reevaluation_dir"])
    csv_path = write_csv(merged, reevaluation_dir / f"physical_stack_candidate_probe{suffix}.csv")
    summary = {
        "request_path": str(local_request),
        "result_path": str(local_result),
        "csv_path": str(csv_path),
        "count": int(len(merged)),
        "limit": int(limit),
    }
    write_json(summary, reevaluation_dir / f"physical_stack_candidate_probe_summary{suffix}.json")
    return merged, summary

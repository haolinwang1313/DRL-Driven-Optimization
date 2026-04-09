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
import time
from pathlib import Path
from statistics import mean

from honeybee.room import Room
from honeybee.model import Model
from honeybee.boundarycondition import boundary_conditions
from honeybee_energy.lib.programtypes import office_program
from honeybee_energy.hvac.idealair import IdealAirSystem
from honeybee_energy.load.people import People
from honeybee_energy.load.lighting import Lighting
from honeybee_energy.load.equipment import ElectricEquipment
from honeybee_energy.load.infiltration import Infiltration
from honeybee_energy.load.setpoint import Setpoint
from honeybee_energy.load.ventilation import Ventilation
from honeybee_energy.material.opaque import EnergyMaterialNoMass
from honeybee_energy.construction.opaque import OpaqueConstruction
from honeybee_energy.material.glazing import EnergyWindowMaterialSimpleGlazSys
from honeybee_energy.construction.window import WindowConstruction
from honeybee_energy.schedule.ruleset import ScheduleRuleset, ScheduleDay, ScheduleRule
from ladybug.dt import Date
from ladybug.epw import EPW
from ladybug_geometry.geometry3d.pointvector import Point3D

REQUEST_PATH = Path({request_remote_path!r})
RESULT_PATH = Path({result_remote_path!r})
STATUS_PATH = Path({status_remote_path!r})
PROJECT_ROOT = REQUEST_PATH.parents[2]
PAYLOAD = json.loads(REQUEST_PATH.read_text(encoding='utf-8'))
PROTOTYPES = {json.dumps(prototype_payload, ensure_ascii=False)}
BLOCK_COORD = {{
    0: (0, 0), 1: (1, 0), 2: (2, 0),
    3: (0, 1), 4: (1, 1), 5: (2, 1),
    6: (0, 2), 7: (1, 2), 8: (2, 2),
}}
TIMEOUTS = {{
    'cleanup': int(PAYLOAD.get('timeouts', {{}}).get('cleanup', 60)),
    'energy_simulate': int(PAYLOAD.get('timeouts', {{}}).get('energy_simulate', 1800)),
    'energy_result': int(PAYLOAD.get('timeouts', {{}}).get('energy_result', 180)),
    'radiance_translate': int(PAYLOAD.get('timeouts', {{}}).get('radiance_translate', 180)),
    'radiance_sunpath': int(PAYLOAD.get('timeouts', {{}}).get('radiance_sunpath', 120)),
    'radiance_sky': int(PAYLOAD.get('timeouts', {{}}).get('radiance_sky', 120)),
    'radiance_octree': int(PAYLOAD.get('timeouts', {{}}).get('radiance_octree', 180)),
    'radiance_point_in_time': int(PAYLOAD.get('timeouts', {{}}).get('radiance_point_in_time', 180)),
    'radiance_scontrib': int(PAYLOAD.get('timeouts', {{}}).get('radiance_scontrib', 180)),
}}
WALL_CONSTRUCTION = OpaqueConstruction(
    'paper_wall_u_0_8',
    [EnergyMaterialNoMass('paper_wall_layer', 1.0 / 0.8)],
)
ROOF_CONSTRUCTION = OpaqueConstruction(
    'paper_roof_u_0_5',
    [EnergyMaterialNoMass('paper_roof_layer', 1.0 / 0.5)],
)
FLOOR_CONSTRUCTION = OpaqueConstruction(
    'paper_floor_u_1_5',
    [EnergyMaterialNoMass('paper_floor_layer', 1.0 / 1.5)],
)
WINDOW_CONSTRUCTION = WindowConstruction(
    'paper_window_u_2_7_shgc_0_78',
    [EnergyWindowMaterialSimpleGlazSys('paper_window_glz', 2.7, 0.78, 0.6)],
)

def log_event(message):
    print(message, flush=True)

def update_status(**extra):
    payload = {{
        'status': 'running',
        'remote_request': str(REQUEST_PATH),
        'timestamp': time.time(),
    }}
    payload.update(extra)
    STATUS_PATH.write_text(json.dumps(payload, ensure_ascii=False), encoding='utf-8')

def run_cmd(cmd, *, env, stage, timeout, current_case_index=None, total_cases=None, completed_cases=None, matched_sample_id=None):
    update_status(
        current_case_index=current_case_index,
        total_cases=total_cases,
        completed_cases=completed_cases,
        matched_sample_id=matched_sample_id,
        stage=stage,
        timeout_seconds=timeout,
    )
    log_event(f"[{{stage}}] " + " ".join(str(part) for part in cmd))
    try:
        return subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{{stage}}_timeout_after_{{timeout}}s") from exc

def run_cmd_to_file(cmd, *, env, stage, timeout, output_path, current_case_index=None, total_cases=None, completed_cases=None, matched_sample_id=None):
    update_status(
        current_case_index=current_case_index,
        total_cases=total_cases,
        completed_cases=completed_cases,
        matched_sample_id=matched_sample_id,
        stage=stage,
        timeout_seconds=timeout,
    )
    log_event(f"[{{stage}}] " + " ".join(str(part) for part in cmd))
    try:
        with Path(output_path).open('wb') as out_handle:
            proc = subprocess.run(cmd, stdout=out_handle, stderr=subprocess.PIPE, env=env, timeout=timeout)
        if proc.returncode != 0:
            raise RuntimeError(proc.stderr.decode('utf-8', errors='replace'))
        return proc
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError(f"{{stage}}_timeout_after_{{timeout}}s") from exc

update_status(stage='boot')

def make_models(block_record):
    energy_rooms = []
    radiance_rooms = []
    land_unit = float(block_record.get('land_unit_size_m', 80.0))
    theta = float(block_record.get('theta_deg', 0.0)) % 360.0
    floor_height = float(block_record.get('floor_height_m', 3.0))
    window_ratio_ns = 0.4
    window_ratio_ew = 0.1
    people_per_area = 0.03
    lighting_w_per_area = 5.0
    equipment_w_per_area = 1.9
    ventilation_m3s_per_person = 30.0 / 3600.0
    occupancy_schedule_fraction = 0.35
    lighting_schedule_fraction = 0.25
    equipment_schedule_fraction = 0.30
    infiltration_schedule_fraction = 0.50
    heating_c = 18.0
    cooling_c = 26.0
    heating_off_c = 12.0
    cooling_off_c = 30.0
    max_south_points_per_face = 5
    roof_area_total = 0.0

    def apply_window_ratios(room, *, collect_south_points=False, simple_windows=False):
        south_points = []
        for face in room.faces:
            if str(face.boundary_condition) != 'Outdoors' or str(face.type) != 'Wall':
                continue
            face.properties.energy.construction = WALL_CONSTRUCTION
            nx = getattr(face.normal, 'x', 0.0)
            ny = getattr(face.normal, 'y', 0.0)
            ratio = window_ratio_ns if abs(ny) >= abs(nx) else window_ratio_ew
            if ratio > 0 and not face.apertures:
                if simple_windows:
                    face.apertures_by_ratio(ratio, rect_split=False)
                else:
                    face.apertures_by_ratio_rectangle(ratio, 1.5, 0.8, 0.1)
            for aperture in face.apertures:
                aperture.properties.energy.construction = WINDOW_CONSTRUCTION
            if collect_south_points and abs(ny) >= abs(nx) and ny < 0:
                lower_left = face.geometry.lower_left_corner
                lower_right = face.geometry.lower_right_corner
                sill_height = min(lower_left.z, lower_right.z) + 0.8
                point_count = max_south_points_per_face
                if point_count == 1:
                    fractions = [0.5]
                else:
                    fractions = [i / (point_count - 1) for i in range(point_count)]
                for frac in fractions:
                    x = lower_left.x + frac * (lower_right.x - lower_left.x) - face.normal.x * 0.1
                    y = lower_left.y + frac * (lower_right.y - lower_left.y) - face.normal.y * 0.1
                    z = sill_height
                    south_points.append((Point3D(x, y, z), face.normal))
        return south_points

    def configure_energy_room(room, *, tag_suffix, multiplier=1, top_bc=None, bottom_bc=None):
        heating_default = ScheduleDay(f\"heat_default_{{tag_suffix}}\", [heating_off_c])
        heating_on = ScheduleDay(f\"heat_on_{{tag_suffix}}\", [heating_c])
        cooling_default = ScheduleDay(f\"cool_default_{{tag_suffix}}\", [cooling_off_c])
        cooling_on = ScheduleDay(f\"cool_on_{{tag_suffix}}\", [cooling_c])
        heating_schedule = ScheduleRuleset(
            f\"heat_{{tag_suffix}}\",
            heating_default,
            schedule_rules=[
                ScheduleRule(
                    heating_on,
                    apply_sunday=True,
                    apply_monday=True,
                    apply_tuesday=True,
                    apply_wednesday=True,
                    apply_thursday=True,
                    apply_friday=True,
                    apply_saturday=True,
                    start_date=Date(12, 1),
                    end_date=Date(2, 28),
                )
            ],
        )
        cooling_schedule = ScheduleRuleset(
            f\"cool_{{tag_suffix}}\",
            cooling_default,
            schedule_rules=[
                ScheduleRule(
                    cooling_on,
                    apply_sunday=True,
                    apply_monday=True,
                    apply_tuesday=True,
                    apply_wednesday=True,
                    apply_thursday=True,
                    apply_friday=True,
                    apply_saturday=True,
                    start_date=Date(6, 15),
                    end_date=Date(8, 31),
                )
            ],
        )
        room.story = tag_suffix
        room.multiplier = multiplier
        if bottom_bc is not None:
            room.faces[0].boundary_condition = bottom_bc
        if top_bc is not None:
            room.faces[-1].boundary_condition = top_bc
        room.faces[0].properties.energy.construction = FLOOR_CONSTRUCTION
        room.faces[-1].properties.energy.construction = ROOF_CONSTRUCTION
        room.properties.energy.program_type = office_program
        room.properties.energy.reset_loads_to_program()
        room.properties.energy.hvac = IdealAirSystem(f\"ideal_{{tag_suffix}}\")
        room.properties.energy.people = People(
            f\"people_{{tag_suffix}}\",
            people_per_area,
            occupancy_schedule=ScheduleRuleset.from_constant_value(f\"occ_sched_{{tag_suffix}}\", occupancy_schedule_fraction),
        )
        room.properties.energy.lighting = Lighting(
            f\"lights_{{tag_suffix}}\",
            lighting_w_per_area,
            schedule=ScheduleRuleset.from_constant_value(f\"light_sched_{{tag_suffix}}\", lighting_schedule_fraction),
        )
        room.properties.energy.electric_equipment = ElectricEquipment(
            f\"equip_{{tag_suffix}}\",
            equipment_w_per_area,
            schedule=ScheduleRuleset.from_constant_value(f\"equip_sched_{{tag_suffix}}\", equipment_schedule_fraction),
        )
        exterior_area = sum(face.area for face in room.faces if str(face.boundary_condition) == 'Outdoors')
        infiltration_per_area = (1.0 * room.volume / 3600.0) / max(exterior_area, 1e-6)
        room.properties.energy.infiltration = Infiltration(
            f\"infil_{{tag_suffix}}\",
            infiltration_per_area,
            schedule=ScheduleRuleset.from_constant_value(f\"infil_sched_{{tag_suffix}}\", infiltration_schedule_fraction),
        )
        room.properties.energy.ventilation = Ventilation(f\"vent_{{tag_suffix}}\", flow_per_person=ventilation_m3s_per_person)
        room.properties.energy.setpoint = Setpoint(
            f\"setpoint_{{tag_suffix}}\",
            heating_schedule,
            cooling_schedule,
        )
        apply_window_ratios(room, collect_south_points=False, simple_windows=True)

    for assignment in block_record['assignments']:
        proto = PROTOTYPES[assignment['prototype_name']]
        cell_x, cell_y = BLOCK_COORD[int(assignment['block_index'])]
        width = float(proto['width_m'])
        depth = float(proto['depth_m'])
        height = float(assignment['floors']) * floor_height
        origin_x = cell_x * land_unit + (land_unit - width) / 2.0
        origin_y = cell_y * land_unit + (land_unit - depth) / 2.0
        floors = int(assignment['floors'])
        tag_base = f\"b{{assignment['block_index']}}\"
        if floors <= 1:
            energy_room = Room.from_box(
                f\"energy_room_{{assignment['block_index']}}_single\",
                width=width,
                depth=depth,
                height=floor_height,
                orientation_angle=theta,
                origin=Point3D(origin_x, origin_y, 0.0),
            )
            configure_energy_room(energy_room, tag_suffix=f\"{{tag_base}}_single\", multiplier=1)
            energy_rooms.append(energy_room)
        else:
            ground_room = Room.from_box(
                f\"energy_room_{{assignment['block_index']}}_ground\",
                width=width,
                depth=depth,
                height=floor_height,
                orientation_angle=theta,
                origin=Point3D(origin_x, origin_y, 0.0),
            )
            configure_energy_room(
                ground_room,
                tag_suffix=f\"{{tag_base}}_ground\",
                multiplier=1,
                top_bc=boundary_conditions.adiabatic,
            )
            energy_rooms.append(ground_room)

            if floors > 2:
                middle_room = Room.from_box(
                    f\"energy_room_{{assignment['block_index']}}_middle\",
                    width=width,
                    depth=depth,
                    height=floor_height,
                    orientation_angle=theta,
                    origin=Point3D(origin_x, origin_y, floor_height),
                )
                configure_energy_room(
                    middle_room,
                    tag_suffix=f\"{{tag_base}}_middle\",
                    multiplier=floors - 2,
                    top_bc=boundary_conditions.adiabatic,
                    bottom_bc=boundary_conditions.adiabatic,
                )
                energy_rooms.append(middle_room)

            top_room = Room.from_box(
                f\"energy_room_{{assignment['block_index']}}_top\",
                width=width,
                depth=depth,
                height=floor_height,
                orientation_angle=theta,
                origin=Point3D(origin_x, origin_y, max(floors - 1, 0) * floor_height),
            )
            configure_energy_room(
                top_room,
                tag_suffix=f\"{{tag_base}}_top\",
                multiplier=1,
                bottom_bc=boundary_conditions.adiabatic,
            )
            energy_rooms.append(top_room)

        radiance_room = Room.from_box(
            f\"radiance_room_{{assignment['block_index']}}\",
            width=width,
            depth=depth,
            height=height,
            orientation_angle=theta,
            origin=Point3D(origin_x, origin_y, 0.0),
        )
        radiance_rooms.append(radiance_room)

    energy_model = Model(f\"energy_candidate_{{block_record['sample_id']}}\", energy_rooms)
    radiance_model = Model(f\"radiance_candidate_{{block_record['sample_id']}}\", radiance_rooms)
    south_aperture_points = []
    for room in radiance_model.rooms:
        south_aperture_points.extend(apply_window_ratios(room, collect_south_points=True, simple_windows=False))
        for face in room.faces:
            if str(face.boundary_condition) != 'Outdoors' or str(face.type) != 'Wall':
                if str(face.boundary_condition) == 'Outdoors' and str(face.type) == 'RoofCeiling':
                    roof_area_total += face.area
            continue
    return energy_model, radiance_model, south_aperture_points, roof_area_total

def run_case(case, *, current_case_index, total_cases, completed_cases):
    sample_id = case['matched_sample_id']
    block_record = case['block_record']
    case_dir = PROJECT_ROOT / 'artifacts' / 'physical_stack_candidates' / f'sample_{{sample_id}}'
    base_env = os.environ.copy()
    base_env['PATH'] = str(PROJECT_ROOT / '.venv' / 'bin') + ':' + base_env.get('PATH', '')
    rad_env = os.environ.copy()
    rad_env['PATH'] = PAYLOAD['radiance_env']['PATH'].replace('$' + '{{PATH}}', base_env.get('PATH', ''))
    rad_env['RAYPATH'] = PAYLOAD['radiance_env']['RAYPATH']
    update_status(
        current_case_index=current_case_index,
        total_cases=total_cases,
        completed_cases=completed_cases,
        matched_sample_id=sample_id,
        stage='prepare_case',
    )
    if case_dir.exists():
        subprocess.run(['rm', '-rf', str(case_dir)], check=False, timeout=TIMEOUTS['cleanup'])
    case_dir.mkdir(parents=True, exist_ok=True)
    energy_model, radiance_model, south_aperture_points, roof_area_total = make_models(block_record)
    energy_hbjson_base = case_dir / 'energy_model'
    radiance_hbjson_base = case_dir / 'radiance_model'
    energy_model.to_hbjson(name=str(energy_hbjson_base), folder='.')
    radiance_model.to_hbjson(name=str(radiance_hbjson_base), folder='.')
    energy_hbjson_path = case_dir / 'energy_model.hbjson'
    radiance_hbjson_path = case_dir / 'radiance_model.hbjson'

    energy_cmd = [
        'honeybee-energy', 'simulate', 'model',
        str(energy_hbjson_path),
        str(PROJECT_ROOT / PAYLOAD['epw_relpath']),
        '-f', str(case_dir / 'sim_out'),
        '-log', str(case_dir / 'sim_log.json'),
    ]
    energy = run_cmd(
        energy_cmd,
        env=base_env,
        stage='energyplus_simulate',
        timeout=TIMEOUTS['energy_simulate'],
        current_case_index=current_case_index,
        total_cases=total_cases,
        completed_cases=completed_cases,
        matched_sample_id=sample_id,
    )
    eui = None
    generation = None
    pv_generation_million_kwh = None
    if energy.returncode == 0:
        eui_cmd = [
            'honeybee-energy', 'result', 'energy-use-intensity',
            str(case_dir / 'sim_out' / 'run' / 'eplusout.sql'),
        ]
        eui_out = run_cmd(
            eui_cmd,
            env=base_env,
            stage='energyplus_parse_eui',
            timeout=TIMEOUTS['energy_result'],
            current_case_index=current_case_index,
            total_cases=total_cases,
            completed_cases=completed_cases,
            matched_sample_id=sample_id,
        )
        if eui_out.returncode == 0:
            eui = json.loads(eui_out.stdout).get('eui')
        gen_cmd = [
            'honeybee-energy', 'result', 'generation-summary',
            str(case_dir / 'sim_out' / 'run' / 'eplusout.sql'),
        ]
        gen_out = run_cmd(
            gen_cmd,
            env=base_env,
            stage='energyplus_parse_generation',
            timeout=TIMEOUTS['energy_result'],
            current_case_index=current_case_index,
            total_cases=total_cases,
            completed_cases=completed_cases,
            matched_sample_id=sample_id,
        )
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
    rad_translate = run_cmd(
        ['honeybee-radiance', 'translate', 'model-to-rad-folder', str(radiance_hbjson_path), '--folder', str(rad_dir), '-cg', '--log-file', str(case_dir / 'rad_folder_log.json')],
        env=base_env,
        stage='radiance_translate',
        timeout=TIMEOUTS['radiance_translate'],
        current_case_index=current_case_index,
        total_cases=total_cases,
        completed_cases=completed_cases,
        matched_sample_id=sample_id,
    )
    radiance_mean = None
    sunlight_hours = None
    if rad_translate.returncode == 0:
        sensor_path = case_dir / 'south_window.pts'
        if south_aperture_points:
            with sensor_path.open('w', encoding='utf-8') as sensor_file:
                for center, normal in south_aperture_points:
                    sensor_file.write(f"{{center.x}} {{center.y}} {{center.z}} {{-normal.x}} {{-normal.y}} {{-normal.z}}\\n")
        if south_aperture_points:
            try:
                sun_dir = case_dir / 'sunpath'
                sun_dir.mkdir(parents=True, exist_ok=True)
                sun_cmd = [
                    'honeybee-radiance', 'sunpath', 'epw',
                    str(PROJECT_ROOT / PAYLOAD['epw_relpath']),
                    '--start-date', 'JAN-20',
                    '--start-time', '08:00',
                    '--end-date', 'JAN-20',
                    '--end-time', '16:00',
                    '--folder', str(sun_dir),
                    '--name', 'jan20sun',
                    '--reverse-vectors',
                ]
                run_cmd(
                    sun_cmd,
                    env=base_env,
                    stage='radiance_sunpath',
                    timeout=TIMEOUTS['radiance_sunpath'],
                    current_case_index=current_case_index,
                    total_cases=total_cases,
                    completed_cases=completed_cases,
                    matched_sample_id=sample_id,
                )
                octree_path = case_dir / 'jan20sun.oct'
                run_cmd_to_file(
                    [
                        'oconv',
                        '-f',
                        str(rad_dir / 'model' / 'scene' / 'envelope.mat'),
                        str(rad_dir / 'model' / 'scene' / 'envelope.rad'),
                        str(rad_dir / 'model' / 'aperture' / 'aperture.mat'),
                        str(rad_dir / 'model' / 'aperture' / 'aperture.rad'),
                        str(sun_dir / 'jan20sun.rad'),
                    ],
                    env=rad_env,
                    stage='radiance_octree_direct_sun',
                    timeout=TIMEOUTS['radiance_octree'],
                    output_path=octree_path,
                    current_case_index=current_case_index,
                    total_cases=total_cases,
                    completed_cases=completed_cases,
                    matched_sample_id=sample_id,
                )
                mtx_path = case_dir / 'jan20sun.mtx'
                run_cmd(
                    [
                        'honeybee-radiance', 'dc', 'scontrib',
                        str(octree_path),
                        str(sensor_path),
                        str(sun_dir / 'jan20sun.mod'),
                        '--value',
                        '--output', str(mtx_path),
                    ],
                    env=rad_env,
                    stage='radiance_scontrib',
                    timeout=TIMEOUTS['radiance_scontrib'],
                    current_case_index=current_case_index,
                    total_cases=total_cases,
                    completed_cases=completed_cases,
                    matched_sample_id=sample_id,
                )
                lines = mtx_path.read_text(encoding='utf-8', errors='replace').splitlines()
                data_start = 0
                for line_index, line in enumerate(lines):
                    if line and (line[0].isdigit() or line[0] == '-' or line[0] == '.'):
                        data_start = line_index
                        break
                hourly_counts = []
                scalar_means = []
                for line in lines[data_start:]:
                    values = [float(x) for x in line.split()]
                    if not values:
                        continue
                    triplets = [values[idx] for idx in range(0, len(values), 3)]
                    scalar_means.append(mean(triplets))
                    hourly_counts.append(sum(1 for value in triplets if value > 0.0))
                if scalar_means:
                    radiance_mean = mean(scalar_means)
                    sunlight_hours = float(mean(hourly_counts))
            except Exception:
                sampled_hours = list(range(8, 17))
                hour_indices = [(19 * 24) + (hour - 1) for hour in sampled_hours]
                hourly_means = []
                point_hour_totals = None
                for idx, epw_index in enumerate(hour_indices):
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
                    sky = run_cmd(
                        sky_cmd,
                        env=base_env,
                        stage=f'radiance_sky_{{idx}}',
                        timeout=TIMEOUTS['radiance_sky'],
                        current_case_index=current_case_index,
                        total_cases=total_cases,
                        completed_cases=completed_cases,
                        matched_sample_id=sample_id,
                    )
                    if sky.returncode != 0:
                        continue
                    octree_cmd = [
                        'honeybee-radiance', 'octree', 'from-folder-static',
                        str(rad_dir),
                        '--add-before', str(sky_dir / 'test_sky'),
                        '-o', str(case_dir / 'test.oct'),
                    ]
                    octree = run_cmd(
                        octree_cmd,
                        env=rad_env,
                        stage=f'radiance_octree_{{idx}}',
                        timeout=TIMEOUTS['radiance_octree'],
                        current_case_index=current_case_index,
                        total_cases=total_cases,
                        completed_cases=completed_cases,
                        matched_sample_id=sample_id,
                    )
                    if octree.returncode != 0:
                        continue
                    pt_out = case_dir / f'pt_{{idx}}.res'
                    point_in_time = run_cmd(
                        [
                            'honeybee-radiance', 'raytrace', 'point-in-time',
                            str(case_dir / 'test.oct'),
                            str(sensor_path),
                            '-m', 'illuminance',
                            '-o', str(pt_out),
                        ],
                        env=rad_env,
                        stage=f'radiance_point_in_time_{{idx}}',
                        timeout=TIMEOUTS['radiance_point_in_time'],
                        current_case_index=current_case_index,
                        total_cases=total_cases,
                        completed_cases=completed_cases,
                        matched_sample_id=sample_id,
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
                        if point_hour_totals is None:
                            point_hour_totals = [0.0] * len(values)
                        for value_index, value in enumerate(values):
                            if value >= 1000.0:
                                point_hour_totals[value_index] += 1.0
                if hourly_means:
                    radiance_mean = mean(hourly_means)
                    if point_hour_totals:
                        sunlight_hours = float(mean(point_hour_totals))

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
        update_status(
            current_case_index=index,
            total_cases=total_cases,
            completed_cases=len(results),
            matched_sample_id=case['matched_sample_id'],
            stage='dispatch_case',
        )
        result = run_case(
            case,
            current_case_index=index,
            total_cases=total_cases,
            completed_cases=len(results),
        )
        results.append(result)
        RESULT_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding='utf-8')
        update_status(
            current_case_index=index,
            total_cases=total_cases,
            completed_cases=len(results),
            matched_sample_id=case['matched_sample_id'],
            stage='case_completed',
            result_path=str(RESULT_PATH),
        )
    STATUS_PATH.write_text(json.dumps({{'status': 'completed', 'count': len(results), 'result_path': str(RESULT_PATH), 'timestamp': time.time()}}, ensure_ascii=False), encoding='utf-8')
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

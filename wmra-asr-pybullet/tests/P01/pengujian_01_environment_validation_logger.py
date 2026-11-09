#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P01 — validasi environment final dosbim 2 (dua rak, mobile WMRA).

Simpan script ini di scripts/testv2/pengujian_01_environment_validation_logger.py
Script MAIN tidak diubah. Output TERPISAH dari pengujian lama:
  data/test_logs/dosbim2_revision/P01_environment/P01_YYYYmmdd_HHMMSS/

Jalankan dari root repository:
  python scripts/testv2/pengujian_01_environment_validation_logger.py \\
    --source src/main_simulation/07_indonesian_store_jaco_camera_NEW.py

Default mode DIRECT untuk validasi otomatis. Tambah --gui untuk inspeksi visual.
P01 memeriksa struktur dan perpindahan environment, bukan akurasi CV/pick.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
import pybullet as p
import pybullet_data


def timestamp():
    return time.strftime('%Y%m%d_%H%M%S')


def load_source(path):
    spec = importlib.util.spec_from_file_location('scriptis_p01_main', str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError('Gagal import source: ' + str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def body_pos(body_id):
    return list(p.getBasePositionAndOrientation(int(body_id))[0])


def check(rows, name, criterion, fn):
    start = time.monotonic()
    try:
        success, detail, evidence = fn()
        row = dict(component=name, criterion=criterion,
                   status='Berhasil' if success else 'Gagal',
                   detail=str(detail), evidence=evidence or {}, error='',
                   elapsed_sec=round(time.monotonic() - start, 3))
    except Exception as exc:
        row = dict(component=name, criterion=criterion, status='Gagal',
                   detail=str(exc), evidence={}, error=traceback.format_exc(),
                   elapsed_sec=round(time.monotonic() - start, 3))
    rows.append(row)
    print(f"[{'PASS' if row['status'] == 'Berhasil' else 'FAIL'}] {name}: {row['detail']}")
    return row['status'] == 'Berhasil'


def shelf_distribution(sim, objects):
    distribution = Counter()
    detail = []
    for name, item in objects.items():
        pos = body_pos(item['id'])
        rack = sim.get_rack_id_from_y(pos[1])
        level = min(range(len(sim.SHELF_Z_LEVELS)),
                    key=lambda i: abs(pos[2] - (sim.SHELF_Z_LEVELS[i] + 0.065))) + 1
        distribution[(rack, level)] += 1
        detail.append({'object': name, 'rack': rack, 'level': level,
                       'xyz_m': [round(v, 5) for v in pos]})
    counts = {f'rack{rack}_shelf{level}': distribution[(rack, level)]
              for rack in (1, 2) for level in (1, 2, 3)}
    valid = len(objects) == 12 and all(count == 2 for count in counts.values())
    return valid, counts, detail


def camera_eye_expected(sim, robot_data):
    robot_id = robot_data['robot_id']
    link = sim.get_wrist_pov_mount_link_index(robot_data)
    state = p.getLinkState(robot_id, link, computeForwardKinematics=True)
    eye, _ = p.multiplyTransforms(state[4], state[5], sim.WRIST_POV_LOCAL_OFFSET,
                                  [0, 0, 0, 1])
    return np.asarray(eye, dtype=float)


def main():
    parser = argparse.ArgumentParser(description='P01 environment final dosbim 2')
    parser.add_argument('--source', default=str(Path(__file__).resolve().parents[2] / "src" / "main_simulation" / "07_indonesian_store_jaco_camera_NEW.py"))
    parser.add_argument('--output_dir', default='data/test_logs/dosbim2_revision/P01_environment')
    parser.add_argument('--gui', action='store_true')
    parser.add_argument('--settle_steps', type=int, default=240)
    parser.add_argument('--move_duration', type=float, default=0.35,
                        help='Durasi pindah station untuk validasi; tidak digunakan sebagai metrik kecepatan')
    args = parser.parse_args()

    root = Path.cwd()
    source = Path(args.source).expanduser().resolve()
    run_dir = (root / args.output_dir / f'P01_{timestamp()}').resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    sim = None
    objects, initial_poses, robot_data = {}, {}, None
    connected = False

    try:
        if not check(rows, 'Source simulasi', 'File main tersedia',
                     lambda: (source.is_file(), source, {'source': str(source)})):
            return

        def do_import():
            nonlocal sim
            sim = load_source(source)
            return True, 'Import berhasil', {'source': str(source)}
        if not check(rows, 'Import modul simulasi', 'File dapat di-import tanpa menjalankan main()', do_import):
            return

        check(rows, 'Konfigurasi environment', '2 rak × 3 shelf × 2 objek', lambda: (
            bool(sim.MULTI_RACK_MODE) and sim.SHELF_LEVELS == 3
            and len(sim.SHELF_Z_LEVELS) == 3 and sim.OBJECT_RANDOMIZE_EQUAL_2_PER_SHELF,
            'multi_rack=%s, shelf_levels=%s, equal_2=%s' %
            (sim.MULTI_RACK_MODE, sim.SHELF_LEVELS, sim.OBJECT_RANDOMIZE_EQUAL_2_PER_SHELF),
            {'rack2_offset_y_m': sim.RACK_2_Y_OFFSET}))
        check(rows, 'Konfigurasi camera', 'Wrist camera realtime, bukan platform camera', lambda: (
            sim.WRIST_POV_CAMERA_ENABLED and sim.WRIST_POV_LIVE_DURING_MOTION
            and sim.WRIST_POV_LOCK_WORLD_ORIENTATION and not sim.REALTIME_FIXED_CAMERA_MODE,
            'wrist=%s realtime=%s world_lock=%s platform_mode=%s' %
            (sim.WRIST_POV_CAMERA_ENABLED, sim.WRIST_POV_LIVE_DURING_MOTION,
             sim.WRIST_POV_LOCK_WORLD_ORIENTATION, sim.REALTIME_FIXED_CAMERA_MODE), {}))
        check(rows, 'Konfigurasi scan', '6 titik scan untuk setiap rak aktif', lambda: (
            len(sim.build_visual_search_scan_plan()) == 6,
            'jumlah titik scan station1=%d' % len(sim.build_visual_search_scan_plan()), {}))
        check(rows, 'Release guard', 'Aktif dan max align iterasi = 12', lambda: (
            sim.RELEASE_GUARD_ENABLED and sim.RELEASE_GUARD_MAX_ALIGN_ITERS == 12,
            'enabled=%s, max_iters=%s' %
            (sim.RELEASE_GUARD_ENABLED, sim.RELEASE_GUARD_MAX_ALIGN_ITERS), {}))
        check(rows, 'URDF JACO', 'Path URDF robot tersedia', lambda: (
            Path(sim.JACO_URDF).expanduser().is_file(), sim.JACO_URDF, {}))

        client = p.connect(p.GUI if args.gui else p.DIRECT)
        connected = client >= 0
        if not check(rows, 'PyBullet client', 'Physics client terhubung',
                     lambda: (connected, f'client_id={client}', {'gui': args.gui})):
            return
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)
        p.setPhysicsEngineParameter(fixedTimeStep=1 / 240, numSolverIterations=150,
                                    numSubSteps=2)

        # Mencegah log randomization P01 tercampur output lama.
        sim.OBJECT_RANDOM_LOG_DIR = str(run_dir / 'randomization')
        def scene():
            nonlocal objects, initial_poses
            objects, initial_poses = sim.create_indonesian_store_scene()
            for _ in range(max(1, args.settle_steps)):
                p.stepSimulation()
            ok, counts, poses = shelf_distribution(sim, objects)
            return ok, f'{len(objects)} objek; {counts}', {'counts': counts, 'objects': poses}
        scene_ok = check(rows, 'Scene awal', '12 objek; masing-masing 2 per shelf di kedua rak', scene)

        def static_rack():
            names = list(sim.BODY_NAME_BY_ID.values())
            racks = {}
            for rid in (1, 2):
                racks[str(rid)] = sorted(name for name in names
                    if name.startswith(f'rak{rid}_level_'))
            ok = all(len(racks[str(rid)]) == 3 for rid in (1, 2))
            return ok, f'Level terbuat per rak: {racks}', {'levels': racks}
        check(rows, 'Geometri dua rak', 'Masing-masing rak memiliki 3 papan level', static_rack)

        def stable():
            problems = []
            for name, item in objects.items():
                pos = body_pos(item['id'])
                if pos[2] < 0.45 or pos[2] > 1.40 or sim.get_rack_id_from_y(pos[1]) is None:
                    problems.append({'object': name, 'xyz_m': pos})
            return not problems, f'Objek di luar area rak: {len(problems)}', {'problems': problems}
        if scene_ok:
            check(rows, 'Stabilitas objek', 'Objek tidak jatuh atau keluar dari area rak', stable)

        def randomized():
            # Preserve origin slots from stabilized scene, as in main().
            for name, item in objects.items():
                pos, orn = p.getBasePositionAndOrientation(item['id'])
                initial_poses[name]['position'] = list(pos)
                initial_poses[name]['orientation'] = list(orn)
            sim.reset_objects_randomized(objects, initial_poses, reason='P01_dosbim2')
            ok, counts, poses = shelf_distribution(sim, objects)
            original_rack = {name: sim.get_rack_id_from_y(initial_poses[name]['position'][1])
                             for name in objects}
            actual_rack = {entry['object']: entry['rack'] for entry in poses}
            preserved = all(actual_rack[name] == original_rack[name] for name in objects)
            return (ok and preserved, f'2 objek/shelf setelah random={ok}; tetap di rak asal={preserved}',
                    {'counts': counts, 'original_rack': original_rack, 'actual_rack': actual_rack,
                     'objects': poses})
        if scene_ok:
            check(rows, 'Randomisasi', '2 objek/shelf tetap dan objek tidak tertukar antarrak', randomized)

        def robot():
            nonlocal robot_data
            robot_data = sim.load_jaco_robot()
            sim.register_mobile_platform_body(robot_data['robot_id'])
            sim.reset_robot(robot_data)
            count = len(robot_data['arm_joints'])
            return (count == 6, f'Arm joints={count}, robot_id={robot_data["robot_id"]}',
                    {'robot_id': robot_data['robot_id'], 'arm_joints': robot_data['arm_joints']})
        if not check(rows, 'JACO mobile mount', 'Robot 6-DoF terdaftar sebagai mobile body', robot):
            return

        def mobile_registry():
            ids = sim.MOBILE_PLATFORM_BODY_IDS
            names = [sim.get_body_name(i) for i in ids]
            chassis = any(name == 'electric_wheelchair_chassis' for name in names)
            basket = any('handover_basket' in name for name in names)
            robot_registered = robot_data['robot_id'] in ids
            return (chassis and basket and robot_registered,
                    f'chassis={chassis}, basket={basket}, robot={robot_registered}; mobile_bodies={len(ids)}',
                    {'names': names})
        check(rows, 'Registrasi mobile platform', 'Chassis, JACO, dan basket ikut platform', mobile_registry)

        def camera_capture():
            sim.set_live_camera_context(robot_data=robot_data, objects=objects, selected_name=None)
            rgb, depth, seg = sim.capture_synthetic_camera(
                width=160, height=120, robot_data=robot_data, objects=objects)
            ok = rgb.shape == (120, 160, 3) and depth.shape == (120, 160) \
                and seg.shape == (120, 160)
            return (ok, f'RGB={rgb.shape}, depth={depth.shape}, segmentation={seg.shape}',
                    {'rgb_shape': list(rgb.shape), 'depth_shape': list(depth.shape),
                     'segmentation_shape': list(seg.shape)})
        check(rows, 'Capture sensor wrist', 'RGB, depth, segmentation berhasil diambil', camera_capture)

        def eye_matches_wrist():
            actual_eye, target, up = sim.get_wrist_pov_camera_pose(robot_data=robot_data)
            wrist_eye = camera_eye_expected(sim, robot_data)
            distance = float(np.linalg.norm(np.asarray(actual_eye) - wrist_eye))
            return (distance < 1e-5, f'camera_to_wrist_eye_error={distance:.8f} m',
                    {'camera_eye': actual_eye, 'expected_wrist_eye': wrist_eye.tolist(),
                     'look_target': target, 'up_vector': up})
        check(rows, 'Posisi kamera wrist', 'Eye kamera terpasang pada link wrist (bukan bokong/platform)', eye_matches_wrist)

        def real_wrist_tracking():
            joint = robot_data['arm_joints'][0]
            original = p.getJointState(robot_data['robot_id'], joint)[0]
            before = np.asarray(sim.get_wrist_pov_camera_pose(robot_data=robot_data)[0])
            try:
                p.resetJointState(robot_data['robot_id'], joint, original + 0.20, 0.0)
                after = np.asarray(sim.get_wrist_pov_camera_pose(robot_data=robot_data)[0])
            finally:
                p.resetJointState(robot_data['robot_id'], joint, original, 0.0)
            movement = float(np.linalg.norm(after - before))
            return (movement > 0.001, f'camera eye bergeser {movement:.5f} m saat joint lengan berubah',
                    {'eye_before': before.tolist(), 'eye_after': after.tolist(), 'delta_m': movement})
        check(rows, 'Kamera mengikuti wrist', 'Eye berubah mengikuti gerak joint wrist/arm', real_wrist_tracking)

        # Snap posisi mobile dan static untuk mengecek perpindahan 1->2->1.
        original_mobile = {bid: body_pos(bid) for bid in sim.MOBILE_PLATFORM_BODY_IDS}
        original_objects = {name: body_pos(item['id']) for name, item in objects.items()}
        original_rack_bodies = {bid: body_pos(bid) for bid, name in sim.BODY_NAME_BY_ID.items()
                                if name.startswith('rak1_') or name.startswith('rak2_')}

        def station2():
            moved = sim.move_wheelchair_to_station(robot_data, 2, duration=args.move_duration)
            dy = float(sim.RACK_2_Y_OFFSET)
            errors = {sim.get_body_name(bid): round(body_pos(bid)[1] - start[1] - dy, 5)
                      for bid, start in original_mobile.items()}
            ok = moved and sim.CURRENT_STATION_ID == 2 and sim.ACTIVE_RACK_ID == 2 \
                 and all(abs(err) < 0.015 for err in errors.values())
            return (ok, f'station={sim.CURRENT_STATION_ID}, active_rack={sim.ACTIVE_RACK_ID}; '
                    f'max_delta_error={max((abs(v) for v in errors.values()), default=0):.5f} m',
                    {'mobile_dy_error_m': errors, 'expected_platform_dy_m': dy})
        check(rows, 'Wheelchair maju', 'Station 1→2; semua mobile body maju 1,15 m', station2)

        def rack2_plan():
            plan = sim.build_visual_search_scan_plan()
            ok = len(plan) == 6 and all(item['rack_id'] == 2 for item in plan)
            return (ok, f'scan_count={len(plan)}, rack_ids={sorted(set(item["rack_id"] for item in plan))}',
                    {'plan': plan})
        check(rows, 'Scan station 2', 'Scan plan berisi 6 titik dan menunjuk rak 2', rack2_plan)

        def static_objects():
            errors = {name: round(float(np.linalg.norm(
                      np.asarray(body_pos(item['id'])) - np.asarray(original_objects[name]))), 5)
                      for name, item in objects.items()}
            rack_errors = {sim.get_body_name(bid): round(float(np.linalg.norm(
                           np.asarray(body_pos(bid)) - np.asarray(pos))), 5)
                           for bid, pos in original_rack_bodies.items()}
            # Objek bisa tersentuh gripper saat reset; laporkan secara jujur jika bergeser.
            ok = all(err < 0.02 for err in errors.values()) \
                 and all(err < 1e-5 for err in rack_errors.values())
            return (ok, f'obj_max_delta={max(errors.values(), default=0):.5f} m; '
                    f'rak_max_delta={max(rack_errors.values(), default=0):.5f} m',
                    {'object_displacements_m': errors, 'rack_displacements_m': rack_errors})
        check(rows, 'Rak/objek tetap diam', 'Hanya platform mobile yang berpindah', static_objects)

        def station1():
            moved = sim.move_wheelchair_to_station(robot_data, 1, duration=args.move_duration)
            errors = {sim.get_body_name(bid): round(body_pos(bid)[1] - start[1], 5)
                      for bid, start in original_mobile.items()}
            ok = moved and sim.CURRENT_STATION_ID == 1 and sim.ACTIVE_RACK_ID == 1 \
                 and all(abs(err) < 0.015 for err in errors.values())
            return (ok, f'station={sim.CURRENT_STATION_ID}; '
                    f'max_return_error={max((abs(v) for v in errors.values()), default=0):.5f} m',
                    {'mobile_return_error_y_m': errors})
        check(rows, 'Wheelchair mundur', 'Station 2→1; platform kembali ke posisi awal', station1)

    except Exception:
        check(rows, 'Eksekusi P01', 'Seluruh tahap pengujian dieksekusi tanpa exception tak tertangani',
              lambda: (_ for _ in ()).throw(RuntimeError(traceback.format_exc())))
    finally:
        if connected and p.isConnected():
            p.disconnect()
        csv_path = run_dir / 'pengujian_01_environment_validation.csv'
        json_path = run_dir / 'pengujian_01_environment_validation_summary.json'
        md_path = run_dir / 'pengujian_01_environment_validation_summary.md'
        fields = ['no', 'component', 'criterion', 'status', 'detail', 'evidence_json', 'elapsed_sec', 'error']
        with csv_path.open('w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fields)
            writer.writeheader()
            for index, row in enumerate(rows, 1):
                writer.writerow({**{key: row.get(key, '') for key in fields},
                                 'no': index,
                                 'evidence_json': json.dumps(row['evidence'], ensure_ascii=False, default=str)})
        passed = sum(row['status'] == 'Berhasil' for row in rows)
        summary = dict(test='P01', revision='dosbim2',
                       timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
                       source=str(source), source_sha256=hashlib.sha256(source.read_bytes()).hexdigest()
                       if source.is_file() else None,
                       total=len(rows), passed=passed, failed=len(rows)-passed,
                       status='VALID' if rows and passed == len(rows) else 'PERLU_PERBAIKAN',
                       output_dir=str(run_dir), rows=rows)
        json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
        md_lines = ['# P01 — Validasi Environment Final Dosbim 2', '',
                    f"Waktu: {summary['timestamp']}", f"Source: `{source}`",
                    f"SHA256 source: `{summary['source_sha256']}`",
                    f"Status: **{summary['status']}** ({passed}/{len(rows)} lulus)", '',
                    '| No | Pemeriksaan | Kriteria | Hasil | Detail |',
                    '|---:|---|---|---|---|']
        for i, row in enumerate(rows, 1):
            detail = row['detail'].replace('|', '\\|').replace('\n', ' ')
            md_lines.append(f"| {i} | {row['component']} | {row['criterion']} | {row['status']} | {detail} |")
        md_lines += ['', 'Catatan: P01 menguji lingkungan dan konfigurasi, bukan keberhasilan CV, grasp, atau end-to-end.']
        md_path.write_text('\n'.join(md_lines) + '\n', encoding='utf-8')
        print('\n=== P01 ENVIRONMENT VALIDATION — DOSBIM 2 ===')
        print(f"Status : {summary['status']} ({passed}/{len(rows)} lulus)")
        print(f'CSV    : {csv_path}')
        print(f'JSON   : {json_path}')
        print(f'MD     : {md_path}')


if __name__ == '__main__':
    main()

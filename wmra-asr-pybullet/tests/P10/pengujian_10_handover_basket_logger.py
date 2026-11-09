#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P10 DOSBIM2 — handover, release guard, natural drop ke keranjang mobile.

Jalankan dari root repository:
  python scripts/testv2/pengujian_10_handover_basket_logger.py --limit 2 --gui
  python scripts/testv2/pengujian_10_handover_basket_logger.py --repeat 4 --gui

36 trial formal = 9 kelas (3 bentuk x 3 warna) x 4 repetisi, setiap trial
randomisasi dalam rak asal; ground-truth pose objek dipakai untuk mengisolasi
handover dari CV/ASR. Menjalankan fungsi simulasi UTAMA untuk IK, virtual grasp,
release guard (max 12), dan natural gravity drop; TIDAK teleport objek ke basket.
Kriteria basket: seluruh proyeksi AABB XY objek ada dalam ruang dalam keranjang
(+margin opsional), dasar objek dekat lantai basket setelah settling, tidak lagi
attached. Keberhasilan release guard dan objek masuk basket dicatat terpisah.
Collision objek saat transport dimatikan dalam simulasi utama; tidak mengevaluasi
force-closure grasp fisik ataupun keselamatan collision (sudah di P09).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import random
import statistics
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import pybullet as p
import pybullet_data

SHAPES = ("balok", "silinder", "segitiga")
COLORS = ("merah", "biru", "kuning")
CSV_NAME = "pengujian_10_handover_basket.csv"
JSON_NAME = "pengujian_10_handover_basket_summary.json"
MD_NAME = "pengujian_10_handover_basket_summary.md"
FIELDS = [
    "trial", "repetition", "seed", "object_name", "expected_rack", "actual_rack", "station", "shelf_index_1based",
    "object_start_xyz_m", "object_after_lift_xyz_m", "lift_height_m", "grasp_constraint_created",
    "ready_error_m", "pre_grasp_error_m", "approach_error_m", "retreat_error_m", "lift_error_m",
    "handover_bridge_error_m", "handover_present_error_m", "handover_release_point_error_m",
    "bridge_object_xyz_m", "present_object_xyz_m", "release_point_object_xyz_m",
    "guard_max_iters", "guard_initial_ready", "guard_initial_xy_error_m", "guard_initial_z_error_m",
    "guard_align_return_ready", "guard_after_align_ready", "guard_after_stop_ready", "guard_final_xy_error_m", "guard_final_z_error_m",
    "wait_robot_stop_ok", "release_performed", "constraint_after_release", "object_at_release_xyz_m",
    "basket_center_xyz_m", "basket_floor_top_z_m", "basket_inner_half_xy_m", "basket_floor_body_contact",
    "object_final_xyz_m", "object_final_aabb_min_xyz_m", "object_final_aabb_max_xyz_m",
    "basket_inside_xy_aabb", "basket_floor_z_error_m", "basket_floor_z_ok", "basket_success",
    "handover_basket_success", "failure_stage", "status", "elapsed_sec", "error"
]


def xyz(value):
    return json.dumps([round(float(v), 6) for v in value])


def pos(body_id):
    return [float(v) for v in p.getBasePositionAndOrientation(body_id)[0]]


def import_sim(path):
    spec = importlib.util.spec_from_file_location("scriptis_p10_dosbim2_main", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Tidak dapat mengimpor simulasi: {path}")
    sim = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = sim
    spec.loader.exec_module(sim)
    return sim


def constraint_exists(cid):
    if cid is None:
        return False
    try:
        return any(p.getConstraintUniqueId(i) == cid for i in range(p.getNumConstraints()))
    except p.error:
        return False


def move_stage(sim, robot_data, row, label, target, *, duration, force=170, velocity=0.15,
               target_id=None, constraint_id=None, require_constraint=False):
    err, next_id, dropped = sim.move_robot_to_position(
        robot_data=robot_data,
        target_position=target,
        duration=duration,
        attached_object_id=target_id,
        grasp_constraint_id=constraint_id,
        release_on_collision=False,
        motor_force=force,
        max_velocity=velocity,
        error_print_limit=float(getattr(sim, "HANDOVER_ERROR_LIMIT", 0.42))
    )
    actual, _ = sim.get_end_effector_position(robot_data)
    error = math.dist([float(v) for v in actual], [float(v) for v in target])
    row[f"{label}_error_m"] = round(error, 6)
    if require_constraint and (bool(dropped) or not constraint_exists(constraint_id)):
        row["failure_stage"] = f"{label}_drop_or_constraint_lost"
        return next_id, False
    if dropped:
        row["failure_stage"] = f"{label}_stopped"
        return next_id, False
    return next_id, True


def check_basket(sim, target_id, row, args):
    """Evaluasi posisi fisik objek setelah drop; tanpa mengubah posisi objek."""
    if sim.BASKET_INFO is None:
        row["failure_stage"] = "basket_info_missing"
        return False
    info = sim.BASKET_INFO
    center = [float(v) for v in sim.BASKET_CENTER]
    floor_top = float(info["floor_top_z"])
    half_x = float(info["inner_size"][0]) / 2.0
    half_y = float(info["inner_size"][1]) / 2.0
    aabb_min, aabb_max = p.getAABB(target_id)
    x0, y0, z0 = [float(v) for v in aabb_min]
    x1, y1, z1 = [float(v) for v in aabb_max]
    margin = float(args.xy_margin)
    inside = bool(x0 >= center[0] - half_x + margin and x1 <= center[0] + half_x - margin
                  and y0 >= center[1] - half_y + margin and y1 <= center[1] + half_y - margin)
    floor_error = z0 - floor_top
    floor_ok = bool(abs(floor_error) <= args.floor_tolerance)
    floor_contact = False
    for body_id in info.get("parts", []):
        if sim.get_body_name(body_id) == "handover_basket_floor":
            floor_contact = bool(p.getContactPoints(bodyA=target_id, bodyB=body_id))
            break
    row.update({
        "basket_center_xyz_m": xyz(center),
        "basket_floor_top_z_m": round(floor_top, 6),
        "basket_inner_half_xy_m": json.dumps([round(half_x, 6), round(half_y, 6)]),
        "basket_floor_body_contact": floor_contact,
        "object_final_xyz_m": xyz(pos(target_id)),
        "object_final_aabb_min_xyz_m": xyz(aabb_min),
        "object_final_aabb_max_xyz_m": xyz(aabb_max),
        "basket_inside_xy_aabb": inside,
        "basket_floor_z_error_m": round(floor_error, 6),
        "basket_floor_z_ok": floor_ok,
    })
    row["basket_success"] = bool(inside and floor_ok and not constraint_exists(row.get("_active_constraint")))
    return bool(row["basket_success"])


def run_trial(sim, robot_data, objects, initial_poses, row, args):
    name = row["object_name"]
    target_id = objects[name]["id"]
    random.seed(row["seed"])
    sim.reset_objects_randomized(objects, initial_poses, reason=f"P10_trial_{row['trial']:03d}")
    if sim.move_wheelchair_to_station(robot_data, row["expected_rack"], duration=0.35) is not True:
        row["failure_stage"] = "station_move"
        return
    sim.reset_robot(robot_data)
    row["station"] = int(sim.CURRENT_STATION_ID)
    for _ in range(args.settle_steps):
        p.stepSimulation()
    start = pos(target_id)
    row["object_start_xyz_m"] = xyz(start)
    row["actual_rack"] = int(sim.get_rack_id_from_y(start[1]))
    if row["actual_rack"] != row["expected_rack"] or row["station"] != row["expected_rack"]:
        row["failure_stage"] = "rack_or_station_mismatch"
        return
    if sim.BASKET_INFO is None or int(sim.RELEASE_GUARD_MAX_ALIGN_ITERS) != 12:
        row["failure_stage"] = "basket_or_release_guard_configuration"
        return
    points = sim.get_shelf_aware_motion_points(start)
    shelf = int(points["shelf_index"])
    row["shelf_index_1based"] = shelf + 1
    sim.rotate_joint1_to_face_object(robot_data, start, duration=sim.JOINT1_ALIGN_DURATION)
    for label, duration, force, velocity, limit in (
        ("ready", 1.8, 200, 0.25, sim.READY_ERROR_LIMIT),
        ("pre_grasp", 2.5, 190, 0.18, sim.REACH_ERROR_LIMIT),
        ("approach", 2.2, 170, 0.12, sim.BOTTOM_APPROACH_ERROR_LIMIT if shelf == 0 else sim.APPROACH_ERROR_LIMIT),
    ):
        if label == "pre_grasp":
            sim.set_gripper_open(robot_data)
        _, moved = move_stage(sim, robot_data, row, label, points[label],
                              duration=duration, force=force, velocity=velocity)
        if not moved or row[f"{label}_error_m"] > float(limit):
            row["failure_stage"] = row["failure_stage"] or f"{label}_position_error"
            return
    cid = sim.attach_object_to_end_effector(robot_data=robot_data, object_id=target_id)
    row["_active_constraint"] = cid
    row["grasp_constraint_created"] = constraint_exists(cid)
    if not row["grasp_constraint_created"]:
        row["failure_stage"] = "virtual_grasp"
        return
    for _ in range(60):
        p.stepSimulation()
    for label, duration in (("retreat", 2.0), ("lift", 2.5)):
        next_id, moved = move_stage(sim, robot_data, row, label, points[label], duration=duration,
                                     target_id=target_id, constraint_id=cid, require_constraint=True)
        if not moved:
            return
        cid = next_id if next_id is not None else cid
    row["object_after_lift_xyz_m"] = xyz(pos(target_id))
    row["lift_height_m"] = round(pos(target_id)[2] - start[2], 6)
    if row["lift_height_m"] < args.min_lift_height:
        row["failure_stage"] = "lift_height"
        return

    # Jalur handover adalah jalur dari simulasi utama, memakai referensi base
    # dan HANDOVER_POS setelah station dipindah (basket ikut wheelchair).
    bridge = [sim.ROBOT_BASE_POS[0] + 0.30, sim.ROBOT_BASE_POS[1] + 0.02,
              sim.HANDOVER_POS[2] + 0.16]
    present = [sim.HANDOVER_POS[0] + 0.12, sim.HANDOVER_POS[1] - 0.10,
               sim.HANDOVER_POS[2] + 0.12]
    release_point = [sim.HANDOVER_POS[0] + 0.06, sim.HANDOVER_POS[1] - 0.04,
                     sim.HANDOVER_POS[2] + 0.07]
    for label, target, duration, force, velocity, settle in (
        ("handover_bridge", bridge, 1.80, 180, 0.20, 0),
        ("handover_present", present, 1.90, 175, 0.16, 60),
        ("handover_release_point", release_point, 1.20, 155, 0.13, 0),
    ):
        next_id, moved = move_stage(sim, robot_data, row, label, target, duration=duration,
                                     force=force, velocity=velocity, target_id=target_id,
                                     constraint_id=cid, require_constraint=True)
        if not moved:
            return
        cid = next_id if next_id is not None else cid
        if label == "handover_bridge":
            row["bridge_object_xyz_m"] = xyz(pos(target_id))
        elif label == "handover_present":
            row["present_object_xyz_m"] = xyz(pos(target_id))
        else:
            row["release_point_object_xyz_m"] = xyz(pos(target_id))
        for _ in range(settle):
            p.stepSimulation()

    # Release guard yang SAMA dengan simulasi utama, bukan menggeser objek
    # secara langsung. Catat guard sebelum alignment dan sesudah robot stop.
    initial_ready, initial_info = sim.check_object_ready_to_release_into_basket(target_id, verbose=False)
    row["guard_initial_ready"] = bool(initial_ready)
    row["guard_initial_xy_error_m"] = round(float(initial_info.get("xy_error", math.nan)), 6)
    row["guard_initial_z_error_m"] = round(float(initial_info.get("z_error", math.nan)), 6)
    next_id, dropped, ready = sim.align_attached_object_over_basket_release_guard(
        robot_data=robot_data, object_id=target_id, grasp_constraint=cid,
        max_iters=sim.RELEASE_GUARD_MAX_ALIGN_ITERS)
    if next_id is not None:
        cid = next_id if next_id is not None else cid
    row["guard_align_return_ready"] = bool(ready)
    ready_now, _ = sim.check_object_ready_to_release_into_basket(target_id, verbose=False)
    row["guard_after_align_ready"] = bool(ready_now)
    if dropped or not constraint_exists(cid):
        row["failure_stage"] = "release_guard_drop_or_constraint_lost"
        return
    if not ready or not ready_now:
        row["failure_stage"] = "release_guard_not_ready"
        return   # DILARANG release bila guard gagal
    stopped = sim.wait_until_robot_stops_before_release(robot_data=robot_data, object_id=target_id)
    row["wait_robot_stop_ok"] = bool(stopped)
    if not stopped:
        row["failure_stage"] = "robot_not_stable_before_release"
        return  # P10 mensyaratkan robot berhenti; simulasi utama bisa lanjut pada timeout.
    # Ikuti main: delay opsional (biasanya off), lalu final check.
    sim.delay_after_robot_stop_before_release(robot_data=robot_data, object_id=target_id,
                                              seconds=sim.RELEASE_DELAY_AFTER_STOP_SEC)
    final_ready, final_info = sim.check_object_ready_to_release_into_basket(target_id, verbose=False)
    row["guard_after_stop_ready"] = bool(final_ready)
    row["guard_final_xy_error_m"] = round(float(final_info.get("xy_error", math.nan)), 6)
    row["guard_final_z_error_m"] = round(float(final_info.get("z_error", math.nan)), 6)
    if not final_ready or not constraint_exists(cid):
        row["failure_stage"] = "final_release_guard_not_ready"
        return  # Guard gagal: objek tetap ditahan selama trial

    row["object_at_release_xyz_m"] = xyz(pos(target_id))
    sim.enable_object_collision_for_natural_drop(target_id)
    sim.release_object(cid)
    row["release_performed"] = True
    row["constraint_after_release"] = constraint_exists(cid)
    if row["constraint_after_release"]:
        row["failure_stage"] = "constraint_not_released"
        return
    row["_active_constraint"] = None
    sim.set_gripper_open(robot_data)
    sim.natural_gravity_drop_to_basket(target_id, steps=sim.BASKET_DROP_SETTLE_STEPS)
    inside = check_basket(sim, target_id, row, args)
    if not inside:
        row["failure_stage"] = "object_not_inside_basket_after_drop"
        return
    row["handover_basket_success"] = True
    row["status"] = "Berhasil"


def clean_trial(sim, robot_data, objects, row):
    cid = row.get("_active_constraint")
    target_id = objects[row["object_name"]]["id"]
    if cid is not None:
        try:
            sim.release_object(cid)
        except Exception:
            if constraint_exists(cid):
                p.removeConstraint(cid)
    sim.set_object_transport_mode(target_id, enabled=False)
    sim.set_gripper_open(robot_data)
    row.pop("_active_constraint", None)


def summary_write(rows, run_dir, source, source_hash, args, planned, completed):
    success = sum(r.get("handover_basket_success") is True for r in rows)
    total = len(rows)
    by_shelf = defaultdict(list)
    by_rack = defaultdict(list)
    for r in rows:
        by_shelf[str(r.get("shelf_index_1based", ""))].append(r)
        by_rack[str(r.get("actual_rack", ""))].append(r)
    def group_stats(groups):
        return {k: {"trials": len(group), "success": sum(r.get("handover_basket_success") is True for r in group)}
                for k, group in sorted(groups.items())}
    times = [float(r["elapsed_sec"]) for r in rows if r.get("elapsed_sec") not in (None, "")]
    report = {
        "test": "P10", "revision": "dosbim2", "completed": bool(completed), "source": str(source),
        "source_sha256": source_hash, "mode": "GUI" if args.gui else "DIRECT",
        "planned_repetitions": args.repeat, "planned_trials_this_run": planned,
        "trial_count": total, "handover_basket_success": success, "failed": total - success,
        "success_rate_percent": round(100 * success / total, 3) if total else None,
        "release_guard_ready_after_stop": sum(r.get("guard_after_stop_ready") is True for r in rows),
        "release_performed": sum(r.get("release_performed") is True for r in rows),
        "basket_success": sum(r.get("basket_success") is True for r in rows),
        "by_rack": group_stats(by_rack), "by_shelf": group_stats(by_shelf),
        "avg_elapsed_sec": round(statistics.mean(times), 3) if times else None,
        "run_dir": str(run_dir),
        "failure_stages": [{"trial": r["trial"], "object_name": r["object_name"],
                            "stage": r["failure_stage"], "error": r["error"]}
                           for r in rows if r.get("handover_basket_success") is not True],
        "measurement_notes": [
            "Objek awal memakai ground truth PyBullet. P10 tidak menguji ASR atau visual search.",
            "Virtual fixed-constraint grasp mengandung alignment/seating; collision objek dimatikan selama transport.",
            "P10 lebih ketat daripada main saat stop timeout: P10 membatalkan release jika wait-stop mengembalikan False.",
            "Release dilarang bila guard tidak ready. Max alignment iteration = 12 di source utama.",
            "Objek masuk basket dinilai dari AABB XY seluruh objek dan posisi dasar dekat floor, bukan jarak XY <= 0.16 saja.",
            "Pengujian script P10 tidak mengklaim force-closure fisik atau collision-free transport.",
        ],
    }
    (run_dir / JSON_NAME).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# P10 — Handover dan keranjang (dosbim2)", "",
             f"Source hash: `{source_hash}`. Mode: {report['mode']}.",
             f"Trial: {total}/{planned}; completed={completed}.",
             f"Release guard ready: {report['release_guard_ready_after_stop']}/{total}.",
             f"Release dilakukan: {report['release_performed']}/{total}.",
             f"Berhasil masuk basket: {success}/{total} ({report['success_rate_percent']}%).", "",
             "| Rak | Trial | Berhasil |", "|---|---:|---:|"]
    for k, v in report["by_rack"].items():
        lines.append(f"| {k} | {v['trials']} | {v['success']} |")
    lines.extend(["", "| Shelf (1=bawah) | Trial | Berhasil |", "|---|---:|---:|"])
    for k, v in report["by_shelf"].items():
        lines.append(f"| {k} | {v['trials']} | {v['success']} |")
    lines.extend(["", *report["measurement_notes"]])
    (run_dir / MD_NAME).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description="P10 — handover/release guard/basket dua rak")
    parser.add_argument("--source", default=str(Path(__file__).resolve().parents[2] / "src" / "main_simulation" / "07_indonesian_store_jaco_camera_NEW.py"))
    parser.add_argument("--output_dir", default="data/test_logs/dosbim2_revision/P10_handover_basket")
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="N trial untuk uji coba; 0=seluruh trial")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--settle_steps", type=int, default=120)
    parser.add_argument("--min_lift_height", type=float, default=0.05)
    parser.add_argument("--xy_margin", type=float, default=0.002)
    parser.add_argument("--floor_tolerance", type=float, default=0.035)
    parser.add_argument("--gui", action="store_true")
    args = parser.parse_args()
    if args.repeat < 1 or args.limit < 0 or args.settle_steps < 0 or args.min_lift_height <= 0 or args.xy_margin < 0 or args.floor_tolerance <= 0:
        parser.error("Periksa --repeat, --limit, --min_lift_height, --xy_margin, --floor_tolerance")
    source = Path(args.source).expanduser().resolve()
    if not source.is_file():
        parser.error(f"Simulasi utama tidak ditemukan: {source}")
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    run_dir = Path(args.output_dir).expanduser().resolve() / time.strftime("P10_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    sim = import_sim(source)
    sim.SHOW_CAMERA_WINDOWS = False
    sim.WRIST_POV_LIVE_DURING_MOTION = False
    sim.OBJECT_RANDOM_LOG_DIR = str(run_dir / "randomization")
    client = p.connect(p.GUI if args.gui else p.DIRECT)
    if client < 0:
        raise RuntimeError("PyBullet gagal connect")
    rows = []
    try:
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)
        p.setPhysicsEngineParameter(fixedTimeStep=1.0 / 240.0, numSolverIterations=150, numSubSteps=2)
        objects, initial_poses = sim.create_indonesian_store_scene()
        robot_data = sim.load_jaco_robot()
        sim.register_mobile_platform_body(robot_data["robot_id"])
        sim.set_live_camera_context(robot_data=robot_data, objects=objects)
        sim.reset_robot(robot_data)
        for _ in range(max(1, args.settle_steps)):
            p.stepSimulation()
        targets = []
        for shape in SHAPES:
            for color in COLORS:
                name = f"{shape}_{color}"
                if name not in objects:
                    raise RuntimeError(f"Objek tidak ditemukan: {name}")
                rack = int(sim.get_rack_id_from_y(initial_poses[name]["position"][1]))
                if rack not in (1, 2):
                    raise RuntimeError(f"Rak target di luar 1/2: {name}: {rack}")
                targets.append((name, rack))
        trials = [(rep, name, rack) for rep in range(1, args.repeat + 1) for name, rack in targets]
        if args.limit:
            trials = trials[:args.limit]
        print(f"\n=== P10 DOSBIM2 | {len(trials)} trial | source sha256={sha} ===")
        print("Output:", run_dir)
        csv_path = run_dir / CSV_NAME
        for idx, (rep, name, rack) in enumerate(trials, 1):
            row = {field: "" for field in FIELDS}
            row.update({"trial": idx, "repetition": rep, "seed": args.seed + idx, "object_name": name,
                        "expected_rack": rack, "guard_max_iters": int(sim.RELEASE_GUARD_MAX_ALIGN_ITERS),
                        "grasp_constraint_created": False, "release_performed": False,
                        "basket_success": False, "handover_basket_success": False,
                        "status": "Gagal", "failure_stage": ""})
            started = time.monotonic()
            try:
                run_trial(sim, robot_data, objects, initial_poses, row, args)
            except Exception:
                row["failure_stage"] = row["failure_stage"] or "runtime_error"
                row["error"] = traceback.format_exc()
                print(row["error"])
            finally:
                try:
                    clean_trial(sim, robot_data, objects, row)
                except Exception:
                    row["failure_stage"] = row["failure_stage"] or "cleanup_error"
                    row["error"] = (row.get("error") or "") + "\nCLEANUP ERROR:\n" + traceback.format_exc()
                    row["handover_basket_success"] = False
                    row["status"] = "Gagal"
                row["elapsed_sec"] = round(time.monotonic() - started, 3)
                rows.append(row)
                with csv_path.open("w", newline="", encoding="utf-8") as file:
                    writer = csv.DictWriter(file, fieldnames=FIELDS, extrasaction="ignore")
                    writer.writeheader()
                    writer.writerows(rows)
                summary_write(rows, run_dir, source, sha, args, len(trials), completed=False)
                print(f"[P10 {idx}/{len(trials)}] {name} | rack={row['actual_rack'] or '?'} "
                      f"shelf={row['shelf_index_1based'] or '?'} | guard={row['guard_after_stop_ready']} "
                      f"release={row['release_performed']} | basket={row['basket_success']} "
                      f"status={row['status']} | reason={row['failure_stage'] or '-'}")
        final = summary_write(rows, run_dir, source, sha, args, len(trials), completed=True)
        print("\n=== P10 SELESAI ===")
        print(json.dumps({k: final[k] for k in ("trial_count", "release_guard_ready_after_stop", "release_performed", "handover_basket_success", "failed", "success_rate_percent", "by_shelf")}, ensure_ascii=False, indent=2))
    finally:
        if p.isConnected(client):
            p.disconnect(client)


if __name__ == "__main__":
    main()

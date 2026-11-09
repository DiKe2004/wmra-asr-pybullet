#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P08 DOSBIM2 — virtual grasp dan lift JACO2 pada dua rak.

Jalankan dari root repository:
  python scripts/testv2/pengujian_08_grasp_lift_logger.py --limit 2 --gui
  python scripts/testv2/pengujian_08_grasp_lift_logger.py --repeat 4 --gui

Menguji ready -> pre-grasp -> approach -> virtual attach -> retreat -> lift.
Tidak menjalankan ASR, CV, handover, ataupun release guard. Posisi objek diketahui
oleh penguji dari PyBullet untuk mengisolasi grasp/lift. Sistem simulasi memakai
virtual fixed constraint, object seating dan collision off saat transport; P08
BUKAN bukti physical force-closure grasp atau bebas tabrakan.

Objek di-randomisasi setiap trial tanpa pindah antarrak. Hasil gagal tidak diubah
menjadi sukses dan setiap trial dicatat langsung ke CSV + summary JSON.
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
STAGES = ("ready", "pre_grasp", "approach", "retreat", "lift")
CSV_NAME = "pengujian_08_grasp_lift.csv"
JSON_NAME = "pengujian_08_grasp_lift_summary.json"
MD_NAME = "pengujian_08_grasp_lift_summary.md"
FIELDS = [
    "trial", "repetition", "seed", "object_name", "expected_rack", "actual_rack", "station",
    "shelf_index_1based", "object_initial_xyz_m", "object_pre_attach_xyz_m",
    "object_after_attach_xyz_m", "object_after_retreat_xyz_m", "object_after_lift_xyz_m",
    "ready_target_xyz_m", "ready_actual_xyz_m", "ready_error_m", "ready_limit_m", "ready_ok", "ready_stopped",
    "pre_grasp_target_xyz_m", "pre_grasp_actual_xyz_m", "pre_grasp_error_m", "pre_grasp_limit_m", "pre_grasp_ok", "pre_grasp_stopped",
    "approach_target_xyz_m", "approach_actual_xyz_m", "approach_error_m", "approach_limit_m", "approach_ok", "approach_stopped",
    "slot_object_distance_before_attach_m", "object_shift_on_attach_m", "object_initial_gripper_contacts",
    "constraint_created", "constraint_after_retreat", "constraint_after_lift", "grasp_constraint_id",
    "retreat_target_xyz_m", "retreat_actual_xyz_m", "retreat_error_m", "retreat_stopped", "retreat_dropped",
    "lift_target_xyz_m", "lift_actual_xyz_m", "lift_error_m", "lift_stopped", "lift_dropped",
    "lift_height_m", "min_lift_height_m", "lift_height_ok", "grasp_lift_success",
    "failure_stage", "status", "elapsed_sec", "error"
]


def compact_xyz(xyz):
    return json.dumps([round(float(v), 6) for v in xyz])


def get_pos(body_id):
    pos, _ = p.getBasePositionAndOrientation(body_id)
    return [float(v) for v in pos]


def import_sim(path: Path):
    spec = importlib.util.spec_from_file_location("scriptis_p08_dosbim2_main", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Tidak bisa import simulasi: {path}")
    sim = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = sim
    spec.loader.exec_module(sim)
    return sim


def constraint_exists(constraint_id):
    if constraint_id is None:
        return False
    try:
        return any(p.getConstraintUniqueId(i) == constraint_id
                   for i in range(p.getNumConstraints()))
    except p.error:
        return False


def record_motion(row, sim, robot_data, stage, point, *, limit=None,
                  object_id=None, constraint_id=None, duration=2.0, force=200,
                  velocity=0.25):
    kwargs = dict(
        robot_data=robot_data, target_position=point, duration=duration,
        attached_object_id=object_id, grasp_constraint_id=constraint_id,
        release_on_collision=False, motor_force=force, max_velocity=velocity,
        error_print_limit=limit if limit is not None else 0.35,
    )
    _, next_constraint, stopped = sim.move_robot_to_position(**kwargs)
    actual, _ = sim.get_end_effector_position(robot_data)
    error = math.dist([float(v) for v in actual], [float(v) for v in point])
    row[f"{stage}_target_xyz_m"] = compact_xyz(point)
    row[f"{stage}_actual_xyz_m"] = compact_xyz(actual)
    row[f"{stage}_error_m"] = round(error, 6)
    row[f"{stage}_stopped"] = bool(stopped)
    if limit is not None:
        row[f"{stage}_limit_m"] = float(limit)
        row[f"{stage}_ok"] = bool(error <= limit and not stopped)
    return next_constraint, bool(stopped)


def run_trial(sim, robot_data, objects, initial_poses, row, args):
    name = row["object_name"]
    target_id = objects[name]["id"]
    random.seed(row["seed"])
    sim.reset_objects_randomized(objects, initial_poses, reason=f"P08_trial_{row['trial']:03d}")
    # Pindahkan wheelchair, robot dan basket; rak/objek tetap pada posisi dunia.
    if sim.move_wheelchair_to_station(robot_data, row["expected_rack"], duration=0.35) is not True:
        row["failure_stage"] = "station_move"
        return
    sim.reset_robot(robot_data)
    row["station"] = int(sim.CURRENT_STATION_ID)
    for _ in range(args.settle_steps):
        p.stepSimulation()

    start = get_pos(target_id)
    row["object_initial_xyz_m"] = compact_xyz(start)
    row["actual_rack"] = int(sim.get_rack_id_from_y(start[1]))
    if row["actual_rack"] != row["expected_rack"] or row["station"] != row["expected_rack"]:
        row["failure_stage"] = "target_rack_mismatch"
        return
    points = sim.get_shelf_aware_motion_points(start)
    shelf = int(points["shelf_index"])
    row["shelf_index_1based"] = shelf + 1
    limits = {
        "ready": float(sim.READY_ERROR_LIMIT),
        "pre_grasp": float(sim.REACH_ERROR_LIMIT),
        "approach": float(sim.BOTTOM_APPROACH_ERROR_LIMIT if shelf == 0 else sim.APPROACH_ERROR_LIMIT),
    }

    sim.rotate_joint1_to_face_object(robot_data, start, duration=sim.JOINT1_ALIGN_DURATION)
    row["object_initial_gripper_contacts"] = len(
        p.getContactPoints(bodyA=target_id, bodyB=robot_data["robot_id"]))
    for stage, duration, force, velocity in (
        ("ready", 1.8, 200, 0.25),
        ("pre_grasp", 2.5, 190, 0.18),
        ("approach", 2.2, 170, 0.12),
    ):
        if stage == "pre_grasp":
            sim.set_gripper_open(robot_data)
        _, stopped = record_motion(
            row, sim, robot_data, stage, points[stage], limit=limits[stage],
            duration=duration, force=force, velocity=velocity)
        if not row[f"{stage}_ok"]:
            row["failure_stage"] = stage + ("_stopped" if stopped else "_position_error")
            return

    # Ukur jarak SEBELUM fungsi attach (yang boleh meng-align robot dan men-seat objek).
    before_attach = get_pos(target_id)
    row["object_pre_attach_xyz_m"] = compact_xyz(before_attach)
    slot, _, _, _ = sim.get_grasp_slot_world_pose(robot_data, target_id)
    row["slot_object_distance_before_attach_m"] = round(math.dist(slot, before_attach), 6)
    row["object_initial_gripper_contacts"] = len(
        p.getContactPoints(bodyA=target_id, bodyB=robot_data["robot_id"]))

    # Mekanisme yang sama dengan simulasi utama (virtual grasp + fixed constraint).
    constraint_id = sim.attach_object_to_end_effector(robot_data=robot_data, object_id=target_id)
    row["grasp_constraint_id"] = constraint_id if constraint_id is not None else ""
    after_attach = get_pos(target_id)
    row["object_after_attach_xyz_m"] = compact_xyz(after_attach)
    row["object_shift_on_attach_m"] = round(math.dist(after_attach, before_attach), 6)
    row["constraint_created"] = constraint_exists(constraint_id)
    if not row["constraint_created"]:
        row["failure_stage"] = "grasp_attach"
        return

    # Simpan ID untuk cleanup meskipun motion berikutnya melempar exception.
    row["_active_constraint"] = constraint_id
    for _ in range(60):
        p.stepSimulation()
    retreat_constraint, stopped = record_motion(
        row, sim, robot_data, "retreat", points["retreat"],
        object_id=target_id, constraint_id=constraint_id, duration=2.0)
    row["retreat_dropped"] = bool(stopped or not constraint_exists(constraint_id))
    row["constraint_after_retreat"] = constraint_exists(constraint_id)
    row["object_after_retreat_xyz_m"] = compact_xyz(get_pos(target_id))
    row["_active_constraint"] = retreat_constraint if retreat_constraint is not None else constraint_id
    if row["retreat_dropped"]:
        row["failure_stage"] = "retreat_drop_or_stop"
        return

    lift_constraint, stopped = record_motion(
        row, sim, robot_data, "lift", points["lift"], object_id=target_id,
        constraint_id=retreat_constraint, duration=2.5)
    row["lift_dropped"] = bool(stopped or not constraint_exists(constraint_id))
    row["constraint_after_lift"] = constraint_exists(constraint_id)
    row["_active_constraint"] = lift_constraint if lift_constraint is not None else constraint_id
    final = get_pos(target_id)
    row["object_after_lift_xyz_m"] = compact_xyz(final)
    row["lift_height_m"] = round(final[2] - start[2], 6)
    row["lift_height_ok"] = (row["lift_height_m"] >= args.min_lift_height)
    if row["lift_dropped"]:
        row["failure_stage"] = "lift_drop_or_stop"
        return
    if not row["lift_height_ok"]:
        row["failure_stage"] = "insufficient_lift_height"
        return

    # Sukses P08 = virtual constraint bertahan dan object naik >= batas.
    row["grasp_lift_success"] = True
    row["status"] = "Berhasil"


def clean_trial(sim, robot_data, objects, row):
    constraint_id = row.get("_active_constraint")
    target_id = objects[row["object_name"]]["id"]
    if constraint_id is not None:
        try:
            sim.release_object(constraint_id)
        except Exception as exc:
            print("WARNING cleanup release:", exc)
            try:
                if constraint_exists(constraint_id):
                    p.removeConstraint(constraint_id)
            except Exception:
                pass
    try:
        sim.set_object_transport_mode(target_id, enabled=False)
    except Exception as exc:
        print("WARNING cleanup transport:", exc)
    try:
        sim.set_gripper_open(robot_data)
    except Exception:
        pass
    row.pop("_active_constraint", None)


def summarize(rows, *, run_dir, source, sha, args, planned, completed):
    total = len(rows)
    success = sum(row.get("grasp_lift_success") is True for row in rows)
    def by_key(key):
        d = defaultdict(list)
        for row in rows:
            d[str(row.get(key, ""))].append(row)
        return {k: {
            "trials": len(group),
            "success": sum(row.get("grasp_lift_success") is True for row in group),
            "rate_percent": round(100 * sum(row.get("grasp_lift_success") is True for row in group) / len(group), 3),
        } for k, group in sorted(d.items())}
    heights = [float(row["lift_height_m"]) for row in rows
               if isinstance(row.get("lift_height_m"), (int, float))]
    summary = {
        "test": "P08", "revision": "dosbim2", "completed": completed,
        "source": str(source), "source_sha256": sha, "run_dir": str(run_dir),
        "mode": "GUI" if args.gui else "DIRECT", "planned_repetitions": args.repeat,
        "planned_trials_this_run": planned, "trial_count": total,
        "min_lift_height_m": args.min_lift_height,
        "virtual_grasp_lift_success": success, "failed": total - success,
        "success_rate_percent": round(100 * success / total, 3) if total else None,
        "constraint_created": sum(r.get("constraint_created") is True for r in rows),
        "constraint_after_lift": sum(r.get("constraint_after_lift") is True for r in rows),
        "lift_height_reached": sum(r.get("lift_height_ok") is True for r in rows),
        "average_lift_height_m": round(statistics.mean(heights), 6) if heights else None,
        "by_rack": by_key("actual_rack"), "by_shelf": by_key("shelf_index_1based"),
        "failures": [{"trial": r["trial"], "object": r["object_name"],
                      "rack": r["actual_rack"], "shelf": r["shelf_index_1based"],
                      "stage": r["failure_stage"], "error": r["error"]}
                     for r in rows if r.get("grasp_lift_success") is not True],
        "interpretation_note": "Posisi objek memakai ground truth PyBullet; grasp merupakan virtual fixed constraint dengan bantuan alignment/seating dan transport collision off. Hasil bukan performa deteksi CV, force-closure fisik, handover, atau bukti collision-free.",
    }
    (run_dir / JSON_NAME).write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    text = [
        "# P08 — Virtual Grasp dan Lift (Dosbim 2)", "",
        f"Source SHA256: `{sha}`; mode: {summary['mode']}; trial: {total}/{planned}; selesai: {completed}.",
        f"Berhasil: {success}/{total} ({summary['success_rate_percent']}%).", "",
        "| Rak | Trial | Berhasil | Persentase |", "|---|---:|---:|---:|",
    ]
    for k, v in summary["by_rack"].items():
        text.append(f"| {k} | {v['trials']} | {v['success']} | {v['rate_percent']}% |")
    text += ["", "| Shelf (1=bawah) | Trial | Berhasil | Persentase |", "|---|---:|---:|---:|"]
    for k, v in summary["by_shelf"].items():
        text.append(f"| {k} | {v['trials']} | {v['success']} | {v['rate_percent']}% |")
    text.extend(["", summary["interpretation_note"]])
    (run_dir / MD_NAME).write_text("\n".join(text) + "\n", encoding="utf-8")
    return summary


def main():
    ap = argparse.ArgumentParser(description="P08 Virtual Grasp + Lift — 2 rak / 3 shelf")
    ap.add_argument("--source", default=str(Path(__file__).resolve().parents[2] / "src" / "main_simulation" / "07_indonesian_store_jaco_camera_NEW.py"))
    ap.add_argument("--output_dir", default="data/test_logs/dosbim2_revision/P08_grasp_lift")
    ap.add_argument("--repeat", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="Uji coba N trial (0=semua)")
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--settle_steps", type=int, default=120)
    ap.add_argument("--min_lift_height", type=float, default=0.05)
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()
    if args.repeat < 1 or args.limit < 0 or args.settle_steps < 0 or args.min_lift_height <= 0:
        ap.error("--repeat >= 1; --limit/--settle_steps >=0; --min_lift_height >0")
    source = Path(args.source).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Simulasi utama tidak ditemukan: {source}")
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    root_output = Path(args.output_dir).expanduser().resolve()
    run_dir = root_output / time.strftime("P08_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    csv_path = run_dir / CSV_NAME

    sim = import_sim(source)
    # Matikan window OpenCV saat logger; tidak mengubah pose, solver, atau alur grasp.
    sim.SHOW_CAMERA_WINDOWS = False
    sim.WRIST_POV_LIVE_DURING_MOTION = False
    sim.OBJECT_RANDOM_LOG_DIR = str(run_dir / "randomization")
    client = p.connect(p.GUI if args.gui else p.DIRECT)
    if client < 0:
        raise RuntimeError("Tidak dapat menghubungkan PyBullet")
    rows = []
    try:
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)
        p.setPhysicsEngineParameter(fixedTimeStep=1.0 / 240.0,
                                    numSolverIterations=150, numSubSteps=2)
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
                    raise RuntimeError(f"Objek formal tidak ada: {name}")
                rack = int(sim.get_rack_id_from_y(initial_poses[name]["position"][1]))
                if rack not in (1, 2):
                    raise RuntimeError(f"Rak objek tidak valid: {name} -> {rack}")
                targets.append((name, rack))
        trials = [(rep, name, rack) for rep in range(1, args.repeat + 1)
                  for name, rack in targets]
        if args.limit:
            trials = trials[:args.limit]
        print(f"\n=== P08 DOSBIM2 | {len(trials)} trial | 9 kelas target ===")
        print("Source SHA256:", sha)
        print("Output:", run_dir)
        print("Min lift:", args.min_lift_height, "m")
        for index, (rep, name, rack) in enumerate(trials, 1):
            row = {field: "" for field in FIELDS}
            row.update({"trial": index, "repetition": rep, "seed": args.seed + index,
                        "object_name": name, "expected_rack": rack,
                        "min_lift_height_m": args.min_lift_height,
                        "constraint_created": False, "constraint_after_retreat": False,
                        "constraint_after_lift": False, "lift_height_ok": False,
                        "grasp_lift_success": False, "status": "Gagal"})
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
                    row["error"] += "\nCLEANUP ERROR:\n" + traceback.format_exc()
                    row["grasp_lift_success"] = False
                    row["status"] = "Gagal"
                row["elapsed_sec"] = round(time.monotonic() - started, 3)
                rows.append(row)
                with csv_path.open("w", encoding="utf-8", newline="") as fp:
                    w = csv.DictWriter(fp, fieldnames=FIELDS, extrasaction="ignore")
                    w.writeheader()
                    w.writerows(rows)
                summarize(rows, run_dir=run_dir, source=source, sha=sha,
                          args=args, planned=len(trials), completed=False)
                print(f"[P08 {index}/{len(trials)}] {row['status']} | {name} "
                      f"rack={row['actual_rack'] or '?'} shelf={row['shelf_index_1based'] or '?'} "
                      f"lift={row['lift_height_m'] if row['lift_height_m'] != '' else '-'} "
                      f"failure={row['failure_stage'] or '-'}")
        final = summarize(rows, run_dir=run_dir, source=source, sha=sha,
                          args=args, planned=len(trials), completed=True)
        print("\n=== P08 SELESAI ===")
        print(json.dumps({key: final[key] for key in (
            "trial_count", "virtual_grasp_lift_success", "failed", "success_rate_percent", "by_rack", "by_shelf")},
            ensure_ascii=False, indent=2))
    finally:
        if p.isConnected(client):
            p.disconnect(client)


if __name__ == "__main__":
    main()

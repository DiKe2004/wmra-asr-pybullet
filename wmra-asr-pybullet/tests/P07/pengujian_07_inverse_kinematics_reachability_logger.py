#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P07 DOSBIM 2 — IK reachability 2 rak, 3 shelf, 9 kelas x 4 repetisi.

Jalankan dari root proyek:
  python scripts/testv2/pengujian_07_inverse_kinematics_reachability_logger.py --limit 2 --gui
  python scripts/testv2/pengujian_07_inverse_kinematics_reachability_logger.py --repeat 4 --gui

Uji IK ke posisi objek yang diketahui (ground truth untuk penguji), BUKAN
pencarian kamera, grasp/lift, atau success rate end-to-end. Parameter dan
threshold dipakai dari main simulator; semua gagal/exception tetap dicatat.
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
STAGES = ("ready", "pre_grasp", "approach")
FIELDNAMES = [
    "trial", "repetition", "seed", "object_name", "expected_rack", "actual_rack",
    "station", "shelf_index_1based", "object_xyz_m", "joint1_prealign_ok",
    "ready_target_xyz_m", "ready_actual_xyz_m", "ready_error_m", "ready_limit_m", "ready_ok", "ready_motion_stopped",
    "pre_grasp_target_xyz_m", "pre_grasp_actual_xyz_m", "pre_grasp_error_m", "pre_grasp_limit_m", "pre_grasp_ok", "pre_grasp_motion_stopped",
    "approach_target_xyz_m", "approach_actual_xyz_m", "approach_error_m", "approach_limit_m", "approach_ok", "approach_motion_stopped",
    "ik_success", "elapsed_sec", "failure_stage", "status", "error",
]


def load_sim(path: Path):
    spec = importlib.util.spec_from_file_location("scriptis_p07_dosbim2_main", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Tidak dapat import simulasi: {path}")
    sim = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = sim
    spec.loader.exec_module(sim)
    return sim


def dump_csv(path: Path, rows: list[dict]):
    with path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=FIELDNAMES, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows: list[dict], *, planned: int, source: Path, sha: str,
              run_dir: Path, gui: bool, completed: bool, repeat: int) -> dict:
    n = len(rows)
    successes = sum(r["ik_success"] is True for r in rows)
    def group_metrics(field):
        groups = defaultdict(list)
        for r in rows:
            groups[str(r.get(field, ""))].append(r)
        return {key: {"trials": len(group),
                      "success": sum(item["ik_success"] is True for item in group),
                      "rate_percent": round(100 * sum(item["ik_success"] is True for item in group) / len(group), 3)}
                for key, group in sorted(groups.items())}
    stage_results = {}
    for stage in STAGES:
        values = [float(r[f"{stage}_error_m"]) for r in rows
                  if isinstance(r.get(f"{stage}_error_m"), (int, float))]
        stage_results[stage] = {
            "within_tolerance": sum(r.get(f"{stage}_ok") is True for r in rows),
            "mean_error_m": round(statistics.mean(values), 6) if values else None,
            "max_error_m": round(max(values), 6) if values else None,
            "number_measured": len(values),
        }
    summary = {
        "test": "P07", "revision": "dosbim2", "completed": completed,
        "source": str(source), "source_sha256": sha, "run_dir": str(run_dir),
        "mode": "GUI" if gui else "DIRECT", "planned_repetitions": repeat,
        "planned_trials_this_run": planned, "trial_count": n,
        "ik_success": successes, "ik_failed": n - successes,
        "ik_success_rate_percent": round(100 * successes / n, 3) if n else None,
        "by_rack": group_metrics("actual_rack"),
        "by_shelf": group_metrics("shelf_index_1based"),
        "stage_metrics": stage_results,
        "failures": [{"trial": r["trial"], "target": r["object_name"],
                       "rack": r["actual_rack"], "shelf": r["shelf_index_1based"],
                       "failure_stage": r["failure_stage"], "error": r["error"]}
                      for r in rows if r["ik_success"] is not True],
        "measurement_note": "Penguji mengetahui posisi objek dari PyBullet. Keberhasilan = ketiga tahap ready/pre-grasp/approach mencapai toleransi posisi main simulator. Tidak menguji deteksi CV, orientasi gripper, grasp, atau bebas tabrakan.",
    }
    (run_dir / "pengujian_07_inverse_kinematics_reachability_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = [
        "# P07 — Inverse Kinematics (Dosbim 2)", "",
        f"- Source SHA256: `{sha}`", f"- Mode: {summary['mode']}",
        f"- Trial: {n}/{planned}; selesai: {completed}",
        f"- Semua tahap dalam toleransi: {successes}/{n} ({summary['ik_success_rate_percent']}%)", "",
        "## Per rak", "", "| Rak | Trial | Berhasil | Persentase |", "|---|---:|---:|---:|",
    ]
    for k, v in summary["by_rack"].items():
        markdown.append(f"| {k} | {v['trials']} | {v['success']} | {v['rate_percent']}% |")
    markdown.extend(["", "## Per shelf", "", "| Shelf (1=bawah) | Trial | Berhasil | Persentase |", "|---|---:|---:|---:|"])
    for k, v in summary["by_shelf"].items():
        markdown.append(f"| {k} | {v['trials']} | {v['success']} | {v['rate_percent']}% |")
    markdown.extend(["", "Pengujian IK memakai ground-truth posisi objek dan toleransi main simulator. Tidak termasuk CV/grasp/handover/collision-free."])
    (run_dir / "pengujian_07_inverse_kinematics_reachability_summary.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8")
    return summary


def measure_stage(sim, robot_data, stage: str, point, limit: float):
    duration, force, velocity = {
        "ready": (1.8, 200, 0.25),
        "pre_grasp": (2.5, 190, 0.18),
        "approach": (2.2, 170, 0.12),
    }[stage]
    returned_error, _, stopped = sim.move_robot_to_position(
        robot_data=robot_data, target_position=point, duration=duration,
        attached_object_id=None, grasp_constraint_id=None,
        release_on_collision=False, motor_force=force,
        max_velocity=velocity, error_print_limit=limit,
    )
    actual_pos, _ = sim.get_end_effector_position(robot_data)
    measured_error = math.dist([float(x) for x in actual_pos], [float(x) for x in point])
    # Recompute against actual end-effector pose; do not trust only solver return.
    return {
        f"{stage}_target_xyz_m": json.dumps([round(float(x), 6) for x in point]),
        f"{stage}_actual_xyz_m": json.dumps([round(float(x), 6) for x in actual_pos]),
        f"{stage}_error_m": round(measured_error, 6),
        f"{stage}_limit_m": round(float(limit), 6),
        f"{stage}_motion_stopped": bool(stopped),
        f"{stage}_ok": bool(measured_error <= limit and not stopped),
    }


def main():
    ap = argparse.ArgumentParser(description="P07 IK dua rak, dosbim 2")
    ap.add_argument("--source", default=str(Path(__file__).resolve().parents[2] / "src" / "main_simulation" / "07_indonesian_store_jaco_camera_NEW.py"))
    ap.add_argument("--output_dir", default="data/test_logs/dosbim2_revision/P07_inverse_kinematics")
    ap.add_argument("--repeat", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="Uji coba N trial (0 = semua)")
    ap.add_argument("--seed", type=int, default=20260921)
    ap.add_argument("--settle_steps", type=int, default=120)
    ap.add_argument("--gui", action="store_true")
    args = ap.parse_args()
    if args.repeat < 1 or args.limit < 0 or args.settle_steps < 0:
        ap.error("--repeat harus >=1, --limit dan --settle_steps harus >=0")
    source = Path(args.source).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Source simulasi tidak ditemukan: {source}")
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    run_dir = Path(args.output_dir).expanduser().resolve() / time.strftime("P07_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    csv_path = run_dir / "pengujian_07_inverse_kinematics_reachability.csv"

    sim = load_sim(source)
    # P07 mengukur gerak IK saja; visualisasi OpenCV tidak diperlukan.
    # Tidak mengubah fisika, solver, toleransi, atau konfigurasi source main.
    sim.SHOW_CAMERA_WINDOWS = False
    sim.WRIST_POV_LIVE_DURING_MOTION = False
    sim.OBJECT_RANDOM_LOG_DIR = str(run_dir / "randomization")
    client = p.connect(p.GUI if args.gui else p.DIRECT)
    if client < 0:
        raise RuntimeError("Gagal menghubungkan PyBullet")
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
                    raise RuntimeError(f"Kelas target formal hilang: {name}")
                rack = sim.get_rack_id_from_y(initial_poses[name]["position"][1])
                if rack not in (1, 2):
                    raise RuntimeError(f"Rak target tidak valid: {name} -> {rack}")
                targets.append((name, rack))
        trials = [(rep, name, rack) for rep in range(1, args.repeat + 1)
                  for name, rack in targets]
        if args.limit:
            trials = trials[:args.limit]
        print(f"\n=== P07 IK DOSBIM2 | {len(trials)} TRIAL | 9 TARGET ===")
        print("Source SHA256:", sha)
        print("Output:", run_dir)
        for idx, (rep, name, rack) in enumerate(trials, start=1):
            row = {field: "" for field in FIELDNAMES}
            row.update({"trial": idx, "repetition": rep, "seed": args.seed + idx,
                        "object_name": name, "expected_rack": rack,
                        "joint1_prealign_ok": False, "ik_success": False, "status": "Gagal"})
            started = time.monotonic()
            try:
                random.seed(args.seed + idx)
                sim.reset_objects_randomized(objects, initial_poses,
                                             reason=f"P07_trial_{idx:03d}")
                if sim.move_wheelchair_to_station(robot_data, rack, duration=0.35) is not True:
                    raise RuntimeError(f"Wheelchair gagal ke station {rack}")
                sim.reset_robot(robot_data)
                row["station"] = int(sim.CURRENT_STATION_ID)
                obj_pos, _ = p.getBasePositionAndOrientation(objects[name]["id"])
                row["object_xyz_m"] = json.dumps([round(float(x), 6) for x in obj_pos])
                row["actual_rack"] = sim.get_rack_id_from_y(obj_pos[1])
                if row["station"] != rack or row["actual_rack"] != rack:
                    raise RuntimeError(f"Mismatch station/rak: {row['station']}, {row['actual_rack']} vs {rack}")
                points = sim.get_shelf_aware_motion_points(obj_pos)
                shelf_index = int(points["shelf_index"])
                row["shelf_index_1based"] = shelf_index + 1
                limits = {
                    "ready": float(sim.READY_ERROR_LIMIT),
                    "pre_grasp": float(sim.REACH_ERROR_LIMIT),
                    "approach": float(sim.BOTTOM_APPROACH_ERROR_LIMIT if shelf_index == 0
                                      else sim.APPROACH_ERROR_LIMIT),
                }
                # Sama dengan stage 0a pada prosedur picking utama.
                sim.rotate_joint1_to_face_object(robot_data, obj_pos,
                                                  duration=sim.JOINT1_ALIGN_DURATION)
                row["joint1_prealign_ok"] = True
                for stage in STAGES:
                    # Jangan melanjutkan jika stage sebelumnya gagal; penilaian
                    # tetap FAIL dan tahap yang dilewati ditandai sebagai skipped.
                    if stage != "ready" and not row[f"{STAGES[STAGES.index(stage)-1]}_ok"]:
                        row["failure_stage"] = STAGES[STAGES.index(stage)-1]
                        break
                    row.update(measure_stage(sim, robot_data, stage, points[stage], limits[stage]))
                    if not row[f"{stage}_ok"]:
                        row["failure_stage"] = stage
                        break
                row["ik_success"] = all(row.get(f"{s}_ok") is True for s in STAGES)
                if row["ik_success"]:
                    row["status"] = "Berhasil"
                elif not row["failure_stage"]:
                    row["failure_stage"] = "stage_not_completed"
            except Exception:
                row["failure_stage"] = row["failure_stage"] or "runtime_error"
                row["error"] = traceback.format_exc()
                print(row["error"])
            finally:
                row["elapsed_sec"] = round(time.monotonic() - started, 3)
                rows.append(row)
                dump_csv(csv_path, rows)
                summarize(rows, planned=len(trials), source=source, sha=sha,
                          run_dir=run_dir, gui=args.gui, completed=False, repeat=args.repeat)
                print(f"[P07 {idx}/{len(trials)}] {row['status']} | {name} | "
                      f"rak={row['actual_rack'] or '?'} shelf={row['shelf_index_1based'] or '?'} "
                      f"ready={row['ready_error_m'] or '-'} "
                      f"pre={row['pre_grasp_error_m'] or '-'} "
                      f"approach={row['approach_error_m'] or '-'} "
                      f"fail={row['failure_stage'] or '-'}")
        final = summarize(rows, planned=len(trials), source=source, sha=sha,
                          run_dir=run_dir, gui=args.gui, completed=True, repeat=args.repeat)
        print("\n=== P07 SELESAI ===")
        print(json.dumps({k: final[k] for k in ("trial_count", "ik_success",
                       "ik_failed", "ik_success_rate_percent", "by_rack", "by_shelf")},
                         ensure_ascii=False, indent=2))
    finally:
        if p.isConnected(client):
            p.disconnect(client)


if __name__ == "__main__":
    main()

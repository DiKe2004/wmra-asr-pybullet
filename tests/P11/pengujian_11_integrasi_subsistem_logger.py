#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P11 DOSBIM2 — integrasi command teks -> multi-rack visual search -> pick -> basket.

Script di: scripts/testv2/pengujian_11_integrasi_subsistem_logger.py
Dari root Scriptis4_PyBullet:
    python scripts/testv2/pengujian_11_integrasi_subsistem_logger.py --limit 1 --gui
    python scripts/testv2/pengujian_11_integrasi_subsistem_logger.py --gui

Tiga SKENARIO yang sengaja dipilih untuk menguji jalur integrasi:
    1. Rak 1, shelf bawah : balok merah (tanpa fallback antarrak).
    2. Rak 2, shelf bawah : silinder kuning (harus fallback dari rak 1).
    3. Rak 2, shelf atas  : segitiga kuning (harus fallback dari rak 1).

Setiap trial membuat scene baru dengan penempatan awal yang TERDOKUMENTASI,
bukan randomisasi. Randomisasi dan 27 target formal dievaluasi terpisah pada P12.
P11 memakai fungsi pipeline UTAMA (tidak mengerjakan pick lewat ground truth).
Ground truth objek hanya dibaca SETELAH dijalankan untuk evaluasi.
Pipeline menggunakan virtual grasp fixed-constraint dan gate body ID PyBullet;
tidak boleh diklaim sebagai RGB-only ataupun force-closure grasp fisik.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import time
import traceback
from pathlib import Path

import pybullet as p
import pybullet_data

SCENARIOS = (
    ("rak1_shelf_bawah", "ambil balok merah", "balok_merah", 1, 1),
    ("rak2_shelf_bawah_fallback", "ambil silinder kuning", "silinder_kuning", 2, 1),
    ("rak2_shelf_atas_fallback", "ambil segitiga kuning", "segitiga_kuning", 2, 3),
)
FIELDS = (
    "trial", "scenario", "command", "expected_name", "expected_rack", "expected_shelf",
    "source_sha256", "parsed_action", "parsed_shape", "parsed_color", "parser_valid",
    "initial_station", "scan_stations", "scan_attempts", "detected_names_by_station",
    "rack2_reached", "wheelchair_moved", "detected_name", "detected_rack",
    "object_match", "candidate_body_id", "candidate_visual_shape_label", "candidate_score",
    "grasp_attempted", "grasp_constraint_created", "guard_align_calls", "guard_align_last_ready",
    "guard_final_ready", "release_called", "release_performed", "natural_drop_called",
    "object_final_xyz_m", "basket_inside_xy_aabb", "basket_floor_error_m",
    "basket_floor_ok", "basket_success", "task_success", "failure_stage",
    "elapsed_sec", "exception",
)


def import_sim(source: Path, trial: int):
    module_name = f"scriptis_p11_main_trial_{trial}_{time.monotonic_ns()}"
    spec = importlib.util.spec_from_file_location(module_name, str(source))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Tidak bisa load source: {source}")
    sim = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = sim
    spec.loader.exec_module(sim)
    return sim


def object_pos(body_id):
    return [float(v) for v in p.getBasePositionAndOrientation(body_id)[0]]


def basket_audit(sim, target_id, margin_m=0.002, floor_tolerance_m=0.035):
    """Baca kondisi fisik aktual pasca-drop; tidak menggeser objek."""
    if sim.BASKET_INFO is None:
        return False, False, math.nan, False
    basket = sim.BASKET_INFO
    center = [float(x) for x in sim.BASKET_CENTER]
    half_x = float(basket["inner_size"][0]) / 2.0
    half_y = float(basket["inner_size"][1]) / 2.0
    lo, hi = p.getAABB(target_id)
    inside = (
        lo[0] >= center[0] - half_x + margin_m
        and hi[0] <= center[0] + half_x - margin_m
        and lo[1] >= center[1] - half_y + margin_m
        and hi[1] <= center[1] + half_y - margin_m
    )
    floor_error = float(lo[2]) - float(basket["floor_top_z"])
    floor_ok = abs(floor_error) <= floor_tolerance_m
    return bool(inside and floor_ok), bool(inside), round(floor_error, 6), bool(floor_ok)


def record_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_summary(run_dir, source, sha, rows, planned_trials, complete, args):
    successes = sum(r.get("task_success") is True for r in rows)
    n = len(rows)
    report = {
        "test": "P11", "revision": "dosbim2", "completed": complete,
        "source": str(source), "source_sha256": sha,
        "run_dir": str(run_dir), "mode": "GUI" if args.gui else "DIRECT",
        "show_camera_window": bool(args.show_camera),
        "planned_trials_this_run": planned_trials, "trial_count": n,
        "task_success": successes, "failed": n - successes,
        "task_success_rate_percent": round(100.0 * successes / n, 3) if n else None,
        "parser_valid": sum(r.get("parser_valid") is True for r in rows),
        "correct_target_selected": sum(r.get("object_match") is True for r in rows),
        "basket_success": sum(r.get("basket_success") is True for r in rows),
        "rack2_reached": sum(r.get("rack2_reached") is True for r in rows),
        "release_performed": sum(r.get("release_performed") is True for r in rows),
        "average_elapsed_sec": round(statistics.mean(
            float(r["elapsed_sec"]) for r in rows if r.get("elapsed_sec") not in (None, "")
        ), 3) if any(r.get("elapsed_sec") not in (None, "") for r in rows) else None,
        "failures": [{"trial": r["trial"], "scenario": r["scenario"],
                      "stage": r["failure_stage"], "error": r["exception"]}
                     for r in rows if r.get("task_success") is not True],
        "measurement_notes": [
            "P11 mencakup 3 skenario tetap: rak1 bawah, rak2 bawah fallback, rak2 atas fallback; tanpa randomisasi objek.",
            "Masukan berupa teks, tidak menguji ASR; sistem utama mengeksekusi scan rak1 terlebih dahulu dan fallback ke rak2.",
            "Hasil deteksi bukan pembuktian klasifikasi RGB murni: pipeline memakai body ID/metadata PyBullet sebagai gate.",
            "Return nama dari run_multi_station_visual_search_and_pick TIDAK dipakai sendirian sebagai kriteria sukses task.",
            "Task sukses wajib target benar + station benar + virtual grasp + guard ready + natural drop + AABB objek di dalam basket.",
            "Grasp memakai virtual fixed constraint/alignment dan collision objek saat transport dinonaktifkan oleh main.",
            "Nilai P11 bukan pengganti hasil P10 ataupun P12 formal 27 trial.",
        ],
    }
    (run_dir / "pengujian_11_integrasi_subsistem_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = ["# P11 — Integrasi Subsistem (dosbim2)", "",
          f"Source SHA256: `{sha}`", f"Selesai: {complete}; trial: {n}/{planned_trials}.",
          f"Task selesai dan objek di keranjang: {successes}/{n} ({report['task_success_rate_percent']}%).", "",
          "| Trial | Skenario | Target ditemukan | Basket | Task | Kegagalan |",
          "|---:|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['trial']} | {r['scenario']} | {r['detected_name']} | "
                  f"{r['basket_success']} | {r['task_success']} | {r['failure_stage']} |")
    md.extend(["", "## Batasan pengukuran", *[f"- {s}" for s in report["measurement_notes"]]])
    (run_dir / "pengujian_11_integrasi_subsistem_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return report


def run_trial(sim, robot_data, objects, row, args):
    """Jalankan pipeline asli; patch hanya menyadap hasil stage tanpa mengubah return."""
    target_name = row["expected_name"]
    target_id = objects[target_name]["id"]
    row["initial_station"] = int(sim.CURRENT_STATION_ID)
    pos0 = object_pos(target_id)
    rack_actual = int(sim.get_rack_id_from_y(pos0[1]))
    shelf_actual = 1 + min(range(len(sim.SHELF_Z_LEVELS)), key=lambda k: abs(
        pos0[2] - (sim.SHELF_Z_LEVELS[k] + 0.065)))
    if rack_actual != row["expected_rack"] or shelf_actual != row["expected_shelf"]:
        raise RuntimeError(f"Konfigurasi scene tidak cocok: {target_name}: rack={rack_actual}, shelf={shelf_actual}")

    parsed = sim.parse_terminal_target_command(row["command"])
    row["parsed_action"] = parsed.get("action", "") if isinstance(parsed, dict) else ""
    row["parsed_shape"] = parsed.get("shape", "") if isinstance(parsed, dict) else ""
    row["parsed_color"] = parsed.get("color", "") if isinstance(parsed, dict) else ""
    row["parser_valid"] = bool(parsed and parsed.get("action") == "visual_search_pick")
    if not row["parser_valid"]:
        row["failure_stage"] = "parser"
        return

    events = {"scan": [], "grasp": [], "guard": [], "final_guard": [],
              "release": [], "natural_drop": []}
    original_search = sim.run_visual_search_for_target
    original_attach = sim.attach_object_to_end_effector
    original_align = sim.align_attached_object_over_basket_release_guard
    original_guard_check = sim.check_object_ready_to_release_into_basket
    original_release = sim.release_object
    original_drop = sim.natural_gravity_drop_to_basket

    def watch_search(*a, **kw):
        station = int(sim.CURRENT_STATION_ID)
        name, candidate = original_search(*a, **kw)
        events["scan"].append((station, name, candidate))
        return name, candidate

    def watch_attach(*a, **kw):
        cid = original_attach(*a, **kw)
        events["grasp"].append(cid)
        return cid

    def watch_align(*a, **kw):
        result = original_align(*a, **kw)
        events["guard"].append(bool(result[2]))
        return result

    def watch_guard_check(*a, **kw):
        result = original_guard_check(*a, **kw)
        events["final_guard"].append(bool(result[0]))
        return result

    def watch_release(*a, **kw):
        cid = a[0] if a else kw.get("constraint_id")
        result = original_release(*a, **kw)
        events["release"].append(cid)
        return result

    def watch_drop(*a, **kw):
        object_id = a[0] if a else kw.get("object_id")
        result = original_drop(*a, **kw)
        events["natural_drop"].append(object_id)
        return result

    sim.run_visual_search_for_target = watch_search
    sim.attach_object_to_end_effector = watch_attach
    sim.align_attached_object_over_basket_release_guard = watch_align
    sim.check_object_ready_to_release_into_basket = watch_guard_check
    sim.release_object = watch_release
    sim.natural_gravity_drop_to_basket = watch_drop
    try:
        # Jalur asli: rak 1 -> (jika belum ketemu) pindah rak 2 -> pick -> basket.
        result_name = sim.run_multi_station_visual_search_and_pick(
            robot_data=robot_data, objects=objects,
            target_shape=parsed["shape"], target_color=parsed["color"])
    finally:
        sim.run_visual_search_for_target = original_search
        sim.attach_object_to_end_effector = original_attach
        sim.align_attached_object_over_basket_release_guard = original_align
        sim.check_object_ready_to_release_into_basket = original_guard_check
        sim.release_object = original_release
        sim.natural_gravity_drop_to_basket = original_drop
        # Tulis hasil sebagian bahkan ketika pipeline terhenti exception.
        row["scan_stations"] = json.dumps([v[0] for v in events["scan"]])
        row["scan_attempts"] = len(events["scan"])
        row["detected_names_by_station"] = json.dumps(
            [{"station": s, "name": n or ""} for s, n, _ in events["scan"]], ensure_ascii=False)
        row["rack2_reached"] = any(v[0] == 2 for v in events["scan"])
        row["wheelchair_moved"] = row["rack2_reached"]
        row["grasp_attempted"] = len(events["grasp"]) > 0
        row["grasp_constraint_created"] = any(cid is not None for cid in events["grasp"])
        row["guard_align_calls"] = len(events["guard"])
        row["guard_align_last_ready"] = events["guard"][-1] if events["guard"] else False
        row["guard_final_ready"] = events["final_guard"][-1] if events["final_guard"] else False
        row["release_called"] = bool(events["release"])
        row["natural_drop_called"] = target_id in events["natural_drop"]
        row["release_performed"] = bool(row["natural_drop_called"] and events["release"])
        row["object_final_xyz_m"] = json.dumps([round(x, 6) for x in object_pos(target_id)])
        basket_ok, inside, floor_error, floor_ok = basket_audit(
            sim, target_id, margin_m=args.xy_margin,
            floor_tolerance_m=args.floor_tolerance)
        row["basket_inside_xy_aabb"] = inside
        row["basket_floor_error_m"] = floor_error
        row["basket_floor_ok"] = floor_ok
        row["basket_success"] = bool(basket_ok and row["release_performed"])

        if events["scan"]:
            accepted = [(s, n, c) for s, n, c in events["scan"] if n is not None]
            if accepted:
                station, name, candidate = accepted[-1]
                row["detected_name"] = name
                row["detected_rack"] = station
                if candidate is not None:
                    row["candidate_body_id"] = candidate.get("body_id", "")
                    row["candidate_visual_shape_label"] = candidate.get("shape_label", "")
                    row["candidate_score"] = candidate.get("score", "")

        row["object_match"] = row["detected_name"] == target_name
        station_path_ok = ([1] if row["expected_rack"] == 1 else [1, 2]) == [
            s for s, _, _ in events["scan"]]
        row["task_success"] = bool(
            row["parser_valid"]
            and station_path_ok
            and row["object_match"]
            and row["detected_rack"] == row["expected_rack"]
            and row["grasp_constraint_created"]
            and row["guard_align_last_ready"]
            and row["guard_final_ready"]
            and row["release_performed"]
            and row["basket_success"]
        )
        # A returned object name means VISUAL SEARCH succeeded; does NOT
        # automatically mean the subsequent pick/handover succeeded.
        if not row["task_success"] and not row["failure_stage"]:
            if not station_path_ok:
                row["failure_stage"] = "multi_station_search_order"
            elif not row["object_match"]:
                row["failure_stage"] = "visual_search_or_wrong_object"
            elif result_name != target_name:
                row["failure_stage"] = "multi_station_result"
            elif not row["grasp_constraint_created"]:
                row["failure_stage"] = "virtual_grasp_or_ik"
            elif not row["guard_align_last_ready"] or not row["guard_final_ready"]:
                row["failure_stage"] = "release_guard"
            elif not row["release_performed"]:
                row["failure_stage"] = "handover_or_release"
            elif not row["basket_success"]:
                row["failure_stage"] = "basket_placement"
            else:
                row["failure_stage"] = "integration_other"


def main():
    parser = argparse.ArgumentParser(description="P11 integrasi subsistem dua rak — dosbim2")
    parser.add_argument("--source", default="scripts/07_indonesian_store_jaco_camera_NEW.py")
    parser.add_argument("--output_dir", default="data/test_logs/dosbim2_revision/P11_integrasi_subsistem")
    parser.add_argument("--limit", type=int, default=0, help="Jalankan N skenario pertama; 0 = semua 3")
    parser.add_argument("--repeat", type=int, default=1, help="Ulangi 3 skenario; default 1")
    parser.add_argument("--settle_steps", type=int, default=120)
    parser.add_argument("--floor_tolerance", type=float, default=0.035)
    parser.add_argument("--xy_margin", type=float, default=0.002)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--show-camera", action="store_true", help="Tampilkan camera window realtime, disarankan bersama --gui")
    args = parser.parse_args()
    if args.limit < 0 or args.repeat < 1 or args.settle_steps < 0:
        parser.error("--limit >= 0, --repeat >= 1, --settle_steps >= 0")
    source = Path(args.source).expanduser().resolve()
    if not source.is_file():
        parser.error(f"Simulasi utama tidak ditemukan: {source}")
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    run_root = Path(args.output_dir).expanduser().resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    run_dir = run_root / time.strftime("P11_%Y%m%d_%H%M%S")
    run_dir.mkdir(exist_ok=False)
    trial_plan = [(rep, scenario) for rep in range(1, args.repeat + 1) for scenario in SCENARIOS]
    if args.limit:
        trial_plan = trial_plan[:args.limit]
    rows = []
    csv_path = run_dir / "pengujian_11_integrasi_subsistem.csv"
    print(f"=== P11 DOSBIM2 | {len(trial_plan)} TRIAL | sha256={sha} ===")
    print(f"Output: {run_dir}")
    try:
        for trial_num, (rep, scenario) in enumerate(trial_plan, 1):
            name, command, target_name, rack, shelf = scenario
            row = {key: "" for key in FIELDS}
            row.update({"trial": trial_num, "scenario": name, "command": command,
                        "expected_name": target_name, "expected_rack": rack,
                        "expected_shelf": shelf, "source_sha256": sha,
                        "task_success": False, "failure_stage": ""})
            t0 = time.monotonic()
            client = None
            try:
                sim = import_sim(source, trial_num)
                sim.SHOW_CAMERA_WINDOWS = bool(args.show_camera)
                sim.WRIST_POV_LIVE_DURING_MOTION = bool(args.show_camera)
                sim.VISUAL_SEARCH_LOG_DIR = str(run_dir / "debug_images" / f"trial_{trial_num:03d}")
                sim.OBJECT_RANDOM_LOG_DIR = str(run_dir / "randomization")
                sim.COLLISION_LOG_DIR = str(run_dir / "collision_logs" / f"trial_{trial_num:03d}")
                client = p.connect(p.GUI if args.gui else p.DIRECT)
                if client < 0:
                    raise RuntimeError("PyBullet tidak bisa connect")
                p.resetSimulation()
                p.setAdditionalSearchPath(pybullet_data.getDataPath())
                p.setGravity(0, 0, -9.81)
                p.setRealTimeSimulation(0)
                p.setPhysicsEngineParameter(fixedTimeStep=1/240, numSolverIterations=150, numSubSteps=2)
                objects, initial_poses = sim.create_indonesian_store_scene()
                robot_data = sim.load_jaco_robot()
                sim.register_mobile_platform_body(robot_data["robot_id"])
                sim.set_live_camera_context(robot_data=robot_data, objects=objects)
                sim.reset_robot(robot_data)
                for _ in range(args.settle_steps):
                    p.stepSimulation()
                if target_name not in objects:
                    raise RuntimeError(f"Target tidak ada: {target_name}")
                print(f"\n--- P11 trial {trial_num}/{len(trial_plan)}: {name}; command='{command}' ---")
                run_trial(sim, robot_data, objects, row, args)
            except Exception as exc:
                row["failure_stage"] = row["failure_stage"] or "runtime_exception"
                row["exception"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=5)}"
                print(row["exception"])
            finally:
                row["elapsed_sec"] = round(time.monotonic() - t0, 3)
                rows.append(row)
                record_csv(csv_path, rows)
                write_summary(run_dir, source, sha, rows, len(trial_plan), False, args)
                if client is not None and client >= 0 and p.isConnected(client):
                    p.disconnect(client)
                print(f"P11 trial {trial_num}: {'SUCCESS' if row['task_success'] else 'FAILED'} "
                      f"stage={row['failure_stage'] or '-'} elapsed={row['elapsed_sec']}s")
        report = write_summary(run_dir, source, sha, rows, len(trial_plan), True, args)
        print(f"P11 selesai: {report['task_success']}/{report['trial_count']} sukses; summary: "
              f"{run_dir / 'pengujian_11_integrasi_subsistem_summary.json'}")
    finally:
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()

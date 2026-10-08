#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P05 dosbim 2: visual detection, dua rak (TANPA pick/handover).

Tempat script: scripts/testv2/pengujian_05_visual_search_detection_logger.py
Output: data/test_logs/dosbim2_revision/P05_visual_search/P05_<timestamp>/

Dari akar project:
  python scripts/testv2/pengujian_05_visual_search_detection_logger.py --limit 2
  python scripts/testv2/pengujian_05_visual_search_detection_logger.py --repeat 4

9 kelas formal (3 bentuk x 3 warna) x 4 repetisi = 36 trial.
Hijau tetap ada di scene sebagai distractor, bukan kelas target formal.

Pada P05, posisi rak target dipakai HANYA oleh penguji untuk menempatkan
wheelchair di station pengujian; tidak diberikan sebagai input deteksi CV.
Perpindahan otomatis rack1 -> rack2 diuji terpisah pada P11/P12.
Keberhasilan dicatat dari hasil fungsi visual search aktual, bukan ditebak
berdasarkan keberadaan objek di scene. Tidak ada pemanggilan pick/grasp.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import random
import statistics
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import pybullet as p
import pybullet_data


FORMAL_SHAPES = ("box", "cylinder", "triangle")
FORMAL_COLORS = ("red", "blue", "yellow")
ID_SHAPE = {"box": "balok", "cylinder": "silinder", "triangle": "segitiga"}
ID_COLOR = {"red": "merah", "blue": "biru", "yellow": "kuning"}
FIELDS = (
    "trial", "repetition", "seed", "target_shape", "target_color", "target_name",
    "expected_rack", "expected_shelf", "station_scanned", "total_planned_scans",
    "scans_performed", "candidate_counts_by_scan", "detected_name", "detected_rack",
    "detected_shape", "detected_color", "visual_shape_label", "visual_score",
    "visual_shape_score", "contour_area_px", "depth_world_xyz_m", "found",
    "shape_correct", "color_correct", "detection_correct", "status",
    "failure_reason", "search_time_sec", "error",
)


def load_sim(path):
    spec = importlib.util.spec_from_file_location("scriptis_p05_dosbim2_main", str(path))
    if spec is None or spec.loader is None:
        raise RuntimeError("Gagal import main simulation: " + str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def name_to_shape_color(name):
    s = str(name or "").lower()
    for prefix, shape in (("balok_", "box"), ("silinder_", "cylinder"), ("segitiga_", "triangle")):
        if s.startswith(prefix):
            color = s[len(prefix):]
            return shape, {"merah": "red", "biru": "blue", "kuning": "yellow", "hijau": "green"}.get(color, color)
    return "", ""


def csv_write(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summary_write(run_dir, rows, sim_path, source_sha, args, finished=False):
    n = len(rows)
    found = sum(bool(r["found"]) for r in rows)
    correct = sum(bool(r["detection_correct"]) for r in rows)
    shape_ok = sum(bool(r["shape_correct"]) for r in rows)
    color_ok = sum(bool(r["color_correct"]) for r in rows)
    times = [float(r["search_time_sec"]) for r in rows if r["search_time_sec"] is not None]
    rack_metrics = {}
    for rack in (1, 2):
        subset = [r for r in rows if r["expected_rack"] == rack]
        rack_metrics[f"rack{rack}"] = {
            "trials": len(subset),
            "found": sum(bool(r["found"]) for r in subset),
            "correct": sum(bool(r["detection_correct"]) for r in subset),
        }
    result = {
        "test": "P05", "revision": "dosbim2", "completed": finished,
        "source": str(sim_path), "source_sha256": source_sha,
        "run_dir": str(run_dir.resolve()), "planned_repetitions": args.repeat,
        "formal_classes": 9, "green_objects": "distractor_only",
        "mode": "GUI" if args.gui else "DIRECT", "trial_count": n,
        "found": found, "correct": correct, "shape_correct": shape_ok,
        "color_correct": color_ok, "not_found": n - found,
        "detection_success_rate_percent": round(100 * correct / n, 3) if n else None,
        "visual_found_rate_percent": round(100 * found / n, 3) if n else None,
        "average_search_time_sec": round(statistics.mean(times), 3) if times else None,
        "median_search_time_sec": round(statistics.median(times), 3) if times else None,
        "by_rack": rack_metrics,
        "failures": [{"trial": r["trial"], "target": r["target_name"],
                      "reason": r["failure_reason"], "detected_name": r["detected_name"]}
                     for r in rows if not r["detection_correct"]],
        "note": "P05 menguji visual search pada station rak target. CV main memiliki active-rack/body strict gate; metrik ini adalah hasil pipeline visual search lengkap, bukan akurasi klasifikasi RGB mandiri.",
    }
    (run_dir / "pengujian_05_visual_search_detection_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    md = [
        "# P05 — Visual Search Detection (Dosbim 2)", "",
        f"Source SHA-256: `{source_sha}`", "",
        f"Trial: {n}; found: {found}; correct: {correct}; gagal: {n-correct}.",
        f"Detection success rate: {result['detection_success_rate_percent']}%.",
        f"Average search time: {result['average_search_time_sec']} s.", "",
        "Rak target ditentukan penguji untuk pengaturan station, bukan input detektor.",
        "Kamera wrist dipakai untuk pemilihan target aktual; pengambilan objek tidak dilakukan.",
        "Hijau tersedia sebagai distractor, bukan kelas target formal.",
        "Pengujian fallback otomatis antarrak dilakukan pada P11/P12.",
        "Kegagalan CV tetap dicatat, tidak diganti menjadi sukses.",
    ]
    (run_dir / "pengujian_05_visual_search_detection_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return result


def main():
    ap = argparse.ArgumentParser(description="P05 visual search dua rak, dosbim 2")
    ap.add_argument("--source", default="scripts/07_indonesian_store_jaco_camera_NEW.py")
    ap.add_argument("--output_dir", default="data/test_logs/dosbim2_revision/P05_visual_search")
    ap.add_argument("--repeat", type=int, default=4, help="4 repetisi x 9 kelas = 36 trial")
    ap.add_argument("--limit", type=int, default=0, help="smoke test N trial pertama; 0 = semua")
    ap.add_argument("--seed", type=int, default=20260920)
    ap.add_argument("--settle_steps", type=int, default=120)
    ap.add_argument("--gui", action="store_true")
    ap.add_argument("--show-camera", action="store_true", help="Tampilkan window OpenCV; opsional, bisa berat")
    args = ap.parse_args()
    if args.repeat < 1 or args.limit < 0:
        ap.error("--repeat harus >=1 dan --limit harus >=0")
    source = Path(args.source).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError("Source simulasi tidak ditemukan: " + str(source))
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    run_dir = Path(args.output_dir).expanduser().resolve() / time.strftime("P05_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "pengujian_05_visual_search_detection.csv"
    sim = load_sim(source)

    sim.SHOW_CAMERA_WINDOWS = bool(args.show_camera)
    sim.VISUAL_SEARCH_LOG_DIR = str(run_dir / "debug_images")
    sim.OBJECT_RANDOM_LOG_DIR = str(run_dir / "randomization")
    # Pertahankan aturan pipeline visual asli; tidak mengubah threshold, body gate,
    # parameter motor, ataupun hasil kandidat hanya demi meluluskan P05.

    client = p.connect(p.GUI if args.gui else p.DIRECT)
    if client < 0:
        raise RuntimeError("Gagal menghubungkan PyBullet")

    rows = []
    original_capture = sim.capture_wrist_pov_for_visual_search
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

        # Setiap kelas formal harus tersedia tepat satu objek pada scene.
        targets = []
        for shape in FORMAL_SHAPES:
            for color in FORMAL_COLORS:
                name = ID_SHAPE[shape] + "_" + ID_COLOR[color]
                if name not in objects:
                    raise RuntimeError("Kelas formal tidak ditemukan di scene: " + name)
                target_pos = initial_poses[name]["position"]
                target_rack = sim.get_rack_id_from_y(target_pos[1])
                if target_rack not in (1, 2):
                    raise RuntimeError("Rack invalid untuk " + name)
                targets.append((shape, color, name, target_rack))

        trials = [(rep, *target) for rep in range(1, args.repeat + 1) for target in targets]
        if args.limit:
            trials = trials[:args.limit]
        print(f"\n=== P05 DOSBIM2 | {len(trials)} TRIAL | 9 FORMAL CLASSES ===")
        print("Source hash:", source_sha)
        print("Output:", run_dir)

        for trial_i, (rep, target_shape, target_color, name, expected_rack) in enumerate(trials, 1):
            seed = args.seed + trial_i
            random.seed(seed)
            row = {field: "" for field in FIELDS}
            row.update({
                "trial": trial_i, "repetition": rep, "seed": seed,
                "target_shape": target_shape, "target_color": target_color,
                "target_name": name, "expected_rack": expected_rack,
                "found": False, "shape_correct": False, "color_correct": False,
                "detection_correct": False, "scans_performed": 0,
                "search_time_sec": None, "status": "Gagal",
            })
            scan_candidate_counts = []
            start = None
            try:
                # Randomisasi dijalankan antartrial, selalu 2 objek/shelf dan
                # tiap identitas objek tetap dalam grup rak yang sama.
                sim.reset_objects_randomized(objects, initial_poses,
                                             reason=f"P05_trial_{trial_i:03d}")
                # Simpan gambar debug setiap trial di folder berbeda agar nama file
                # dengan timestamp detik yang sama tidak saling menimpa.
                sim.VISUAL_SEARCH_LOG_DIR = str(run_dir / "debug_images" / f"trial_{trial_i:03d}")
                sim.move_wheelchair_to_station(robot_data, expected_rack, duration=0.35)
                sim.reset_robot(robot_data)
                sim.set_wrist_pov_forced_target(None)
                sim.ACTIVE_RACK_ID = expected_rack
                sim.ACTIVE_RACK_Y_OFFSET = sim.WHEELCHAIR_STATIONS[expected_rack]["rack_y_offset"]
                obj_pos, _ = p.getBasePositionAndOrientation(objects[name]["id"])
                row["expected_shelf"] = 1 + min(range(len(sim.SHELF_Z_LEVELS)),
                    key=lambda k: abs(obj_pos[2] - (sim.SHELF_Z_LEVELS[k] + 0.065)))
                row["station_scanned"] = sim.CURRENT_STATION_ID
                row["total_planned_scans"] = len(sim.build_visual_search_scan_plan())
                if row["station_scanned"] != expected_rack or row["total_planned_scans"] != 6:
                    raise RuntimeError("Station/scan plan tidak sesuai: " + str(row))

                # Capture wrapper hanya mengukur berapa kali frame aktual dibaca;
                # tidak mengganti isi/hasil frame asli.
                def counted_capture(*a, **kw):
                    frame = original_capture(*a, **kw)
                    scan_candidate_counts.append(None)
                    return frame

                original_detect = sim.detect_target_from_visual_frame
                def counted_detect(*a, **kw):
                    result = original_detect(*a, **kw)
                    if scan_candidate_counts:
                        scan_candidate_counts[-1] = len(result[0])
                    return result

                sim.capture_wrist_pov_for_visual_search = counted_capture
                sim.detect_target_from_visual_frame = counted_detect
                start = time.monotonic()
                detected_name, candidate = sim.run_visual_search_for_target(
                    robot_data=robot_data, objects=objects,
                    target_shape=target_shape, target_color=target_color)
                row["search_time_sec"] = round(time.monotonic() - start, 3)
                row["scans_performed"] = len(scan_candidate_counts)
                row["candidate_counts_by_scan"] = json.dumps(scan_candidate_counts)
                row["detected_name"] = detected_name or ""
                row["found"] = bool(detected_name)

                if candidate is not None:
                    row["visual_shape_label"] = candidate.get("shape_label", "")
                    row["visual_score"] = round(float(candidate.get("score", 0.0)), 5)
                    row["visual_shape_score"] = round(float(candidate.get("target_shape_score", 0.0)), 5)
                    row["contour_area_px"] = round(float(candidate.get("area", 0.0)), 2)
                    row["depth_world_xyz_m"] = json.dumps(candidate.get("world_pos_from_depth"))

                if detected_name:
                    det_shape, det_color = name_to_shape_color(detected_name)
                    row["detected_shape"] = det_shape
                    row["detected_color"] = det_color
                    body_pos, _ = p.getBasePositionAndOrientation(objects[detected_name]["id"])
                    row["detected_rack"] = sim.get_rack_id_from_y(body_pos[1])
                    row["shape_correct"] = det_shape == target_shape
                    row["color_correct"] = det_color == target_color
                    row["detection_correct"] = bool(
                        detected_name == name and row["detected_rack"] == expected_rack)
                row["status"] = "Berhasil" if row["detection_correct"] else "Gagal"
                row["failure_reason"] = (
                    "" if row["detection_correct"] else
                    "target_not_found" if not detected_name else
                    "wrong_object_or_rack"
                )
            except Exception:
                row["error"] = traceback.format_exc()
                row["failure_reason"] = "runtime_error"
                if start is not None and row["search_time_sec"] is None:
                    row["search_time_sec"] = round(time.monotonic() - start, 3)
                print(row["error"])
            finally:
                sim.capture_wrist_pov_for_visual_search = original_capture
                if "original_detect" in locals():
                    sim.detect_target_from_visual_frame = original_detect
                sim.set_wrist_pov_forced_target(None)
                rows.append(row)
                csv_write(csv_path, rows)  # simpan progres setiap trial
                summary_write(run_dir, rows, source, source_sha, args, finished=False)
                print(f"[P05 {trial_i}/{len(trials)}] {row['status']} "
                      f"target={name} rack={expected_rack} "
                      f"detected={row['detected_name'] or '-'} "
                      f"scans={row['scans_performed']} "
                      f"time={row['search_time_sec']}s")
        summary = summary_write(run_dir, rows, source, source_sha, args, finished=True)
        print("\n=== P05 SELESAI ===")
        print(json.dumps({k: summary[k] for k in ("trial_count", "found", "correct",
                           "not_found", "detection_success_rate_percent", "by_rack")},
                         ensure_ascii=False, indent=2))
    finally:
        sim.capture_wrist_pov_for_visual_search = original_capture
        sim.set_wrist_pov_forced_target(None)
        if p.isConnected(client):
            p.disconnect(client)


if __name__ == "__main__":
    main()

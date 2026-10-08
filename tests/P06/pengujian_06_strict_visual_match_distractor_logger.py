#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P06 dosbim2 — strict visual match with *observed* distractors.

Place at scripts/testv2/pengujian_06_strict_visual_match_distractor_logger.py
Run from the project root:
  python scripts/testv2/pengujian_06_strict_visual_match_distractor_logger.py --limit 3 --gui
  python scripts/testv2/pengujian_06_strict_visual_match_distractor_logger.py --repeat 4 --gui

Formal attempt plan: 9 target classes x 3 distractor focus conditions x 4 repeats
= 108 attempts. All twelve objects remain in the original two racks throughout:
NO cross-rack swaps, object additions/removals, recoloring, detector threshold changes,
or use of ground-truth object name to decide the CV result.

IMPORTANT: A label (e.g. same_color) does NOT mean the corresponding distractor
was actually visible at that station. The output separately reports presence
in the active rack and visibility through the camera segmentation image.
Use scenario_exposed=True only if a distractor of the intended type really
appeared in a scan image. A visible distractor is not necessarily a detected
visual *candidate*, so both metrics are logged separately.

This is a pipeline evaluation. The main code uses PyBullet segmentation body-ID
and an object-identity strict gate; DO NOT interpret its success as pure RGB
shape classification accuracy. P06 does not test wheelchair rack fallback or
robot grasp; these belong to later integration/manipulation tests.
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

import numpy as np
import pybullet as p
import pybullet_data

SHAPES = ("box", "cylinder", "triangle")
COLORS = ("red", "blue", "yellow")
SHAPE_ID = {"box": "balok", "cylinder": "silinder", "triangle": "segitiga"}
COLOR_ID = {"red": "merah", "blue": "biru", "yellow": "kuning", "green": "hijau"}
ID_SHAPE = {v: k for k, v in SHAPE_ID.items()}
ID_COLOR = {v: k for k, v in COLOR_ID.items()}
SCENARIOS = ("same_color_different_shape", "same_shape_different_color", "mixed_clutter")
FIELDS = (
    "trial", "repetition", "seed", "scenario", "target_name", "target_shape", "target_color",
    "target_rack", "target_shelf", "station_scanned", "all_objects_in_scene",
    "same_color_other_shape_active", "same_shape_other_color_active",
    "same_color_other_shape_other_rack", "same_shape_other_color_other_rack",
    "active_rack_distractors", "observed_distractors", "same_color_visible",
    "same_shape_visible", "other_visible", "scenario_exposed", "scenario_eligibility_reason",
    "visual_candidate_distractors", "same_color_candidate_distractors",
    "candidate_counts_by_scan", "scans_performed", "detected_name", "detected_rack",
    "visual_shape_label", "visual_shape_agrees", "visual_score", "target_shape_score",
    "found", "strict_match_success", "false_positive", "false_positive_name",
    "status", "failure_reason", "search_time_sec", "error",
)


def load_sim(source):
    spec = importlib.util.spec_from_file_location("scriptis_p06_dosbim2_main", str(source))
    if spec is None or spec.loader is None:
        raise RuntimeError("Gagal membuat loader untuk " + str(source))
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def shape_color(name):
    text = str(name or "")
    for prefix, shape in ID_SHAPE.items():
        if text.startswith(prefix + "_"):
            return shape, ID_COLOR.get(text[len(prefix) + 1:], "")
    return "", ""


def rack_of(sim, data):
    pos, _ = p.getBasePositionAndOrientation(data["id"])
    return int(sim.get_rack_id_from_y(pos[1]))


def shelf_of(sim, data):
    pos, _ = p.getBasePositionAndOrientation(data["id"])
    return 1 + min(range(len(sim.SHELF_Z_LEVELS)),
                   key=lambda i: abs(pos[2] - (sim.SHELF_Z_LEVELS[i] + .065)))


def in_scenario(scenario, target_shape, target_color, candidate_name):
    shape, color = shape_color(candidate_name)
    same_color = bool(color == target_color and shape and shape != target_shape)
    same_shape = bool(shape == target_shape and color and color != target_color)
    if scenario == "same_color_different_shape":
        return same_color
    if scenario == "same_shape_different_color":
        return same_shape
    return same_color or same_shape


def dump_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, run_dir, source, digest, args, finished):
    def rate(ok, total):
        return round(100.0 * ok / total, 3) if total else None

    valid = [r for r in rows if r["scenario_exposed"] is True]
    times = [r["search_time_sec"] for r in rows if isinstance(r["search_time_sec"], (int, float))]
    scenarios = {}
    for scenario in SCENARIOS:
        sub = [r for r in rows if r["scenario"] == scenario]
        exposed = [r for r in sub if r["scenario_exposed"] is True]
        scenarios[scenario] = {
            "attempts": len(sub),
            "scenario_exposed": len(exposed),
            "not_exposed": len(sub) - len(exposed),
            "strict_match_success_all": sum(r["strict_match_success"] is True for r in sub),
            "strict_match_success_exposed": sum(r["strict_match_success"] is True for r in exposed),
            "success_rate_exposed_percent": rate(sum(r["strict_match_success"] is True for r in exposed), len(exposed)),
            "false_positive_exposed": sum(r["false_positive"] is True for r in exposed),
        }
    data = {
        "test": "P06", "revision": "dosbim2", "completed": bool(finished),
        "source": str(source), "source_sha256": digest, "run_dir": str(run_dir),
        "mode": "GUI" if args.gui else "DIRECT", "repetitions": args.repeat,
        "formal_classes": 9, "planned_attempts": 9 * len(SCENARIOS) * args.repeat,
        "completed_attempts": len(rows),
        "strict_match_success_all": sum(r["strict_match_success"] is True for r in rows),
        "success_rate_all_percent": rate(sum(r["strict_match_success"] is True for r in rows), len(rows)),
        "false_positive_all": sum(r["false_positive"] is True for r in rows),
        "scenario_exposed_count": len(valid),
        "strict_match_success_exposed": sum(r["strict_match_success"] is True for r in valid),
        "success_rate_exposed_percent": rate(sum(r["strict_match_success"] is True for r in valid), len(valid)),
        "false_positive_exposed": sum(r["false_positive"] is True for r in valid),
        "average_search_time_sec": round(statistics.mean(times), 3) if times else None,
        "by_scenario": scenarios,
        "failures": [{"trial": r["trial"], "scenario": r["scenario"], "target": r["target_name"],
                      "detected": r["detected_name"], "reason": r["failure_reason"],
                      "scenario_exposed": r["scenario_exposed"]}
                     for r in rows if r["strict_match_success"] is not True],
        "method_note": (
            "All 12 objects remain in their original rack groups; every trial contains the same full-clutter "
            "scene with randomized within-rack placement. The 3 scenario labels indicate distinct distractor "
            "exposure analyses, NOT three isolated/reconstructed scenes. Not all scenario labels are physically "
            "exposed in every trial: interpret scenario-conditioned metrics only on observed (visible) distractors. "
            "Observed means visible in wrist-camera PyBullet segmentation, not necessarily detected as a color/shape candidate. "
            "Strict gate uses PyBullet body-ID and object identity; results are pipeline success, not pure RGB shape classification. "
            "No grasp/handover is performed."
        ),
    }
    (run_dir / "pengujian_06_strict_visual_match_distractor_summary.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = [
        "# P06 — Strict Visual Match with Distractor (dosbim2)", "",
        f"Source SHA256: `{digest}`", "",
        f"Completed attempts: {len(rows)}; match: {data['strict_match_success_all']}; false positive: {data['false_positive_all']}.",
        f"Exposed attempts: {len(valid)}; match on exposed: {data['strict_match_success_exposed']}.",
        "", "## Breakdown by scenario", "",
        "| Scenario | Attempts | Exposed | Match (exposed) | False positive (exposed) |", "|---|---:|---:|---:|---:|",
    ]
    for key, val in scenarios.items():
        lines.append(f"| {key} | {val['attempts']} | {val['scenario_exposed']} | "
                     f"{val['strict_match_success_exposed']} | {val['false_positive_exposed']} |")
    lines += ["", "## Limitasi", "", data["method_note"], "",
              "`scenario_exposed=False` berarti distractor sesuai label tidak benar-benar terlihat di kamera "
              "selama trial, sehingga trial itu tidak membuktikan kemampuan membedakan distractor jenis tersebut."]
    (run_dir / "pengujian_06_strict_visual_match_distractor_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    return data


def main():
    parser = argparse.ArgumentParser(description="P06 dosbim2 strict match + observed distractor logging")
    parser.add_argument("--source", default="scripts/07_indonesian_store_jaco_camera_NEW.py")
    parser.add_argument("--output_dir", default="data/test_logs/dosbim2_revision/P06_strict_visual_match")
    parser.add_argument("--repeat", type=int, default=4)
    parser.add_argument("--limit", type=int, default=0, help="Jalankan N trial pertama (smoke test)")
    parser.add_argument("--seed", type=int, default=20260921)
    parser.add_argument("--settle_steps", type=int, default=120)
    parser.add_argument("--min_visible_pixels", type=int, default=50)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--show-camera", action="store_true")
    args = parser.parse_args()
    if args.repeat < 1 or args.limit < 0 or args.min_visible_pixels < 1:
        parser.error("--repeat >= 1, --limit >= 0, --min_visible_pixels >= 1")
    source = Path(args.source).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError("Simulasi utama tidak ditemukan: " + str(source))
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    run_dir = Path(args.output_dir).expanduser().resolve() / time.strftime("P06_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "pengujian_06_strict_visual_match_distractor.csv"
    sim = load_sim(source)
    sim.SHOW_CAMERA_WINDOWS = bool(args.show_camera)
    sim.VISUAL_SEARCH_LOG_DIR = str(run_dir / "debug_images")
    sim.OBJECT_RANDOM_LOG_DIR = str(run_dir / "randomization")
    client = p.connect(p.GUI if args.gui else p.DIRECT)
    if client < 0:
        raise RuntimeError("PyBullet connect failed")
    rows = []
    orig_capture = sim.capture_wrist_pov_for_visual_search
    orig_detect = sim.detect_target_from_visual_frame
    try:
        p.resetSimulation()
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)
        p.setPhysicsEngineParameter(fixedTimeStep=1 / 240., numSolverIterations=150, numSubSteps=2)
        objects, initial_poses = sim.create_indonesian_store_scene()
        robot_data = sim.load_jaco_robot()
        sim.register_mobile_platform_body(robot_data["robot_id"])
        sim.set_live_camera_context(robot_data=robot_data, objects=objects)
        sim.reset_robot(robot_data)
        for _ in range(args.settle_steps):
            p.stepSimulation()
        targets = [(shape, color, SHAPE_ID[shape] + "_" + COLOR_ID[color])
                   for shape in SHAPES for color in COLORS]
        for _, _, name in targets:
            if name not in objects:
                raise RuntimeError("Formal target not found in scene: " + name)
        trials = [(rep, scenario, shape, color, name)
                  for rep in range(1, args.repeat + 1)
                  for shape, color, name in targets
                  for scenario in SCENARIOS]
        if args.limit:
            trials = trials[:args.limit]
        print(f"\n=== P06 DOSBIM2 | {len(trials)} attempts | all twelve objects remain in scene ===")
        print("Source hash:", digest)
        print("Output:", run_dir)
        for index, (rep, scenario, shape, color, name) in enumerate(trials, 1):
            random.seed(args.seed + index)
            row = {field: "" for field in FIELDS}
            row.update({"trial": index, "repetition": rep, "seed": args.seed + index,
                        "scenario": scenario, "target_name": name, "target_shape": shape,
                        "target_color": color, "found": False, "strict_match_success": False,
                        "false_positive": False, "scenario_exposed": False, "status": "Gagal"})
            started = None
            frames_seen = 0
            candidate_counts = []
            visible_pixels = Counter()
            candidate_distractor_names = set()
            same_color_candidate_names = set()
            try:
                sim.reset_objects_randomized(objects, initial_poses, reason=f"P06_trial_{index:03d}")
                sim.VISUAL_SEARCH_LOG_DIR = str(run_dir / "debug_images" / f"trial_{index:03d}")
                target_rack = rack_of(sim, objects[name])
                row["target_rack"] = target_rack
                row["target_shelf"] = shelf_of(sim, objects[name])
                if target_rack not in (1, 2):
                    raise RuntimeError("Target rack tidak valid " + str(target_rack))
                active_names = [n for n, data in objects.items() if n != name and rack_of(sim, data) == target_rack]
                other_names = [n for n, data in objects.items() if n != name and rack_of(sim, data) != target_rack]
                same_color_names = [n for n in active_names
                                    if in_scenario("same_color_different_shape", shape, color, n)]
                same_shape_names = [n for n in active_names
                                    if in_scenario("same_shape_different_color", shape, color, n)]
                row["all_objects_in_scene"] = len(objects)
                row["active_rack_distractors"] = json.dumps(active_names, ensure_ascii=False)
                row["same_color_other_shape_active"] = json.dumps(same_color_names, ensure_ascii=False)
                row["same_shape_other_color_active"] = json.dumps(same_shape_names, ensure_ascii=False)
                row["same_color_other_shape_other_rack"] = json.dumps(
                    [n for n in other_names if in_scenario("same_color_different_shape", shape, color, n)], ensure_ascii=False)
                row["same_shape_other_color_other_rack"] = json.dumps(
                    [n for n in other_names if in_scenario("same_shape_different_color", shape, color, n)], ensure_ascii=False)
                sim.move_wheelchair_to_station(robot_data, target_rack, duration=.35)
                sim.reset_robot(robot_data)
                sim.set_wrist_pov_forced_target(None)
                sim.ACTIVE_RACK_ID = target_rack
                sim.ACTIVE_RACK_Y_OFFSET = sim.WHEELCHAIR_STATIONS[target_rack]["rack_y_offset"]
                row["station_scanned"] = sim.CURRENT_STATION_ID
                if sim.CURRENT_STATION_ID != target_rack or len(sim.build_visual_search_scan_plan()) != 6:
                    raise RuntimeError("Station or 6-scan plan invalid")
                body_id_to_name = {int(data["id"]): n for n, data in objects.items()}

                def measured_capture(*a, **kw):
                    nonlocal frames_seen
                    frame = orig_capture(*a, **kw)
                    frames_seen += 1
                    seg = np.asarray(frame["seg_body"])
                    # Only count visible pixels of *physical* distractors at active station.
                    for n in active_names:
                        num = int(np.count_nonzero(seg == int(objects[n]["id"])))
                        visible_pixels[n] = max(visible_pixels[n], num)
                    return frame

                def measured_detect(*a, **kw):
                    result = orig_detect(*a, **kw)
                    candidates = result[0]
                    candidate_counts.append(len(candidates))
                    for candidate in candidates:
                        candidate_name = body_id_to_name.get(candidate.get("body_id"))
                        if candidate_name in active_names:
                            candidate_distractor_names.add(candidate_name)
                            if candidate_name in same_color_names:
                                same_color_candidate_names.add(candidate_name)
                    return result

                sim.capture_wrist_pov_for_visual_search = measured_capture
                sim.detect_target_from_visual_frame = measured_detect
                started = time.monotonic()
                detected_name, selected_candidate = sim.run_visual_search_for_target(
                    robot_data=robot_data, objects=objects, target_shape=shape, target_color=color)
                row["search_time_sec"] = round(time.monotonic() - started, 3)
                row["scans_performed"] = frames_seen
                row["candidate_counts_by_scan"] = json.dumps(candidate_counts)
                row["visual_candidate_distractors"] = json.dumps(sorted(candidate_distractor_names), ensure_ascii=False)
                row["same_color_candidate_distractors"] = json.dumps(sorted(same_color_candidate_names), ensure_ascii=False)
                visible = {n for n in active_names if visible_pixels[n] >= args.min_visible_pixels}
                vis_color = sorted(n for n in same_color_names if n in visible)
                vis_shape = sorted(n for n in same_shape_names if n in visible)
                row["observed_distractors"] = json.dumps(
                    {n: visible_pixels[n] for n in sorted(visible)}, ensure_ascii=False)
                row["same_color_visible"] = json.dumps(vis_color, ensure_ascii=False)
                row["same_shape_visible"] = json.dumps(vis_shape, ensure_ascii=False)
                row["other_visible"] = json.dumps(sorted(visible - set(vis_color) - set(vis_shape)), ensure_ascii=False)
                if scenario == "same_color_different_shape":
                    exposed = bool(vis_color)
                elif scenario == "same_shape_different_color":
                    exposed = bool(vis_shape)
                else:
                    # Full clutter is genuinely observed if >=2 non-target objects are visible.
                    exposed = len(visible) >= 2
                row["scenario_exposed"] = exposed
                row["scenario_eligibility_reason"] = (
                    "distractor_observed_in_camera" if exposed else
                    "matching_distractor_absent_from_active_rack" if
                    (scenario == "same_color_different_shape" and not same_color_names or
                     scenario == "same_shape_different_color" and not same_shape_names) else
                    "matching_distractor_not_visible_in_scan"
                )
                row["detected_name"] = detected_name or ""
                row["found"] = bool(detected_name)
                row["strict_match_success"] = bool(detected_name == name)
                row["false_positive"] = bool(detected_name and detected_name != name)
                row["false_positive_name"] = detected_name if row["false_positive"] else ""
                if detected_name and detected_name in objects:
                    row["detected_rack"] = rack_of(sim, objects[detected_name])
                if selected_candidate is not None:
                    row["visual_shape_label"] = selected_candidate.get("shape_label", "")
                    row["visual_shape_agrees"] = selected_candidate.get("shape_label") == shape
                    row["visual_score"] = float(selected_candidate.get("score", 0.))
                    row["target_shape_score"] = float(selected_candidate.get("target_shape_score", 0.))
                row["status"] = "Berhasil" if row["strict_match_success"] else "Gagal"
                row["failure_reason"] = "" if row["strict_match_success"] else (
                    "wrong_object_selected" if row["false_positive"] else "target_not_found")
            except Exception:
                row["error"] = traceback.format_exc()
                row["failure_reason"] = "runtime_error"
                if started is not None and row["search_time_sec"] == "":
                    row["search_time_sec"] = round(time.monotonic() - started, 3)
                print(row["error"])
            finally:
                sim.capture_wrist_pov_for_visual_search = orig_capture
                sim.detect_target_from_visual_frame = orig_detect
                sim.set_wrist_pov_forced_target(None)
                rows.append(row)
                dump_csv(csv_path, rows)
                summarize(rows, run_dir, source, digest, args, finished=False)
                print(f"[P06 {index}/{len(trials)}] {row['status']} "
                      f"scenario={scenario} exposed={row['scenario_exposed']} "
                      f"target={name} rack={row['target_rack']} "
                      f"detected={row['detected_name'] or '-'} scans={row['scans_performed']}")
        final = summarize(rows, run_dir, source, digest, args, finished=True)
        print("\n=== P06 SELESAI ===")
        print(json.dumps({k: final[k] for k in
                         ("completed_attempts", "strict_match_success_all", "false_positive_all",
                          "scenario_exposed_count", "strict_match_success_exposed", "by_scenario")},
                         ensure_ascii=False, indent=2))
    finally:
        sim.capture_wrist_pov_for_visual_search = orig_capture
        sim.detect_target_from_visual_frame = orig_detect
        sim.set_wrist_pov_forced_target(None)
        if p.isConnected(client):
            p.disconnect(client)


if __name__ == "__main__":
    main()

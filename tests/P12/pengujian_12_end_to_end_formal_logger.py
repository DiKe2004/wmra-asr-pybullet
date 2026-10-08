#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""P12 DOSBIM2: 9 target x 3 repetisi = 27 trial end-to-end berbasis perintah teks.

Simpan sebagai ~/Scriptis4_PyBullet/scripts/testv2/pengujian_12_end_to_end_formal_logger.py
Dari root proyek:
  python scripts/testv2/pengujian_12_end_to_end_formal_logger.py --limit 1 --gui   # PILOT terpisah
  python scripts/testv2/pengujian_12_end_to_end_formal_logger.py --gui             # FORMAL 27
  python scripts/testv2/pengujian_12_end_to_end_formal_logger.py                   # FORMAL DIRECT

Script ini tidak mengubah file simulasi utama ataupun parameter baseline v12.
Setiap trial membuat scene baru, melakukan satu randomisasi in-rack dengan seed
tercatat, mulai dari station 1, lalu memanggil pipeline utama tanpa menyalip
visual search dengan metadata ground truth. Ground truth dibaca untuk audit.

Keberhasilan yang diukur ialah penyelesaian *pipeline virtual grasp*, bukan
pengenalan RGB murni, genggaman force-closure, atau collision-free fisik.
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
from collections import Counter
from datetime import datetime
from pathlib import Path

import pybullet as p
import pybullet_data

FORMAL_TARGETS = (
    ("balok_merah", "ambil balok merah", "box", "red", 1),
    ("balok_biru", "ambil balok biru", "box", "blue", 2),
    ("balok_kuning", "ambil balok kuning", "box", "yellow", 1),
    ("silinder_merah", "ambil silinder merah", "cylinder", "red", 2),
    ("silinder_biru", "ambil silinder biru", "cylinder", "blue", 1),
    ("silinder_kuning", "ambil silinder kuning", "cylinder", "yellow", 2),
    ("segitiga_merah", "ambil segitiga merah", "triangle", "red", 2),
    ("segitiga_biru", "ambil segitiga biru", "triangle", "blue", 1),
    ("segitiga_kuning", "ambil segitiga kuning", "triangle", "yellow", 2),
)
# expected_rack diperoleh dari penempatan awal baseline; hanya untuk audit
# sesudah randomisasi (randomisasi tidak boleh memindahkan objek lintas rak).
FIELDS = (
    "trial", "target_class", "repetition", "command", "expected_name", "expected_rack",
    "shape", "color",
    "random_seed", "source_sha256", "initial_station", "initial_rack", "initial_shelf",
    "initial_object_xyz_m", "initial_object_quat", "initial_objects_json",
    "parsed_action", "parsed_shape", "parsed_color", "parser_valid",
    "station_commands", "scan_stations", "scan_attempts", "detected_names_by_station",
    "rack2_reached", "wheelchair_moved", "final_station", "station_path_ok",
    "detected_name", "detected_rack", "object_match", "candidate_body_id",
    "candidate_visual_shape_label", "candidate_visual_shape_matches_command", "candidate_score",
    "visual_pick_position_xyz_m", "grasp_attempted", "grasp_constraint_created",
    "lift_observed", "lift_height_m", "lift_pass_5cm", "lift_ik_error_m",
    "guard_align_calls", "guard_align_last_ready", "guard_final_ready",
    "release_called", "release_performed", "natural_drop_called", "active_grasp_after",
    "object_final_xyz_m", "basket_inside_xy_aabb", "basket_floor_error_m",
    "basket_floor_ok", "basket_success", "collision_contact_entries",
    "collision_near_entries", "collision_rack_contact_entries", "collision_mount_contact_entries",
    "collision_other_contact_entries", "collision_unique_contact_pairs",
    "collision_events_by_body_json", "collision_events_by_stage_json",
    "task_success", "failure_stage", "end_to_end_sec", "setup_sec", "exception"
)


def trial_plan():
    return [
        {"trial": i * 3 + repetition,
         "target_class": name, "repetition": repetition, "command": command,
         "expected_name": name, "shape": shape, "color": color,
         "expected_rack": rack}
        for i, (name, command, shape, color, rack) in enumerate(FORMAL_TARGETS)
        for repetition in (1, 2, 3)
    ]


def import_sim(source: Path, trial: int):
    name = f"scriptis_p12_trial_{trial}_{time.monotonic_ns()}"
    spec = importlib.util.spec_from_file_location(name, str(source))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Tidak bisa memuat simulasi utama: {source}")
    sim = importlib.util.module_from_spec(spec)
    sys.modules[name] = sim
    spec.loader.exec_module(sim)
    return sim


def xyz(body_id):
    return [float(v) for v in p.getBasePositionAndOrientation(body_id)[0]]


def audit_basket(sim, object_id, margin=0.002, floor_tol=0.035):
    """Audit AABB objek asli setelah drop, tanpa memindahkan objek."""
    if sim.BASKET_INFO is None:
        return False, False, None, False
    info = sim.BASKET_INFO
    center = [float(v) for v in sim.BASKET_CENTER]
    half_x = float(info["inner_size"][0]) / 2.0
    half_y = float(info["inner_size"][1]) / 2.0
    lo, hi = p.getAABB(object_id)
    inside = (lo[0] >= center[0] - half_x + margin
              and hi[0] <= center[0] + half_x - margin
              and lo[1] >= center[1] - half_y + margin
              and hi[1] <= center[1] + half_y - margin)
    floor_error = float(lo[2]) - float(info["floor_top_z"])
    floor_ok = abs(floor_error) <= floor_tol
    return bool(inside and floor_ok), bool(inside), round(floor_error, 6), bool(floor_ok)


def write_csv(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summary(run_dir, source, sha, rows, planned_count, is_formal, complete, args):
    successes = sum(r.get("task_success") is True for r in rows)
    duration = [float(r["end_to_end_sec"]) for r in rows
                if r.get("end_to_end_sec") not in (None, "")]
    stages = Counter(str(r.get("failure_stage") or "") for r in rows
                     if r.get("task_success") is not True)
    by_target = []
    for name, _, _, _, _ in FORMAL_TARGETS:
        subset = [r for r in rows if r.get("expected_name") == name]
        by_target.append({"target": name, "completed": len(subset),
                          "success": sum(r.get("task_success") is True for r in subset)})
    report = {
        "test": "P12", "revision": "dosbim2", "formal_run": is_formal,
        "completed": bool(complete), "source": str(source), "source_sha256": sha,
        "run_dir": str(run_dir), "mode": "GUI" if args.gui else "DIRECT",
        "seed_base": args.seed_base, "planned_formal_trials": 27,
        "planned_trials_this_run": planned_count, "trial_count": len(rows),
        "task_success": successes, "failed": len(rows) - successes,
        "task_success_rate_percent": round(100 * successes / len(rows), 3) if rows else None,
        "parser_valid": sum(r.get("parser_valid") is True for r in rows),
        "correct_target_selected": sum(r.get("object_match") is True for r in rows),
        "basket_success": sum(r.get("basket_success") is True for r in rows),
        "wheelchair_moved": sum(r.get("wheelchair_moved") is True for r in rows),
        "release_performed": sum(r.get("release_performed") is True for r in rows),
        "trials_with_observed_rack_contact": sum(
            int(r.get("collision_rack_contact_entries") or 0) > 0 for r in rows),
        "trials_with_observed_other_contact": sum(
            int(r.get("collision_other_contact_entries") or 0) > 0 for r in rows),
        "average_end_to_end_sec": round(statistics.mean(duration), 3) if duration else None,
        "median_end_to_end_sec": round(statistics.median(duration), 3) if duration else None,
        "per_target": by_target, "failure_stages": dict(stages),
        "failures": [{"trial": r["trial"], "target": r["expected_name"],
                      "stage": r.get("failure_stage", ""), "error": r.get("exception", "")}
                     for r in rows if r.get("task_success") is not True],
        "measurement_notes": [
            "Input: perintah teks valid; ASR dan speaker recognition tidak diuji ulang.",
            "Setiap trial scene baru; randomisasi slot hanya dalam rak asal, jitter dan yaw v12 tidak diubah.",
            "End-to-end wall time dimulai tepat sebelum parsing perintah, berakhir setelah audit basket; setup terpisah.",
            "Visual pipeline memakai segmen body ID/metadata sebagai gate; bukan akurasi RGB-only.",
            "Grasp bersifat virtual fixed constraint dengan alignment; collision objek saat transport dinonaktifkan.",
            "Collision entries berasal dari log monitor selama IK_MOVE: dapat berulang untuk satu kontak, tidak mencakup seluruh collision.",
            "Keberhasilan mensyaratkan target dan station benar, virtual grasp, lift >=5 cm teramati, guard ready, release, dan AABB objek di keranjang.",
            "Pilot (--limit) tidak boleh digabung ke hasil formal 27 trial. Kegagalan formal tidak dihapus atau diulang diam-diam.",
        ],
    }
    (run_dir / "pengujian_12_end_to_end_summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    lines = ["# P12 — End-to-End Formal", "",
             f"Jenis run: {'FORMAL' if is_formal else 'PILOT'}; lengkap: {complete}.",
             f"SHA256 sumber: `{sha}`", f"Trial: {len(rows)}/{planned_count}; sukses: {successes}.",
             "", "| Trial | Target | Repetisi | Rak | Shelf | Station | Basket | Task | Gagal di | E2E (detik) |",
             "|---:|---|---:|---:|---:|---|---|---|---|---:|"]
    for r in rows:
        lines.append(f"| {r['trial']} | {r['expected_name']} | {r['repetition']} | "
                     f"{r.get('initial_rack', '')} | {r.get('initial_shelf', '')} | "
                     f"{r.get('scan_stations', '')} | {r.get('basket_success', '')} | "
                     f"{r.get('task_success', '')} | {r.get('failure_stage', '')} | "
                     f"{r.get('end_to_end_sec', '')} |")
    lines += ["", "## Batasan", *[f"- {note}" for note in report["measurement_notes"]]]
    (run_dir / "pengujian_12_end_to_end_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8")
    return report


def audit_initial_scene(sim, objects, row):
    """Ground truth audit saja; tidak digunakan untuk memilih objek saat search."""
    name = row["expected_name"]
    if name not in objects:
        raise RuntimeError(f"Target tidak ada dalam scene: {name}")
    poses = {}
    for obj_name, obj in objects.items():
        pos, orn = p.getBasePositionAndOrientation(obj["id"])
        poses[obj_name] = {
            "rack": sim.get_rack_id_from_y(pos[1]),
            "xyz_m": [round(float(v), 6) for v in pos],
            "quat": [round(float(v), 6) for v in orn],
        }
    pos = poses[name]["xyz_m"]
    rack = poses[name]["rack"]
    shelf = 1 + min(range(len(sim.SHELF_Z_LEVELS)), key=lambda k: abs(
        pos[2] - (float(sim.SHELF_Z_LEVELS[k]) + 0.065)))
    row.update(initial_station=int(sim.CURRENT_STATION_ID), initial_rack=rack,
               initial_shelf=shelf, initial_object_xyz_m=json.dumps(pos),
               initial_object_quat=json.dumps(poses[name]["quat"]),
               initial_objects_json=json.dumps(poses, ensure_ascii=False))
    if rack != row["expected_rack"]:
        raise RuntimeError(f"Objek {name} berpindah rak saat randomisasi: {rack} vs {row['expected_rack']}")
    if len(objects) != 12 or any(v["rack"] not in (1, 2) for v in poses.values()):
        raise RuntimeError("Scene randomisasi tidak memiliki 12 objek valid pada dua rak")
    if sim.CURRENT_STATION_ID != 1 or sim.RELEASE_GUARD_MAX_ALIGN_ITERS != 12:
        raise RuntimeError("Baseline tidak sesuai: station awal harus 1 dan release guard 12")
    if list(sim.WRIST_POV_LOCAL_OFFSET) != [0.04, 0.14, 0.075]:
        raise RuntimeError("Offset kamera wrist bukan baseline v12")
    if not sim.OBJECT_RANDOMIZE_KEEP_RACK_GROUP or not sim.OBJECT_RANDOMIZE_EQUAL_2_PER_SHELF:
        raise RuntimeError("Mode randomisasi bukan baseline v12")
    # Setiap shelf harus memiliki dua objek pada tiap rak setelah penempatan acak.
    counts = Counter((v["rack"], 1 + min(range(3), key=lambda k: abs(
        v["xyz_m"][2] - (sim.SHELF_Z_LEVELS[k] + 0.065)))) for v in poses.values())
    if any(counts[(rack_id, level)] != 2 for rack_id in (1, 2) for level in (1, 2, 3)):
        raise RuntimeError(f"Jumlah objek per rak/shelf tidak sesuai 2: {dict(counts)}")
    return objects[name]["id"], float(pos[2])


def run_trial(sim, robot_data, objects, row, args):
    """Sadap fungsi pipeline asli untuk logging; fungsi pergerakan/deteksi tidak diubah."""
    target_id = objects[row["expected_name"]]["id"]
    start_z = float(json.loads(row["initial_object_xyz_m"])[2])
    events = {"station": [], "scan": [], "grasp": [], "guard": [], "final_guard": [],
              "release": [], "natural_drop": [], "lift": [], "lift_point": None}
    contact_by_body, contact_by_stage, contact_pairs = Counter(), Counter(), set()
    counts = Counter()
    names = (
        "move_wheelchair_to_station", "run_visual_search_for_target",
        "attach_object_to_end_effector", "get_shelf_aware_motion_points",
        "move_robot_to_position", "align_attached_object_over_basket_release_guard",
        "check_object_ready_to_release_into_basket", "release_object",
        "natural_gravity_drop_to_basket", "write_collision_event_to_csv"
    )
    original = {key: getattr(sim, key) for key in names}

    def watch_station(*a, **kw):
        station_id = int(kw.get("station_id", a[1] if len(a) >= 2 else -1))
        previous = int(sim.CURRENT_STATION_ID)
        result = original["move_wheelchair_to_station"](*a, **kw)
        events["station"].append({"from": previous, "requested": station_id,
                                  "reached": int(sim.CURRENT_STATION_ID), "ok": bool(result)})
        return result

    def watch_search(*a, **kw):
        station = int(sim.CURRENT_STATION_ID)
        name, candidate = original["run_visual_search_for_target"](*a, **kw)
        events["scan"].append((station, name, candidate))
        return name, candidate

    def watch_grasp(*a, **kw):
        cid = original["attach_object_to_end_effector"](*a, **kw)
        events["grasp"].append(cid)
        return cid

    def watch_points(*a, **kw):
        points = original["get_shelf_aware_motion_points"](*a, **kw)
        events["lift_point"] = tuple(float(v) for v in points["lift"])
        return points

    def watch_move(*a, **kw):
        target = kw.get("target_position", a[1] if len(a) > 1 else None)
        lift = events["lift_point"]
        is_lift = (lift is not None and target is not None
                   and len(target) == 3 and all(abs(float(v) - t) < 1e-6
                                                for v, t in zip(target, lift)))
        result = original["move_robot_to_position"](*a, **kw)
        if is_lift:
            events["lift"].append({"height": xyz(target_id)[2] - start_z,
                                   "error": float(result[0]), "dropped": bool(result[2])})
        return result

    def watch_align(*a, **kw):
        result = original["align_attached_object_over_basket_release_guard"](*a, **kw)
        events["guard"].append(bool(result[2]))
        return result

    def watch_final_guard(*a, **kw):
        result = original["check_object_ready_to_release_into_basket"](*a, **kw)
        events["final_guard"].append(bool(result[0]))
        return result

    def watch_release(*a, **kw):
        cid = a[0] if a else kw.get("constraint_id")
        result = original["release_object"](*a, **kw)
        still_exists = False
        if cid is not None:
            try:
                still_exists = int(cid) in [p.getConstraintUniqueId(k)
                                            for k in range(p.getNumConstraints())]
            except Exception:
                still_exists = True  # jangan mengklaim release jika audit constraint error
        events["release"].append({"constraint": cid, "removed": cid is not None and not still_exists})
        return result

    def watch_drop(*a, **kw):
        obj_id = a[0] if a else kw.get("object_id")
        result = original["natural_gravity_drop_to_basket"](*a, **kw)
        events["natural_drop"].append(int(obj_id))
        return result

    def watch_collision(event):
        severity = str(event.get("severity", ""))
        body = str(event.get("other_body", ""))
        stage = str(event.get("stage", ""))
        if severity == "CONTACT":
            counts["contact"] += 1
            contact_by_body[body] += 1
            contact_by_stage[stage] += 1
            contact_pairs.add((body, str(event.get("robot_link", ""))))
            if str(event.get("other_role", "")) == "shelf":
                counts["rack_contact"] += 1
            elif "mount" in body.lower():
                counts["mount_contact"] += 1
            else:
                counts["other_contact"] += 1
        elif severity == "NEAR":
            counts["near"] += 1
        return original["write_collision_event_to_csv"](event)

    watchers = {
        "move_wheelchair_to_station": watch_station,
        "run_visual_search_for_target": watch_search,
        "attach_object_to_end_effector": watch_grasp,
        "get_shelf_aware_motion_points": watch_points,
        "move_robot_to_position": watch_move,
        "align_attached_object_over_basket_release_guard": watch_align,
        "check_object_ready_to_release_into_basket": watch_final_guard,
        "release_object": watch_release,
        "natural_gravity_drop_to_basket": watch_drop,
        "write_collision_event_to_csv": watch_collision,
    }
    for key, callback in watchers.items():
        setattr(sim, key, callback)

    result_name = None
    pipeline_error = None
    t0 = time.monotonic()  # WAKTU E2E: sesaat sebelum perintah diparse
    try:
        parsed = sim.parse_terminal_target_command(row["command"])
        row["parsed_action"] = parsed.get("action", "") if isinstance(parsed, dict) else ""
        row["parsed_shape"] = parsed.get("shape", "") if isinstance(parsed, dict) else ""
        row["parsed_color"] = parsed.get("color", "") if isinstance(parsed, dict) else ""
        row["parser_valid"] = bool(parsed and parsed.get("action") == "visual_search_pick"
                                   and parsed.get("shape") == row["shape"]
                                   and parsed.get("color") == row["color"])
        if row["parser_valid"]:
            result_name = sim.run_multi_station_visual_search_and_pick(
                robot_data=robot_data, objects=objects,
                target_shape=parsed["shape"], target_color=parsed["color"])
        else:
            row["failure_stage"] = "parser"
    except Exception as exc:
        pipeline_error = exc
        row["exception"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}"
    finally:
        # Pulihkan semua wrapper; baca state setelah pipeline dan drop selesai.
        for key, callback in original.items():
            setattr(sim, key, callback)
        row["end_to_end_sec"] = round(time.monotonic() - t0, 3)
        row["station_commands"] = json.dumps(events["station"])
        row["scan_stations"] = json.dumps([s for s, _, _ in events["scan"]])
        row["scan_attempts"] = len(events["scan"])
        row["detected_names_by_station"] = json.dumps(
            [{"station": s, "name": n or ""} for s, n, _ in events["scan"]],
            ensure_ascii=False)
        row["rack2_reached"] = any(s == 2 for s, _, _ in events["scan"])
        row["wheelchair_moved"] = any(v["from"] != v["reached"] and v["reached"] == 2
                                      and v["ok"] for v in events["station"])
        row["final_station"] = int(sim.CURRENT_STATION_ID)
        scan_path = [s for s, _, _ in events["scan"]]
        row["station_path_ok"] = scan_path == (
            [1] if row["initial_rack"] == 1 else [1, 2])
        row["grasp_attempted"] = bool(events["grasp"])
        row["grasp_constraint_created"] = any(cid is not None for cid in events["grasp"])
        row["lift_observed"] = bool(events["lift"])
        row["lift_height_m"] = (round(events["lift"][-1]["height"], 6)
                                 if events["lift"] else "")
        row["lift_ik_error_m"] = (round(events["lift"][-1]["error"], 6)
                                   if events["lift"] else "")
        row["lift_pass_5cm"] = bool(events["lift"] and
                                      events["lift"][-1]["height"] >= 0.05 and
                                      not events["lift"][-1]["dropped"])
        row["guard_align_calls"] = len(events["guard"])
        row["guard_align_last_ready"] = events["guard"][-1] if events["guard"] else False
        row["guard_final_ready"] = events["final_guard"][-1] if events["final_guard"] else False
        row["release_called"] = bool(events["release"])
        row["natural_drop_called"] = target_id in events["natural_drop"]
        row["release_performed"] = bool(row["natural_drop_called"] and
            any(r["removed"] for r in events["release"]))
        row["active_grasp_after"] = sim.ACTIVE_GRASP_OBJECT_ID is not None
        row["collision_contact_entries"] = counts["contact"]
        row["collision_near_entries"] = counts["near"]
        row["collision_rack_contact_entries"] = counts["rack_contact"]
        row["collision_mount_contact_entries"] = counts["mount_contact"]
        row["collision_other_contact_entries"] = counts["other_contact"]
        row["collision_unique_contact_pairs"] = len(contact_pairs)
        row["collision_events_by_body_json"] = json.dumps(contact_by_body, ensure_ascii=False)
        row["collision_events_by_stage_json"] = json.dumps(contact_by_stage, ensure_ascii=False)
        accepted = [(s, n, c) for s, n, c in events["scan"] if n is not None]
        if accepted:
            station, name, candidate = accepted[-1]
            row["detected_name"] = name
            row["detected_rack"] = station
            if candidate is not None:
                row["candidate_body_id"] = candidate.get("body_id", "")
                row["candidate_visual_shape_label"] = candidate.get("shape_label", "")
                row["candidate_visual_shape_matches_command"] = (
                    candidate.get("shape_label") == row["shape"])
                row["candidate_score"] = candidate.get("score", "")
        row["object_match"] = (row.get("detected_name") == row["expected_name"])
        row["visual_pick_position_xyz_m"] = json.dumps(
            objects[row["expected_name"]].get("visual_pick_position", None))
        try:
            row["object_final_xyz_m"] = json.dumps([round(v, 6) for v in xyz(target_id)])
            basket_ok, inside, floor_error, floor_ok = audit_basket(
                sim, target_id, margin=args.xy_margin, floor_tol=args.floor_tolerance)
            row["basket_inside_xy_aabb"] = inside
            row["basket_floor_error_m"] = floor_error
            row["basket_floor_ok"] = floor_ok
            row["basket_success"] = bool(basket_ok and row["release_performed"])
        except Exception as exc:
            row["exception"] += f"\nPost-task audit: {type(exc).__name__}: {exc}"
            row["basket_success"] = False
        # Kesuksesan tidak disimpulkan dari return nama objek semata.
        row["task_success"] = bool(
            pipeline_error is None and row["parser_valid"] and row["station_path_ok"]
            and row["object_match"] and row["detected_rack"] == row["initial_rack"]
            and (row["wheelchair_moved"] == (row["initial_rack"] == 2))
            and result_name == row["expected_name"]
            and row["grasp_constraint_created"] and row["lift_pass_5cm"]
            and row["guard_align_last_ready"] and row["guard_final_ready"]
            and row["release_performed"] and row["basket_success"]
            and not row["active_grasp_after"])
        if not row["task_success"] and not row.get("failure_stage"):
            if pipeline_error is not None:
                row["failure_stage"] = "runtime_exception"
            elif not row["parser_valid"]:
                row["failure_stage"] = "parser"
            elif not row["station_path_ok"] or row["wheelchair_moved"] != (row["initial_rack"] == 2):
                row["failure_stage"] = "multi_station_search_or_movement"
            elif not row["object_match"] or row["detected_rack"] != row["initial_rack"]:
                row["failure_stage"] = "visual_search_or_wrong_object"
            elif result_name != row["expected_name"]:
                row["failure_stage"] = "multi_station_result"
            elif not row["grasp_constraint_created"]:
                row["failure_stage"] = "virtual_grasp_or_ik"
            elif not row["lift_pass_5cm"]:
                row["failure_stage"] = "lift"
            elif not row["guard_align_last_ready"] or not row["guard_final_ready"]:
                row["failure_stage"] = "release_guard"
            elif not row["release_performed"] or row["active_grasp_after"]:
                row["failure_stage"] = "handover_or_release"
            elif not row["basket_success"]:
                row["failure_stage"] = "basket_placement"
            else:
                row["failure_stage"] = "integration_other"
    if pipeline_error is not None:
        print(row["exception"])


def main():
    parser = argparse.ArgumentParser(description="P12 — 27 trial end-to-end formal v12")
    parser.add_argument("--source", default="scripts/07_indonesian_store_jaco_camera_NEW.py")
    parser.add_argument("--output-dir", default="data/test_logs/dosbim2_revision/P12_end_to_end")
    parser.add_argument("--limit", type=int, default=0,
                        help="N trial pertama sebagai PILOT terpisah, 0 = FORMAL 27 trial")
    parser.add_argument("--seed-base", type=int, default=120900,
                        help="seed trial = seed-base + nomor trial - 1")
    parser.add_argument("--settle-steps", type=int, default=120)
    parser.add_argument("--xy-margin", type=float, default=0.002)
    parser.add_argument("--floor-tolerance", type=float, default=0.035)
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--show-camera", action="store_true",
                        help="Tampilkan jendela POV pada GUI (lebih lambat)")
    args = parser.parse_args()
    if args.limit < 0 or args.limit > 27 or args.settle_steps < 0:
        parser.error("--limit harus 0..27 dan --settle-steps tidak boleh negatif")
    if args.xy_margin < 0 or args.floor_tolerance < 0:
        parser.error("Toleransi audit harus nonnegatif")
    if args.show_camera and not args.gui:
        parser.error("--show-camera memerlukan --gui")
    source = Path(args.source).expanduser().resolve()
    if not source.is_file():
        parser.error(f"Script simulasi utama tidak ditemukan: {source}")
    sha = hashlib.sha256(source.read_bytes()).hexdigest()
    base = Path(args.output_dir).expanduser().resolve()
    base.mkdir(parents=True, exist_ok=True)
    is_formal = args.limit == 0
    plan = trial_plan() if is_formal else trial_plan()[:args.limit]
    tag = "P12_FORMAL" if is_formal else "P12_PILOT"
    run_dir = base / (tag + "_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    run_dir.mkdir(exist_ok=False)
    (run_dir / "test_plan.json").write_text(
        json.dumps({"formal_run": is_formal, "source_sha256": sha, "seed_base": args.seed_base,
                    "trials": plan}, indent=2, ensure_ascii=False), encoding="utf-8")
    csv_path = run_dir / "pengujian_12_end_to_end.csv"
    rows = []
    print(f"=== P12 {'FORMAL' if is_formal else 'PILOT'} | {len(plan)} TRIAL ===")
    print("Source SHA256:", sha, "\nOutput:", run_dir)
    try:
        for planned in plan:
            num = planned["trial"]
            row = {key: "" for key in FIELDS}
            row.update({k: v for k, v in planned.items() if k in FIELDS})
            row.update(random_seed=args.seed_base + num - 1, source_sha256=sha,
                       task_success=False, failure_stage="")
            client = None
            t_setup = time.monotonic()
            try:
                random.seed(row["random_seed"])
                sim = import_sim(source, num)
                sim.SHOW_CAMERA_WINDOWS = bool(args.show_camera)
                sim.WRIST_POV_LIVE_DURING_MOTION = bool(args.show_camera)
                sim.VISUAL_SEARCH_LOG_DIR = str(run_dir / "debug_images" / f"trial_{num:03d}")
                sim.OBJECT_RANDOM_LOG_DIR = str(run_dir / "randomization" / f"trial_{num:03d}")
                sim.COLLISION_LOG_DIR = str(run_dir / "collision_logs" / f"trial_{num:03d}")
                client = p.connect(p.GUI if args.gui else p.DIRECT)
                if client < 0:
                    raise RuntimeError("PyBullet tidak berhasil connect")
                p.resetSimulation()
                p.setAdditionalSearchPath(pybullet_data.getDataPath())
                p.setGravity(0, 0, -9.81)
                p.setRealTimeSimulation(0)
                p.setPhysicsEngineParameter(fixedTimeStep=1/240,
                                            numSolverIterations=150, numSubSteps=2)
                objects, initial_poses = sim.create_indonesian_store_scene()
                # Sesuai main(): settle objek lalu simpan pose slot yang sudah stabil.
                for _ in range(240):
                    p.stepSimulation()
                for name, obj in objects.items():
                    pos, orn = p.getBasePositionAndOrientation(obj["id"])
                    initial_poses[name]["position"] = list(pos)
                    initial_poses[name]["orientation"] = list(orn)
                sim.reset_objects_randomized(objects, initial_poses,
                                             reason=f"P12_trial_{num:03d}_seed_{row['random_seed']}")
                robot_data = sim.load_jaco_robot()
                sim.register_mobile_platform_body(robot_data["robot_id"])
                sim.set_live_camera_context(robot_data=robot_data, objects=objects)
                sim.reset_robot(robot_data)
                for _ in range(args.settle_steps):
                    p.stepSimulation()
                target_id, start_z = audit_initial_scene(sim, objects, row)
                row["setup_sec"] = round(time.monotonic() - t_setup, 3)
                print(f"\n--- P12 trial {num}/27: {row['expected_name']} "
                      f"rep={row['repetition']} rack={row['initial_rack']} "
                      f"shelf={row['initial_shelf']} seed={row['random_seed']} ---")
                run_trial(sim, robot_data, objects, row, args)
            except KeyboardInterrupt:
                row["failure_stage"] = row.get("failure_stage") or "interrupted"
                row["exception"] = row.get("exception") or "KeyboardInterrupt"
                print("P12 dihentikan pengguna; hasil yang sudah direkam tetap disimpan.")
                raise
            except Exception as exc:
                row["failure_stage"] = row.get("failure_stage") or "setup_or_runtime_exception"
                row["exception"] = row.get("exception") or (
                    f"{type(exc).__name__}: {exc}\n{traceback.format_exc(limit=8)}")
                print(row["exception"])
            finally:
                if row["setup_sec"] == "":
                    row["setup_sec"] = round(time.monotonic() - t_setup, 3)
                rows.append(row)
                write_csv(csv_path, rows)
                summary(run_dir, source, sha, rows, len(plan), is_formal, False, args)
                if client is not None and client >= 0 and p.isConnected(client):
                    p.disconnect(client)
                print(f"P12 trial {num}: {'SUCCESS' if row['task_success'] else 'FAILED'} "
                      f"stage={row['failure_stage'] or '-'} e2e={row['end_to_end_sec']}")
        report = summary(run_dir, source, sha, rows, len(plan), is_formal, True, args)
        print("\nP12 selesai:", report["task_success"], "/", report["trial_count"],
              "| summary:", run_dir / "pengujian_12_end_to_end_summary.json")
    finally:
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main()

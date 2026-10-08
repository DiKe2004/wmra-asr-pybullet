#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P09 — Collision Rate Test Logger

Tujuan:
- Menguji apakah gerakan robot menuju target menyebabkan collision yang tidak diinginkan.
- Tahapan yang diuji:
  1. reset scene
  2. joint-1 pre-align
  3. shelf-ready
  4. pre-grasp
  5. approach
  6. attach/grasp
  7. retreat
  8. lift

Catatan:
- P09 tidak menjalankan ASR.
- P09 tidak menjalankan parser.
- P09 tidak menjalankan visual search.
- P09 tidak menjalankan handover/basket.
- P09 fokus mencatat collision/contacts selama gerak robot.
- Collision antara gripper dan target saat tahap grasp boleh dianggap expected contact.
- Collision dengan rak, objek non-target, atau lingkungan dicatat sebagai collision risk.
"""

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import re
import statistics
import sys
import time
import traceback

import pybullet as p
import pybullet_data


def now_tag():
    return time.strftime("%Y%m%d_%H%M%S")


def normalize_text(text):
    text = str(text or "").strip().lower()
    text = text.replace("_", " ")
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_shape(value):
    raw = str(value or "").strip().lower()
    v = normalize_text(value)

    if raw == "prisma_segitiga":
        return "prisma_segitiga"
    if v in ["silinder", "cylinder", "tabung", "cyl"]:
        return "silinder"
    if v in ["balok", "kotak", "box", "cube", "kubus", "rectangular", "rectangular box", "persegi panjang"]:
        return "balok"
    if v in ["prisma segitiga", "prisma", "segitiga", "triangle", "triangular prism", "prism"]:
        return "prisma_segitiga"
    return v.replace(" ", "_")


def normalize_color(value):
    v = normalize_text(value)
    if v in ["merah", "red"]:
        return "merah"
    if v in ["biru", "blue"]:
        return "biru"
    if v in ["kuning", "yellow"]:
        return "kuning"
    if v in ["hijau", "green"]:
        return "hijau"
    return v.replace(" ", "_")


def object_shape_from_name_or_meta(name, meta):
    if isinstance(meta, dict):
        value = meta.get("shape") or meta.get("shape_id") or meta.get("object_shape") or meta.get("type")
        if value:
            return normalize_shape(value)

    n = normalize_text(name)
    if "silinder" in n or "cylinder" in n:
        return "silinder"
    if "balok" in n or "kotak" in n or "box" in n:
        return "balok"
    if "segitiga" in n or "prisma" in n or "triangle" in n:
        return "prisma_segitiga"
    return ""


def object_color_from_name_or_meta(name, meta):
    if isinstance(meta, dict):
        value = meta.get("color") or meta.get("color_id") or meta.get("object_color")
        if value:
            return normalize_color(value)

    n = normalize_text(name)
    if "merah" in n or "red" in n:
        return "merah"
    if "biru" in n or "blue" in n:
        return "biru"
    if "kuning" in n or "yellow" in n:
        return "kuning"
    if "hijau" in n or "green" in n:
        return "hijau"
    return ""


def class_from_shape_color(shape, color):
    return f"{normalize_shape(shape)}_{normalize_color(color)}"


def load_sim_module(source_path, gui=False):
    spec = importlib.util.spec_from_file_location("scriptis4_main_sim", str(source_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Gagal import source simulasi: {source_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules["scriptis4_main_sim"] = module
    spec.loader.exec_module(module)

    if not gui:
        for attr, value in [
            ("SHOW_CAMERA_WINDOWS", False),
            ("WRIST_POV_LIVE_DURING_MOTION", False),
            ("COLLISION_VERBOSE_DURING_MOTION", False),
        ]:
            if hasattr(module, attr):
                try:
                    setattr(module, attr, value)
                except Exception:
                    pass

    return module


def build_body_name_map(objects, robot_data):
    mapping = {}

    robot_id = robot_data.get("robot_id")
    if robot_id is not None:
        mapping[int(robot_id)] = "robot"

    for name, meta in objects.items():
        if isinstance(meta, dict) and "id" in meta:
            try:
                mapping[int(meta["id"])] = name
            except Exception:
                pass

    # Tambahkan body lain yang bukan robot/objects sebagai environment.
    # Ini biasanya plane, rack/shelf, basket, table/chair, dsb.
    try:
        for i in range(p.getNumBodies()):
            bid = p.getBodyUniqueId(i)
            if bid not in mapping:
                try:
                    raw = p.getBodyInfo(bid)[1].decode("utf-8", errors="ignore")
                    mapping[bid] = raw if raw else f"environment_body_{bid}"
                except Exception:
                    mapping[bid] = f"environment_body_{bid}"
    except Exception:
        pass

    return mapping


def body_label(body_id, body_map):
    try:
        body_id = int(body_id)
    except Exception:
        return str(body_id)

    label = body_map.get(body_id, f"body_{body_id}")

    if label in ["", "plane"]:
        return label or "plane"

    low = normalize_text(label)
    if "plane" in low:
        return "plane"
    if "floor" in low:
        return "floor"
    if "rack" in low or "shelf" in low or "rak" in low:
        return "rack/shelf"
    if "basket" in low or "handover" in low:
        return "basket/handover_area"
    return label


def is_robot_env_collision(contact, robot_id, target_id, body_map, allow_target_contact=False):
    """
    Return:
    - collision_risk: bool
    - expected_contact: bool
    - label
    """
    body_a = contact[1]
    body_b = contact[2]
    link_a = contact[3]
    link_b = contact[4]
    normal_force = contact[9] if len(contact) > 9 else 0.0

    if body_a != robot_id and body_b != robot_id:
        return False, False, ""

    other = body_b if body_a == robot_id else body_a

    # Abaikan kontak robot dengan target jika tahap grasp/after grasp memperbolehkan.
    if allow_target_contact and target_id is not None and other == target_id:
        return False, True, f"expected target contact force={normal_force:.4f}"

    other_label = body_label(other, body_map)

    # Abaikan plane/floor kecil kalau hanya kontak base/roda/platform.
    # Jika gripper/arm nyentuh rak/objek lain tetap dihitung.
    if other_label in ["plane", "floor"]:
        return False, False, ""

    return True, False, f"robot contact with {other_label} | link {link_a}/{link_b} | force={normal_force:.4f}"


def scan_contacts(robot_id, target_id, body_map, allow_target_contact=False):
    contacts = p.getContactPoints()
    risk_events = []
    expected_events = []

    for c in contacts:
        risk, expected, label = is_robot_env_collision(
            contact=c,
            robot_id=robot_id,
            target_id=target_id,
            body_map=body_map,
            allow_target_contact=allow_target_contact,
        )

        if risk:
            risk_events.append(label)
        elif expected:
            expected_events.append(label)

    # Deduplicate while preserving order.
    risk_events = list(dict.fromkeys(risk_events))
    expected_events = list(dict.fromkeys(expected_events))
    return risk_events, expected_events


def monitored_move(sim, robot_data, target_position, stage, body_map, target_id=None,
                   allow_target_contact=False, duration=2.0, motor_force=None,
                   max_velocity=None, error_print_limit=0.35):
    """
    Wrapper move_robot_to_position.
    Karena fungsi simulasi internal sudah melakukan step, logger ini mengecek contact
    sesudah movement. Untuk monitoring detail per-step, source simulasi perlu hook khusus.
    """
    kwargs = {
        "robot_data": robot_data,
        "target_position": target_position,
        "duration": duration,
        "attached_object_id": target_id if allow_target_contact else None,
        "grasp_constraint_id": None,
        "release_on_collision": False,
        "monitor_start_delay_steps": 40,
        "error_print_limit": error_print_limit,
    }
    if motor_force is not None:
        kwargs["motor_force"] = motor_force
    if max_velocity is not None:
        kwargs["max_velocity"] = max_velocity

    result = sim.move_robot_to_position(**kwargs)

    if isinstance(result, tuple):
        error = result[0]
        dropped = bool(result[2]) if len(result) >= 3 else False
    else:
        error = result
        dropped = False

    risk_events, expected_events = scan_contacts(
        robot_id=robot_data["robot_id"],
        target_id=target_id,
        body_map=body_map,
        allow_target_contact=allow_target_contact,
    )

    return float(error), dropped, risk_events, expected_events


def build_object_index(objects, formal_colors=None):
    rows = []
    allowed_colors = None
    if formal_colors:
        allowed_colors = {normalize_color(c) for c in formal_colors}

    for name, meta in objects.items():
        shape = object_shape_from_name_or_meta(name, meta)
        color = object_color_from_name_or_meta(name, meta)

        if not shape or not color:
            continue

        if allowed_colors is not None and color not in allowed_colors:
            continue

        obj_id = meta.get("id", "") if isinstance(meta, dict) else ""
        pos, _ = p.getBasePositionAndOrientation(obj_id)

        rows.append({
            "object_name": name,
            "object_id": obj_id,
            "shape": shape,
            "color": color,
            "target_class": class_from_shape_color(shape, color),
            "object_x": pos[0],
            "object_y": pos[1],
            "object_z": pos[2],
        })

    shape_order = {"silinder": 0, "balok": 1, "prisma_segitiga": 2}
    color_order = {"merah": 0, "biru": 1, "kuning": 2, "hijau": 3}
    rows = sorted(
        rows,
        key=lambda r: (
            shape_order.get(r["shape"], 99),
            color_order.get(r["color"], 99),
            r["object_name"],
        )
    )
    return rows


def step_sim(steps, sleep=True):
    for _ in range(max(0, int(steps))):
        p.stepSimulation()
        if sleep:
            time.sleep(1 / 240)


def reset_between_trials(sim, robot_data, objects, initial_poses, settle_steps, gui=False):
    try:
        sim.reset_robot(robot_data)
    except Exception as exc:
        print("WARNING reset_robot gagal:", exc)

    try:
        sim.reset_objects(objects, initial_poses)
    except Exception as exc:
        print("WARNING reset_objects gagal:", exc)

    if hasattr(sim, "set_live_camera_context"):
        try:
            sim.set_live_camera_context(robot_data=robot_data, objects=objects, selected_name=None)
        except Exception:
            pass

    step_sim(settle_steps, sleep=gui)


def safe_release_after_trial(sim, robot_data, object_id, grasp_constraint):
    try:
        if grasp_constraint is not None and hasattr(sim, "release_object"):
            sim.release_object(grasp_constraint)
        elif grasp_constraint is not None:
            p.removeConstraint(grasp_constraint)
    except Exception:
        pass

    try:
        if hasattr(sim, "set_object_transport_mode"):
            sim.set_object_transport_mode(object_id, enabled=False)
    except Exception:
        pass

    try:
        if hasattr(sim, "set_gripper_open"):
            sim.set_gripper_open(robot_data)
    except Exception:
        pass

    try:
        p.resetBaseVelocity(object_id, [0, 0, 0], [0, 0, 0])
    except Exception:
        pass


def get_thresholds(sim, shelf_index):
    ready_limit = getattr(sim, "READY_ERROR_LIMIT", 0.42)
    pre_limit = getattr(sim, "REACH_ERROR_LIMIT", 0.26)
    if shelf_index == 0:
        approach_limit = getattr(sim, "BOTTOM_APPROACH_ERROR_LIMIT", 0.28)
    else:
        approach_limit = getattr(sim, "APPROACH_ERROR_LIMIT", 0.24)
    return ready_limit, pre_limit, approach_limit


def run_one_collision_trial(sim, robot_data, objects, target_name, body_map):
    object_id = objects[target_name]["id"]
    object_start_pos, _ = p.getBasePositionAndOrientation(object_id)

    if not hasattr(sim, "get_shelf_aware_motion_points"):
        raise RuntimeError("Source simulasi tidak punya fungsi get_shelf_aware_motion_points().")
    if not hasattr(sim, "move_robot_to_position"):
        raise RuntimeError("Source simulasi tidak punya fungsi move_robot_to_position().")

    points = sim.get_shelf_aware_motion_points(object_start_pos)
    shelf_index = points["shelf_index"]
    ready_limit, pre_limit, approach_limit = get_thresholds(sim, shelf_index)

    all_collision_events = []
    all_expected_events = []
    stage_collision_map = {}

    result = {
        "shelf_index": shelf_index,
        "object_start_x": round(float(object_start_pos[0]), 5),
        "object_start_y": round(float(object_start_pos[1]), 5),
        "object_start_z": round(float(object_start_pos[2]), 5),
        "ready_error": "",
        "pre_grasp_error": "",
        "approach_error": "",
        "retreat_error": "",
        "lift_error": "",
        "ready_success": "Tidak",
        "pre_grasp_success": "Tidak",
        "approach_success": "Tidak",
        "grasp_attach_success": "Tidak",
        "retreat_success": "Tidak",
        "lift_success": "Tidak",
        "collision_detected": "Tidak",
        "collision_event_count": 0,
        "expected_contact_count": 0,
        "collision_stages": "",
        "collision_events_json": "[]",
        "expected_contacts_json": "[]",
        "collision_free_success": "Tidak",
        "failure_stage": "",
        "catatan": "",
        "grasp_constraint_id": "",
    }

    grasp_constraint = None

    print("\n=== P09 COLLISION TRIAL ===")
    print("Target:", target_name)
    print("Object start:", object_start_pos)
    print("Shelf index:", shelf_index)

    def add_stage_contacts(stage, risk_events, expected_events):
        nonlocal all_collision_events, all_expected_events, stage_collision_map
        if risk_events:
            stage_collision_map.setdefault(stage, [])
            stage_collision_map[stage].extend(risk_events)
            for e in risk_events:
                all_collision_events.append({"stage": stage, "event": e})
        if expected_events:
            for e in expected_events:
                all_expected_events.append({"stage": stage, "event": e})

    # baseline contacts setelah reset
    risk, exp = scan_contacts(robot_data["robot_id"], object_id, body_map, allow_target_contact=False)
    add_stage_contacts("baseline_after_reset", risk, exp)

    # 0. Joint-1 pre-align
    if hasattr(sim, "rotate_joint1_to_face_object"):
        sim.rotate_joint1_to_face_object(
            robot_data,
            object_start_pos,
            duration=getattr(sim, "JOINT1_ALIGN_DURATION", 0.8)
        )
        risk, exp = scan_contacts(robot_data["robot_id"], object_id, body_map, allow_target_contact=False)
        add_stage_contacts("joint1_prealign", risk, exp)

    # 1. Shelf-ready
    ready_error, dropped, risk, exp = monitored_move(
        sim=sim,
        robot_data=robot_data,
        target_position=points["ready"],
        stage="ready",
        body_map=body_map,
        target_id=object_id,
        allow_target_contact=False,
        duration=1.8,
        error_print_limit=ready_limit,
    )
    add_stage_contacts("shelf_ready", risk, exp)
    result["ready_error"] = round(float(ready_error), 5)
    result["ready_success"] = "Ya" if ready_error <= ready_limit else "Tidak"

    if ready_error > ready_limit:
        result["failure_stage"] = "shelf_ready"
        result["catatan"] = "Shelf-ready belum reachable"
        return result, grasp_constraint, all_collision_events, all_expected_events, stage_collision_map

    # 2. Pre-grasp
    if hasattr(sim, "set_gripper_open"):
        sim.set_gripper_open(robot_data)

    pre_error, dropped, risk, exp = monitored_move(
        sim=sim,
        robot_data=robot_data,
        target_position=points["pre_grasp"],
        stage="pre_grasp",
        body_map=body_map,
        target_id=object_id,
        allow_target_contact=False,
        duration=2.5,
        motor_force=190,
        max_velocity=0.18,
        error_print_limit=pre_limit,
    )
    add_stage_contacts("pre_grasp", risk, exp)
    result["pre_grasp_error"] = round(float(pre_error), 5)
    result["pre_grasp_success"] = "Ya" if pre_error <= pre_limit else "Tidak"

    if pre_error > pre_limit:
        result["failure_stage"] = "pre_grasp"
        result["catatan"] = "Pre-grasp belum reachable"
        return result, grasp_constraint, all_collision_events, all_expected_events, stage_collision_map

    # 3. Approach
    approach_error, dropped, risk, exp = monitored_move(
        sim=sim,
        robot_data=robot_data,
        target_position=points["approach"],
        stage="approach",
        body_map=body_map,
        target_id=object_id,
        allow_target_contact=True,
        duration=2.2,
        motor_force=170,
        max_velocity=0.12,
        error_print_limit=approach_limit,
    )
    add_stage_contacts("approach", risk, exp)
    result["approach_error"] = round(float(approach_error), 5)
    result["approach_success"] = "Ya" if approach_error <= approach_limit else "Tidak"

    if approach_error > approach_limit:
        result["failure_stage"] = "approach"
        result["catatan"] = "Approach terlalu jauh dari objek"
        return result, grasp_constraint, all_collision_events, all_expected_events, stage_collision_map

    # 4. Attach / grasp
    if not hasattr(sim, "attach_object_to_end_effector"):
        raise RuntimeError("Source simulasi tidak punya fungsi attach_object_to_end_effector().")

    grasp_constraint = sim.attach_object_to_end_effector(
        robot_data=robot_data,
        object_id=object_id
    )
    result["grasp_constraint_id"] = str(grasp_constraint) if grasp_constraint is not None else ""
    result["grasp_attach_success"] = "Ya" if grasp_constraint is not None else "Tidak"

    risk, exp = scan_contacts(robot_data["robot_id"], object_id, body_map, allow_target_contact=True)
    add_stage_contacts("grasp_attach", risk, exp)

    if grasp_constraint is None:
        result["failure_stage"] = "grasp_attach"
        result["catatan"] = "Objek tidak berhasil di-grasp/attach"
        return result, grasp_constraint, all_collision_events, all_expected_events, stage_collision_map

    step_sim(60, sleep=True)

    # 5. Retreat
    retreat_result = sim.move_robot_to_position(
        robot_data=robot_data,
        target_position=points["retreat"],
        duration=2.0,
        attached_object_id=object_id,
        grasp_constraint_id=grasp_constraint,
        release_on_collision=False,
        monitor_start_delay_steps=120,
        error_print_limit=0.35
    )
    if isinstance(retreat_result, tuple):
        retreat_error = retreat_result[0]
        grasp_constraint = retreat_result[1] if len(retreat_result) > 1 else grasp_constraint
        dropped = bool(retreat_result[2]) if len(retreat_result) > 2 else False
    else:
        retreat_error = retreat_result
        dropped = False

    risk, exp = scan_contacts(robot_data["robot_id"], object_id, body_map, allow_target_contact=True)
    add_stage_contacts("retreat", risk, exp)

    result["retreat_error"] = round(float(retreat_error), 5)
    result["retreat_success"] = "Ya" if not dropped else "Tidak"

    if dropped:
        result["failure_stage"] = "retreat_drop"
        result["catatan"] = "Objek lepas saat retreat"
        return result, grasp_constraint, all_collision_events, all_expected_events, stage_collision_map

    # 6. Lift
    lift_result = sim.move_robot_to_position(
        robot_data=robot_data,
        target_position=points["lift"],
        duration=2.5,
        attached_object_id=object_id,
        grasp_constraint_id=grasp_constraint,
        release_on_collision=False,
        monitor_start_delay_steps=40,
        error_print_limit=0.35
    )
    if isinstance(lift_result, tuple):
        lift_error = lift_result[0]
        grasp_constraint = lift_result[1] if len(lift_result) > 1 else grasp_constraint
        dropped = bool(lift_result[2]) if len(lift_result) > 2 else False
    else:
        lift_error = lift_result
        dropped = False

    risk, exp = scan_contacts(robot_data["robot_id"], object_id, body_map, allow_target_contact=True)
    add_stage_contacts("lift", risk, exp)

    result["lift_error"] = round(float(lift_error), 5)
    result["lift_success"] = "Ya" if not dropped else "Tidak"

    if dropped:
        result["failure_stage"] = "lift_drop"
        result["catatan"] = "Objek lepas saat lift"
        return result, grasp_constraint, all_collision_events, all_expected_events, stage_collision_map

    result["failure_stage"] = ""
    result["catatan"] = "Gerak grasp-lift selesai; collision dianalisis dari contact log"
    return result, grasp_constraint, all_collision_events, all_expected_events, stage_collision_map


def write_csv(path, rows):
    fields = [
        "no",
        "repeat",
        "object_name",
        "target_class",
        "shape",
        "color",
        "shelf_index",
        "object_start_x",
        "object_start_y",
        "object_start_z",
        "ready_error",
        "pre_grasp_error",
        "approach_error",
        "retreat_error",
        "lift_error",
        "ready_success",
        "pre_grasp_success",
        "approach_success",
        "grasp_attach_success",
        "retreat_success",
        "lift_success",
        "collision_detected",
        "collision_event_count",
        "expected_contact_count",
        "collision_stages",
        "collision_events_json",
        "expected_contacts_json",
        "collision_free_success",
        "motion_success",
        "elapsed_sec",
        "status_final",
        "failure_stage",
        "catatan",
        "grasp_constraint_id",
        "error",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for i, r in enumerate(rows, start=1):
            row = dict(r)
            row["no"] = i
            w.writerow(row)


def percent(n, d):
    return (n / d * 100.0) if d else 0.0


def md_escape(value):
    return str(value if value is not None else "").replace("|", "\\|")


def avg(values):
    vals = []
    for v in values:
        try:
            vals.append(float(v))
        except Exception:
            pass
    return statistics.mean(vals) if vals else None


def write_summary_md(path, rows, args, csv_path):
    total = len(rows)
    motion_ok = sum(1 for r in rows if r["motion_success"] == "Ya")
    collision_free = sum(1 for r in rows if r["collision_free_success"] == "Ya")
    collision_trials = sum(1 for r in rows if r["collision_detected"] == "Ya")
    final_ok = sum(1 for r in rows if r["status_final"] == "Berhasil")
    avg_time = avg([r.get("elapsed_sec") for r in rows])

    by_stage = {}
    for r in rows:
        stages = str(r.get("collision_stages", "")).split(";")
        for s in stages:
            s = s.strip()
            if not s:
                continue
            by_stage[s] = by_stage.get(s, 0) + 1

    by_shelf = {}
    for r in rows:
        shelf = str(r.get("shelf_index", ""))
        if shelf not in by_shelf:
            by_shelf[shelf] = {"total": 0, "collision_free": 0, "collision": 0}
        by_shelf[shelf]["total"] += 1
        if r["collision_free_success"] == "Ya":
            by_shelf[shelf]["collision_free"] += 1
        if r["collision_detected"] == "Ya":
            by_shelf[shelf]["collision"] += 1

    lines = [
        "# P09 — Pengujian Collision Rate",
        "",
        f"- Waktu pengujian: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Source simulasi: `{args.source}`",
        f"- CSV detail: `{csv_path}`",
        f"- Repeat: `{args.repeat}`",
        f"- Mode PyBullet: `{'GUI' if args.gui else 'DIRECT'}`",
        "",
        "## Ringkasan Metrik",
        "",
        f"- Total trial: {total}",
        f"- Motion success: {motion_ok}/{total} = {percent(motion_ok, total):.2f}%",
        f"- Collision-free success: {collision_free}/{total} = {percent(collision_free, total):.2f}%",
        f"- Collision detected rate: {collision_trials}/{total} = {percent(collision_trials, total):.2f}%",
        f"- Final P09 success: {final_ok}/{total} = {percent(final_ok, total):.2f}%",
        f"- Average elapsed time: {avg_time:.3f} detik" if avg_time is not None else "- Average elapsed time: -",
        "",
        "## Collision Stage",
        "",
        "| Stage | Jumlah Trial dengan Collision |",
        "|---|---:|",
    ]

    if by_stage:
        for st, n in sorted(by_stage.items(), key=lambda x: x[0]):
            lines.append(f"| {md_escape(st)} | {n} |")
    else:
        lines.append("| - | 0 |")

    lines += [
        "",
        "## Ringkasan per Level Rak",
        "",
        "| Shelf Index | Trial | Collision-free | Collision | Collision-free Rate |",
        "|---:|---:|---:|---:|---:|",
    ]

    for shelf, data in sorted(by_shelf.items(), key=lambda x: x[0]):
        lines.append(
            f"| {shelf} | {data['total']} | {data['collision_free']} | {data['collision']} | "
            f"{percent(data['collision_free'], data['total']):.2f}% |"
        )

    lines += [
        "",
        "## Detail Trial",
        "",
        "| No | Target | Rak | Motion | Collision | Status | Collision Stage | Catatan |",
        "|---:|---|---:|---|---|---|---|---|",
    ]

    for i, r in enumerate(rows[:60], start=1):
        lines.append(
            f"| {i} | {md_escape(r['object_name'])} | {r.get('shelf_index','')} | "
            f"{r['motion_success']} | {r['collision_detected']} | {r['status_final']} | "
            f"{md_escape(r.get('collision_stages',''))} | {md_escape(r.get('catatan',''))} |"
        )

    lines += [
        "",
        "## Narasi Singkat Bab 4",
        "",
        (
            "Pengujian collision rate dilakukan untuk mengevaluasi apakah lintasan robot saat "
            "mendekati objek, melakukan grasp, retreat, dan lift menimbulkan kontak yang tidak "
            "diinginkan dengan rak, objek lain, atau lingkungan simulasi."
        ),
        "",
        (
            "Pada pengujian ini, kontak antara gripper dan objek target pada tahap approach/grasp "
            "dipisahkan sebagai expected contact. Sementara itu, kontak dengan rak, objek non-target, "
            "atau komponen lingkungan lain dicatat sebagai collision risk."
        ),
        "",
        (
            "Satu trial dinyatakan berhasil apabila gerakan sub-sistem dapat diselesaikan dan tidak "
            "terdeteksi collision risk. Pengujian ini penting karena gerakan yang berhasil secara IK "
            "atau grasp belum tentu aman apabila lintasannya masih menyentuh rak atau objek lain."
        ),
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="scripts/07_indonesian_store_jaco_camera_NEW.py")
    parser.add_argument("--output_dir", default="data/test_logs_timestamped")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--repeat", type=int, default=3)
    parser.add_argument("--limit_targets", type=int, default=0)
    parser.add_argument("--settle_steps", type=int, default=120)
    parser.add_argument("--formal_colors", default="", help="Contoh: merah,biru,kuning. Kosong = semua warna.")
    args = parser.parse_args()

    root = Path.cwd()
    source_path = Path(args.source)
    if not source_path.is_absolute():
        source_path = root / source_path

    if not source_path.exists():
        raise FileNotFoundError(f"Source simulasi tidak ditemukan: {source_path}")

    sim = load_sim_module(source_path, gui=args.gui)

    run_dir = root / args.output_dir / f"P09_collision_rate_{now_tag()}"
    csv_path = run_dir / "pengujian_09_collision_rate.csv"
    md_path = run_dir / "pengujian_09_collision_rate_summary.md"
    json_path = run_dir / "pengujian_09_collision_rate_summary.json"
    run_dir.mkdir(parents=True, exist_ok=True)

    client = p.connect(p.GUI if args.gui else p.DIRECT)
    if client < 0:
        raise RuntimeError("Gagal connect PyBullet")

    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(0)
    p.setPhysicsEngineParameter(
        fixedTimeStep=1.0 / 240.0,
        numSolverIterations=150,
        numSubSteps=2,
    )

    results = []

    try:
        objects, initial_poses = sim.create_indonesian_store_scene()

        step_sim(240, sleep=args.gui)

        # Simpan ulang pose awal setelah objek stabil.
        for name, obj in objects.items():
            pos, orn = p.getBasePositionAndOrientation(obj["id"])
            initial_poses[name]["position"] = list(pos)
            initial_poses[name]["orientation"] = list(orn)

        robot_data = sim.load_jaco_robot()
        sim.reset_robot(robot_data)

        body_map = build_body_name_map(objects, robot_data)

        if hasattr(sim, "set_live_camera_context"):
            sim.set_live_camera_context(robot_data=robot_data, objects=objects, selected_name=None)

        formal_colors = None
        if args.formal_colors.strip():
            formal_colors = [c.strip() for c in args.formal_colors.split(",") if c.strip()]

        targets = build_object_index(objects, formal_colors=formal_colors)

        if args.limit_targets and args.limit_targets > 0:
            targets = targets[:args.limit_targets]

        trial_plan = []
        for rep in range(1, max(1, int(args.repeat)) + 1):
            for target in targets:
                row = dict(target)
                row["repeat"] = rep
                trial_plan.append(row)

        print("\n=== P09 COLLISION RATE TEST START ===")
        print("Source:", source_path)
        print("Targets:", len(targets))
        print("Trials:", len(trial_plan))
        print("Repeat:", args.repeat)
        print("Mode:", "GUI" if args.gui else "DIRECT")
        print("Formal colors:", formal_colors if formal_colors else "all colors in scene")

        for idx, t in enumerate(trial_plan, start=1):
            started = time.time()
            object_name = t["object_name"]
            object_id = objects[object_name]["id"]
            grasp_constraint = None

            try:
                reset_between_trials(
                    sim=sim,
                    robot_data=robot_data,
                    objects=objects,
                    initial_poses=initial_poses,
                    settle_steps=args.settle_steps,
                    gui=args.gui,
                )

                # Body map setelah reset/object creation tetap sama, tapi refresh untuk aman.
                body_map = build_body_name_map(objects, robot_data)

                if hasattr(sim, "set_live_camera_context"):
                    sim.set_live_camera_context(
                        robot_data=robot_data,
                        objects=objects,
                        selected_name=object_name
                    )

                trial_result, grasp_constraint, collision_events, expected_contacts, stage_collision_map = run_one_collision_trial(
                    sim=sim,
                    robot_data=robot_data,
                    objects=objects,
                    target_name=object_name,
                    body_map=body_map,
                )

                motion_success = (
                    trial_result.get("ready_success") == "Ya"
                    and trial_result.get("pre_grasp_success") == "Ya"
                    and trial_result.get("approach_success") == "Ya"
                    and trial_result.get("grasp_attach_success") == "Ya"
                    and trial_result.get("retreat_success") == "Ya"
                    and trial_result.get("lift_success") == "Ya"
                )

                collision_detected = len(collision_events) > 0
                collision_free = motion_success and not collision_detected
                status_final = "Berhasil" if collision_free else "Gagal"

                trial_result["collision_detected"] = "Ya" if collision_detected else "Tidak"
                trial_result["collision_event_count"] = len(collision_events)
                trial_result["expected_contact_count"] = len(expected_contacts)
                trial_result["collision_stages"] = ";".join(sorted(stage_collision_map.keys()))
                trial_result["collision_events_json"] = json.dumps(collision_events, ensure_ascii=False)
                trial_result["expected_contacts_json"] = json.dumps(expected_contacts, ensure_ascii=False)
                trial_result["collision_free_success"] = "Ya" if collision_free else "Tidak"
                trial_result["motion_success"] = "Ya" if motion_success else "Tidak"
                trial_result["status_final"] = status_final
                trial_result["error"] = ""

                if collision_detected and motion_success:
                    trial_result["catatan"] = "Motion selesai tetapi terdeteksi collision risk"
                elif collision_free:
                    trial_result["catatan"] = "Motion selesai tanpa collision risk"

            except Exception:
                trial_result = {
                    "shelf_index": "",
                    "object_start_x": "",
                    "object_start_y": "",
                    "object_start_z": "",
                    "ready_error": "",
                    "pre_grasp_error": "",
                    "approach_error": "",
                    "retreat_error": "",
                    "lift_error": "",
                    "ready_success": "Tidak",
                    "pre_grasp_success": "Tidak",
                    "approach_success": "Tidak",
                    "grasp_attach_success": "Tidak",
                    "retreat_success": "Tidak",
                    "lift_success": "Tidak",
                    "collision_detected": "Tidak",
                    "collision_event_count": 0,
                    "expected_contact_count": 0,
                    "collision_stages": "",
                    "collision_events_json": "[]",
                    "expected_contacts_json": "[]",
                    "collision_free_success": "Tidak",
                    "motion_success": "Tidak",
                    "status_final": "Gagal",
                    "failure_stage": "exception",
                    "catatan": "Exception saat pengujian collision",
                    "grasp_constraint_id": str(grasp_constraint) if grasp_constraint is not None else "",
                    "error": traceback.format_exc(),
                }

            finally:
                try:
                    safe_release_after_trial(sim, robot_data, object_id, grasp_constraint)
                except Exception:
                    pass

            row = {
                "repeat": t["repeat"],
                "object_name": object_name,
                "target_class": t["target_class"],
                "shape": t["shape"],
                "color": t["color"],
                **trial_result,
                "elapsed_sec": round(time.time() - started, 4),
            }
            results.append(row)

            print(
                f"[{idx}/{len(trial_plan)}] {row['status_final']} | "
                f"target={object_name} | collision={row.get('collision_detected')} | "
                f"motion={row.get('motion_success')} | stage={row.get('collision_stages','') or '-'}"
            )

    finally:
        if p.isConnected(client):
            p.disconnect()

    write_csv(csv_path, results)
    write_summary_md(md_path, results, args, csv_path)

    total = len(results)
    motion_ok = sum(1 for r in results if r["motion_success"] == "Ya")
    collision_free = sum(1 for r in results if r["collision_free_success"] == "Ya")
    collision_trials = sum(1 for r in results if r["collision_detected"] == "Ya")
    final_ok = sum(1 for r in results if r["status_final"] == "Berhasil")

    summary = {
        "timestamp": now_tag(),
        "source": str(source_path),
        "output_dir": str(run_dir),
        "csv": str(csv_path),
        "markdown": str(md_path),
        "total": total,
        "repeat": args.repeat,
        "motion_success_percent": round(percent(motion_ok, total), 4),
        "collision_free_success_percent": round(percent(collision_free, total), 4),
        "collision_detected_rate_percent": round(percent(collision_trials, total), 4),
        "final_p09_success_percent": round(percent(final_ok, total), 4),
        "motion_success": motion_ok,
        "collision_free_success": collision_free,
        "collision_detected": collision_trials,
        "final_success": final_ok,
        "mode": "GUI" if args.gui else "DIRECT",
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== P09 COLLISION RATE TEST DONE ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

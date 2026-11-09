#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P03 — Command Parsing Test Logger

Tujuan:
- Menguji modul parsing perintah sebagai pengujian satuan.
- Input: teks command/ground truth dari metadata, misalnya prompt_text.
- Output: shape, color, command_class, valid/invalid, akurasi shape/color/class.

Catatan:
- P03 TIDAK menjalankan ASR.
- P03 TIDAK menjalankan robot/visual search.
- P03 hanya menguji apakah teks dapat dipahami menjadi bentuk + warna.

Untuk metadata Scriptis:
- text_col = prompt_text
- shape_col = shape
- color_col = color
- class_col = command_class
"""

import argparse
import csv
import json
from pathlib import Path
import re
import time

PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)

SHAPE_ALIASES = {
    "silinder": ["silinder", "cylinder", "cyl", "tabung", "benda silinder", "objek silinder"],
    "balok": ["balok", "kotak", "box", "cube", "kubus", "persegi panjang", "benda kotak", "objek kotak", "rectangular box", "rectangular"],
    "prisma_segitiga": ["prisma segitiga", "prisma_segitiga", "prisma triangular", "triangular prism", "segitiga", "triangle", "prism", "prisma", "benda segitiga", "objek segitiga"],
}

COLOR_ALIASES = {
    "merah": ["merah", "red"],
    "biru": ["biru", "blue"],
    "kuning": ["kuning", "yellow"],
    "hijau": ["hijau", "green"],
}


def now_tag():
    return time.strftime("%Y%m%d_%H%M%S")


def normalize_text(text: str) -> str:
    text = str(text or "").strip().lower()
    text = text.replace("_", " ")
    text = PUNCT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def normalize_shape(value: str) -> str:
    value_norm = normalize_text(value)
    if value_norm in ["prisma segitiga", "prisma", "segitiga", "triangle", "triangular prism", "prism"]:
        return "prisma_segitiga"
    if value_norm in ["silinder", "cylinder", "tabung", "cyl"]:
        return "silinder"
    if value_norm in ["balok", "kotak", "box", "cube", "kubus", "rectangular box", "rectangular", "persegi panjang"]:
        return "balok"
    raw = str(value or "").strip().lower()
    if raw == "prisma_segitiga":
        return "prisma_segitiga"
    return value_norm.replace(" ", "_")


def normalize_color(value: str) -> str:
    value_norm = normalize_text(value)
    if value_norm in ["merah", "red"]:
        return "merah"
    if value_norm in ["biru", "blue"]:
        return "biru"
    if value_norm in ["kuning", "yellow"]:
        return "kuning"
    if value_norm in ["hijau", "green"]:
        return "hijau"
    return value_norm.replace(" ", "_")


def contains_alias(text_norm: str, alias: str) -> bool:
    alias_norm = normalize_text(alias)
    if not alias_norm:
        return False
    return f" {alias_norm} " in f" {text_norm} "


def find_shape(text_norm: str):
    matches = []
    for canonical, aliases in SHAPE_ALIASES.items():
        for alias in sorted(aliases, key=lambda x: len(normalize_text(x)), reverse=True):
            if contains_alias(text_norm, alias):
                matches.append({"canonical": canonical, "alias": alias, "alias_len": len(normalize_text(alias))})
                break
    if not matches:
        return None, None, []
    matches = sorted(matches, key=lambda m: m["alias_len"], reverse=True)
    return matches[0]["canonical"], matches[0]["alias"], matches


def find_color(text_norm: str):
    matches = []
    for canonical, aliases in COLOR_ALIASES.items():
        for alias in sorted(aliases, key=lambda x: len(normalize_text(x)), reverse=True):
            if contains_alias(text_norm, alias):
                matches.append({"canonical": canonical, "alias": alias, "alias_len": len(normalize_text(alias))})
                break
    if not matches:
        return None, None, []
    matches = sorted(matches, key=lambda m: m["alias_len"], reverse=True)
    return matches[0]["canonical"], matches[0]["alias"], matches


def parse_command_text(text: str):
    text_norm = normalize_text(text)
    shape, matched_shape_alias, shape_matches = find_shape(text_norm)
    color, matched_color_alias, color_matches = find_color(text_norm)
    valid = bool(shape and color)
    command_class = f"{shape}_{color}" if valid else ""

    if valid:
        status = "valid"
        reason = "shape dan color ditemukan"
    elif not shape and not color:
        status = "invalid"
        reason = "shape dan color tidak ditemukan"
    elif not shape:
        status = "invalid"
        reason = "shape tidak ditemukan"
    else:
        status = "invalid"
        reason = "color tidak ditemukan"

    return {
        "input_text": text,
        "normalized_text": text_norm,
        "parse_status": status,
        "valid": valid,
        "shape": shape or "",
        "color": color or "",
        "command_class": command_class,
        "matched_shape_alias": matched_shape_alias or "",
        "matched_color_alias": matched_color_alias or "",
        "reason": reason,
        "shape_matches": shape_matches,
        "color_matches": color_matches,
    }


def read_metadata(metadata_path: Path):
    with metadata_path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if not rows:
        raise RuntimeError(f"Metadata kosong: {metadata_path}")
    return rows, fieldnames


def pick_col(fieldnames, candidates, override=None, required=True, label="kolom"):
    if override:
        if override not in fieldnames:
            raise RuntimeError(f"{label} override tidak ditemukan: {override}\nKolom tersedia: {', '.join(fieldnames)}")
        return override
    lower = {c.lower(): c for c in fieldnames}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    if required:
        raise RuntimeError(f"{label} tidak ditemukan.\nKolom tersedia: {', '.join(fieldnames)}\nKandidat otomatis: {', '.join(candidates)}")
    return None


def make_alias_extra_cases():
    cases = [
        ("alias_001", "ambil cylinder merah", "silinder", "merah"),
        ("alias_002", "ambil red cylinder", "silinder", "merah"),
        ("alias_003", "tolong ambil box biru", "balok", "biru"),
        ("alias_004", "ambil blue box", "balok", "biru"),
        ("alias_005", "ambil triangular prism kuning", "prisma_segitiga", "kuning"),
        ("alias_006", "ambil triangle yellow", "prisma_segitiga", "kuning"),
        ("alias_007", "ambil segitiga hijau", "prisma_segitiga", "hijau"),
        ("alias_008", "ambil green triangle", "prisma_segitiga", "hijau"),
    ]
    return [{
        "audio_id": audio_id, "prompt_text": text, "shape": shape, "color": color,
        "command_class": f"{shape}_{color}", "audio_condition": "alias_test",
        "prompt_variant": "alias_test", "repetition": "0", "source_type": "alias_extra",
    } for audio_id, text, shape, color in cases]


def write_csv(path: Path, rows):
    fields = [
        "no", "source_type", "sample_id", "audio_condition", "prompt_variant", "repetition",
        "input_text", "normalized_text", "shape_gt", "color_gt", "command_class_gt",
        "shape_pred", "color_pred", "command_class_pred", "parse_status", "command_valid",
        "shape_correct", "color_correct", "command_class_correct", "matched_shape_alias",
        "matched_color_alias", "reason", "status_final", "catatan", "parse_detail_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for i, row in enumerate(rows, start=1):
            r = dict(row)
            r["no"] = i
            w.writerow(r)


def percent(n, d):
    return (n / d * 100.0) if d else 0.0


def write_summary_md(path: Path, rows, args, csv_path: Path, used_cols):
    total = len(rows)
    valid = sum(1 for r in rows if r["command_valid"] == "Ya")
    shape_ok = sum(1 for r in rows if r["shape_correct"] == "Ya")
    color_ok = sum(1 for r in rows if r["color_correct"] == "Ya")
    class_ok = sum(1 for r in rows if r["command_class_correct"] == "Ya")
    final_ok = sum(1 for r in rows if r["status_final"] == "Benar")
    invalid_rows = [r for r in rows if r["command_valid"] != "Ya"]
    wrong_rows = [r for r in rows if r["status_final"] != "Benar"]

    lines = [
        "# P03 — Pengujian Satuan Parsing Perintah",
        "",
        f"- Waktu pengujian: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Metadata: `{args.metadata}`",
        f"- CSV detail: `{csv_path}`",
        f"- Unique prompts: `{args.unique_prompts}`",
        f"- Include alias extra: `{args.include_alias_tests}`",
        "",
        "## Kolom Metadata yang Dipakai",
        "",
        f"- Kolom teks perintah: `{used_cols.get('text_col')}`",
        f"- Kolom shape GT: `{used_cols.get('shape_col')}`",
        f"- Kolom color GT: `{used_cols.get('color_col')}`",
        f"- Kolom command class GT: `{used_cols.get('class_col')}`",
        "",
        "## Ringkasan Metrik",
        "",
        f"- Total command diuji: {total}",
        f"- Command valid rate: {valid}/{total} = {percent(valid, total):.2f}%",
        f"- Shape accuracy: {shape_ok}/{total} = {percent(shape_ok, total):.2f}%",
        f"- Color accuracy: {color_ok}/{total} = {percent(color_ok, total):.2f}%",
        f"- Command class accuracy: {class_ok}/{total} = {percent(class_ok, total):.2f}%",
        f"- Final parser accuracy: {final_ok}/{total} = {percent(final_ok, total):.2f}%",
        "",
        "## Contoh Hasil",
        "",
        "| No | Input Text | Shape GT | Color GT | Shape Pred | Color Pred | Status |",
        "|---:|---|---|---|---|---|---|",
    ]
    for idx, r in enumerate(rows[:15], start=1):
        inp = str(r["input_text"]).replace("|", "\\|")
        lines.append(f"| {idx} | {inp} | {r['shape_gt']} | {r['color_gt']} | {r['shape_pred']} | {r['color_pred']} | {r['status_final']} |")

    if invalid_rows:
        lines += ["", "## Command Tidak Valid", "", "| No | Input Text | Reason |", "|---:|---|---|"]
        for idx, r in enumerate(invalid_rows[:20], start=1):
            inp = str(r["input_text"]).replace("|", "\\|")
            reason = str(r["reason"]).replace("|", "\\|")
            lines.append(f"| {idx} | {inp} | {reason} |")

    if wrong_rows:
        lines += ["", "## Contoh Kesalahan Parser", "", "| No | Input Text | GT | Prediksi | Catatan |", "|---:|---|---|---|---|"]
        for idx, r in enumerate(wrong_rows[:20], start=1):
            inp = str(r["input_text"]).replace("|", "\\|")
            gt = f"{r['shape_gt']}_{r['color_gt']}"
            pred = f"{r['shape_pred']}_{r['color_pred']}"
            cat = str(r["catatan"]).replace("|", "\\|")
            lines.append(f"| {idx} | {inp} | {gt} | {pred} | {cat} |")

    lines += [
        "",
        "## Narasi Singkat Bab 4",
        "",
        "Pengujian satuan parsing perintah dilakukan untuk mengevaluasi kemampuan sistem dalam mengekstraksi informasi bentuk dan warna dari teks perintah. Pada tahap ini, input yang digunakan berupa teks perintah, sehingga modul ASR dan modul robot belum dilibatkan.",
        "",
        "Hasil parsing dinyatakan benar apabila bentuk dan warna yang diprediksi sama dengan ground truth. Selain itu, sistem juga membentuk kelas perintah berdasarkan kombinasi bentuk dan warna, misalnya `silinder_merah` atau `prisma_segitiga_kuning`.",
        "",
        "Pengujian ini penting karena hasil transkripsi ASR tidak selalu sama persis dengan ground truth. Beberapa kata dapat muncul dalam variasi bahasa Inggris atau sinonim, misalnya `cylinder`, `box`, `triangle`, atau `kotak`. Istilah `kotak` dan `balok` diperlakukan sebagai kelas bentuk yang sama. Dengan adanya normalisasi kata, variasi tersebut tetap dapat dipetakan ke bentuk target yang sesuai.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/audio_dataset/metadata.csv")
    parser.add_argument("--output_dir", default="data/test_logs_timestamped")
    parser.add_argument("--text_col", default=None)
    parser.add_argument("--shape_col", default=None)
    parser.add_argument("--color_col", default=None)
    parser.add_argument("--class_col", default=None)
    parser.add_argument("--id_col", default=None)
    parser.add_argument("--condition", default="all", help="Filter audio_condition, pakai all untuk semua")
    parser.add_argument("--unique_prompts", action="store_true", help="Gunakan prompt unik saja")
    parser.add_argument("--include_alias_tests", action="store_true", help="Tambah kasus cylinder/box/triangle")
    parser.add_argument("--show_columns", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    root = Path.cwd()
    metadata_path = Path(args.metadata)
    if not metadata_path.is_absolute():
        metadata_path = root / metadata_path

    run_dir = root / args.output_dir / f"P03_command_parsing_{now_tag()}"
    run_dir.mkdir(parents=True, exist_ok=True)
    csv_path = run_dir / "pengujian_03_command_parsing.csv"
    md_path = run_dir / "pengujian_03_command_parsing_summary.md"
    json_path = run_dir / "pengujian_03_command_parsing_summary.json"

    rows_meta, fieldnames = read_metadata(metadata_path)

    if args.show_columns:
        print("\n=== METADATA COLUMNS ===")
        for i, col in enumerate(fieldnames, start=1):
            print(f"{i:02d}. {col}")
        print("========================\n")

    text_col = pick_col(fieldnames, ["prompt_text", "command", "command_text", "text", "sentence", "ground_truth", "gt", "teks", "perintah", "kalimat_perintah"], override=args.text_col, label="Kolom teks perintah")
    shape_col = pick_col(fieldnames, ["shape", "shape_gt", "target_shape", "bentuk"], override=args.shape_col, label="Kolom shape GT")
    color_col = pick_col(fieldnames, ["color", "color_gt", "target_color", "warna"], override=args.color_col, label="Kolom color GT")
    class_col = pick_col(fieldnames, ["command_class", "class", "target_class", "label"], override=args.class_col, required=False, label="Kolom command class GT")
    id_col = pick_col(fieldnames, ["audio_id", "id", "sample_id", "no"], override=args.id_col, required=False, label="Kolom ID")

    if args.condition.lower() != "all" and "audio_condition" in fieldnames:
        rows_meta = [r for r in rows_meta if str(r.get("audio_condition", "")).strip().lower() == args.condition.lower()]

    if "valid_audio" in fieldnames:
        rows_meta = [r for r in rows_meta if str(r.get("valid_audio", "")).strip().lower() in ["true", "1", "yes", "ya"]]

    converted = []
    for idx, row in enumerate(rows_meta, start=1):
        shape_gt = normalize_shape(row.get(shape_col, ""))
        color_gt = normalize_color(row.get(color_col, ""))
        class_gt_raw = normalize_text(row.get(class_col, "")) if class_col else ""
        class_gt_raw = class_gt_raw.replace(" ", "_")
        # Untuk P03, command_class_gt dibuat dari shape/color yang sudah dinormalisasi.
        # Ini membuat "kotak_merah" dan "balok_merah" dihitung setara karena
        # normalize_shape("kotak") -> "balok".
        class_gt = f"{shape_gt}_{color_gt}"

        converted.append({
            "source_type": "metadata",
            "sample_id": row.get(id_col, idx) if id_col else idx,
            "audio_condition": row.get("audio_condition", ""),
            "prompt_variant": row.get("prompt_variant", ""),
            "repetition": row.get("repetition", ""),
            "input_text": row.get(text_col, ""),
            "shape_gt": shape_gt,
            "color_gt": color_gt,
            "command_class_gt": class_gt,
            "command_class_gt_raw": class_gt_raw,
        })

    if args.unique_prompts:
        seen = set()
        unique = []
        for row in converted:
            key = (normalize_text(row["input_text"]), row["shape_gt"], row["color_gt"], row["command_class_gt"])
            if key not in seen:
                seen.add(key)
                unique.append(row)
        converted = unique

    if args.include_alias_tests:
        for r in make_alias_extra_cases():
            shape_gt = normalize_shape(r["shape"])
            color_gt = normalize_color(r["color"])
            converted.append({
                "source_type": r.get("source_type", "alias_extra"),
                "sample_id": r["audio_id"],
                "audio_condition": r.get("audio_condition", ""),
                "prompt_variant": r.get("prompt_variant", ""),
                "repetition": r.get("repetition", ""),
                "input_text": r["prompt_text"],
                "shape_gt": shape_gt,
                "color_gt": color_gt,
                "command_class_gt": f"{shape_gt}_{color_gt}",
                "command_class_gt_raw": f"{shape_gt}_{color_gt}",
            })

    if args.limit and args.limit > 0:
        converted = converted[:args.limit]

    results = []
    print("\n=== P03 COMMAND PARSING TEST START ===")
    print("Metadata:", metadata_path)
    print("Rows tested:", len(converted))
    print("text_col:", text_col)
    print("shape_col:", shape_col)
    print("color_col:", color_col)
    print("class_col:", class_col)
    print("condition:", args.condition)
    print("unique_prompts:", args.unique_prompts)
    print("include_alias_tests:", args.include_alias_tests)

    for i, row in enumerate(converted, start=1):
        parsed = parse_command_text(row["input_text"])
        shape_pred = parsed["shape"]
        color_pred = parsed["color"]
        class_pred = parsed["command_class"]
        shape_correct = shape_pred == row["shape_gt"]
        color_correct = color_pred == row["color_gt"]
        class_correct = class_pred == row["command_class_gt"]
        command_valid = bool(parsed["valid"])
        final_ok = command_valid and shape_correct and color_correct and class_correct
        catatan = "Parser benar" if final_ok else (parsed["reason"] if not command_valid else "Prediksi tidak sesuai ground truth")

        results.append({
            "source_type": row["source_type"],
            "sample_id": row["sample_id"],
            "audio_condition": row["audio_condition"],
            "prompt_variant": row["prompt_variant"],
            "repetition": row["repetition"],
            "input_text": row["input_text"],
            "normalized_text": parsed["normalized_text"],
            "shape_gt": row["shape_gt"],
            "color_gt": row["color_gt"],
            "command_class_gt": row["command_class_gt"],
            "command_class_gt_raw": row.get("command_class_gt_raw", ""),
            "shape_pred": shape_pred,
            "color_pred": color_pred,
            "command_class_pred": class_pred,
            "parse_status": parsed["parse_status"],
            "command_valid": "Ya" if command_valid else "Tidak",
            "shape_correct": "Ya" if shape_correct else "Tidak",
            "color_correct": "Ya" if color_correct else "Tidak",
            "command_class_correct": "Ya" if class_correct else "Tidak",
            "matched_shape_alias": parsed["matched_shape_alias"],
            "matched_color_alias": parsed["matched_color_alias"],
            "reason": parsed["reason"],
            "status_final": "Benar" if final_ok else "Salah",
            "catatan": catatan,
            "parse_detail_json": json.dumps(parsed, ensure_ascii=False),
        })
        print(f"[{i}/{len(converted)}] {results[-1]['status_final']} | input='{row['input_text']}' | pred={class_pred} | gt={row['command_class_gt']}")

    write_csv(csv_path, results)
    write_summary_md(md_path, results, args, csv_path, used_cols={"text_col": text_col, "shape_col": shape_col, "color_col": color_col, "class_col": class_col})

    total = len(results)
    valid = sum(1 for r in results if r["command_valid"] == "Ya")
    shape_ok = sum(1 for r in results if r["shape_correct"] == "Ya")
    color_ok = sum(1 for r in results if r["color_correct"] == "Ya")
    class_ok = sum(1 for r in results if r["command_class_correct"] == "Ya")
    final_ok = sum(1 for r in results if r["status_final"] == "Benar")

    summary = {
        "timestamp": now_tag(),
        "metadata": str(metadata_path),
        "output_dir": str(run_dir),
        "csv": str(csv_path),
        "markdown": str(md_path),
        "total": total,
        "command_valid_rate_percent": round(percent(valid, total), 4),
        "shape_accuracy_percent": round(percent(shape_ok, total), 4),
        "color_accuracy_percent": round(percent(color_ok, total), 4),
        "command_class_accuracy_percent": round(percent(class_ok, total), 4),
        "final_parser_accuracy_percent": round(percent(final_ok, total), 4),
        "valid": valid,
        "shape_correct": shape_ok,
        "color_correct": color_ok,
        "class_correct": class_ok,
        "final_correct": final_ok,
        "text_col": text_col,
        "shape_col": shape_col,
        "color_col": color_col,
        "class_col": class_col,
        "condition": args.condition,
        "unique_prompts": args.unique_prompts,
        "include_alias_tests": args.include_alias_tests,
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== P03 COMMAND PARSING TEST DONE ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P04 — ASR-to-Parser Integration Test Logger

Tujuan:
- Menguji integrasi output ASR ke parser perintah.
- Input utama: CSV hasil P02 (`pengujian_02_asr_transcription.csv`).
- Proses: ASR transcript -> parser -> shape/color/command_class.
- Output: command valid rate, shape accuracy, color accuracy, command class accuracy,
  serta relasi dengan WER.

Catatan metodologi:
- P02 hanya menguji kualitas transkripsi ASR / WER.
- P03 hanya menguji parser dari teks ground truth/variasi teks.
- P04 menguji apakah transcript ASR dari P02 masih dapat dipahami parser.
"""

import argparse
import csv
import json
from pathlib import Path
import re
import statistics
import time


PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


SHAPE_ALIASES = {
    "silinder": [
        "silinder", "cylinder", "cyl", "tabung", "benda silinder", "objek silinder"
    ],
    "balok": [
        "balok", "kotak", "box", "cube", "kubus", "persegi panjang",
        "rectangular box", "rectangular", "benda kotak", "objek kotak"
    ],
    "prisma_segitiga": [
        "prisma segitiga", "prisma_segitiga", "triangular prism",
        "prisma triangular", "segitiga", "triangle", "prism", "prisma",
        "benda segitiga", "objek segitiga"
    ],
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
    raw = str(value or "").strip().lower()
    value_norm = normalize_text(value)

    if raw == "prisma_segitiga":
        return "prisma_segitiga"

    if value_norm in ["silinder", "cylinder", "tabung", "cyl"]:
        return "silinder"

    if value_norm in ["balok", "kotak", "box", "cube", "kubus", "rectangular box", "rectangular", "persegi panjang"]:
        return "balok"

    if value_norm in ["prisma segitiga", "prisma", "segitiga", "triangle", "triangular prism", "prism"]:
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
                matches.append({
                    "canonical": canonical,
                    "alias": alias,
                    "alias_norm": normalize_text(alias),
                    "alias_len": len(normalize_text(alias)),
                })
                break

    if not matches:
        return "", "", []

    matches = sorted(matches, key=lambda m: m["alias_len"], reverse=True)
    return matches[0]["canonical"], matches[0]["alias"], matches


def find_color(text_norm: str):
    matches = []
    for canonical, aliases in COLOR_ALIASES.items():
        for alias in sorted(aliases, key=lambda x: len(normalize_text(x)), reverse=True):
            if contains_alias(text_norm, alias):
                matches.append({
                    "canonical": canonical,
                    "alias": alias,
                    "alias_norm": normalize_text(alias),
                    "alias_len": len(normalize_text(alias)),
                })
                break

    if not matches:
        return "", "", []

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
        "shape": shape,
        "color": color,
        "command_class": command_class,
        "matched_shape_alias": matched_shape_alias,
        "matched_color_alias": matched_color_alias,
        "reason": reason,
        "shape_matches": shape_matches,
        "color_matches": color_matches,
    }


def read_csv(path: Path):
    with path.open("r", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        fieldnames = reader.fieldnames or []
    if not rows:
        raise RuntimeError(f"CSV kosong: {path}")
    return rows, fieldnames


def find_latest_p02(root: Path):
    base = root / "data" / "test_logs_timestamped"
    if not base.exists():
        return None

    candidates = sorted(
        base.glob("P02_asr_transcription_*/pengujian_02_asr_transcription.csv"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def pick_col(fieldnames, candidates, override=None, required=True, label="kolom"):
    if override:
        if override not in fieldnames:
            raise RuntimeError(
                f"{label} override tidak ditemukan: {override}\n"
                f"Kolom tersedia: {', '.join(fieldnames)}"
            )
        return override

    lower = {c.lower(): c for c in fieldnames}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]

    if required:
        raise RuntimeError(
            f"{label} tidak ditemukan.\n"
            f"Kolom tersedia: {', '.join(fieldnames)}\n"
            f"Kandidat otomatis: {', '.join(candidates)}"
        )
    return None


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "no",
        "source_p02_csv",
        "audio_id",
        "audio_condition",
        "command_class_gt",
        "shape_gt",
        "color_gt",
        "ground_truth",
        "asr_transcript",
        "normalized_asr_transcript",
        "wer_percent",
        "S",
        "D",
        "I",
        "N",
        "shape_pred",
        "color_pred",
        "command_class_pred",
        "command_valid",
        "shape_correct",
        "color_correct",
        "command_class_correct",
        "status_final",
        "matched_shape_alias",
        "matched_color_alias",
        "reason",
        "catatan",
        "parse_detail_json",
    ]

    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for i, row in enumerate(rows, start=1):
            r = dict(row)
            r["no"] = i
            w.writerow(r)


def percent(n, d):
    return (n / d * 100.0) if d else 0.0


def safe_float(x, default=None):
    try:
        if x in [None, ""]:
            return default
        return float(x)
    except Exception:
        return default


def write_summary_md(path: Path, rows, args, p02_csv: Path, csv_path: Path):
    total = len(rows)
    valid = sum(1 for r in rows if r["command_valid"] == "Ya")
    shape_ok = sum(1 for r in rows if r["shape_correct"] == "Ya")
    color_ok = sum(1 for r in rows if r["color_correct"] == "Ya")
    class_ok = sum(1 for r in rows if r["command_class_correct"] == "Ya")
    final_ok = sum(1 for r in rows if r["status_final"] == "Berhasil")

    wers_all = [safe_float(r.get("wer_percent")) for r in rows]
    wers_all = [v for v in wers_all if v is not None]

    rows_success = [r for r in rows if r["status_final"] == "Berhasil"]
    rows_failed = [r for r in rows if r["status_final"] != "Berhasil"]

    wer_success = [safe_float(r.get("wer_percent")) for r in rows_success]
    wer_success = [v for v in wer_success if v is not None]
    wer_failed = [safe_float(r.get("wer_percent")) for r in rows_failed]
    wer_failed = [v for v in wer_failed if v is not None]

    lines = [
        "# P04 — Pengujian Integrasi ASR dan Parsing",
        "",
        f"- Waktu pengujian: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Input P02 CSV: `{p02_csv}`",
        f"- Output CSV detail: `{csv_path}`",
        f"- Filter condition: `{args.condition}`",
        "",
        "## Ringkasan Metrik",
        "",
        f"- Total transcript ASR diuji: {total}",
        f"- Command valid rate: {valid}/{total} = {percent(valid, total):.2f}%",
        f"- Shape slot accuracy: {shape_ok}/{total} = {percent(shape_ok, total):.2f}%",
        f"- Color slot accuracy: {color_ok}/{total} = {percent(color_ok, total):.2f}%",
        f"- Command class accuracy: {class_ok}/{total} = {percent(class_ok, total):.2f}%",
        f"- Final ASR-to-parser success rate: {final_ok}/{total} = {percent(final_ok, total):.2f}%",
        f"- Average WER seluruh data: {statistics.mean(wers_all):.2f}%" if wers_all else "- Average WER seluruh data: -",
        f"- Average WER data berhasil: {statistics.mean(wer_success):.2f}%" if wer_success else "- Average WER data berhasil: -",
        f"- Average WER data gagal: {statistics.mean(wer_failed):.2f}%" if wer_failed else "- Average WER data gagal: -",
        "",
        "## Contoh Hasil",
        "",
        "| No | Ground Truth | ASR Transcript | WER | Prediksi Parser | GT | Status |",
        "|---:|---|---|---:|---|---|---|",
    ]

    for idx, r in enumerate(rows[:15], start=1):
        gt_text = str(r.get("ground_truth", "")).replace("|", "\\|")
        hyp = str(r.get("asr_transcript", "")).replace("|", "\\|")
        pred = str(r.get("command_class_pred", "")).replace("|", "\\|")
        gt_class = str(r.get("command_class_gt", "")).replace("|", "\\|")
        wer = r.get("wer_percent", "")
        lines.append(f"| {idx} | {gt_text} | {hyp} | {wer} | {pred} | {gt_class} | {r['status_final']} |")

    if rows_failed:
        lines += [
            "",
            "## Contoh Kegagalan Integrasi",
            "",
            "| No | Ground Truth | ASR Transcript | Prediksi Parser | GT | Alasan |",
            "|---:|---|---|---|---|---|",
        ]
        for idx, r in enumerate(rows_failed[:20], start=1):
            gt_text = str(r.get("ground_truth", "")).replace("|", "\\|")
            hyp = str(r.get("asr_transcript", "")).replace("|", "\\|")
            pred = str(r.get("command_class_pred", "")).replace("|", "\\|")
            gt_class = str(r.get("command_class_gt", "")).replace("|", "\\|")
            reason = str(r.get("reason", "")).replace("|", "\\|")
            lines.append(f"| {idx} | {gt_text} | {hyp} | {pred} | {gt_class} | {reason} |")

    lines += [
        "",
        "## Narasi Singkat Bab 4",
        "",
        (
            "Pengujian integrasi ASR dan parsing dilakukan untuk mengetahui apakah teks hasil "
            "transkripsi ASR dapat diproses menjadi informasi bentuk dan warna target. "
            "Input pada pengujian ini berasal dari hasil pengujian P02, yaitu transcript ASR "
            "dan nilai WER, kemudian transcript tersebut diproses oleh parser perintah."
        ),
        "",
        (
            "Satu data uji dinyatakan berhasil apabila parser dapat menghasilkan bentuk, warna, "
            "dan command class yang sesuai dengan ground truth. Pengujian ini menunjukkan bahwa "
            "nilai WER yang tidak nol belum tentu menyebabkan kegagalan sistem, selama informasi "
            "utama berupa bentuk dan warna masih dapat dikenali oleh parser."
        ),
        "",
        (
            "Contohnya, apabila ground truth adalah `ambil silinder merah` tetapi ASR menghasilkan "
            "`ambil cylinder merah`, maka P02 menghitung perbedaan tersebut sebagai kesalahan "
            "transkripsi. Namun pada P04, command tetap dapat dinyatakan berhasil apabila parser "
            "menormalisasi `cylinder` menjadi `silinder` dan warna `merah` tetap dikenali."
        ),
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--p02_csv", default=None, help="Path CSV hasil P02. Jika kosong, pakai latest P02.")
    parser.add_argument("--output_dir", default="data/test_logs_timestamped")
    parser.add_argument("--condition", default="all", help="Filter audio_condition. Pakai all untuk semua dari P02.")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--show_columns", action="store_true")
    parser.add_argument("--transcript_col", default=None)
    parser.add_argument("--shape_col", default=None)
    parser.add_argument("--color_col", default=None)
    parser.add_argument("--class_col", default=None)
    parser.add_argument("--id_col", default=None)
    args = parser.parse_args()

    root = Path.cwd()

    if args.p02_csv:
        p02_csv = Path(args.p02_csv)
        if not p02_csv.is_absolute():
            p02_csv = root / p02_csv
    else:
        p02_csv = find_latest_p02(root)
        if p02_csv is None:
            raise RuntimeError(
                "CSV P02 tidak ditemukan otomatis.\n"
                "Jalankan P02 dulu atau berikan --p02_csv path/ke/pengujian_02_asr_transcription.csv"
            )

    rows_p02, fieldnames = read_csv(p02_csv)

    if args.show_columns:
        print("\n=== P02 CSV COLUMNS ===")
        for i, col in enumerate(fieldnames, start=1):
            print(f"{i:02d}. {col}")
        print("=======================\n")

    transcript_col = pick_col(
        fieldnames,
        ["asr_transcript", "transcript", "hypothesis", "hyp", "predicted_text"],
        override=args.transcript_col,
        label="Kolom transcript ASR",
    )
    shape_col = pick_col(
        fieldnames,
        ["shape", "shape_gt", "target_shape"],
        override=args.shape_col,
        label="Kolom shape GT",
    )
    color_col = pick_col(
        fieldnames,
        ["color", "color_gt", "target_color"],
        override=args.color_col,
        label="Kolom color GT",
    )
    class_col = pick_col(
        fieldnames,
        ["command_class", "command_class_gt", "class", "target_class", "label"],
        override=args.class_col,
        required=False,
        label="Kolom command class GT",
    )
    id_col = pick_col(
        fieldnames,
        ["audio_id", "id", "sample_id", "no"],
        override=args.id_col,
        required=False,
        label="Kolom ID",
    )

    if args.condition.lower() != "all" and "audio_condition" in fieldnames:
        rows_p02 = [
            r for r in rows_p02
            if str(r.get("audio_condition", "")).strip().lower() == args.condition.lower()
        ]

    # Abaikan baris P02 yang gagal transcribe, kecuali transcript tetap ada.
    if "status" in fieldnames:
        rows_p02 = [
            r for r in rows_p02
            if str(r.get("status", "")).strip().upper() in ["OK", ""] or str(r.get(transcript_col, "")).strip()
        ]

    if args.limit and args.limit > 0:
        rows_p02 = rows_p02[:args.limit]

    run_dir = root / args.output_dir / f"P04_asr_parser_integration_{now_tag()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_dir / "pengujian_04_asr_parser_integration.csv"
    md_path = run_dir / "pengujian_04_asr_parser_integration_summary.md"
    json_path = run_dir / "pengujian_04_asr_parser_integration_summary.json"

    results = []

    print("\n=== P04 ASR-TO-PARSER INTEGRATION TEST START ===")
    print("P02 CSV:", p02_csv)
    print("Rows tested:", len(rows_p02))
    print("transcript_col:", transcript_col)
    print("shape_col:", shape_col)
    print("color_col:", color_col)
    print("class_col:", class_col)
    print("condition:", args.condition)

    for i, row in enumerate(rows_p02, start=1):
        transcript = row.get(transcript_col, "")
        parsed = parse_command_text(transcript)

        shape_gt = normalize_shape(row.get(shape_col, ""))
        color_gt = normalize_color(row.get(color_col, ""))
        command_class_gt = f"{shape_gt}_{color_gt}"

        # Jika P02 menyimpan command_class_gt, tetap dinormalisasi dari shape/color
        # agar kotak dan balok dihitung setara secara semantik.
        shape_pred = parsed["shape"]
        color_pred = parsed["color"]
        command_class_pred = parsed["command_class"]

        command_valid = bool(parsed["valid"])
        shape_correct = shape_pred == shape_gt
        color_correct = color_pred == color_gt
        class_correct = command_class_pred == command_class_gt
        final_ok = command_valid and shape_correct and color_correct and class_correct

        if final_ok:
            catatan = "Transcript ASR berhasil dipahami parser"
        elif not command_valid:
            catatan = parsed["reason"]
        else:
            catatan = "Parser menghasilkan shape/color yang tidak sesuai GT"

        result = {
            "source_p02_csv": str(p02_csv),
            "audio_id": row.get(id_col, i) if id_col else i,
            "audio_condition": row.get("audio_condition", ""),
            "command_class_gt": command_class_gt,
            "shape_gt": shape_gt,
            "color_gt": color_gt,
            "ground_truth": row.get("ground_truth", ""),
            "asr_transcript": transcript,
            "normalized_asr_transcript": parsed["normalized_text"],
            "wer_percent": row.get("wer_percent", ""),
            "S": row.get("S", ""),
            "D": row.get("D", ""),
            "I": row.get("I", ""),
            "N": row.get("N", ""),
            "shape_pred": shape_pred,
            "color_pred": color_pred,
            "command_class_pred": command_class_pred,
            "command_valid": "Ya" if command_valid else "Tidak",
            "shape_correct": "Ya" if shape_correct else "Tidak",
            "color_correct": "Ya" if color_correct else "Tidak",
            "command_class_correct": "Ya" if class_correct else "Tidak",
            "status_final": "Berhasil" if final_ok else "Gagal",
            "matched_shape_alias": parsed["matched_shape_alias"],
            "matched_color_alias": parsed["matched_color_alias"],
            "reason": parsed["reason"],
            "catatan": catatan,
            "parse_detail_json": json.dumps(parsed, ensure_ascii=False),
        }
        results.append(result)

        print(
            f"[{i}/{len(rows_p02)}] {result['status_final']} | "
            f"ASR='{transcript}' | pred={command_class_pred} | gt={command_class_gt} | WER={result['wer_percent']}"
        )

    write_csv(csv_path, results)
    write_summary_md(md_path, results, args, p02_csv, csv_path)

    total = len(results)
    valid = sum(1 for r in results if r["command_valid"] == "Ya")
    shape_ok = sum(1 for r in results if r["shape_correct"] == "Ya")
    color_ok = sum(1 for r in results if r["color_correct"] == "Ya")
    class_ok = sum(1 for r in results if r["command_class_correct"] == "Ya")
    final_ok = sum(1 for r in results if r["status_final"] == "Berhasil")

    wers = [safe_float(r.get("wer_percent")) for r in results]
    wers = [v for v in wers if v is not None]

    summary = {
        "timestamp": now_tag(),
        "p02_csv": str(p02_csv),
        "output_dir": str(run_dir),
        "csv": str(csv_path),
        "markdown": str(md_path),
        "total": total,
        "command_valid_rate_percent": round(percent(valid, total), 4),
        "shape_slot_accuracy_percent": round(percent(shape_ok, total), 4),
        "color_slot_accuracy_percent": round(percent(color_ok, total), 4),
        "command_class_accuracy_percent": round(percent(class_ok, total), 4),
        "final_asr_parser_success_rate_percent": round(percent(final_ok, total), 4),
        "average_wer_percent": round(statistics.mean(wers), 4) if wers else None,
        "valid": valid,
        "shape_correct": shape_ok,
        "color_correct": color_ok,
        "class_correct": class_ok,
        "final_success": final_ok,
        "condition": args.condition,
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== P04 ASR-TO-PARSER INTEGRATION TEST DONE ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

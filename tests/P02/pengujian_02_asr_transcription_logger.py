#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
P02 — ASR Transcription Test Logger v3

P02 fokus pada pengujian satuan ASR:
Audio perintah suara -> ASR transcript -> WER.

Metadata yang didukung:
- audio path: wav_path, audio_path, path, filename, file, dll.
- ground truth text: prompt_text, ground_truth, command, text, perintah, dll.

Untuk dataset Scriptis sekarang:
- audio_col = wav_path
- text_col  = prompt_text
- id_col    = audio_id
- P02 disarankan pakai audio_condition = clean
"""

import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics
import time
import wave

PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)


def now_tag():
    return time.strftime("%Y%m%d_%H%M%S")


def normalize_text(text: str) -> str:
    text = str(text or "").strip().lower()
    text = text.replace("_", " ")
    text = PUNCT_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def levenshtein_alignment(ref_tokens, hyp_tokens):
    n = len(ref_tokens)
    m = len(hyp_tokens)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    bt = [[None] * (m + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        dp[i][0] = i
        bt[i][0] = "D"
    for j in range(1, m + 1):
        dp[0][j] = j
        bt[0][j] = "I"

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if ref_tokens[i - 1] == hyp_tokens[j - 1]:
                choices = [
                    (dp[i - 1][j - 1], "C"),
                    (dp[i - 1][j] + 1, "D"),
                    (dp[i][j - 1] + 1, "I"),
                ]
            else:
                choices = [
                    (dp[i - 1][j - 1] + 1, "S"),
                    (dp[i - 1][j] + 1, "D"),
                    (dp[i][j - 1] + 1, "I"),
                ]
            best_cost, best_op = min(choices, key=lambda x: x[0])
            dp[i][j] = best_cost
            bt[i][j] = best_op

    i, j = n, m
    s = d = ins = c = 0
    ops = []

    while i > 0 or j > 0:
        op = bt[i][j]
        if op == "C":
            c += 1
            ops.append(("C", ref_tokens[i - 1], hyp_tokens[j - 1]))
            i -= 1
            j -= 1
        elif op == "S":
            s += 1
            ops.append(("S", ref_tokens[i - 1], hyp_tokens[j - 1]))
            i -= 1
            j -= 1
        elif op == "D":
            d += 1
            ops.append(("D", ref_tokens[i - 1], ""))
            i -= 1
        elif op == "I":
            ins += 1
            ops.append(("I", "", hyp_tokens[j - 1]))
            j -= 1
        else:
            break

    ops.reverse()
    wer = ((s + d + ins) / n * 100.0) if n > 0 else (0.0 if m == 0 else 100.0)

    return {
        "S": s,
        "D": d,
        "I": ins,
        "C": c,
        "N": n,
        "WER": wer,
        "ops": ops,
    }


def wav_basic_info(path: Path):
    try:
        import numpy as np
        with wave.open(str(path), "rb") as wf:
            channels = wf.getnchannels()
            sampwidth = wf.getsampwidth()
            framerate = wf.getframerate()
            frames = wf.getnframes()
            raw = wf.readframes(frames)

        duration = frames / float(framerate) if framerate else 0.0

        if sampwidth == 2:
            data = np.frombuffer(raw, dtype="<i2").astype("float32")
            denom = 32768.0
        elif sampwidth == 4:
            data = np.frombuffer(raw, dtype="<i4").astype("float32")
            denom = 2147483648.0
        elif sampwidth == 1:
            data = np.frombuffer(raw, dtype="uint8").astype("float32") - 128.0
            denom = 128.0
        else:
            return {"duration_sec": round(duration, 4), "rms_dbfs": None, "sample_rate": framerate, "channels": channels}

        if channels > 1 and data.size >= channels:
            data = data.reshape(-1, channels).mean(axis=1)

        rms = math.sqrt(float((data * data).mean())) / denom if data.size else 0.0
        dbfs = 20.0 * math.log10(max(rms, 1e-12))

        return {
            "duration_sec": round(duration, 4),
            "rms_dbfs": round(dbfs, 4),
            "sample_rate": framerate,
            "channels": channels,
        }
    except Exception:
        return {"duration_sec": None, "rms_dbfs": None, "sample_rate": None, "channels": None}


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
            f"Gunakan override manual. Kandidat otomatis: {', '.join(candidates)}"
        )
    return None


class ASRBackend:
    def __init__(self, backend, model_name, device, compute_type, language):
        self.backend = backend
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.model = None

    def load(self):
        if self.backend == "faster-whisper":
            from faster_whisper import WhisperModel
            self.model = WhisperModel(
                self.model_name,
                device=self.device,
                compute_type=self.compute_type
            )
            return

        if self.backend == "whisper":
            import whisper
            self.model = whisper.load_model(self.model_name, device=self.device)
            return

        raise RuntimeError(f"Backend tidak dikenali: {self.backend}")

    def transcribe(self, audio_path: Path):
        if self.model is None:
            self.load()

        start = time.time()

        if self.backend == "faster-whisper":
            segments, info = self.model.transcribe(
                str(audio_path),
                language=self.language,
                vad_filter=False,
                beam_size=5
            )
            text = " ".join(seg.text.strip() for seg in segments).strip()
            elapsed = time.time() - start
            return text, elapsed, {
                "language": getattr(info, "language", None),
                "language_probability": getattr(info, "language_probability", None),
            }

        if self.backend == "whisper":
            result = self.model.transcribe(
                str(audio_path),
                language=self.language,
                fp16=(self.device == "cuda")
            )
            elapsed = time.time() - start
            return str(result.get("text", "")).strip(), elapsed, {
                "language": result.get("language")
            }

        raise RuntimeError(f"Backend tidak dikenali: {self.backend}")


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "no",
        "audio_id",
        "speaker_id",
        "gender",
        "accent",
        "audio_condition",
        "command_class",
        "shape",
        "color",
        "prompt_variant",
        "repetition",
        "audio_path",
        "ground_truth",
        "asr_transcript",
        "normalized_ground_truth",
        "normalized_asr_transcript",
        "S",
        "D",
        "I",
        "C",
        "N",
        "wer_percent",
        "audio_duration_sec",
        "audio_rms_dbfs",
        "sample_rate",
        "channels",
        "asr_time_sec",
        "status",
        "catatan",
        "alignment_ops_json",
        "backend_info_json",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i, row in enumerate(rows, start=1):
            r = dict(row)
            r["no"] = i
            w.writerow(r)


def write_summary_md(path: Path, rows, args, csv_path: Path, used_cols):
    total = len(rows)
    valid = [r for r in rows if r["status"] == "OK"]
    failed = total - len(valid)

    wers = [float(r["wer_percent"]) for r in valid if r.get("wer_percent") not in [None, ""]]
    avg_wer = statistics.mean(wers) if wers else None
    median_wer = statistics.median(wers) if wers else None

    sum_s = sum(int(r["S"]) for r in valid if r["S"] != "")
    sum_d = sum(int(r["D"]) for r in valid if r["D"] != "")
    sum_i = sum(int(r["I"]) for r in valid if r["I"] != "")
    sum_n = sum(int(r["N"]) for r in valid if r["N"] != "")
    corpus_wer = ((sum_s + sum_d + sum_i) / sum_n * 100.0) if sum_n > 0 else None

    times = [float(r["asr_time_sec"]) for r in valid if r.get("asr_time_sec") not in [None, ""]]
    avg_time = statistics.mean(times) if times else None

    lines = [
        "# P02 — Pengujian Satuan ASR Transcription",
        "",
        f"- Waktu pengujian: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"- Metadata: `{args.metadata}`",
        f"- Audio root: `{args.audio_root}`",
        f"- Audio condition filter: `{args.condition}`",
        f"- Valid audio only: `{not args.include_invalid_audio}`",
        f"- Backend: `{args.backend}`",
        f"- Model: `{args.model}`",
        f"- Device: `{args.device}`",
        f"- Compute type: `{args.compute_type}`",
        f"- CSV detail: `{csv_path}`",
        "",
        "## Kolom Metadata yang Dipakai",
        "",
        f"- Kolom audio: `{used_cols.get('audio_col')}`",
        f"- Kolom ground truth: `{used_cols.get('text_col')}`",
        f"- Kolom ID: `{used_cols.get('id_col')}`",
        "",
        "## Ringkasan Metrik",
        "",
        f"- Total audio diuji: {total}",
        f"- Audio berhasil diproses: {len(valid)}",
        f"- Audio gagal diproses: {failed}",
        f"- Average WER: {avg_wer:.2f}%" if avg_wer is not None else "- Average WER: -",
        f"- Median WER: {median_wer:.2f}%" if median_wer is not None else "- Median WER: -",
        f"- Corpus WER: {corpus_wer:.2f}%" if corpus_wer is not None else "- Corpus WER: -",
        f"- Average ASR time: {avg_time:.3f} detik" if avg_time is not None else "- Average ASR time: -",
        "",
        "## Rekap Substitution/Deletion/Insertion",
        "",
        f"- Substitution (S): {sum_s}",
        f"- Deletion (D): {sum_d}",
        f"- Insertion (I): {sum_i}",
        f"- Jumlah kata referensi (N): {sum_n}",
        "",
        "## Contoh Hasil",
        "",
        "| No | Audio ID | Ground Truth | ASR Transcript | WER | Catatan |",
        "|---:|---|---|---|---:|---|",
    ]

    for idx, r in enumerate(rows[:10], start=1):
        aid = str(r.get("audio_id", "")).replace("|", "\\|")
        gt = str(r.get("ground_truth", "")).replace("|", "\\|")
        hyp = str(r.get("asr_transcript", "")).replace("|", "\\|")
        cat = str(r.get("catatan", "")).replace("|", "\\|")
        wer = r.get("wer_percent", "")
        lines.append(f"| {idx} | {aid} | {gt} | {hyp} | {wer} | {cat} |")

    lines += [
        "",
        "## Narasi Singkat Bab 4",
        "",
        (
            "Pengujian satuan ASR dilakukan untuk mengevaluasi kemampuan sistem dalam "
            "mengubah perintah suara menjadi teks. Hasil transkripsi dibandingkan dengan "
            "ground truth menggunakan metrik Word Error Rate (WER), yang menghitung jumlah "
            "substitution, deletion, dan insertion terhadap jumlah kata referensi."
        ),
        "",
        (
            "Hasil pengujian ini digunakan sebagai dasar untuk mengetahui kualitas keluaran ASR "
            "sebelum teks hasil transkripsi diproses pada tahap parsing perintah. Kesalahan "
            "transkripsi pada tahap ini belum langsung dinyatakan sebagai kegagalan sistem secara "
            "keseluruhan, karena tahap parsing dan normalisasi kata diuji secara terpisah pada "
            "pengujian berikutnya."
        ),
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", default="data/audio_dataset/metadata.csv")
    parser.add_argument("--audio_root", default=".")
    parser.add_argument("--output_dir", default="data/test_logs_timestamped")
    parser.add_argument("--backend", choices=["faster-whisper", "whisper"], default="faster-whisper")
    parser.add_argument("--model", default="small")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute_type", default="float16")
    parser.add_argument("--language", default="id")
    parser.add_argument("--limit", type=int, default=0, help="0 = semua data")
    parser.add_argument("--condition", default="clean", help="Filter audio_condition. Pakai all untuk semua.")
    parser.add_argument("--include_invalid_audio", action="store_true", help="Ikutkan valid_audio=False")
    parser.add_argument("--audio_col", default=None)
    parser.add_argument("--text_col", default=None)
    parser.add_argument("--id_col", default=None)
    parser.add_argument("--show_columns", action="store_true")
    args = parser.parse_args()

    root = Path.cwd()
    metadata_path = Path(args.metadata)
    if not metadata_path.is_absolute():
        metadata_path = root / metadata_path

    audio_root = Path(args.audio_root)
    if not audio_root.is_absolute():
        audio_root = root / audio_root

    run_dir = root / args.output_dir / f"P02_asr_transcription_{now_tag()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    csv_path = run_dir / "pengujian_02_asr_transcription.csv"
    md_path = run_dir / "pengujian_02_asr_transcription_summary.md"
    json_path = run_dir / "pengujian_02_asr_transcription_summary.json"

    rows_meta, fieldnames = read_metadata(metadata_path)

    if args.show_columns:
        print("\n=== METADATA COLUMNS ===")
        for i, col in enumerate(fieldnames, start=1):
            print(f"{i:02d}. {col}")
        print("========================\n")

    audio_col = pick_col(
        fieldnames,
        ["wav_path", "audio_path", "path", "file_path", "filename", "file", "audio", "audio_file", "audio_filename", "nama_file"],
        override=args.audio_col,
        label="Kolom audio",
    )
    text_col = pick_col(
        fieldnames,
        ["prompt_text", "ground_truth", "groundtruth", "gt", "command", "command_text", "text", "sentence", "reference", "label", "teks", "teks_perintah", "perintah", "kalimat", "kalimat_perintah"],
        override=args.text_col,
        label="Kolom ground truth",
    )
    id_col = pick_col(
        fieldnames,
        ["audio_id", "id", "sample_id", "no", "filename", "file"],
        override=args.id_col,
        required=False,
        label="Kolom ID",
    )

    before_filter = len(rows_meta)

    if args.condition.lower() != "all" and "audio_condition" in fieldnames:
        rows_meta = [
            r for r in rows_meta
            if str(r.get("audio_condition", "")).strip().lower() == args.condition.lower()
        ]

    if (not args.include_invalid_audio) and "valid_audio" in fieldnames:
        rows_meta = [
            r for r in rows_meta
            if str(r.get("valid_audio", "")).strip().lower() in ["true", "1", "yes", "ya"]
        ]

    if args.limit and args.limit > 0:
        rows_meta = rows_meta[:args.limit]

    asr = ASRBackend(
        backend=args.backend,
        model_name=args.model,
        device=args.device,
        compute_type=args.compute_type,
        language=args.language,
    )

    results = []

    print("\n=== P02 ASR TRANSCRIPTION TEST START ===")
    print("Metadata:", metadata_path)
    print("Audio root:", audio_root)
    print("Rows metadata before filter:", before_filter)
    print("Rows tested:", len(rows_meta))
    print("audio_col:", audio_col)
    print("text_col:", text_col)
    print("id_col:", id_col)
    print("condition:", args.condition)
    print("Backend:", args.backend)
    print("Model:", args.model)
    print("Device:", args.device)

    for idx, row in enumerate(rows_meta, start=1):
        audio_rel = str(row.get(audio_col, "")).strip()
        gt = str(row.get(text_col, "")).strip()
        audio_id = str(row.get(id_col, idx)).strip() if id_col else str(idx)

        audio_path = Path(audio_rel)
        if not audio_path.is_absolute():
            audio_path = audio_root / audio_path

        base = {
            "audio_id": audio_id,
            "speaker_id": row.get("speaker_id", ""),
            "gender": row.get("gender", ""),
            "accent": row.get("accent", ""),
            "audio_condition": row.get("audio_condition", ""),
            "command_class": row.get("command_class", ""),
            "shape": row.get("shape", ""),
            "color": row.get("color", ""),
            "prompt_variant": row.get("prompt_variant", ""),
            "repetition": row.get("repetition", ""),
            "audio_path": str(audio_path),
            "ground_truth": gt,
            "asr_transcript": "",
            "normalized_ground_truth": normalize_text(gt),
            "normalized_asr_transcript": "",
            "S": "",
            "D": "",
            "I": "",
            "C": "",
            "N": "",
            "wer_percent": "",
            "audio_duration_sec": "",
            "audio_rms_dbfs": "",
            "sample_rate": "",
            "channels": "",
            "asr_time_sec": "",
            "status": "OK",
            "catatan": "",
            "alignment_ops_json": "[]",
            "backend_info_json": "{}",
        }

        try:
            if not audio_path.exists():
                raise FileNotFoundError(f"Audio tidak ditemukan: {audio_path}")

            info = wav_basic_info(audio_path)
            base["audio_duration_sec"] = info.get("duration_sec")
            base["audio_rms_dbfs"] = info.get("rms_dbfs")
            base["sample_rate"] = info.get("sample_rate")
            base["channels"] = info.get("channels")

            hyp, elapsed, backend_info = asr.transcribe(audio_path)
            hyp_norm = normalize_text(hyp)
            gt_norm = normalize_text(gt)

            align = levenshtein_alignment(gt_norm.split(), hyp_norm.split())

            base["asr_transcript"] = hyp
            base["normalized_asr_transcript"] = hyp_norm
            base["S"] = align["S"]
            base["D"] = align["D"]
            base["I"] = align["I"]
            base["C"] = align["C"]
            base["N"] = align["N"]
            base["wer_percent"] = round(float(align["WER"]), 4)
            base["asr_time_sec"] = round(float(elapsed), 4)
            base["alignment_ops_json"] = json.dumps(align["ops"], ensure_ascii=False)
            base["backend_info_json"] = json.dumps(backend_info, ensure_ascii=False)

            if align["WER"] == 0:
                base["catatan"] = "Transkripsi sama dengan ground truth"
            else:
                base["catatan"] = "Terdapat error transkripsi"

            print(f"[{idx}/{len(rows_meta)}] OK | WER={base['wer_percent']} | GT='{gt}' | ASR='{hyp}'")

        except Exception as exc:
            base["status"] = "ERROR"
            base["catatan"] = str(exc)
            print(f"[{idx}/{len(rows_meta)}] ERROR | {audio_path} | {exc}")

        results.append(base)

    write_csv(csv_path, results)
    write_summary_md(
        md_path,
        results,
        args,
        csv_path,
        used_cols={"audio_col": audio_col, "text_col": text_col, "id_col": id_col},
    )

    valid = [r for r in results if r["status"] == "OK"]
    sum_s = sum(int(r["S"]) for r in valid if r["S"] != "")
    sum_d = sum(int(r["D"]) for r in valid if r["D"] != "")
    sum_i = sum(int(r["I"]) for r in valid if r["I"] != "")
    sum_n = sum(int(r["N"]) for r in valid if r["N"] != "")
    corpus_wer = ((sum_s + sum_d + sum_i) / sum_n * 100.0) if sum_n else None
    wers = [float(r["wer_percent"]) for r in valid if r["wer_percent"] != ""]

    summary = {
        "timestamp": now_tag(),
        "metadata": str(metadata_path),
        "audio_root": str(audio_root),
        "output_dir": str(run_dir),
        "csv": str(csv_path),
        "markdown": str(md_path),
        "total": len(results),
        "ok": len(valid),
        "error": len(results) - len(valid),
        "average_wer_percent": round(statistics.mean(wers), 4) if wers else None,
        "median_wer_percent": round(statistics.median(wers), 4) if wers else None,
        "corpus_wer_percent": round(corpus_wer, 4) if corpus_wer is not None else None,
        "S": sum_s,
        "D": sum_d,
        "I": sum_i,
        "N": sum_n,
        "audio_col": audio_col,
        "text_col": text_col,
        "id_col": id_col,
        "condition": args.condition,
    }
    json_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n=== P02 ASR TRANSCRIPTION TEST DONE ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

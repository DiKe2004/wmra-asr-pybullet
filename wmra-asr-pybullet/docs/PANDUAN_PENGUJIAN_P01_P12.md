# Panduan pengujian P01–P12

Dokumen ini dibuat dari **skrip yang tersedia dalam ZIP**, bukan dari eksekusi ulang. Jalankan terminal dari **root repository**, dengan Python virtual environment aktif dan prasyarat terpasang. Banyak pengujian membutuhkan model JACO URDF, yang **belum disertakan**. Semua perintah di bawah bersifat **contoh pelaksanaan/pilot**, bukan rekonstruksi otomatis hasil formal yang dilaporkan dalam skripsi.

## Sebelum memulai

```bash
source .venv/bin/activate
python tools/preflight.py
```

Jika pemeriksaan model gagal, lihat `ASSETS_DAN_DATA.md`. Jangan memakai URDF robot 7-DoF untuk menggantikan model 6-DoF.

## Matriks pengujian

| Kode | Fokus | Input yang diperlukan | Nama skrip |
|---|---|---|---|
| P01 | Validasi lingkungan simulasi | Main source, URDF/mesh | `pengujian_01_environment_validation_logger.py` |
| P02 | Transkripsi ASR dan Word Error Rate (WER) | Metadata CSV, audio WAV, model Whisper | `pengujian_02_asr_transcription_logger.py` |
| P03 | Parser perintah bentuk/warna | Metadata CSV | `pengujian_03_command_parsing_logger.py` |
| P04 | Integrasi transkripsi ASR–parser | CSV keluaran P02 | `pengujian_04_asr_parser_integration_logger.py` |
| P05 | Pencarian objek target | Main source, URDF/mesh | `pengujian_05_visual_search_detection_logger.py` |
| P06 | Strict visual match dengan distractor | Main source, URDF/mesh | `pengujian_06_strict_visual_match_distractor_logger.py` |
| P07 | Ketercapaian inverse kinematics | Main source, URDF/mesh | `pengujian_07_inverse_kinematics_reachability_logger.py` |
| P08 | Virtual grasp & lift | Main source, URDF/mesh | `pengujian_08_grasp_lift_logger.py` |
| P09 | Pemantauan kontak robot | Main source, URDF/mesh | `pengujian_09_collision_rate_logger.py` |
| P10 | Handover & penempatan keranjang | Main source, URDF/mesh | `pengujian_10_handover_basket_logger.py` |
| P11 | Integrasi subsistem manipulasi | Main source, URDF/mesh, **teks valid** | `pengujian_11_integrasi_subsistem_logger.py` |
| P12 | Penyelesaian tugas formal | Main source, URDF/mesh, **teks valid** | `pengujian_12_end_to_end_formal_logger.py` |

## Perintah dasar yang sesuai opsi CLI skrip

### P01 — lingkungan

```bash
python tests/P01/pengujian_01_environment_validation_logger.py \
  --output_dir data/test_logs/pilot_baru/P01
```

P01 tidak memiliki opsi `--limit`; menjalankan perintah ini memulai pemeriksaan lingkungan sesuai kode skrip.

### P02 — ASR, uji awal 1 rekaman

```bash
python tests/P02/pengujian_02_asr_transcription_logger.py \
  --metadata data/audio_dataset/metadata.csv \
  --audio_root . \
  --condition clean --limit 1 \
  --device cpu --compute_type int8 \
  --output_dir data/test_logs/pilot_baru
```

Nilai default original menggunakan `cuda` / `float16`. Contoh CPU hanya untuk mengecek instalasi. Untuk melihat kolom metadata yang terdeteksi, tambahkan `--show_columns` (tetap butuh metadata). Untuk hasil formal, gunakan environment dan versi dataset sama seperti penelitian asli, bukan otomatis contoh ini.

### P03 — parser, 5 perintah pertama

```bash
python tests/P03/pengujian_03_command_parsing_logger.py \
  --metadata data/audio_dataset/metadata.csv --condition all --limit 5 \
  --output_dir data/test_logs/pilot_baru
```

### P04 — integrasi ASR–parser, memakai keluaran P02

```bash
python tests/P04/pengujian_04_asr_parser_integration_logger.py \
  --p02_csv "/path/hasil_P02/pengujian_02_asr_transcription.csv" \
  --limit 5 --output_dir data/test_logs/pilot_baru
```

**Ganti** lokasi CSV dengan path output P02 yang benar. Hindari asumsi bahwa script otomatis mengambil hasil formal lama.

### P05 — visual search

```bash
python tests/P05/pengujian_05_visual_search_detection_logger.py \
  --limit 1 --output_dir data/test_logs/pilot_baru/P05
```

### P06 — strict visual match

```bash
python tests/P06/pengujian_06_strict_visual_match_distractor_logger.py \
  --limit 1 --output_dir data/test_logs/pilot_baru/P06
```

### P07 — inverse kinematics

```bash
python tests/P07/pengujian_07_inverse_kinematics_reachability_logger.py \
  --limit 1 --output_dir data/test_logs/pilot_baru/P07
```

### P08 — virtual grasp dan lift

```bash
python tests/P08/pengujian_08_grasp_lift_logger.py \
  --limit 1 --output_dir data/test_logs/pilot_baru/P08
```

### P09 — monitoring kontak

```bash
python tests/P09/pengujian_09_collision_rate_logger.py \
  --limit_targets 1 --repeat 1 --output_dir data/test_logs/pilot_baru/P09
```

### P10 — handover & keranjang

```bash
python tests/P10/pengujian_10_handover_basket_logger.py \
  --limit 1 --output_dir data/test_logs/pilot_baru/P10
```

### P11 — integrasi subsistem

```bash
python tests/P11/pengujian_11_integrasi_subsistem_logger.py \
  --limit 1 --output_dir data/test_logs/pilot_baru/P11
```

P11 memasukkan **perintah teks valid** pada rangkaian manipulasi, bukan inferensi audio Whisper pada trial yang sama.

### P12 — end-to-end **pilot**, bukan formal lama

```bash
python tests/P12/pengujian_12_end_to_end_formal_logger.py \
  --limit 1 --output-dir data/test_logs/pilot_baru/P12
```

**Penting:** Opsi `--limit 0` pada P12 menjalankan 27 trial sesuai kode, tetapi hasilnya tetap **pengujian baru**. Jangan beri judul `P12_FORMAL_20260923_160925_806445` untuk run baru atau mencampurnya dengan arsip skripsi lama.

## Sesudah pengujian

1. Buka CSV/JSON/Markdown keluaran pada folder yang disebut skrip.
2. Catat tanggal eksekusi, Python, OS, GPU, paket, source SHA-256, seed, dan parameter CLI.
3. Simpan hasil **berhasil dan gagal**, jangan menyaring kegagalan.
4. Bandingkan ringkasan dengan data formal asli **hanya setelah** memastikan model, source, parameter, dataset, dan environment cocok.
5. Jangan menyebut uji yang belum dijalankan sebagai "reproduksi berhasil".

## Catatan tentang checksum source

Normalisasi path untuk struktur GitHub menyebabkan hash file Python aktif berbeda dengan ZIP awal, walaupun definisi fungsi dan kelas utama **tidak berubah**. File asli tersimpan di `archive/zip_asli_2026-11-09/` dan daftar SHA-256 awal ada di `SHA256_CODE_DARI_ZIP_ASLI.txt`. Cocokkan dengan `source_sha256` pada log formal sebelum menyatakan data formal berasal dari file yang identik.

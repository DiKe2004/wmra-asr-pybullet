# Model robot, dataset, dan artefak yang belum disertakan

Dokumen ini membedakan **berkas yang ada di ZIP** dengan **berkas yang wajib disediakan pemilik proyek**. Jangan menyatakan folder kosong sebagai aset lengkap.

## 1. Model Kinova JACO Gen 2

Source utama memanggil `p.loadURDF(JACO_URDF, basePosition=..., useFixedBase=True)` dan memerlukan URDF Kinova JACO Gen 2 **konfigurasi enam sendi dan tiga jari**, beserta semua mesh/material/URDF tambahan yang dirujuk. File dan lisensinya **tidak ada dalam ZIP sumber**.

Lokasi default:

```text
assets/robots/jaco2_6dof/j2s6s300.urdf
```

Untuk menyiapkan aset:

1. Dapatkan model dari distribusi tepercaya yang memang mengizinkan pemakaian terkait. **Lokasi unduhan, versi, serta izin redistribusinya belum diverifikasi di sini**.
2. Simpan `j2s6s300.urdf` bersama seluruh aset yang diacu (`mesh`, `texture`, `material`) dengan struktur folder relatif asli.
3. Jalankan `python tools/preflight.py`, lalu uji P01 atau simulasi utama pada mesin sendiri untuk memastikan model termuat.
4. Catat URL sumber model, commit/revisi, lisensi, dan tanggal akses di `docs/LISENSI_DAN_ATRIBUSI.md` sebelum publikasi.

Jika model tidak boleh di-*commit* ke GitHub, tempatkan di luar repository dan jalankan:

```bash
export WMRA_JACO_URDF="/lokasi/asli/j2s6s300.urdf"
python tools/preflight.py
```

**Jangan** mengganti model dengan JACO 7-DoF atau gripper lain hanya supaya simulasi dapat dijalankan; ini mengubah objek penelitian.

## 2. Mesh prisma segitiga

File `triangle_prism.obj` **tidak perlu diunduh**. Source memanggil `ensure_triangle_prism_obj(...)` dan menghasilkan mesh tersebut secara otomatis. Lokasi default: `assets/objects/generated_meshes/triangle_prism.obj`, yang dikecualikan dari Git karena file keluaran otomatis. Lokasi dapat diubah melalui `WMRA_MESH_DIR`.

## 3. Dataset audio, metadata, hasil P02/P03/P04

Dataset dan metadata tidak ada di ZIP sumber. **Jangan membuat contoh data fiktif lalu mengklaimnya sebagai data pengujian formal.** Siapkan dataset yang benar-benar digunakan dan lakukan pemeriksaan izin berbagi sebelum mengunggah.

Nilai default P02/P03 adalah:

```text
data/audio_dataset/metadata.csv
```

P02 mendeteksi beberapa kemungkinan nama kolom secara otomatis, antara lain:

| Peran kolom | Contoh nama yang dikenali |
|---|---|
| File WAV | `wav_path`, `audio_path`, `path`, `filename` |
|---|---|
| Referensi transkripsi | `prompt_text`, `ground_truth`, `text`, `command` |
|---|---|
| ID audio | `audio_id`, `id`, `sample_id` |
|---|---|
| Kondisi audio (opsional) | `audio_condition` |
|---|---|
| Penanda audio valid (opsional) | `valid_audio` |

P03 juga mengenali kolom atribut, contohnya `shape`, `color`, `command_class` (periksa `--show_columns` dan deteksi kolom pada source untuk nama alternatif). Audio path di CSV harus diselaraskan dengan parameter `--audio_root` pada P02. Jika root adalah folder proyek, path relatif seperti `data/audio_dataset/clean/recording.wav` harus benar-benar ada di bawahnya.

P04 membutuhkan **CSV keluaran P02**, bukan metadata mentah. Berikan lokasi secara eksplisit dengan `--p02_csv /path/ke/pengujian_02_asr_transcription.csv` agar tidak tergantung pencarian file terakhir.

## 4. JSON perintah ASR ke simulasi

Main memakai `data/asr_eval/latest_command.json` (bisa diubah dengan variabel `WMRA_ASR_COMMAND_JSON`) untuk mode ASR yang diaktifkan manual. Komentar source menyebut `evaluate_asr_dataset.py` dan `realtime_asr_command.py`, namun **dua pembuat JSON tersebut tidak disertakan dalam ZIP**. `tests/P02/pengujian_02_asr_transcription_logger.py` adalah penguji transkripsi dan **tidak otomatis menggantikan penghubung JSON**.

Karena format JSON belum diaudit bersama pembangkit aslinya, dokumen ini **tidak memberi contoh payload buatan**. Jangan mengaktifkan mode ASR otomatis dan mengklaim telah diintegrasikan hanya berdasarkan adanya referensi file JSON.

## 5. Log formal, CSV, JSON, video

Skrip uji menulis ke lokasi seperti `data/test_logs/dosbim2_revision/` dan `data/test_logs_timestamped/`. Tidak satu pun arsip formal pengujian disertakan di ZIP ini. Catatan skripsi terpisah menyebut `dosbim2_revision.zip` dan folder P12 `P12_FORMAL_20260923_160925_806445`, tetapi keberadaan serta isinya **belum diverifikasi dalam paket repository ini**.

Untuk publikasi ilmiah:

- Simpan arsip asli pada lokasi akses yang diizinkan (misalnya repositori data institusi atau media penyimpanan yang dapat diakses penguji).
- Sertakan README/indeks data: sumber, tanggal, jumlah sampel, definisi kolom, cara penggunaan, status izin, SHA-256 arsip.
- Anonimkan informasi pembicara, path absolut lokal, identitas pengguna, dan rekaman suara tanpa hak berbagi.
- Simpan hasil gagal sebagaimana data formal; jangan menghapusnya.
- Pisahkan log pengujian ulang (rerun/pilot) dari log P12 formal lama.

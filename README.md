# Simulasi Wheelchair-Mounted Robotic Arm (WMRA) Berbasis Pengenalan Suara

Repository pendamping skripsi S1 Teknik Robotika dan Kecerdasan Buatan, Universitas Airlangga.

**Judul:** *Simulasi Wheelchair-Mounted Robotic Arm Berbasis Automatic Speech Recognition untuk Pengambilan Barang*  
**Penulis:** Andhika Hartawan Saputra  
**Platform:** Python · PyBullet · Kinova JACO Gen 2 (6-DoF, gripper tiga jari) · Whisper melalui `faster-whisper`.

> **Status distribusi:** kode utama dan skrip pengujian P01–P12 tersedia. Model robot Kinova (URDF dan mesh), dataset audio, serta log formal tidak disertakan dalam ZIP sumber. **Karena itu, repository ini belum bisa dijalankan hanya dengan `git clone` sebelum aset dan data yang diperlukan disiapkan.** Lihat [Persiapan aset](docs/ASSETS_DAN_DATA.md).

## Ringkasan penelitian

Simulasi WMRA memodelkan lengan Kinova JACO Gen 2 pada kursi roda elektrik virtual untuk memilih, mengambil, mengangkat, dan memindahkan objek dari rak minimarket. Sistem mencakup pemrosesan perintah berbasis teks/ASR, pencarian target melalui kamera virtual pada wrist, inverse kinematics (solver Damped Least Squares pada PyBullet), virtual grasp, lift, pemantauan kontak, handover, dan penempatan ke keranjang.

Lingkungan baseline v12 menggunakan **dua rak, masing-masing tiga tingkat dengan dua objek per tingkat**. Objek memiliki tiga bentuk (balok/kotak, silinder, prisma segitiga) dan empat warna. Sembilan kelas target formal memakai merah, biru, dan kuning; hijau sebagai *distractor*. Platform dapat berpindah antar dua **station yang telah ditentukan**; navigasi otonom dan pemetaan lingkungan tidak dievaluasi.

### Batas interpretasi yang sangat penting

- **P11 dan P12 menggunakan masukan teks valid**, bukan inferensi audio Whisper pada setiap percobaan. Hasil P12 **bukan** angka keberhasilan audio-ke-grasp penuh.
- **Virtual grasp menggunakan fixed constraint PyBullet**, bukan uji gaya cengkeraman fisik, slip, atau validasi force closure.
- Modul pencarian menggunakan keluaran kamera sintetis PyBullet (RGB/depth/segmentation mask) dan informasi objek simulasi; **bukan detektor YOLO**.
- Tidak tercatatnya kontak pada suatu skenario tidak membuktikan seluruh lintasan bebas tabrakan.
- Ini adalah penelitian **simulasi**, bukan produk assistive robot siap digunakan oleh manusia.

## Struktur repository

```text
wmra-asr-pybullet/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── main_simulation/
│       └── 07_indonesian_store_jaco_camera_NEW.py
├── tests/
│   ├── P01/ ... P12/             # 12 skrip pengujian
├── tools/
│   └── preflight.py             # pemeriksaan prasyarat, tidak menjalankan robot
├── assets/
│   └── robots/jaco2_6dof/      # URDF dan mesh disiapkan terpisah (belum ada)
├── data/
│   └── README.md                # instruksi dataset/log (berkas privat di-ignore)
├── docs/
│   ├── ASSETS_DAN_DATA.md
│   ├── PANDUAN_PENGUJIAN_P01_P12.md
│   ├── HASIL_DAN_BATASAN.md
│   ├── ENVIRONMENT_DAN_DEPENDENSI.md
│   ├── LISENSI_DAN_ATRIBUSI.md
│   ├── CHECKLIST_PUBLIKASI.md
│   └── REVISION_LOG.md
└── archive/
    └── zip_asli_2026-11-09/  # salinan byte-for-byte source utama sebelum normalisasi path
```

Struktur di atas hanya menunjukkan berkas/folder penting; **model URDF, file mesh Kinova, metadata audio, dan log formal belum ada** dalam paket ini.

## Persyaratan

- Laptop/PC dengan Python 3 dan akses terminal; Ubuntu 24.04 adalah lingkungan pengembangan yang dilaporkan, namun paket ini **belum diuji bersih** pada instalasi baru.
- PyBullet, NumPy, OpenCV, `faster-whisper` (lihat `requirements.txt`).
- Untuk mode GUI, diperlukan lingkungan desktop/display aktif.
- Pengujian ASR pada konfigurasi semula memakai GPU CUDA dengan `float16`. **CPU** dapat dipakai untuk uji awal memakai `--device cpu --compute_type int8`; performa dan hasil waktu dapat berbeda.
- Model JACO URDF dan seluruh aset yang dirujuk olehnya (lihat di bawah).

### Instalasi (Linux/Ubuntu)

Jalankan dari **root repository**:

```bash
# Jika repository belum diunduh:
git clone https://github.com/DiKe2004/wmra-asr-pybullet.git
cd wmra-asr-pybullet

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Catatan: versi dalam `requirements.txt` **belum dikunci berdasarkan uji instalasi bersih**; lihat [catatan environment](docs/ENVIRONMENT_DAN_DEPENDENSI.md). Beberapa sistem memerlukan paket OS untuk jendela OpenCV/PyBullet. Unduhan model Whisper pertama kali dapat membutuhkan koneksi internet.

### Siapkan model robot dan data

Model **wajib** ada di lokasi berikut sebelum menjalankan simulasi robotik:

```text
assets/robots/jaco2_6dof/j2s6s300.urdf
```

Lengkapi file mesh/material/URDF lain yang digunakan dalam `j2s6s300.urdf` dengan **struktur relatif yang dipertahankan**. Aset tersebut tidak disertakan karena tidak ada dalam ZIP yang diperiksa. Pastikan hak redistribusinya terlebih dahulu. Jika sudah memiliki model di luar repository:

```bash
export WMRA_JACO_URDF="$HOME/Scriptis4_PyBullet/assets/robots/jaco2_6dof/j2s6s300.urdf"
```

Path lama itu **hanya contoh lokasi**, bukan syarat. Default program sekarang mencari model relatif terhadap root repository. File mesh prisma segitiga **dibangkitkan otomatis oleh simulasi** pada `assets/objects/generated_meshes/`.

Data P02/P03/P04 tidak ada dalam ZIP dan harus disiapkan terpisah. Detail kolom CSV dan cara akses dataset: [ASSETS_DAN_DATA.md](docs/ASSETS_DAN_DATA.md).

### Periksa prasyarat sebelum menjalankan

```bash
python tools/preflight.py
```

Perintah tersebut memeriksa keberadaan source, skrip P01–P12, paket Python, dan file URDF. Ia tidak mengimpor atau mengeksekusi simulasi. Untuk memeriksa dokumentasi saja tanpa mewajibkan URDF/dependensi:

```bash
python tools/preflight.py --docs-only
```

## Menjalankan simulasi utama

```bash
python src/main_simulation/07_indonesian_store_jaco_camera_NEW.py
```

GUI PyBullet dan terminal menerima input. Contoh perintah terminal yang dikenali meliputi `ambil balok merah`, `ambil silinder biru`, dan `ambil segitiga kuning`. Dalam mode manual, perintah diberikan melalui terminal; pada antarmuka GUI tersedia kontrol seperti `n`/`p` untuk memilih objek, `h` untuk pick–lift–handover, `r` untuk reset, dan `q` untuk keluar. Petunjuk lengkap juga dicetak oleh program pada startup.

**ASR otomatis dalam simulasi secara default OFF.** Program bisa membaca `data/asr_eval/latest_command.json`, tetapi skrip pembuat file itu (`evaluate_asr_dataset.py`/`realtime_asr_command.py`) **tidak tersedia dalam ZIP sumber**. Jangan klaim tersedia mode audio langsung yang siap digunakan dari repository ini tanpa menambahkan implementasi dan mengujinya.

Anda dapat mengatur lokasi JSON yang telah dihasilkan sistem luar menggunakan variabel `WMRA_ASR_COMMAND_JSON` (lihat catatan ASR di [ASSETS_DAN_DATA.md](docs/ASSETS_DAN_DATA.md)).

## Menjalankan pengujian P01–P12

Seluruh skrip P01–P12 ada di `tests/Pxx/` dan menggunakan file source utama yang sama. Pelaksanaan dari root repository memudahkan konsistensi folder keluaran dan metadata.

Contoh pengujian paling sederhana **setelah model URDF tersedia**:

```bash
# P01: validasi konfigurasi lingkungan (bukan semua pengujian)
python tests/P01/pengujian_01_environment_validation_logger.py

# P05: pilot SATU percobaan saja, bukan pengulangan hasil formal
python tests/P05/pengujian_05_visual_search_detection_logger.py --limit 1

# P12: pilot SATU percobaan saja; output terpisah
python tests/P12/pengujian_12_end_to_end_formal_logger.py \
  --limit 1 --output-dir data/test_logs/pilot_baru/P12
```

**Jangan menjalankan ulang P12 formal dengan `--limit 0` dan menimpa/menyebutnya sebagai hasil asli.** Rerun pada kode yang path-nya sudah diperbaiki dapat menghasilkan checksum source berbeda dan membutuhkan validasi terpisah. Panduan semua perintah P01–P12 ada di [PANDUAN_PENGUJIAN_P01_P12.md](docs/PANDUAN_PENGUJIAN_P01_P12.md).

## Hasil pengujian dan replikasi

Catatan pengujian skripsi sebelumnya melaporkan **P12: 24 dari 27 percobaan berhasil (88,89%)** dengan input teks valid. **ZIP ini tidak memuat log formal untuk memverifikasi angka tersebut secara independen.** Sebelum mengklaim kesesuaian source dengan hasil, periksa `source_sha256` log P12 formal dan dokumen [HASIL_DAN_BATASAN.md](docs/HASIL_DAN_BATASAN.md).

Dataset suara, gambar, CSV/JSON dan video formal dikelola terpisah; akses publiknya harus ditetapkan pemilik data, bukan diasumsikan tersedia. Untuk skripsi, sertakan tautan dataset/log **setelah izin akses, anonimisasi, dan pemeriksaan privacy**.

## Sitasi, atribusi dan hak pakai

Gunakan judul skripsi dan nama penulis di atas saat merujuk pekerjaan ini. Penjelasan pihak ketiga, model robot, dan langkah pemilihan lisensi ada pada [LISENSI_DAN_ATRIBUSI.md](docs/LISENSI_DAN_ATRIBUSI.md).

> **Belum terdapat file `LICENSE` resmi untuk kode ini.** Jangan menganggap publikasi di GitHub otomatis memberikan izin memodifikasi/redistribusi source atau model robot kepada orang lain. Pilih lisensi setelah memverifikasi kepemilikan dan aturan universitas serta lisensi pihak ketiga.

## Status publikasi

Lihat [CHECKLIST_PUBLIKASI.md](docs/CHECKLIST_PUBLIKASI.md) dan [CARA_MEMPERBARUI_GITHUB.md](docs/CARA_MEMPERBARUI_GITHUB.md) untuk langkah unggah. Perbaikan dokumentasi dan path telah disiapkan pada paket revisi, tetapi **uji runtime bersih, pemeriksaan mesh/URDF, penyediaan data, dan keputusan lisensi masih harus dikerjakan di mesin pemilik proyek**.

# WMRA-ASR-PyBullet

**Simulasi Wheelchair-Mounted Robotic Arm Berbasis Pengenalan Suara untuk Pengambilan Barang**

Repositori ini menyediakan implementasi simulasi *wheelchair-mounted robotic arm* (WMRA) berbasis **PyBullet** untuk tugas pencarian dan pengambilan objek pada rak toko serba ada. Penelitian menggunakan **Kinova JACO Gen 2 (6-DoF)** dengan gripper tiga jari pada platform kursi roda elektrik virtual. Masukan perintah berbahasa Indonesia diproses melalui *automatic speech recognition* (ASR) menggunakan **Whisper** dan parser atribut objek.

Proyek ini merupakan kode pendamping skripsi S1 Teknik Robotika dan Kecerdasan Buatan, Universitas Airlangga.

## Gambaran Sistem

Alur pemrosesan perintah dan manipulasi objek terdiri dari:

1. **Pemrosesan perintah:** transkripsi audio menggunakan Whisper (*faster-whisper*) dan pengenalan atribut bentuk serta warna melalui parser.
2. **Pencarian objek:** akuisisi data kamera virtual PyBullet dan pemilihan target berdasarkan atribut objek di lingkungan simulasi.
3. **Manipulasi:** perhitungan *inverse kinematics* (IK) dengan solver *Damped Least Squares* (DLS), gerak menuju target, *virtual grasp*, pengangkatan, dan pemindahan objek.
4. **Penyelesaian:** *handover* ke area pengguna atau penempatan objek pada keranjang, disertai pemantauan kontak sesuai skenario pengujian.

**Lingkungan simulasi:** dua rak, masing-masing tiga tingkat dan dua objek per tingkat (12 objek). Bentuk objek meliputi balok, silinder, dan prisma segitiga. Warna yang digunakan adalah merah, biru, kuning, dan hijau; pada pengujian formal P12, hijau digunakan sebagai *distractor*. Simulasi juga mendukung perpindahan platform di antara dua posisi (*station*) yang telah ditetapkan, bukan navigasi otonom.

## Teknologi

- **Simulator:** PyBullet
- **Bahasa:** Python
- **Robot:** Kinova JACO Gen 2, 6-DoF, gripper tiga jari
- **ASR:** Whisper `small` melalui `faster-whisper`
- **Visual:** kamera sintetis PyBullet (RGB, depth, segmentation mask) dan atribut objek dari simulator
- **IK:** `calculateInverseKinematics()` dengan solver `p.IK_DLS`
- **Manipulasi:** *virtual grasp* menggunakan *fixed constraint* PyBullet

## Struktur Repositori

```text
wmra-asr-pybullet/
├── src/main_simulation/         # Implementasi simulasi utama
├── tests/P01/ ... tests/P12/    # Dua belas skrip evaluasi
├── tools/preflight.py          # Pemeriksaan prasyarat
├── assets/                     # Petunjuk penyediaan model robot
├── data/                       # Petunjuk dataset dan keluaran pengujian
├── docs/                       # Metodologi, panduan dan hasil penelitian
├── requirements.txt            # Dependensi Python
├── README.md
└── LICENSE                     # MIT License (kode proyek)
```

Model robot Kinova (URDF dan mesh), rekaman audio, dan log pengujian formal **tidak termasuk** dalam paket kode ini. Panduan penyediaan aset dan data tersedia pada [Dokumentasi Aset & Data](docs/ASSETS_DAN_DATA.md).

## Instalasi

Lingkungan pengembangan yang digunakan adalah **Ubuntu 24.04 LTS** dengan Python. Instalasi baru pada komputer lain belum diverifikasi secara menyeluruh.

```bash
git clone https://github.com/DiKe2004/wmra-asr-pybullet.git
cd wmra-asr-pybullet

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Setelah instalasi, tersedia pemeriksaan statis prasyarat:

```bash
python tools/preflight.py
```

Pemeriksaan ini melaporkan ketersediaan modul Python, skrip pengujian, dan file URDF; hasil lulus tidak menggantikan uji eksekusi simulasi.

### Model Robot Kinova JACO Gen 2

Simulasi memerlukan file `j2s6s300.urdf` **dan seluruh mesh/berkas pendukung** yang dirujuk model tersebut. Tempatkan model yang kompatibel pada:

```text
assets/robots/jaco2_6dof/j2s6s300.urdf
```

Alternatifnya, gunakan lokasi model yang sudah tersedia di komputer:

```bash
export WMRA_JACO_URDF="/lokasi/model/jaco2_6dof/j2s6s300.urdf"
```

Model dan aset eksternal mengikuti hak cipta/lisensi sumber masing-masing. Lihat [Dokumentasi Aset & Data](docs/ASSETS_DAN_DATA.md) dan [Lisensi & Atribusi](docs/LISENSI_DAN_ATRIBUSI.md).

## Menjalankan Simulasi

Setelah dependensi dan model robot tersedia, jalankan dari direktori utama repositori:

```bash
python src/main_simulation/07_indonesian_store_jaco_camera_NEW.py
```

Contoh perintah teks berbahasa Indonesia:

```text
ambil balok merah
ambil silinder biru
ambil segitiga kuning
```

Program menampilkan kontrol dan informasi penggunaan saat dijalankan. Mode audio langsung membutuhkan komponen pembuat perintah ASR yang terpisah dan tidak disertakan dalam repositori ini; **pengoperasian lewat perintah teks tetap dapat digunakan**.

## Pengujian P01–P12

| Kode | Fokus Pengujian |
|---|---|
| P01 | Validasi lingkungan simulasi |
| P02 | Transkripsi ASR |
| P03 | Parser perintah |
| P04 | Integrasi ASR dan parser |
| P05 | Pencarian objek target (*visual search*) |
| P06 | Pemilihan target dengan *distractor* |
| P07 | Ketercapaian *inverse kinematics* |
| P08 | *Virtual grasp* dan *lift* |
| P09 | Pemantauan kontak robot–lingkungan |
| P10 | *Handover* dan penempatan ke keranjang |
| P11 | Integrasi subsistem manipulasi |
| P12 | Evaluasi formal penyelesaian tugas (*end-to-end* berbasis teks valid) |

Contoh menjalankan validasi lingkungan:

```bash
python tests/P01/pengujian_01_environment_validation_logger.py
```

Contoh pengujian pilot P12 (satu percobaan, terpisah dari data formal):

```bash
python tests/P12/pengujian_12_end_to_end_formal_logger.py \
  --limit 1 --output-dir data/test_logs/pilot_baru/P12
```

P02–P04 memerlukan dataset/hasil transkripsi eksternal. Perintah lengkap, parameter, dan kebutuhan data setiap pengujian dijelaskan dalam [Panduan Pengujian P01–P12](docs/PANDUAN_PENGUJIAN_P01_P12.md).

## Hasil Penelitian

Ringkasan hasil yang dilaporkan pada penelitian:

| Pengujian | Hasil |
|---|---:|
| P01 — Validasi lingkungan | 21/21 pemeriksaan berhasil |
| P05 — Pencarian target | 34/36 percobaan berhasil (94,44%) |
| P12 — Penyelesaian tugas | 24/27 percobaan berhasil (88,89%) |

Angka di atas merujuk pada catatan pengujian formal penelitian; dataset dan log mentah tidak disertakan di repositori ini. Rincian definisi metrik dan batas interpretasi tersedia di [Hasil & Batasan](docs/HASIL_DAN_BATASAN.md).

### Ruang Lingkup dan Keterbatasan

- **P11 dan P12 menggunakan masukan teks yang valid**, bukan inferensi ASR selama setiap percobaan. Hasil P12 tidak merepresentasikan keberhasilan audio-ke-manipulasi secara penuh.
- **Virtual grasp** menggunakan *fixed constraint* simulator, sehingga tidak membuktikan gaya cengkeraman atau stabilitas grasp fisik pada robot nyata.
- Pencarian target memanfaatkan kamera virtual dan informasi objek dari PyBullet; proyek ini tidak menggunakan model deteksi objek seperti YOLO untuk persepsi dunia nyata.
- Perpindahan kursi roda hanya di antara *station* yang telah ditentukan; navigasi otonom dan pemetaan lingkungan tidak dievaluasi.
- Hasil pemantauan kontak terbatas pada kondisi serta mekanisme pencatatan pengujian, bukan jaminan lintasan bebas tabrakan untuk semua situasi.

## Dokumentasi

- [Panduan Pengujian P01–P12](docs/PANDUAN_PENGUJIAN_P01_P12.md)
- [Persyaratan Aset dan Data](docs/ASSETS_DAN_DATA.md)
- [Catatan Lingkungan dan Dependensi](docs/ENVIRONMENT_DAN_DEPENDENSI.md)
- [Hasil Pengujian dan Keterbatasan](docs/HASIL_DAN_BATASAN.md)
- [Lisensi dan Atribusi Pihak Ketiga](docs/LISENSI_DAN_ATRIBUSI.md)

## Sitasi

Jika menggunakan repository ini untuk referensi akademik, cantumkan:

> Saputra, A. H. (2026). *Simulasi Wheelchair-Mounted Robotic Arm Berbasis Automatic Speech Recognition untuk Pengambilan Barang*. Skripsi S1, Universitas Airlangga.

## Lisensi

Kode proyek ini didistribusikan di bawah **[MIT License](LICENSE)**, sepanjang hak lisensi atas kode tersebut dimiliki oleh kontributor proyek. Lisensi ini tidak otomatis berlaku untuk URDF/mesh Kinova, model Whisper, dataset audio, maupun aset pihak ketiga lainnya.

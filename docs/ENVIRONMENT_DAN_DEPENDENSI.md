# Lingkungan perangkat lunak dan dependensi

## Kode yang diperiksa

Empat paket langsung di `requirements.txt` asli adalah `pybullet`, `numpy`, `opencv-python`, `faster-whisper`. File tersebut **belum berisi versi yang dipastikan dapat berjalan bersama**. Konfigurasi penelitian menyebut Ubuntu 24.04 dan penggunaan CUDA/`float16` untuk pengujian ASR semula. Ini **bukan bukti** bahwa paket otomatis jalan di semua versi Python, GPU, atau driver.

File `docs/environment_snapshot.txt` dari ZIP sumber merupakan daftar panjang paket lingkungan Conda lokal, termasuk banyak entri `@ file:///...` yang **tidak cocok dipakai sebagai `requirements.txt` lintas mesin**.

Versi yang terlihat pada snapshot tersebut:

| Paket | Versi tercatat di snapshot |
|---|---|
| PyBullet | `3.2.7` |
| NumPy | `2.2.6` |
| OpenCV Python | `4.12.0.88` |
| faster-whisper | **tidak ditemukan** pada snapshot |

Tabel ini adalah hasil pemeriksaan berkas snapshot yang dilampirkan, **bukan hasil uji ulang**. Jangan menyalin tabel ke artikel sebagai "versi teruji" sebelum dicoba pada environment yang sesungguhnya.

## Catat environment setelah uji bersih

Dari root repository dan virtual environment aktif:

```bash
python --version
python -m pip --version
python -m pip show pybullet numpy opencv-python faster-whisper
python -m pip freeze > docs/pip_freeze_terverifikasi.txt
python tools/preflight.py
```

Simpan hanya versi paket yang relevan; tinjau data lingkungan agar tidak mempublikasikan token atau jalur internal. Bila proyek memakai Conda, dokumentasikan file environment yang **dibuat ulang khusus proyek**, bukan seluruh environment base Anaconda.

## CPU vs CUDA

Pengujian ASR `P02` secara default memakai `--device cuda --compute_type float16`; CUDA/CTranslate2 harus sesuai perangkat dan driver. Pengujian awal CPU yang **bukan duplikasi hasil formal** dapat menggunakan:

```bash
python tests/P02/pengujian_02_asr_transcription_logger.py \
  --device cpu --compute_type int8 --limit 1 --condition clean
```

Perintah masih membutuhkan dataset pada lokasi default. Hasil waktu/akurasi CPU mungkin berbeda dari konfigurasi CUDA lama. Model Whisper `small` dapat diunduh pada eksekusi pertama dan membutuhkan koneksi internet.

## Batas pemeriksaan paket ini

Kami telah melakukan validasi sintaks Python (`compileall`) dan pemeriksaan statis struktur/path. **Belum ada pengujian instalasi `pip install`, pemuatan model URDF di PyBullet, maupun uji P01–P12 lengkap pada mesin baru.**

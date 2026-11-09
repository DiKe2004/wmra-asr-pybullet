# Cara mengirim paket revisi ke GitHub dengan aman

Gunakan panduan ini **setelah memeriksa file dalam ZIP revisi**. Jangan menganggap perubahan lokal sudah otomatis muncul di repository online.

## 1. Buat salinan cadangan

Simpan ZIP repository asli yang diunduh sebelum melakukan penyesuaian. Jika repository lokal memakai Git, buat branch perubahan agar tidak menimpa baseline:

```bash
git checkout -b docs/rapikan-skripsi
```

Jika nama branch sudah ada, pilih nama branch lain. Hindari `git push --force`.

## 2. Siapkan folder kerja

Cara yang paling mudah adalah mengekstrak paket `WMRA_GitHub_Revisi.zip`. Folder hasil ekstrak bernama `wmra-asr-pybullet/` dan berisi README, `src`, `tests`, `docs`, dan `tools`.

Untuk mengganti isi repository Git lokal, salin **isi folder** hasil ekstrak ke root repository kloning GitHub, bukan mengunggah ZIP sebagai satu berkas. Jangan menghapus folder `.git`. Jangan menyalin atau menyertakan dataset mentah tanpa izin.

Jika terbiasa menggunakan terminal, contoh setelah Anda membuat salinan kerja repository Git:

```bash
# Jalankan dari root repository Git lokal yang sudah ada
rsync -av --exclude '.git' \
  /lokasi/ekstrak/wmra-asr-pybullet/ ./

git status
```

**Ganti** `/lokasi/ekstrak/` dengan folder Anda. Contoh `rsync` sengaja **tanpa `--delete`** agar file lama lain tidak terhapus secara otomatis. Jika ada file lama yang tidak dipakai, tinjau dan hapus secara manual setelah yakin.

## 3. Audit sebelum commit

1. Lihat `git status` dan daftar berkas yang berubah.
2. Jangan commit `data/audio_dataset/`, `data/asr_eval/`, `data/test_logs/`, file WAV, credential, atau token API.
3. Pastikan URDF/mesh **belum** diunggah jika hak redistribusinya belum jelas.
4. Jalankan `python tools/preflight.py --docs-only` dan `python -m compileall -q src tests tools`.
5. Jika model telah tersedia, jalankan pemeriksaan penuh `python tools/preflight.py`.
6. Periksa konsistensi dengan hasil P12 lama, khususnya `source_sha256` yang berbeda setelah normalisasi path.

## 4. Commit dan unggah

```bash
git add README.md .gitignore requirements.txt src tests tools docs assets data archive
git diff --cached --stat
git diff --cached --check
# Tinjau juga perubahan sensitif pada diff sebelum commit:
git diff --cached

git commit -m "docs: lengkapi panduan skripsi dan perbaiki path repository"
git push -u origin HEAD
```

Jika push ditolak karena perubahan remote, lakukan `git fetch` dan rekonsiliasi, **jangan langsung force push**. Anda juga bisa membuka Pull Request dari branch revisi ke branch utama. Jika tidak ingin memakai terminal Git, gunakan menu **Add file → Upload files** pada GitHub dan unggah folder/berkas hasil ekstrak (bukan ZIP sebagai satu berkas), lalu cek struktur.

## 5. Sesudah publikasi

- Coba URL GitHub dalam Incognito tanpa login.
- Klik README, setiap skrip P01–P12, serta tautan dalam README.
- Pastikan tidak ada audio privat atau file `.env` di daftar file.
- Catat URL commit (hash) yang telah final untuk dicantumkan di lampiran skripsi.
- Perbarui `CHECKLIST_PUBLIKASI.md` hanya berdasarkan hasil pemeriksaan yang sungguh dilakukan.

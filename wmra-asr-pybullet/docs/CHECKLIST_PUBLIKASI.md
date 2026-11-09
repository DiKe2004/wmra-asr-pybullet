# Checklist akhir sebelum dipakai sebagai Lampiran Source Code skripsi

Tanda centang hanya jika sudah diperiksa pada repository GitHub yang benar-benar akan dipublikasikan, bukan berdasarkan asumsi dari ZIP.

## Dokumentasi source

- [x] README diisi lengkap dan code fence ditutup.
- [x] Struktur folder, komponen, batas penelitian, dan contoh CLI terdokumentasi.
- [x] Skrip P01–P12 ditemukan di paket.
- [x] Petunjuk pengujian P01–P12 dibuat.
- [x] Default `--source` skrip robotik diarahkan ke path repository.
- [x] Source utama menggunakan path relatif root (dengan pilihan override via environment).
- [x] Source utama asli dari ZIP disimpan untuk penelusuran perubahan path.

## **Belum bisa dicentang hanya dari ZIP**

- [ ] Model `j2s6s300.urdf` **dan semua mesh** tersedia atau diunduh sesuai petunjuk yang legal.
- [ ] Sumber, versi, dan lisensi model JACO dicatat.
- [ ] Dependency Python diinstal pada lingkungan baru tanpa error.
- [ ] Preflight penuh `python tools/preflight.py` lulus.
- [ ] P01 berhasil dijalankan pada model yang sama (uji bersih, bukan klaim pengulangan formal).
- [ ] Salah satu smoke test P05/P12 berhasil dijalankan di folder baru.
- [ ] `requirements.txt`/manifest environment mencerminkan **versi teruji**; belum otomatis terpin.
- [ ] Dataset asli + metadata telah diperiksa, hak aksesnya jelas, tanpa info pribadi yang dipublikasi tanpa izin.
- [ ] Log formal P01–P12 diarsipkan dan dapat diverifikasi secara tepat oleh penguji.
- [ ] `source_sha256` pada log P12 formal dibandingkan dengan baseline source.
- [ ] README/skripsi/slide tidak lagi mengklaim P11/P12 menjalankan audio→robot dalam satu trial.
- [ ] Tidak ada file token/password/credential dalam file yang akan ditrack **maupun riwayat Git**.
- [ ] Lisensi source milik peneliti dipilih setelah memeriksa hak dan ketentuan institusi.
- [ ] Lisensi pihak ketiga sudah diidentifikasi, dan hanya aset yang boleh didistribusikan yang diunggah.
- [ ] Link GitHub dibuka dari jendela incognito (tanpa login).
- [ ] Link pada Lampiran 1 skripsi menuju repository publik yang benar.

## Peringatan integritas

- Jangan menimpa hasil formal P12 (24 sukses, 3 gagal dari 27) dengan hasil pilot atau rerun.
- Jangan mengklaim telah dilakukan *fresh install test* sebelum menjalankan pengujian di mesin sendiri.
- Jika terpaksa mengubah parameter algoritma, dokumentasikan sebagai revisi versi/eksperimen terpisah dari baseline v12.
- Saat sudah siap, buat commit/release yang dapat dilacak dan catat tag/commit SHA di lampiran skripsi.

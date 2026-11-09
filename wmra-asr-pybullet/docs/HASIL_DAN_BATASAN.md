# Hasil penelitian yang dilaporkan dan batas interpretasinya

**Status verifikasi:** isi repository ZIP yang diperiksa berisi source dan penguji P01–P12, **tidak memuat** dataset atau log eksekusi formal. Angka berikut berasal dari **catatan skripsi/riwayat proyek yang tersedia terpisah**, bukan hasil eksekusi atau perhitungan ulang dari repository ini.

| Pengujian | Hasil yang sebelumnya dilaporkan | Status pemeriksaan dalam paket ini |
|---|---|---|
| P01 — Validasi lingkungan | 21/21 pemeriksaan berhasil | Kode tersedia; log formal tidak tersedia |
| P05 — Pencarian target | 34/36 (94,44%) | Kode tersedia; log formal tidak tersedia |
| P12 — Penyelesaian tugas | 24/27 (88,89%), **3 percobaan gagal** | Kode tersedia; log formal tidak tersedia |

Angka P02–P04 dan P06–P11 belum dimasukkan sebagai "nilai terverifikasi" di tabel karena log finalnya tidak tersedia dalam paket ini. Untuk laporan ilmiah, ambil angka tersebut dari arsip output aslinya dan periksa nama metrik serta denominator setiap pengujian.

## Definisi yang tidak boleh tertukar

- **ASR P02:** menilai transkripsi audio menjadi teks dengan Word Error Rate (WER). Kondisi yang direncanakan: clean, SNR 20 dB, SNR 10 dB, dan SNR 5 dB. Ini tidak setara dengan keberhasilan pengambilan barang.
- **P03:** menguji parser **teks**, bukan kinerja Whisper.
- **P04:** mengukur kemampuan hasil transkripsi untuk memuat atribut bentuk/warna yang dapat dipakai parser.
- **P05/P06:** target selection dalam simulasi bukan bukti kemampuan deteksi objek umum dari gambar dunia nyata.
- **P07:** mengevaluasi kemampuan *inverse kinematics* menjangkau pose, bukan bukti keberhasilan grasp.
- **P08:** pengambilan dan pengangkatan menggunakan *virtual grasp* berbasis PyBullet fixed constraint; tidak membuktikan kekuatan gripper nyata atau stabilitas gaya fisik.
- **P09:** evaluasi kontak sesuai definisi dan lingkup pemeriksaan skrip; tidak sama dengan sertifikat bebas tabrakan total.
- **P10:** memeriksa tahapan handover/keranjang dengan kondisi dan toleransi simulasi.
- **P11/P12:** input **teks valid**. Hasil P12 sebesar 88,89% merupakan tingkat keberhasilan rangkaian manipulasi yang diuji, **bukan** keberhasilan ASR+robot secara penuh pada setiap trial audio.

## Konfigurasi yang harus dijaga jika hasil dibandingkan

Catatan baseline v12:

- JACO Gen 2, 6-DoF, gripper tiga jari; kursi roda virtual.
- Dua rak × tiga tingkat × dua objek per tingkat = 12 objek.
- Tiga bentuk; kelas target formal tiga warna (merah/biru/kuning), hijau hanya distractor.
- Dua station, perpindahan yang telah ditentukan sebesar 1,15 m, bukan navigasi otonom.
- Offset kamera wrist lokal: `WRIST_POV_LOCAL_OFFSET = [0.040, 0.140, 0.075]`.
- Maksimal perbaikan alignment saat release: `RELEASE_GUARD_MAX_ALIGN_ITERS = 12`.

Perubahan algoritma/parameter akan menghasilkan **eksperimen baru**. Perubahan path saja tidak otomatis mengubah algoritma, tetapi tetap mengubah checksum file source; verifikasi dulu sebelum mengaitkan dengan rekaman hasil formal.

## Apa yang harus ditambahkan sebelum klaim reproduksibilitas

1. Indeks dan hash arsip hasil P01–P12 (CSV/JSON/log) yang memiliki tanggal dan sumber jelas.
2. Penjelasan hak akses dataset, lisensi, dan anonimisasi rekaman suara.
3. Pembandingan `source_sha256` dari log formal dengan kode baseline asli.
4. Hasil uji instalasi baru (Python/dependensi/model), minimal P01 dan pilot P05/P12.
5. Pemisahan tegas hasil formal historis dan hasil uji ulang pada source yang sudah diberi patch path.

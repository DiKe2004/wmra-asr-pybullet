# Lisensi, sitasi, dan atribusi

**Status:** ZIP awal tidak mengandung file `LICENSE`. Paket revisi **tidak memberikan lisensi otomatis** atas karya penelitian atau model pihak ketiga. Pemilik repository perlu mengambil keputusan izin penggunaan dengan memperhatikan hak universitas/pembimbing, kepemilikan kode, dan lisensi pustaka/model yang digunakan.

## Pemisahan hak cipta

1. **Source code yang dibuat peneliti**: tentukan apakah akan didistribusikan dengan lisensi permisif (misalnya MIT) atau tetap *all rights reserved*. Jangan mencantumkan lisensi yang tidak disetujui pemegang hak.
2. **Model Kinova dan mesh**: berkas tersebut tidak ada dalam ZIP, dan hak untuk mendistribusikannya belum dibuktikan. Periksa lisensi sumber model sebelum menambahkan ke GitHub. Jika redistribusi dilarang, gunakan petunjuk pemasangan terpisah.
3. **Paket Python**: `pybullet`, `numpy`, `opencv-python`, `faster-whisper` adalah dependensi pihak ketiga dengan lisensinya masing-masing. Cantumkan sumber/atribusi yang relevan ketika redistribusi artefak diperlukan.
4. **Dataset suara dan hasil rekaman**: sekalipun penelitian dilakukan sendiri, hak pemilik suara dan privasi harus diperhatikan sebelum unggah publik.

## Tindakan yang perlu dilakukan pemilik

- [ ] Pastikan siapa pemegang hak cipta source asli.
- [ ] Pilih lisensi untuk **source milik sendiri** dan tambahkan file `LICENSE` di root repository sesuai keputusan resmi.
- [ ] Isi sumber model JACO: URL, nama paket/revisi/commit, lisensi, hak redistribusi.
- [ ] Isi sumber model Whisper/weight apabila redistribusi lokal dilakukan.
- [ ] Periksa dataset audio, izin peserta dan status akses publik/terbatas.
- [ ] Sertakan rujukan tesis sesuai peraturan institusi.

Jangan membuat pernyataan seperti "semua aset bebas digunakan secara komersial" sebelum memiliki bukti sumber lisensi masing-masing.

## Contoh format atribusi aset (lengkapi dengan data nyata)

| Aset | Sumber | Versi/commit | Lisensi | Boleh diunggah? |
|---|---|---|---|---|
| Kinova `j2s6s300.urdf` dan mesh | **Belum dicatat** | **Belum dicatat** | **Belum diverifikasi** | **Belum diputuskan** |
| Model Whisper | Dokumentasi paket yang dipakai | **Belum dicatat** | Periksa ketentuan model/berkas aktual | Tidak disertakan di ZIP |
| Dataset suara | Arsip penelitian | **Belum dicatat** | Persetujuan/izin diperlukan | Belum diputuskan |

## Sitasi

Penulis: **Andhika Hartawan Saputra**. Judul skripsi pada README. Tambahkan tahun terbit, institusi, URL/DOI jika sudah resmi. Hindari menulis DOI atau status publikasi yang belum diberikan institusi.

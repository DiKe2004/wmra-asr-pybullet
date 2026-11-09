# Catatan revisi paket (2026-11-09)

Sumber: ZIP yang diberikan pengguna `wmra-asr-pybullet-main(1).zip`.

- README diperbaiki dan diperluas ke Bahasa Indonesia.
- Default path `--source` pada 9 skrip uji disesuaikan ke `src/main_simulation/` menggunakan `__file__`.
- Source utama: hanya penentuan `PROJECT_ROOT`, `ASR_COMMAND_JSON`, `JACO_URDF`, dan `MESH_DIR` dibuat portabel.
- Fungsi dan kelas source utama lolos perbandingan AST identik dengan ZIP asli.
- **Tidak ada perubahan yang disengaja terhadap algoritma, threshold, parameter v12, atau data pengujian.**
- Source utama original (byte-for-byte) disimpan di `archive/zip_asli_2026-11-09/`.
- Original source SHA-256: `e8d8522d9a72c767f3284437ec8b6032eac7efe13f6d1f10a5b314958db46eef`.
- Source aktif SHA-256 setelah perbaikan path: `e2ae172630e5f68dcaf94b6ee08e3a7cbdd985ee71e9ed8a938a396b82fe167e`.
- Karena source berubah pada level penentuan path, **SHA-256 berbeda** dari ZIP asli. Jangan samakan hasil uji formal lama dengan pengujian ulang sampai diperiksa log aslinya.
- Model robot Kinova URDF/mesh serta dataset audio tidak disertakan karena memang tidak ada dalam ZIP awal; harus disediakan secara sah oleh pemiliknya.
- Performa berjalan, pemasangan paket, kesesuaian log formal, dan lisensi aset belum diverifikasi.
- Metadata tanggal dalam nama folder bukan tanggal pelaksanaan pengujian formal.

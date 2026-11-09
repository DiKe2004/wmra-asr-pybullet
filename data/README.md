# Folder data lokal

Berkas audio, metadata, keluaran ASR, log formal, tangkapan layar dan video **tidak disertakan** dalam paket ZIP sumber. Folder di bawah `data/` yang rawan berisi data pribadi dan keluaran eksperimen diabaikan oleh `.gitignore`.

- `audio_dataset/metadata.csv` serta file WAV pendukung diperlukan untuk pengujian P02/P03.
- P04 membaca CSV hasil P02.
- `asr_eval/latest_command.json` adalah antarmuka JSON opsional ke simulasi utama, tetapi generator JSON tidak tersedia di repository.
- `test_logs/` dan `test_logs_timestamped/` menjadi lokasi keluaran beberapa tes.

Jangan commit rekaman suara mentah atau metadata dengan informasi identitas tanpa hak dan izin. Dokumentasi lebih lengkap: `docs/ASSETS_DAN_DATA.md`.

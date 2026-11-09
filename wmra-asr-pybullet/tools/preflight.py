#!/usr/bin/env python3
"""Validasi statis prasyarat repository; tidak mengimpor atau mengeksekusi simulasi."""
from __future__ import annotations
import argparse
import importlib.util
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / 'src/main_simulation/07_indonesian_store_jaco_camera_NEW.py'
URDF_DEFAULT = ROOT / 'assets/robots/jaco2_6dof/j2s6s300.urdf'
REQUIRED = {
    'pybullet': 'pybullet',
    'pybullet_data': 'pybullet',
    'numpy': 'numpy',
    'cv2': 'opencv-python',
    'faster_whisper': 'faster-whisper',
}

def report(status: str, description: str) -> None:
    print(f'[{status:5}] {description}')

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--docs-only', action='store_true', help='Cek kelengkapan source/dokumen tanpa dependensi dan URDF')
    args = parser.parse_args()
    problems = 0
    report('INFO', f'Root: {ROOT}')
    report('INFO', f'Python: {sys.version.split()[0]}')
    if MAIN.is_file():
        report('OK', 'Source simulasi utama tersedia')
    else:
        report('ERROR', f'Source simulasi utama tidak ditemukan: {MAIN}')
        problems += 1
    for n in range(1, 13):
        scripts = list((ROOT / f'tests/P{n:02}').glob('*.py'))
        if len(scripts) == 1:
            report('OK', f'P{n:02}: {scripts[0].name}')
        else:
            report('ERROR', f'P{n:02}: perlu 1 skrip Python, ditemukan {len(scripts)}')
            problems += 1
    needed_docs = [ROOT/'README.md', ROOT/'requirements.txt',
                   ROOT/'docs/PANDUAN_PENGUJIAN_P01_P12.md',
                   ROOT/'docs/ASSETS_DAN_DATA.md',
                   ROOT/'docs/HASIL_DAN_BATASAN.md',
                   ROOT/'docs/LISENSI_DAN_ATRIBUSI.md',
                   ROOT/'docs/CHECKLIST_PUBLIKASI.md']
    for f in needed_docs:
        if f.is_file():
            report('OK', f'Dokumen: {f.relative_to(ROOT)}')
        else:
            report('ERROR', f'Dokumen tidak ada: {f.relative_to(ROOT)}')
            problems += 1
    if not args.docs_only:
        for module, package in REQUIRED.items():
            if importlib.util.find_spec(module) is not None:
                report('OK', f'Modul Python tersedia: {module}')
            else:
                report('ERROR', f'Modul Python tidak ditemukan: {module}; instal paket {package}')
                problems += 1
        urdf = Path(os.path.expanduser(os.environ.get('WMRA_JACO_URDF', str(URDF_DEFAULT))))
        if urdf.is_file():
            report('OK', f'URDF tersedia: {urdf}')
            report('WARN', 'Dependency mesh/texture dan validitas URDF tetap harus dicoba di PyBullet')
        else:
            report('ERROR', f'URDF JACO belum ada: {urdf}')
            report('INFO', 'Lihat docs/ASSETS_DAN_DATA.md atau set WMRA_JACO_URDF')
            problems += 1
    else:
        report('INFO', '--docs-only: pemeriksaan dependensi dan URDF dilewati')
    report('INFO', f'Total isu: {problems}. Preflight bukan jaminan runtime atau reproduksi hasil formal.')
    return 0 if problems == 0 else 1

if __name__ == '__main__':
    raise SystemExit(main())

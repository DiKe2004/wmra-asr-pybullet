import os
import time
import cv2
import math
import numpy as np
import json
import threading
import queue
import pybullet as p
import pybullet_data
import random
import csv

# File hasil ASR terbaru dari evaluate_asr_dataset.py
ASR_COMMAND_JSON = "data/asr_eval/latest_command.json"

# =========================================================
# AUTO ASR → VISUAL SEARCH MODE
# =========================================================
# Jika True, simulasi otomatis membaca latest_command.json saat file berubah.
# Tidak perlu tekan tombol v.
# Penting:
# - command ASR TIDAK langsung memilih objek dari metadata.
# - command ASR hanya menjadi shape+color target.
# - robot tetap harus melakukan visual search memakai kamera wrist POV.
AUTO_ASR_VISUAL_SEARCH_ENABLED = False  # SAFE DEFAULT: tidak auto jalan saat startup
AUTO_ASR_POLL_INTERVAL_SEC = 0.35
AUTO_ASR_RUN_ONLY_ON_NEW_COMMAND = True
AUTO_ASR_LAST_SIGNATURE = None
AUTO_ASR_BUSY = False

# Guard supaya simulasi TIDAK mengeksekusi latest_command.json lama saat startup.
AUTO_ASR_IGNORE_EXISTING_COMMAND_ON_START = True
AUTO_ASR_REQUIRE_JSON_AFTER_SIM_START = True
AUTO_ASR_ARMED_AT_TIME = None

# Hybrid mode:
# - Terminal prompt tetap bisa dipakai setelah ASR.
# - ASR hanya menjalankan command baru satu kali.
# - latest_command.json yang sama tidak dibaca/print berulang-ulang.
AUTO_ASR_PROMPT_PRIORITY = True
AUTO_ASR_PROCESSED_SIGNATURES = set()
AUTO_ASR_DISABLE_AFTER_ONE_COMMAND = False
# =========================================================
# PATH URDF ROBOT
# =========================================================
# Kinova JACO2 / JACO Gen 2 6-DoF, 3-finger gripper
JACO_URDF = os.path.expanduser(
    "~/Scriptis4_PyBullet/assets/robots/jaco2_6dof/j2s6s300.urdf"
)

# Folder untuk mesh segitiga
MESH_DIR = os.path.expanduser("~/Scriptis4_PyBullet/assets/objects/generated_meshes")
os.makedirs(MESH_DIR, exist_ok=True)
TRIANGLE_PRISM_OBJ = os.path.join(MESH_DIR, "triangle_prism.obj")


# =========================================================
# PARAMETER RAK MINIMARKET INDONESIA (ASUMSI SIMULASI)
# =========================================================
SHELF_HEIGHT = 1.40   # meter, rak 3 bagian/tingkat
SHELF_WIDTH  = 0.90   # meter
SHELF_DEPTH  = 0.45   # meter
SHELF_LEVELS = 3

# Posisi rak
SHELF_BACK_CENTER = [0.90, 0.0, SHELF_HEIGHT / 2.0]

# Posisi tingkat rak: bawah, tengah, atas
SHELF_Z_LEVELS = [0.50, 0.82, 1.15]

# Ukuran papan rak
SHELF_BOARD_THICKNESS = 0.03
BACK_PANEL_THICKNESS = 0.04
SIDE_PANEL_THICKNESS = 0.03

# Posisi wheelchair elektrik ringan / mount robot
WHEELCHAIR_BASE_POS = [-0.50, -0.02, 0.18]
WHEELCHAIR_BASE_SIZE = [0.62, 0.70, 0.12]

# Home pose awal JACO 6-DoF agar tidak stuck di posisi nol
JACO6_HOME_POSITIONS = [
    0.0,
    2.6,
    1.3,
    0.0,
    1.4,
    0.0
]



# Posisi robot dibuat lebih tinggi agar reach ke rak atas lebih mudah
ROBOT_BASE_POS = [-0.02, -0.02, 0.56]

# Posisi standby aman di depan robot, bukan joint home pose
SAFE_STANDBY_POSITION = [
    ROBOT_BASE_POS[0] + 0.28,
    ROBOT_BASE_POS[1],
    ROBOT_BASE_POS[2] + 0.32
]

# Handover target final:
# satu definisi saja, tidak boleh ada HANDOVER_POS dobel.
#
# Target dipindahkan:
# - lebih ke kiri dari arah kursi roda / area pengguna
# - lebih rendah dari versi sebelumnya
# - tetap sedikit di atas handrest
#
# Dengan ROBOT_BASE_POS = [-0.02, -0.02, 0.56],
# perkiraan koordinat akhir:
# x ~= 0.18
# y ~= 0.28
# z ~= 0.70
HANDOVER_POS = [
    ROBOT_BASE_POS[0] - 0.20,
    ROBOT_BASE_POS[1] + 0.30,
    ROBOT_BASE_POS[2] + 0.14
]

HANDOVER_SIZE = [0.22, 0.16, 0.02]

# =========================================================
# PARAMETER BASKET / LAP TRAY DI AREA HANDOVER
# =========================================================
# Basket ditambahkan tanpa mengubah logic motion lama.
# HANDOVER_POS tetap menjadi target final-align.
BASKET_ENABLED = True
BASKET_CENTER = [HANDOVER_POS[0], HANDOVER_POS[1], HANDOVER_POS[2] - 0.055]
BASKET_OUTER_SIZE = [0.34, 0.26, 0.12]
BASKET_WALL_THICKNESS = 0.014
BASKET_FLOOR_THICKNESS = 0.012
BASKET_WALL_HEIGHT = 0.105
BASKET_SUPPORT_MARGIN = 0.030
BASKET_INFO = None

# =========================================================
# MULTI RACK + MOBILE WHEELCHAIR STATION MODE
# =========================================================
MULTI_RACK_MODE = True

# Rak 1 tetap di posisi lama.
RACK_1_Y_OFFSET = 0.0

# Rak 2 berada di arah maju wheelchair (+Y).
RACK_2_Y_OFFSET = 1.15

ACTIVE_RACK_ID = 1
ACTIVE_RACK_Y_OFFSET = RACK_1_Y_OFFSET

WHEELCHAIR_STATIONS = {
    1: {
        "rack_id": 1,
        "platform_y_offset": 0.0,
        "rack_y_offset": RACK_1_Y_OFFSET,
    },
    2: {
        "rack_id": 2,
        "platform_y_offset": RACK_2_Y_OFFSET,
        "rack_y_offset": RACK_2_Y_OFFSET,
    },
}

CURRENT_STATION_ID = 1
CURRENT_PLATFORM_Y_OFFSET = 0.0

MOBILE_PLATFORM_BODY_IDS = []
MOBILE_PLATFORM_ORIGINAL_POSES = {}

# Simpan posisi original supaya bisa digeser relatif.
ROBOT_BASE_POS_ORIGINAL = ROBOT_BASE_POS[:]
HANDOVER_POS_ORIGINAL = HANDOVER_POS[:]
BASKET_CENTER_ORIGINAL = BASKET_CENTER[:]

# False = scene polosan, tanpa tulisan nama objek di atas rak.
SHOW_WORLD_OBJECT_LABELS = False

# Hubungkan computer vision dengan station wheelchair.
# Kandidat dari kamera hanya boleh diterima kalau berada di rak aktif.
# Jadi kalau station 1 sedang scan rak 1 tetapi kamera melihat objek rak 2,
# kandidat itu ditolak dulu dan baru diterima setelah wheelchair pindah ke station 2.
ACTIVE_RACK_FILTER_ENABLED = True
ACTIVE_RACK_Y_MARGIN = 0.52

# Randomize tetap aktif, tetapi object tidak ditukar lintas rak.
# Ini menjaga skenario: target tertentu bisa memang berada di rak 2,
# sehingga fallback station 1 -> station 2 terlihat jelas.
OBJECT_RANDOMIZE_KEEP_RACK_GROUP = True

# Slot objek sekarang dibuat rata:
# 2 objek per shelf/level pada rak 1 dan 2 objek per shelf/level pada rak 2.
# Karena randomize menukar objek antar slot di rak yang sama,
# jumlah objek per shelf tetap rata, tetapi susunan bentuk/warna tetap berubah.
OBJECT_RANDOMIZE_EQUAL_2_PER_SHELF = True


# =========================================================
# POV KAMERA MOUNTED DI DEKAT WRIST / LINK AKHIR
# =========================================================
# Revisi:
# - Kamera TIDAK dibuat sebagai body/objek visual di simulasi.
# - Tidak ada model kamera, tidak ada bracket, tidak ada collision, tidak ada constraint.
# - Yang dipakai hanya POV virtual dari posisi dekat wrist/end-effector.
# - RGB window mengikuti posisi link akhir robot secara realtime.
WRIST_POV_CAMERA_ENABLED = True
WRIST_POV_LOOK_AT_SELECTED_OBJECT = True
WRIST_POV_USE_WRIST_LINK = True

# Offset kamera relatif terhadap wrist/last-arm-link.
# Posisi ini dibuat tidak terlalu di ujung gripper, tapi dekat link sebelum end-effector.
# v12 percobaan: kamera virtual dimajukan/digeser ke sisi-atas wrist.
# Offset ini RIGID pada frame link wrist, tidak pindah ke base/wheelchair,
# dan tetap sama pada seluruh scan dan trial. Nilai ini perlu divalidasi
# pada GUI: self-occlusion harus turun tanpa memotong shelf atas/bawah.
# Baseline v11 bila perlu dikembalikan: [0.000000, 0.065000, 0.015000]
WRIST_POV_LOCAL_OFFSET = [0.040000, 0.140000, 0.075000]  # v12 mount trial, meter
# Catatan: log tuning 2026-08-18 adalah referensi offset LAMA (v11),
# bukan hasil validasi offset baru v12. Jangan gunakan world-eye lama
# sebagai bukti posisi kamera setelah revisi mount.


WRIST_POV_FORWARD_AXIS_LOCAL = [1.0, 0.0, 0.0]
WRIST_POV_UP_AXIS_LOCAL = [0.0, 0.0, 1.0]
WRIST_POV_TARGET_Z_OFFSET = 0.030
WRIST_POV_LOOK_DISTANCE = 0.90

# =========================================================
# STATIC OVERVIEW CAMERA FOR VISUAL SEARCH
# =========================================================
# Saat visual search, wrist hanya pindah SEKALI ke pose overview.
# Tidak lagi menyapu kiri-kanan / atas-bawah tiap slot.
VISUAL_SEARCH_STATIC_OVERVIEW_MODE = False  # v8: kembali ke 6 scan, bukan 1 overview

# Jaga horizon kamera tetap tegak saat visual search.
# Ini mencegah gambar berputar mengikuti roll/pitch wrist.
VISUAL_SEARCH_STATIC_WORLD_UP = True

# Visual search tetap 6 scan per rak (2 posisi x 3 shelf),
# tetapi arah kamera dikunci lurus ke +X dunia.
# Jadi kamera tidak yaw/pitch/roll mengikuti setiap target scan.
VISUAL_SEARCH_LOCK_WORLD_VIEW_DIRECTION = True
VISUAL_SEARCH_WORLD_VIEW_DIRECTION = [1.0, 0.0, 0.0]
VISUAL_SEARCH_WORLD_VIEW_DISTANCE = 1.00

# =========================================================
# REALTIME FIXED CAMERA MODE
# =========================================================
# Kamera tetap realtime, tetapi TIDAK mengikuti posisi/orientasi wrist.
# Eye dipasang virtual relatif terhadap base wheelchair/robot dan selalu
# menghadap lurus ke +X dunia. Jadi tidak spin, tidak freeze.
#
# Catatan akademik:
# saat mode ini True, kamera secara fungsi adalah fixed camera pada platform/
# wheelchair, BUKAN lagi wrist-mounted camera.
REALTIME_FIXED_CAMERA_MODE = False
REALTIME_FIXED_CAMERA_EYE_OFFSET = [-0.12, 0.00, 0.28]
REALTIME_FIXED_CAMERA_LOOK_DIRECTION = [1.0, 0.0, 0.0]
REALTIME_FIXED_CAMERA_LOOK_DISTANCE = 1.00
REALTIME_FIXED_CAMERA_WORLD_UP = [0.0, 0.0, 1.0]

# =========================================================
# WRIST CAMERA — REALTIME, TAPI TANPA SPIN
# =========================================================
# Posisi/eye kamera tetap benar-benar mengikuti wrist.
# Hanya ORIENTASI pandang yang dikunci ke world frame agar gambar
# tidak ikut yaw/pitch/roll joint wrist.
WRIST_POV_LOCK_WORLD_ORIENTATION = True
WRIST_POV_WORLD_VIEW_DIRECTION = [1.0, 0.0, 0.0]
WRIST_POV_WORLD_VIEW_DISTANCE = 1.00
WRIST_POV_WORLD_UP = [0.0, 0.0, 1.0]

# =========================================================
# P05 diagnostic revision: downward aim ONLY for lower shelf
# =========================================================
# Kamera tetap berposisi di wrist, realtime, world-up tetap vertikal.
# Saat forced target scan shelf bawah: yaw tetap +X, roll tetap 0,
# hanya sudut elevasi (pitch) diperbolehkan mengikuti tinggi shelf.
# Shelf tengah/atas sengaja TIDAK diubah agar hasil baseline dapat dibandingkan.
VISUAL_SEARCH_LOW_SHELF_AIM_ENABLED = True
VISUAL_SEARCH_LOW_SHELF_TARGET_Z_LIMIT = 0.70  # target shelf bawah ~0.58 m
VISUAL_SEARCH_LOW_SHELF_MAX_DOWN_DEG = 52.0
VISUAL_SEARCH_LOW_SHELF_MAX_UP_DEG = 15.0
VISUAL_SEARCH_LOW_SHELF_MIN_FORWARD_M = 0.18

# Realtime camera tetap aktif selama arm bergerak.
# Frame hasil visual search juga ditampilkan sesudah pose settle.
VISUAL_SEARCH_DISPLAY_ONLY_AFTER_SETTLE = False

# Pose overview relatif terhadap base robot / wheelchair aktif.
# Bisa dituning sedikit jika seluruh 6 objek belum masuk frame.
VISUAL_SEARCH_OVERVIEW_X_OFFSET = 0.14
VISUAL_SEARCH_OVERVIEW_Y_OFFSET = 0.00
VISUAL_SEARCH_OVERVIEW_Z_OFFSET = 0.52

# Kamera melihat pusat rak aktif.
VISUAL_SEARCH_OVERVIEW_TARGET_X = 0.54
VISUAL_SEARCH_OVERVIEW_TARGET_Z = 0.83
WRIST_POV_FOV = 74
WRIST_POV_NEAR = 0.01
WRIST_POV_FAR = 2.80

# Kamera realtime tetap di-update selama motion.
# Pose kamera fixed relatif terhadap wheelchair/base, sehingga tidak ikut spin wrist.
WRIST_POV_LIVE_DURING_MOTION = True
WRIST_POV_UPDATE_EVERY_N_STEPS = 4
CAMERA_WINDOW_NAME = "RGB Wrist POV Camera - Realtime Fixed Orientation"
LIVE_CAMERA_CONTEXT = {
    "robot_data": None,
    "objects": None,
    "selected_name": None
}

# =========================================================
# OPENCV GUI SAFE MODE
# =========================================================
# Beberapa environment Ubuntu/conda memakai opencv headless,
# sehingga cv2.namedWindow/cv2.imshow bisa error.
# Dengan helper ini simulasi tetap jalan; camera window dimatikan otomatis
# kalau backend GUI OpenCV tidak tersedia.
OPENCV_GUI_AVAILABLE = True


def safe_named_window(window_name, flags=cv2.WINDOW_NORMAL):
    global SHOW_CAMERA_WINDOWS, OPENCV_GUI_AVAILABLE

    if not SHOW_CAMERA_WINDOWS:
        return False

    try:
        cv2.namedWindow(window_name, flags)
        OPENCV_GUI_AVAILABLE = True
        return True
    except cv2.error as exc:
        OPENCV_GUI_AVAILABLE = False
        SHOW_CAMERA_WINDOWS = False
        print("\nWARNING: OpenCV GUI backend tidak tersedia.")
        print("Camera window otomatis dimatikan, simulasi tetap lanjut.")
        print("Penyebab umum: package opencv-python-headless atau OpenCV tanpa GTK/Qt support.")
        print("Detail:", exc)
        return False


def safe_imshow(window_name, image):
    global SHOW_CAMERA_WINDOWS, OPENCV_GUI_AVAILABLE

    if not SHOW_CAMERA_WINDOWS or not OPENCV_GUI_AVAILABLE:
        return False

    try:
        cv2.imshow(window_name, image)
        cv2.waitKey(1)
        return True
    except cv2.error as exc:
        OPENCV_GUI_AVAILABLE = False
        SHOW_CAMERA_WINDOWS = False
        print("\nWARNING: cv2.imshow gagal. Camera window dimatikan, simulasi tetap lanjut.")
        print("Detail:", exc)
        return False



# =========================================================
# TUNING POSISI POV KAMERA WRIST
# =========================================================
# Tombol tuning:
# j/l = X -/+
# i/k = Y +/-
# u/o = Z +/-
# y   = print + simpan koordinat kamera ke CSV
WRIST_POV_TUNING_ENABLED = False  # view-only: kamera tidak bisa diubah-ubah saat simulasi
WRIST_POV_TUNE_STEP = 0.015  # 1.5 cm per pencet
WRIST_POV_LOG_DIR = "data/test_logs/camera_tuning"
WRIST_POV_LOG_FILE = None

# =========================================================
# COLLISION DETECTION / PROXIMITY MONITOR
# =========================================================
# Implementasi berbasis:
# - getContactPoints: kontak yang sudah terjadi setelah stepSimulation()
# - getClosestPoints: jarak terdekat/proximity sebelum terjadi tabrakan
#
# Mode default:
# - tidak otomatis menghentikan robot, supaya gerakan baseline tetap bisa diamati.
# - tekan E untuk snapshot collision.
# - tekan F untuk toggle print collision saat robot bergerak.
BODY_NAME_BY_ID = {}
BODY_ROLE_BY_ID = {}

COLLISION_MONITOR_ENABLED = True
COLLISION_VERBOSE_DURING_MOTION = False
COLLISION_PROXIMITY_DISTANCE = 0.035     # 3.5 cm warning jarak dekat
COLLISION_HARD_CONTACT_DISTANCE = 0.000  # <=0 berarti sudah contact/penetration
COLLISION_PRINT_INTERVAL_STEPS = 60
COLLISION_DEBUG_DRAW_ENABLED = True
COLLISION_DEBUG_LIFETIME = 1.2
COLLISION_AUTO_STOP_ON_BAD_CONTACT = False

COLLISION_LOG_DIR = "data/test_logs/collision_monitor"
COLLISION_LOG_FILE = None
COLLISION_LAST_PRINT_SIGNATURE = {}

# =========================================================
# TERMINAL COMMAND + PURE VISUAL SEARCH MODE
# =========================================================
# Tujuan:
# - robot tidak langsung mengambil berdasarkan metadata nama/posisi objek.
# - user mengetik command di terminal, misalnya: ambil balok merah
# - robot scan rak memakai POV kamera wrist.
# - hanya jika objek tampak/terdeteksi dari RGB camera, baru robot mengambil.
#
# Window kamera disembunyikan default agar simulasi lebih "pure" terminal command.
VISUAL_SEARCH_TERMINAL_MODE = True
SHOW_CAMERA_WINDOWS = True   # camera POV ditampilkan lagi, hanya visual/view-only
CAMERA_VIEW_ONLY_MODE = True
CAMERA_DRAW_OBJECT_LABELS = False      # pure visual: tidak tampilkan label/metadata objek pada camera window
CAMERA_DRAW_SELECTED_TEXT = False      # pure visual: tidak tampilkan Selected: xxx

VISUAL_SEARCH_LOG_DIR = "data/test_logs/visual_search"
VISUAL_SEARCH_SAVE_DEBUG_IMAGES = True
VISUAL_SEARCH_MIN_AREA = 250
VISUAL_SEARCH_MIN_SCORE = 0.30

# =========================================================
# STRICT VISUAL SEARCH — PURE VISUAL DECISION
# =========================================================
# Kandidat tidak diterima berdasarkan nama object.
# Nama/body ID hanya dipakai setelah kandidat lolos visual,
# supaya PyBullet tahu body fisik mana yang harus diambil.
VISUAL_SEARCH_STRICT_MIN_AREA = 300
VISUAL_SEARCH_STRICT_MIN_SCORE = 0.48
VISUAL_SEARCH_STRICT_SHAPE_SCORE = 0.35

# Margin antara score shape target dengan shape alternatif.
# Mengurangi kasus satu contour cocok ke beberapa bentuk sekaligus.
VISUAL_SEARCH_SHAPE_MARGIN = -0.35

# Audit:
# ground truth hanya dicetak SETELAH keputusan visual.
VISUAL_SEARCH_AUDIT_ENABLED = True

# =========================================================
# PURE COMPUTER VISION OBJECT DETECTION
# =========================================================
# Prinsip:
# - bentuk dan warna diputuskan dari RGB contour.
# - posisi pick dipakai dari depth camera hasil unproject pixel.
# - object registry/metadata tidak dipakai untuk menentukan benar/salah kandidat.
# - body_id segmentation hanya dipakai setelah kandidat lolos visual,
#   supaya PyBullet tahu body fisik mana yang harus digenggam.

PURE_CV_MODE = True
PURE_CV_USE_DEPTH_PICK_POSITION = True
PURE_CV_MIN_AREA = 260
PURE_CV_MIN_SCORE = 0.46
PURE_CV_MIN_TARGET_SHAPE_SCORE = 0.28
PURE_CV_MIN_SHAPE_MARGIN = -0.08
PURE_CV_MIN_VIEWS = 1
PURE_CV_VOTE_MIN_TOTAL_SCORE = 0.46
PURE_CV_PRINT_ALL_CANDIDATES = True

PURE_CV_PICK_X_OFFSET = 0.000
PURE_CV_PICK_Y_OFFSET = 0.000
PURE_CV_PICK_Z_OFFSET = 0.000

PURE_CV_PICK_X_MIN = 0.45
PURE_CV_PICK_X_MAX = 0.62

# Multi-rack: Y harus mencakup rak 1 dan rak 2.
PURE_CV_PICK_Y_MIN = min(RACK_1_Y_OFFSET, RACK_2_Y_OFFSET) - 0.42
PURE_CV_PICK_Y_MAX = max(RACK_1_Y_OFFSET, RACK_2_Y_OFFSET) + 0.42

PURE_CV_PICK_Z_MIN = 0.52
PURE_CV_PICK_Z_MAX = 1.30

# Strict visual verification:
# kandidat dari kamera wajib cocok dengan shape+color command.
# Contoh command "ambil silinder hijau":
# - balok_hijau / segitiga_hijau yang ikut terdeteksi warna hijau akan ditolak
# - robot hanya pick jika body yang terlihat adalah silinder_hijau
VISUAL_SEARCH_STRICT_OBJECT_MATCH = True
VISUAL_SEARCH_PRINT_REJECTED_CANDIDATES = True
VISUAL_SEARCH_SCAN_DURATION = 0.75
VISUAL_SEARCH_SCAN_MAX_VELOCITY = 0.32
VISUAL_SEARCH_SCAN_FORCE = 200

# =========================================================
# E2E SPEED + RELEASE WAIT TUNING
# =========================================================
# Gerakan dipercepat dengan mengalikan duration lama memakai factor ini.
# 0.60 artinya durasi menjadi 60% dari sebelumnya.
E2E_SPEED_FACTOR = 0.60
E2E_MAX_VELOCITY_SCALE = 1.45
E2E_MOTOR_FORCE_SCALE = 1.15

# Extra delay sebelum release dimatikan.
# Release tetap menunggu robot stop lewat wait_until_robot_stops_before_release().
RELEASE_EXTRA_DELAY_ENABLED = False

# Tuning stop before release.
# threshold joint dibuat sedikit lebih realistis untuk PyBullet JACO,
# karena joint velocity sering tertahan sekitar 0.08 saat motor masih hold target.
ROBOT_STOP_HOLD_FORCE = 180
ROBOT_STOP_HOLD_MAX_VELOCITY = 0.015
ROBOT_STOP_PRINT_EVERY_STEPS = 50
VISUAL_SEARCH_DEBUG_PREFIX = "visual_search"

# Saat search, target kamera diarahkan ke titik scan rak, bukan ke metadata objek.
WRIST_POV_FORCED_TARGET = None

TERMINAL_COMMAND_QUEUE = queue.Queue()
TERMINAL_COMMAND_THREAD_STARTED = False
TERMINAL_COMMAND_STOP_REQUESTED = False
COLLISION_IGNORE_BODY_NAME_KEYWORDS = [
    "plane",
    "handover_area_non_solid_visual_only"
]

# =========================================================
# HANDOVER TEMBUS / NON-SOLID
# =========================================================
# Handover area hijau hanya marker visual.
# Tidak ada collision surface pada handover, jadi objek bisa menembus marker
# dan jatuh ke basket yang solid.
HANDOVER_POS_NON_SOLID = True

# =========================================================
# PARAMETER VIRTUAL GRASP
# =========================================================
MAX_GRASP_DISTANCE = 0.14      # default untuk rak tengah/atas
GRASP_MAX_FORCE = 120          # constraint virtual grasp

# =========================================================
# NATURAL GRASP + NATURAL GRAVITY DROP TUNING
# =========================================================
# Tujuan:
# - objek tidak terlihat melayang jauh dari jari saat attach
# - objek di-seat/dirapikan dulu ke slot genggam dekat palm/finger
# - orientasi relatif objek dipertahankan saat constraint dibuat
# - saat handover, objek tidak lagi di-smooth/teleport diagonal
# - objek dilepas di atas basket dan jatuh dengan gravity
NATURAL_GRASP_ENABLED = True
GRASP_SEAT_OBJECT_IN_GRIPPER = True
GRASP_SEAT_DURATION = 0.35
GRASP_SEAT_LIFT_ARC = 0.010
GRASP_SEAT_STEPS_AFTER = 12
GRASP_VERIFY_MAX_DISTANCE_FROM_SLOT = 0.065
GRASP_SLOT_ALIGN_BEFORE_ATTACH = True
GRASP_SLOT_ALIGN_MAX_ITERS = 5
GRASP_SLOT_ALIGN_TOLERANCE = 0.045
GRASP_SLOT_ALIGN_DURATION = 0.60
GRASP_SEAT_MAX_DISTANCE = 0.045
BOTTOM_GRASP_SEAT_MAX_DISTANCE = 0.165
GRASP_USE_EXACT_RELATIVE_CONSTRAINT = True

NATURAL_DROP_TO_BASKET_ENABLED = True
BASKET_DROP_HEIGHT = 0.090
BASKET_DROP_SETTLE_STEPS = 420
BASKET_DROP_MAX_HORIZONTAL_SPEED = 0.0

# =========================================================
# RELEASE GUARD: JANGAN LEPAS SEBELUM OBJECT DI ATAS BASKET
# =========================================================
# Problem video:
# object dilepas terlalu cepat sebelum masuk basket.
# Fix:
# sebelum release, posisi object yang masih attached wajib dicek.
# Release hanya boleh kalau center object sudah berada di dalam footprint basket.
RELEASE_GUARD_ENABLED = True
RELEASE_GUARD_XY_TOLERANCE = 0.055       # meter; makin kecil makin ketat
RELEASE_GUARD_Z_TOLERANCE = 0.080        # object harus cukup dekat tinggi target drop
RELEASE_GUARD_MAX_ALIGN_ITERS = 12
RELEASE_GUARD_ALIGN_DURATION = 0.70
RELEASE_GUARD_ALIGN_FORCE = 210
RELEASE_GUARD_ALIGN_MAX_VELOCITY = 0.19
RELEASE_GUARD_ABORT_IF_NOT_READY = True
RELEASE_GUARD_PRINT_EVERY_CHECK = False

# =========================================================
# MANUAL RELEASE / EMERGENCY DROP KEY
# =========================================================
# Tombol 'o' dipakai untuk melepas object yang masih attached di gripper.
# Ini berguna kalau release guard membatalkan release dan object masih ditahan.
MANUAL_RELEASE_ENABLED = True
MANUAL_RELEASE_KEY = ord('o')
MANUAL_RELEASE_SETTLE_STEPS = 120
ACTIVE_GRASP_CONSTRAINT_ID = None
ACTIVE_GRASP_OBJECT_ID = None
BOTTOM_GRASP_DISTANCE = 0.240
BOTTOM_GRASP_ASSIST_ENABLED = False  # fix3: jangan teleport object saat bottom grasp
# Offset world untuk merapikan posisi objek saat grasp rak bawah.
# EE pada rak bawah biasanya terlalu tinggi dari pusat objek.
BOTTOM_GRASP_ASSIST_OFFSET = [0.045, 0.0, -0.075]   # lebih halus, tidak membuat objek meloncat/melayang jauh
GRASP_LOCAL_OFFSET = [0.000, -0.026, -0.018]  # fix3: slot lebih rendah/dekat sela jari
APPROACH_ERROR_LIMIT = 0.24    # edge reach tuning    # kalau error approach lebih besar dari ini, jangan attach
REACH_ERROR_LIMIT = 0.26    # edge reach tuning       # kalau error reach lebih besar dari ini, abort tahap pick
READY_ERROR_LIMIT = 0.42    # edge reach tuning       # shelf-ready hanya transisi, jadi toleransi lebih longgar
BOTTOM_APPROACH_ERROR_LIMIT = 0.28    # edge reach tuning
HANDOVER_ERROR_LIMIT = 0.42    # handover lebih toleran karena area kiri lebih sulit dijangkau JACO

# =========================================================
# PARAMETER WAIT UNTIL ROBOT STOP BEFORE RELEASE
# =========================================================
# Tujuan:
# Setelah fine-align handover selesai, objek TIDAK langsung dilepas.
# Robot ditunggu sampai joint velocity dan gerak end-effector benar-benar kecil,
# baru constraint dilepas/release.
WAIT_ROBOT_STOP_BEFORE_RELEASE = True
ROBOT_STOP_JOINT_VELOCITY_THRESHOLD = 0.085   # rad/s, dibuat tunable agar wait stop tidak selalu timeout
ROBOT_STOP_EE_DELTA_THRESHOLD = 0.0008        # meter per simulation step
ROBOT_STOP_MIN_STABLE_STEPS = 40              # lebih banyak step stabil sebelum release
ROBOT_STOP_MAX_WAIT_STEPS = 420               # tuning: max wait 700 step
ROBOT_STOP_MIN_TOTAL_WAIT_STEPS = 40        # fix4: walaupun sudah stabil, tahan sampai sekitar 650 step
ROBOT_STOP_EXTRA_SETTLE_STEPS = 12          # settle kecil setelah dianggap stop

# Delay tambahan setelah robot benar-benar stop, sebelum release objek.
# 1.5 detik dipilih di tengah request 1-2 detik.
RELEASE_DELAY_AFTER_STOP_SEC = 0.0

# =========================================================
# PARAMETER JOINT-1 PRE-ALIGNMENT
# =========================================================
# Tujuan:
# sebelum IK utama, joint pertama lengan dibuat menghadap target dulu.
# Pada tahap ini hanya joint-1 yang bergerak; joint lain ditahan.
ENABLE_JOINT1_PREALIGN = True
JOINT1_ALIGN_DURATION = 0.75
JOINT1_ALIGN_FORCE = 180
JOINT1_ALIGN_MAX_VELOCITY = 0.48

# Home pose aman antar pengambilan objek
HOME_RETURN_DURATION = 0.90
HOME_RETURN_FORCE = 220
HOME_RETURN_MAX_VELOCITY = 0.38

# Home pose dibuat manual.
# Robot tidak otomatis kembali ke home setelah pick/handover/failure.
# Tekan tombol 'm' di simulasi untuk memanggil home pose.
AUTO_HOME_ENABLED = False

# =========================================================
# CUSTOM IDLE / HOME POSE FIX
# =========================================================
# Problem dari video:
# pose home original robot terlalu melipat ke bawah/dekat base, sehingga terlihat stuck.
#
# Fix:
# reset/home tidak lagi memakai pose nol/initial URDF.
# Reset/home memakai CUSTOM_IDLE_JOINTS yang lebih terbuka.
#
# Kalau pose masih kurang pas, ubah CUSTOM_IDLE_PRESET:
# - "low"  : lebih rendah/dekat basket
# - "mid"  : posisi tengah, rekomendasi awal
# - "high" : lebih tinggi/mendekati rak
CUSTOM_IDLE_ENABLED = True
CUSTOM_IDLE_PRESET = "mid"

CUSTOM_IDLE_PRESETS = {
    "low":  [0.0, 1.75, 0.95, 0.0, 0.95, 0.0],
    "mid":  [0.0, 2.05, 1.05, 0.0, 1.10, 0.0],
    "high": [0.0, 2.35, 1.15, 0.0, 1.22, 0.0],
}

CUSTOM_IDLE_SETTLE_STEPS = 180
CUSTOM_IDLE_ARM_FORCE = 230
CUSTOM_IDLE_ARM_MAX_VELOCITY = 0.22
CUSTOM_IDLE_FINGER_FORCE = 80
CUSTOM_IDLE_FINGER_MAX_VELOCITY = 0.45
CUSTOM_IDLE_SLEEP_IN_GUI = True
CUSTOM_IDLE_CLEAR_GRASP = True

# =========================================================
# EDGE REACH TUNING — RAK KIRI/KANAN UJUNG
# =========================================================
# Problem:
# target dengan y besar, misalnya y=+0.33, membuat shelf-ready terlalu jauh ke samping.
# Log user:
# target ready [0.28, 0.33, 0.90]
# end-effector akhir [0.4369, 0.0401, 0.8999]
# error 0.3296 sehingga ready dianggap gagal.
#
# Fix:
# - shelf-ready tidak harus tepat di y objek; cukup titik transisi aman.
# - pre-grasp dibuat mendekati sisi target secara bertahap.
# - approach tetap mendekati y objek, tetapi sedikit dikurangi agar reachable.
EDGE_REACH_TUNING_ENABLED = True
EDGE_Y_THRESHOLD = 0.24
EDGE_READY_Y_LIMIT = 0.18
EDGE_PREGRASP_Y_LIMIT = 0.27
EDGE_APPROACH_Y_LIMIT = 0.315
EDGE_RETREAT_Y_LIMIT = 0.25
EDGE_LIFT_Y_LIMIT = 0.24

# Ready hanya transisi, jadi tolerance dibuat lebih longgar.
# Ini mencegah task batal hanya karena titik transisi belum presisi.
READY_ERROR_LIMIT = 0.42
REACH_ERROR_LIMIT = 0.26
APPROACH_ERROR_LIMIT = 0.24
BOTTOM_APPROACH_ERROR_LIMIT = 0.28


# =========================================================
# RANDOMIZED OBJECT PLACEMENT
# =========================================================
# Setiap simulasi dibuka dan setiap reset:
# - object ditukar antar slot rak
# - posisi diberi jitter kecil
# - yaw diberi variasi kecil
#
# initial_poses tetap menjadi referensi slot aman.

OBJECT_RANDOMIZE_ON_LOAD = True
OBJECT_RANDOMIZE_ON_RESET = True

OBJECT_RANDOMIZE_SHUFFLE_SLOTS = True

# Random posisi kecil agar tetap berada di dalam rak
OBJECT_RANDOM_X_JITTER = 0.010
OBJECT_RANDOM_Y_JITTER = 0.018
OBJECT_RANDOM_Z_JITTER = 0.000

# Random orientasi horizontal
OBJECT_RANDOM_YAW_DEG = 12.0

# Setelah random, beri physics waktu settle
OBJECT_RANDOM_SETTLE_STEPS = 120

# CSV bukti posisi random
OBJECT_RANDOM_LOG_DIR = "data/test_logs/object_randomize"


# =========================================================
# WARNA
# =========================================================
RGBA_RED    = [1.0, 0.2, 0.2, 1.0]
RGBA_GREEN  = [0.2, 1.0, 0.2, 1.0]
RGBA_BLUE   = [0.2, 0.4, 1.0, 1.0]
RGBA_YELLOW = [1.0, 0.85, 0.1, 1.0]
RGBA_ORANGE = [1.0, 0.55, 0.15, 1.0]
RGBA_PURPLE = [0.7, 0.3, 1.0, 1.0]
RGBA_BROWN  = [0.55, 0.35, 0.20, 1.0]
RGBA_GRAY   = [0.15, 0.15, 0.15, 1.0]
RGBA_WHITE  = [1.0, 1.0, 1.0, 1.0]

def infer_body_role_from_name(name):
    """
    Klasifikasi kasar body untuk collision log.
    """
    n = str(name).lower()

    if n == "plane" or "floor" in n:
        return "floor"
    if n.startswith("rak") or "rak_" in n:
        return "shelf"
    if "wheelchair" in n or "handrest" in n or "armrest" in n or "mount_" in n:
        return "wheelchair_or_mount"
    if "basket" in n:
        return "handover_basket"
    if "handover" in n:
        return "handover_area"
    if (
        n.startswith("balok_")
        or n.startswith("silinder_")
        or n.startswith("segitiga_")
    ):
        return "store_object"
    if "jaco" in n or "kinova" in n:
        return "robot"

    return "environment"


def register_body_name(body_id, name, role=None):
    """
    Menyimpan nama body agar collision log tidak hanya menampilkan ID angka.
    """
    try:
        body_id = int(body_id)
        BODY_NAME_BY_ID[body_id] = str(name)
        BODY_ROLE_BY_ID[body_id] = role if role is not None else infer_body_role_from_name(name)
    except Exception:
        pass
    return body_id
def register_mobile_platform_body(body_id):
    """
    Body yang ikut bergerak bersama wheelchair:
    - wheelchair body
    - robot base
    - basket
    - handover marker
    """
    try:
        body_id = int(body_id)
        if body_id not in MOBILE_PLATFORM_BODY_IDS:
            MOBILE_PLATFORM_BODY_IDS.append(body_id)

        pos, orn = p.getBasePositionAndOrientation(body_id)
        MOBILE_PLATFORM_ORIGINAL_POSES[body_id] = {
            "position": list(pos),
            "orientation": list(orn)
        }
    except Exception:
        pass

    return body_id

def get_body_name(body_id):
    if body_id == -1:
        return "world"
    return BODY_NAME_BY_ID.get(int(body_id), f"body_{body_id}")


def get_body_role(body_id):
    if body_id == -1:
        return "world"
    return BODY_ROLE_BY_ID.get(int(body_id), "unknown")


def get_robot_link_name(robot_id, link_index):
    if link_index == -1:
        return "base_link"
    try:
        info = p.getJointInfo(robot_id, link_index)
        return info[12].decode("utf-8")
    except Exception:
        return f"link_{link_index}"


def should_ignore_collision_body(body_id):
    name = get_body_name(body_id).lower()
    for keyword in COLLISION_IGNORE_BODY_NAME_KEYWORDS:
        if keyword.lower() in name:
            return True
    return False


def get_all_body_ids():
    try:
        return [p.getBodyUniqueId(i) for i in range(p.getNumBodies())]
    except Exception:
        return []




# =========================================================
# TULIS FILE OBJ UNTUK TRIANGLE PRISM
# =========================================================
def ensure_triangle_prism_obj(obj_path):
    """
    Membuat mesh triangular prism sederhana jika belum ada.
    Basis mesh berada di origin, ukuran dasar akan diatur oleh meshScale.
    """
    if os.path.exists(obj_path):
        return

    obj_text = """
# triangular prism
v -0.5 -0.5 -0.5
v  0.5 -0.5 -0.5
v  0.0  0.5 -0.5
v -0.5 -0.5  0.5
v  0.5 -0.5  0.5
v  0.0  0.5  0.5

# bottom triangle
f 1 2 3
# top triangle
f 4 6 5
# side faces
f 1 4 5
f 1 5 2
f 2 5 6
f 2 6 3
f 3 6 4
f 3 4 1
""".strip()

    with open(obj_path, "w") as f:
        f.write(obj_text)


# =========================================================
# FUNGSI GENERIK SHAPE
# =========================================================
def create_box(name, position, size, color, mass=0.0):
    collision = p.createCollisionShape(
        p.GEOM_BOX,
        halfExtents=[size[0] / 2, size[1] / 2, size[2] / 2]
    )

    visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[size[0] / 2, size[1] / 2, size[2] / 2],
        rgbaColor=color
    )

    body_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position
    )

    register_body_name(body_id, name)
    print(f"{name} dibuat | ID: {body_id}")
    return body_id



def create_visual_box_marker(name, position, size, color):
    """
    Marker visual tanpa collision agar area hijau tidak menjadi penghalang fisik.
    """
    visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[size[0] / 2, size[1] / 2, size[2] / 2],
        rgbaColor=color
    )
    body_id = p.createMultiBody(
        baseMass=0.0,
        baseCollisionShapeIndex=-1,
        baseVisualShapeIndex=visual,
        basePosition=position
    )
    register_body_name(body_id, name)
    print(f"{name} visual-only marker dibuat | ID: {body_id}")
    return body_id


def create_cylinder(name, position, radius, height, color, mass=0.15):
    collision = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=radius,
        height=height
    )

    visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=radius,
        length=height,
        rgbaColor=color
    )

    body_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position
    )

    register_body_name(body_id, name)
    print(f"{name} dibuat | ID: {body_id}")
    return body_id

def create_cylinder_oriented(name, position, radius, height, color, orientation_euler, mass=0.0):
    """
    Membuat cylinder dengan orientasi tertentu.
    Dipakai untuk roda wheelchair agar cylinder tidur pada sumbu Y.
    """
    orn = p.getQuaternionFromEuler(orientation_euler)

    collision = p.createCollisionShape(
        p.GEOM_CYLINDER,
        radius=radius,
        height=height
    )

    visual = p.createVisualShape(
        p.GEOM_CYLINDER,
        radius=radius,
        length=height,
        rgbaColor=color
    )

    body_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position,
        baseOrientation=orn
    )

    register_body_name(body_id, name)
    print(f"{name} dibuat | ID: {body_id}")
    return body_id

def create_light_electric_wheelchair():
    """
    Wheelchair elektrik ringan berbasis shape PyBullet.
    Konfigurasi final:
    - kursi roda menyamping rak
    - arah maju kursi = +Y
    - sisi kanan kursi = +X (menghadap rak)
    - robot di sisi kanan kursi, di luar handrest kanan
    """
    wheelchair_parts = []

    x0, y0, z0 = WHEELCHAIR_BASE_POS

    # =====================================================
    # CHASSIS / BODY BAWAH
    # =====================================================
    wheelchair_parts.append(
        create_box(
            name="electric_wheelchair_chassis",
            position=[x0, y0, z0],
            size=WHEELCHAIR_BASE_SIZE,
            color=RGBA_GRAY,
            mass=0.0
        )
    )

    # SEAT
    wheelchair_parts.append(
        create_box(
            name="electric_wheelchair_seat",
            position=[x0, y0 + 0.02, z0 + 0.16],
            size=[0.46, 0.56, 0.10],
            color=[0.05, 0.05, 0.05, 1.0],
            mass=0.0
        )
    )

    # BACKREST
    wheelchair_parts.append(
        create_box(
            name="electric_wheelchair_backrest",
            position=[x0, y0 - 0.26, z0 + 0.43],
            size=[0.56, 0.08, 0.50],
            color=[0.04, 0.04, 0.04, 1.0],
            mass=0.0
        )
    )

    # =====================================================
    # ARMREST KIRI
    # kiri kursi = -X
    # dibuat lebih tebal dan lebih tinggi agar terlihat jelas
    # =====================================================
    wheelchair_parts.append(
        create_box(
            name="electric_wheelchair_armrest_left",
            position=[x0 - 0.31, y0 + 0.02, z0 + 0.35],
            size=[0.08, 0.52, 0.08],
            color=[0.18, 0.18, 0.18, 1.0],
            mass=0.0
        )
    )

    # tiang depan armrest kiri
    wheelchair_parts.append(
        create_box(
            name="electric_wheelchair_armrest_left_front_post",
            position=[x0 - 0.31, y0 + 0.20, z0 + 0.24],
            size=[0.06, 0.06, 0.24],
            color=[0.12, 0.12, 0.12, 1.0],
            mass=0.0
        )
    )

    # tiang belakang armrest kiri
    wheelchair_parts.append(
        create_box(
            name="electric_wheelchair_armrest_left_back_post",
            position=[x0 - 0.31, y0 - 0.18, z0 + 0.24],
            size=[0.06, 0.06, 0.24],
            color=[0.12, 0.12, 0.12, 1.0],
            mass=0.0
        )
    )

    # =====================================================
    # ARMREST KANAN
    # kanan kursi = +X
    # robot berada di luar sisi ini
    # =====================================================
    wheelchair_parts.append(
        create_box(
            name="electric_wheelchair_armrest_right",
            position=[x0 + 0.31, y0 + 0.02, z0 + 0.35],
            size=[0.08, 0.52, 0.08],
            color=[0.18, 0.18, 0.18, 1.0],
            mass=0.0
        )
    )

    # tiang depan armrest kanan
    wheelchair_parts.append(
        create_box(
            name="electric_wheelchair_armrest_right_front_post",
            position=[x0 + 0.31, y0 + 0.20, z0 + 0.24],
            size=[0.06, 0.06, 0.24],
            color=[0.12, 0.12, 0.12, 1.0],
            mass=0.0
        )
    )

    # tiang belakang armrest kanan
    wheelchair_parts.append(
        create_box(
            name="electric_wheelchair_armrest_right_back_post",
            position=[x0 + 0.31, y0 - 0.18, z0 + 0.24],
            size=[0.06, 0.06, 0.24],
            color=[0.12, 0.12, 0.12, 1.0],
            mass=0.0
        )
    )

    # =====================================================
    # MOUNT ROBOT DI SISI KANAN, DI LUAR HANDREST
    # =====================================================
    wheelchair_parts.append(
        create_box(
            name="robot_mount_column",
            position=[ROBOT_BASE_POS[0], ROBOT_BASE_POS[1], ROBOT_BASE_POS[2] - 0.23],
            size=[0.10, 0.10, 0.46],
            color=[0.20, 0.20, 0.20, 1.0],
            mass=0.0
        )
    )

    wheelchair_parts.append(
        create_box(
            name="robot_mount_plate",
            position=[ROBOT_BASE_POS[0], ROBOT_BASE_POS[1], ROBOT_BASE_POS[2] - 0.02],
            size=[0.22, 0.22, 0.04],
            color=[0.10, 0.10, 0.10, 1.0],
            mass=0.0
        )
    )

    # bracket samping supaya terlihat mount keluar dari handrest kanan
    wheelchair_parts.append(
        create_box(
            name="robot_mount_side_bracket",
            position=[x0 + 0.40, y0, z0 + 0.22],
            size=[0.05, 0.18, 0.10],
            color=[0.18, 0.18, 0.18, 1.0],
            mass=0.0
        )
    )

    # =====================================================
    # RODA BESAR
    # =====================================================
    wheelchair_parts.append(
        create_cylinder_oriented(
            name="wheel_left_large",
            position=[x0 - 0.34, y0 - 0.12, z0 + 0.02],
            radius=0.20,
            height=0.06,
            color=[0.02, 0.02, 0.02, 1.0],
            orientation_euler=[0, math.pi / 2, 0],
            mass=0.0
        )
    )

    wheelchair_parts.append(
        create_cylinder_oriented(
            name="wheel_right_large",
            position=[x0 + 0.34, y0 - 0.12, z0 + 0.02],
            radius=0.20,
            height=0.06,
            color=[0.02, 0.02, 0.02, 1.0],
            orientation_euler=[0, math.pi / 2, 0],
            mass=0.0
        )
    )

    # =====================================================
    # CASTER DEPAN
    # depan kursi = +Y
    # =====================================================
    wheelchair_parts.append(
        create_cylinder_oriented(
            name="caster_front_left",
            position=[x0 - 0.22, y0 + 0.28, z0 - 0.05],
            radius=0.07,
            height=0.04,
            color=[0.03, 0.03, 0.03, 1.0],
            orientation_euler=[math.pi / 2, 0, 0],
            mass=0.0
        )
    )

    wheelchair_parts.append(
        create_cylinder_oriented(
            name="caster_front_right",
            position=[x0 + 0.22, y0 + 0.28, z0 - 0.05],
            radius=0.07,
            height=0.04,
            color=[0.03, 0.03, 0.03, 1.0],
            orientation_euler=[math.pi / 2, 0, 0],
            mass=0.0
        )
    )

    # =====================================================
    # FOOTREST
    # =====================================================
    wheelchair_parts.append(
        create_box(
            name="footrest_plate",
            position=[x0, y0 + 0.34, z0 + 0.03],
            size=[0.30, 0.12, 0.03],
            color=[0.15, 0.15, 0.15, 1.0],
            mass=0.0
        )
    )

    print("Light electric wheelchair selesai dibuat (menyamping rak, robot di kanan luar handrest).")
    return wheelchair_parts

def create_triangle_prism(name, position, size_xyz, color, mass=0.15):
    """
    size_xyz = [scale_x, scale_y, scale_z]
    """
    ensure_triangle_prism_obj(TRIANGLE_PRISM_OBJ)

    collision = p.createCollisionShape(
        shapeType=p.GEOM_MESH,
        fileName=TRIANGLE_PRISM_OBJ,
        meshScale=size_xyz
    )

    visual = p.createVisualShape(
        shapeType=p.GEOM_MESH,
        fileName=TRIANGLE_PRISM_OBJ,
        meshScale=size_xyz,
        rgbaColor=color
    )

    body_id = p.createMultiBody(
        baseMass=mass,
        baseCollisionShapeIndex=collision,
        baseVisualShapeIndex=visual,
        basePosition=position
    )

    register_body_name(body_id, name)
    print(f"{name} dibuat | ID: {body_id}")
    return body_id


def apply_object_dynamics(body_id):
    """
    Physics tuning supaya objek jatuh lebih realistis:
    - friction cukup besar
    - damping kecil
    - restitution rendah
    """
    p.changeDynamics(
        body_id,
        -1,
        lateralFriction=0.9,
        spinningFriction=0.02,
        rollingFriction=0.01,
        restitution=0.03,
        linearDamping=0.04,
        angularDamping=0.04
    )


def set_object_transport_mode(object_id, enabled=True):
    """
    Mode transport untuk objek yang sedang dibawa gripper.
    Saat enabled:
    - collision objek dimatikan agar tidak terpental saat melewati rak/handrest/marker.
    - velocity dinolkan supaya constraint tidak membawa energi liar.
    Saat disabled:
    - collision diaktifkan lagi sebelum objek diletakkan di handover support.
    """
    if enabled:
        p.setCollisionFilterGroupMask(object_id, -1, 0, 0)
        p.resetBaseVelocity(object_id, [0, 0, 0], [0, 0, 0])
        p.changeDynamics(
            object_id,
            -1,
            linearDamping=0.95,
            angularDamping=0.95,
            restitution=0.0
        )
        print("Object transport mode: ON, collision objek sementara dimatikan.")
    else:
        p.setCollisionFilterGroupMask(object_id, -1, 1, 1)
        p.resetBaseVelocity(object_id, [0, 0, 0], [0, 0, 0])
        p.changeDynamics(
            object_id,
            -1,
            lateralFriction=1.2,
            spinningFriction=0.02,
            rollingFriction=0.01,
            restitution=0.0,
            linearDamping=0.65,
            angularDamping=0.65
        )
        print("Object transport mode: OFF, collision objek diaktifkan kembali.")





# =========================================================
# BASKET / LAP TRAY AREA HANDOVER
# =========================================================
def create_handover_basket():
    """
    Membuat basket fisik di sekitar area handover.
    Motion planning lama tetap dipakai; basket hanya ditambahkan sebagai objek scene.
    """
    global BASKET_INFO

    if not BASKET_ENABLED:
        return None

    cx, cy, cz = BASKET_CENTER
    sx, sy, sz = BASKET_OUTER_SIZE
    t = BASKET_WALL_THICKNESS
    ft = BASKET_FLOOR_THICKNESS
    wh = BASKET_WALL_HEIGHT

    bottom_z = cz - sz / 2.0
    floor_center_z = bottom_z + ft / 2.0
    wall_center_z = bottom_z + ft + wh / 2.0

    parts = []

    # Alas hitam tipis sebagai lap tray di bawah basket.
    parts.append(create_box(
        name="handover_basket_lap_tray_pad",
        position=[cx, cy, bottom_z - 0.010],
        size=[sx + 2 * BASKET_SUPPORT_MARGIN, sy + 2 * BASKET_SUPPORT_MARGIN, 0.020],
        color=[0.04, 0.04, 0.04, 0.75],
        mass=0.0
    ))

    # Lantai basket.
    parts.append(create_box(
        name="handover_basket_floor",
        position=[cx, cy, floor_center_z],
        size=[sx, sy, ft],
        color=[0.85, 0.48, 0.15, 0.90],
        mass=0.0
    ))

    # Dinding basket.
    parts.append(create_box(
        name="handover_basket_wall_front",
        position=[cx, cy - sy / 2.0 + t / 2.0, wall_center_z],
        size=[sx, t, wh],
        color=[1.0, 0.55, 0.12, 0.62],
        mass=0.0
    ))

    parts.append(create_box(
        name="handover_basket_wall_back",
        position=[cx, cy + sy / 2.0 - t / 2.0, wall_center_z],
        size=[sx, t, wh],
        color=[1.0, 0.55, 0.12, 0.62],
        mass=0.0
    ))

    parts.append(create_box(
        name="handover_basket_wall_left",
        position=[cx - sx / 2.0 + t / 2.0, cy, wall_center_z],
        size=[t, sy, wh],
        color=[1.0, 0.55, 0.12, 0.62],
        mass=0.0
    ))

    parts.append(create_box(
        name="handover_basket_wall_right",
        position=[cx + sx / 2.0 - t / 2.0, cy, wall_center_z],
        size=[t, sy, wh],
        color=[1.0, 0.55, 0.12, 0.62],
        mass=0.0
    ))

    for body_id in parts:
        p.changeDynamics(
            body_id,
            -1,
            lateralFriction=1.25,
            spinningFriction=0.02,
            rollingFriction=0.01,
            restitution=0.0
        )

    BASKET_INFO = {
        "parts": parts,
        "center": [cx, cy, cz],
        "floor_top_z": bottom_z + ft,
        "inner_size": [sx - 2 * t, sy - 2 * t, wh]
    }

    try:
        p.addUserDebugText(
            text="HANDOVER BASKET",
            textPosition=[cx - sx / 2.0, cy, wall_center_z + wh / 2.0 + 0.06],
            textColorRGB=[1.0, 0.55, 0.05],
            textSize=1.0
        )
        p.addUserDebugLine(
            [cx, cy, bottom_z],
            HANDOVER_POS,
            lineColorRGB=[0.0, 1.0, 1.0],
            lineWidth=2.0,
            lifeTime=0
        )
    except Exception:
        pass

    print("Handover basket/lap tray selesai dibuat.")
    print("BASKET_CENTER:", BASKET_CENTER)
    print("BASKET_OUTER_SIZE:", BASKET_OUTER_SIZE)
    print("HANDOVER_POS tetap:", HANDOVER_POS)

    return BASKET_INFO



def place_object_in_handover_basket(object_id):
    """
    Debug visual: taruh objek aktif langsung ke dalam basket tanpa mengubah logic pick.
    """
    if BASKET_INFO is None:
        print("Basket belum dibuat.")
        return

    aabb_min, aabb_max = p.getAABB(object_id)
    half_height = max(0.035, (aabb_max[2] - aabb_min[2]) / 2.0)
    target = [
        BASKET_CENTER[0],
        BASKET_CENTER[1],
        BASKET_INFO["floor_top_z"] + half_height + 0.018
    ]

    pos, orn = p.getBasePositionAndOrientation(object_id)
    p.resetBasePositionAndOrientation(object_id, target, orn)
    p.resetBaseVelocity(object_id, [0, 0, 0], [0, 0, 0])
    set_object_transport_mode(object_id, enabled=False)

    for _ in range(80):
        p.stepSimulation()
        time.sleep(1 / 240)

    print("Objek ditaruh langsung ke handover basket:", target)


def place_selected_object_in_handover_basket(objects, selected_name):
    if selected_name not in objects:
        print("Objek aktif tidak ditemukan:", selected_name)
        return
    place_object_in_handover_basket(objects[selected_name]["id"])


# =========================================================
# DECODE SEGMENTATION BODY ID
# =========================================================
def extract_body_id_from_segmentation(segmentation_img):
    """
    Pada PyBullet segmentation mask:
    bodyUniqueId tersimpan pada 24 bit bawah.
    """
    return segmentation_img & ((1 << 24) - 1)


# =========================================================
# MEMBUAT SCENE RAK TOKO
# =========================================================

def create_shelf_module(rack_id=1, y_offset=0.0):
    """
    Membuat satu modul rak pada offset Y tertentu.
    Rak 1: y_offset = 0.0
    Rak 2: y_offset = RACK_2_Y_OFFSET
    """

    suffix = f"rak{rack_id}"

    create_box(
        name=f"{suffix}_back_panel",
        position=[SHELF_BACK_CENTER[0], y_offset, SHELF_HEIGHT / 2.0],
        size=[BACK_PANEL_THICKNESS, SHELF_WIDTH, SHELF_HEIGHT],
        color=RGBA_BROWN,
        mass=0.0
    )

    create_box(
        name=f"{suffix}_side_left",
        position=[0.72, y_offset - SHELF_WIDTH / 2 + SIDE_PANEL_THICKNESS / 2, SHELF_HEIGHT / 2],
        size=[SHELF_DEPTH, SIDE_PANEL_THICKNESS, SHELF_HEIGHT],
        color=RGBA_BROWN,
        mass=0.0
    )

    create_box(
        name=f"{suffix}_side_right",
        position=[0.72, y_offset + SHELF_WIDTH / 2 - SIDE_PANEL_THICKNESS / 2, SHELF_HEIGHT / 2],
        size=[SHELF_DEPTH, SIDE_PANEL_THICKNESS, SHELF_HEIGHT],
        color=RGBA_BROWN,
        mass=0.0
    )

    for i, zc in enumerate(SHELF_Z_LEVELS):
        create_box(
            name=f"{suffix}_level_{i+1}",
            position=[0.72, y_offset, zc],
            size=[SHELF_DEPTH, SHELF_WIDTH, SHELF_BOARD_THICKNESS],
            color=RGBA_BROWN,
            mass=0.0
        )

    create_box(
        name=f"{suffix}_base_kickplate",
        position=[0.90, y_offset, 0.08],
        size=[BACK_PANEL_THICKNESS, SHELF_WIDTH, 0.16],
        color=RGBA_BROWN,
        mass=0.0
    )

def rack_pos(rack_y_offset, local_y, x, z):
    """
    Membuat posisi objek berdasarkan rak aktif.
    local_y = posisi kiri/kanan di dalam rak.
    rack_y_offset = posisi rak 1 atau rak 2.
    """
    return [
        x,
        rack_y_offset + local_y,
        z
    ]

def get_rack_id_from_y(y_value):
    """
    Menentukan Y termasuk area rak 1 atau rak 2.
    Return:
    - 1 jika dekat rak 1
    - 2 jika dekat rak 2
    - None jika berada di luar area dua rak
    """
    try:
        y = float(y_value)
    except Exception:
        return None

    if abs(y - RACK_1_Y_OFFSET) <= ACTIVE_RACK_Y_MARGIN:
        return 1

    if abs(y - RACK_2_Y_OFFSET) <= ACTIVE_RACK_Y_MARGIN:
        return 2

    return None


def get_candidate_rack_info(candidate):
    """
    Mengambil informasi rak dari kandidat visual.
    Prioritas:
    1. body_id segmentation -> posisi body PyBullet
    2. world_pos_from_depth -> posisi hasil depth camera

    Fungsi ini TIDAK memilih object target dari metadata di awal.
    Ini hanya memfilter kandidat agar sesuai station/rak yang sedang discan.
    """
    body_id = candidate.get("body_id")
    body_y = None
    body_rack_id = None

    if body_id is not None:
        try:
            body_pos, _ = p.getBasePositionAndOrientation(int(body_id))
            body_y = float(body_pos[1])
            body_rack_id = get_rack_id_from_y(body_y)
        except Exception:
            body_y = None
            body_rack_id = None

    depth_y = None
    depth_rack_id = None
    world_pos = candidate.get("world_pos_from_depth")

    try:
        if world_pos is not None and len(world_pos) >= 2 and world_pos[1] is not None:
            depth_y = float(world_pos[1])
            depth_rack_id = get_rack_id_from_y(depth_y)
    except Exception:
        depth_y = None
        depth_rack_id = None

    used_y = body_y if body_y is not None else depth_y
    used_rack_id = body_rack_id if body_rack_id is not None else depth_rack_id
    source = "body_segmentation_y" if body_rack_id is not None else "depth_world_y"

    return {
        "used_y": used_y,
        "used_rack_id": used_rack_id,
        "source": source,
        "body_y": body_y,
        "body_rack_id": body_rack_id,
        "depth_y": depth_y,
        "depth_rack_id": depth_rack_id,
        "active_rack_id": ACTIVE_RACK_ID,
        "active_rack_y_offset": ACTIVE_RACK_Y_OFFSET
    }


def candidate_belongs_to_active_rack(candidate):
    """
    True jika kandidat visual berada di rak yang sedang discan oleh station aktif.
    """
    info = get_candidate_rack_info(candidate)
    ok = info.get("used_rack_id") == ACTIVE_RACK_ID
    return bool(ok), info


def create_indonesian_store_scene():
    objects = {}
    initial_poses = {}

    # -------------------------
    # LANTAI & DYNAMICS
    # -------------------------
    plane_id = p.loadURDF("plane.urdf")
    register_body_name(plane_id, "plane", role="floor")
    p.changeDynamics(
        plane_id,
        -1,
        lateralFriction=1.1,
        spinningFriction=0.02,
        rollingFriction=0.01,
        restitution=0.02
    )

    # -------------------------
    # RAK GONDOLA MINIMARKET - MULTI RACK
    # -------------------------
    create_shelf_module(
        rack_id=1,
        y_offset=RACK_1_Y_OFFSET
    )

    if MULTI_RACK_MODE:
        create_shelf_module(
            rack_id=2,
            y_offset=RACK_2_Y_OFFSET
        )


    # -------------------------
    # LIGHT ELECTRIC WHEELCHAIR / MOUNT
    # -------------------------
    wheelchair_parts = create_light_electric_wheelchair()
    for body_id in wheelchair_parts:
        register_mobile_platform_body(body_id)

    # Handover area TEMBUS / non-solid.
    # Tidak ada create_box solid di sini.
    # Marker hijau hanya visual, jadi objek tidak tertahan/melayang di atas handover pos.
    handover_marker_id = create_visual_box_marker(
        name="handover_area_non_solid_visual_only",
        position=[HANDOVER_POS[0], HANDOVER_POS[1], HANDOVER_POS[2] + 0.002],
        size=HANDOVER_SIZE,
        color=[0.0, 1.0, 0.0, 0.30]
    )
    try:
        p.setCollisionFilterGroupMask(handover_marker_id, -1, 0, 0)
    except Exception:
        pass
    register_mobile_platform_body(handover_marker_id)
    # Basket/lap tray tambahan di sekitar area handover.
    basket_info = create_handover_basket()

    if basket_info is not None:
        for body_id in basket_info.get("parts", []):
            register_mobile_platform_body(body_id)



    # -------------------------
    # 3 BENTUK x 4 WARNA
    # total 12 objek
    # -------------------------
    # Bentuk:
    # - rectangle prism / kotak
    # - cylinder / silinder
    # - triangle prism / prisma segitiga
    #
    # Warna:
    # - red, green, blue, yellow
    #
    # Penataan:
    # setiap level rak berisi 2 objek per rak
    object_specs = [
        # =====================================================
        # RAK 1 — 2 objek per shelf/level
        # =====================================================
        # Rak 1 bawah
        ("balok_merah",     "box",      rack_pos(RACK_1_Y_OFFSET, -0.22, 0.515, SHELF_Z_LEVELS[0] + 0.065), RGBA_RED),
        ("silinder_hijau",  "cylinder", rack_pos(RACK_1_Y_OFFSET,  0.22, 0.525, SHELF_Z_LEVELS[0] + 0.065), RGBA_GREEN),

        # Rak 1 tengah
        ("segitiga_biru",   "triangle", rack_pos(RACK_1_Y_OFFSET, -0.22, 0.515, SHELF_Z_LEVELS[1] + 0.065), RGBA_BLUE),
        ("balok_kuning",    "box",      rack_pos(RACK_1_Y_OFFSET,  0.22, 0.515, SHELF_Z_LEVELS[1] + 0.065), RGBA_YELLOW),

        # Rak 1 atas
        ("balok_hijau",     "box",      rack_pos(RACK_1_Y_OFFSET, -0.22, 0.540, SHELF_Z_LEVELS[2] + 0.065), RGBA_GREEN),
        ("silinder_biru",   "cylinder", rack_pos(RACK_1_Y_OFFSET,  0.22, 0.550, SHELF_Z_LEVELS[2] + 0.065), RGBA_BLUE),

        # =====================================================
        # RAK 2 — 2 objek per shelf/level
        # =====================================================
        # Rak 2 bawah
        ("segitiga_merah",  "triangle", rack_pos(RACK_2_Y_OFFSET, -0.22, 0.540, SHELF_Z_LEVELS[0] + 0.065), RGBA_RED),
        ("silinder_kuning", "cylinder", rack_pos(RACK_2_Y_OFFSET,  0.22, 0.550, SHELF_Z_LEVELS[0] + 0.065), RGBA_YELLOW),

        # Rak 2 tengah
        ("balok_biru",      "box",      rack_pos(RACK_2_Y_OFFSET, -0.22, 0.540, SHELF_Z_LEVELS[1] + 0.065), RGBA_BLUE),
        ("silinder_merah",  "cylinder", rack_pos(RACK_2_Y_OFFSET,  0.22, 0.550, SHELF_Z_LEVELS[1] + 0.065), RGBA_RED),

        # Rak 2 atas
        ("segitiga_hijau",  "triangle", rack_pos(RACK_2_Y_OFFSET, -0.22, 0.540, SHELF_Z_LEVELS[2] + 0.065), RGBA_GREEN),
        ("segitiga_kuning", "triangle", rack_pos(RACK_2_Y_OFFSET,  0.22, 0.540, SHELF_Z_LEVELS[2] + 0.065), RGBA_YELLOW),
    ]

    for name, shape_type, pos, color in object_specs:
        if shape_type == "box":
            body_id = create_box(
                name=name,
                position=pos,
                size=[0.065, 0.045, 0.085],
                color=color,
                mass=0.10
    )
        elif shape_type == "cylinder":
            body_id = create_cylinder(
                name=name,
                position=pos,
                radius=0.028,
                height=0.090,
                color=color,
                mass=0.09
    )
        elif shape_type == "triangle":
            body_id = create_triangle_prism(
                name=name,
                position=pos,
                size_xyz=[0.070, 0.055, 0.085],
                color=color,
                mass=0.10
            )
        else:
            continue

        apply_object_dynamics(body_id)

        objects[name] = {
            "id": body_id,
            "shape": shape_type,
            "position": pos[:],
            "orientation": [0, 0, 0, 1]
        }

        initial_poses[name] = {
            "position": pos[:],
            "orientation": [0, 0, 0, 1]
        }

        if SHOW_WORLD_OBJECT_LABELS:
            p.addUserDebugText(
                text=name,
                textPosition=[pos[0], pos[1], pos[2] + 0.10],
                textColorRGB=[1, 1, 1],
                textSize=1.0
            )
    return objects, initial_poses


# =========================================================
# LOAD JACO ROBOT
# =========================================================
def load_jaco_robot():
    if not os.path.exists(JACO_URDF):
        raise FileNotFoundError(
            f"URDF JACO tidak ditemukan:\n{JACO_URDF}"
        )

    robot_id = p.loadURDF(
        JACO_URDF,
        basePosition=ROBOT_BASE_POS,
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase=True
    )

    register_body_name(robot_id, "Kinova_JACO2_6DoF", role="robot")
    print("\nRobot Kinova JACO Gen 2 / JACO2 berhasil dimuat.")
    print("Robot ID:", robot_id)
    print("URDF:", JACO_URDF)

    num_joints = p.getNumJoints(robot_id)

    movable_joints = []
    arm_joints = []
    finger_joints = []
    ee_candidates = []

    print("\nDaftar joint / link robot:")
    for i in range(num_joints):
        info = p.getJointInfo(robot_id, i)
        joint_name = info[1].decode("utf-8")
        link_name = info[12].decode("utf-8")
        joint_type = info[2]

        joint_name_l = joint_name.lower()
        link_name_l = link_name.lower()

        print(
            f"Index: {i:2d} | "
            f"Joint: {joint_name:35s} | "
            f"Link: {link_name:35s} | "
            f"Type: {joint_type}"
        )

        if joint_type in [p.JOINT_REVOLUTE, p.JOINT_PRISMATIC]:
            movable_joints.append(i)

        # Ambil hanya joint lengan, bukan finger
        if joint_type == p.JOINT_REVOLUTE:
            if "finger" in joint_name_l or "finger" in link_name_l:
                finger_joints.append(i)
            else:
                arm_joints.append(i)

        # Cari end-effector / palm / hand, jangan pakai finger tip
        if (
            "end_effector" in link_name_l
            or "end-effector" in link_name_l
            or "palm" in link_name_l
            or "hand" in link_name_l
        ):
            if "finger" not in link_name_l:
                ee_candidates.append(i)

    # Untuk JACO 6-DoF, pakai 6 joint lengan pertama
    arm_joints = arm_joints[:6]

    if len(ee_candidates) > 0:
        end_effector_link_index = ee_candidates[-1]
    else:
        # fallback: pakai link dari joint lengan terakhir, bukan num_joints - 1
        end_effector_link_index = arm_joints[-1]

    initial_arm_positions = [
        p.getJointState(robot_id, joint_index)[0]
        for joint_index in arm_joints
    ]
    initial_finger_positions = [
        p.getJointState(robot_id, joint_index)[0]
        for joint_index in finger_joints
    ]

    print("\nMode robot: Kinova JACO2 / JACO Gen 2 6-DoF")
    print("Arm joints yang dipakai:", arm_joints)
    print("Finger joints:", finger_joints)
    print("End-effector link index:", end_effector_link_index)
    print("Initial arm positions disimpan:", initial_arm_positions)

    return {
        "robot_id": robot_id,
        "movable_joints": movable_joints,
        "arm_joints": arm_joints,
        "finger_joints": finger_joints,
        "end_effector_link_index": end_effector_link_index,
        "initial_arm_positions": initial_arm_positions,
        "initial_finger_positions": initial_finger_positions
    }


def custom_idle_sleep_step():
    if CUSTOM_IDLE_SLEEP_IN_GUI:
        try:
            info = p.getConnectionInfo()
            if int(info.get("connectionMethod", -1)) == p.GUI:
                time.sleep(1 / 240)
        except Exception:
            pass


def get_custom_idle_arm_targets(robot_data):
    arm_joints = robot_data.get("arm_joints", [])
    preset = CUSTOM_IDLE_PRESETS.get(CUSTOM_IDLE_PRESET, CUSTOM_IDLE_PRESETS["mid"])

    targets = []
    for i, _ in enumerate(arm_joints):
        if i < len(preset):
            targets.append(float(preset[i]))
        else:
            targets.append(0.0)

    return targets


def clear_active_grasp_for_custom_idle():
    """
    Bersihkan constraint grasp lama sebelum robot dipaksa ke idle.
    """
    global ACTIVE_GRASP_CONSTRAINT_ID, ACTIVE_GRASP_OBJECT_ID, WRIST_POV_FORCED_TARGET

    try:
        WRIST_POV_FORCED_TARGET = None
    except Exception:
        pass

    if not CUSTOM_IDLE_CLEAR_GRASP:
        return

    if ACTIVE_GRASP_CONSTRAINT_ID is not None:
        try:
            p.removeConstraint(ACTIVE_GRASP_CONSTRAINT_ID)
            print("Custom idle: active grasp constraint removed:", ACTIVE_GRASP_CONSTRAINT_ID)
        except Exception as exc:
            print("Custom idle: constraint lama gagal/hilang:", exc)

    if ACTIVE_GRASP_OBJECT_ID is not None:
        try:
            p.setCollisionFilterGroupMask(ACTIVE_GRASP_OBJECT_ID, -1, 1, 1)
            p.resetBaseVelocity(ACTIVE_GRASP_OBJECT_ID, [0, 0, 0], [0, 0, 0])
        except Exception:
            pass

    ACTIVE_GRASP_CONSTRAINT_ID = None
    ACTIVE_GRASP_OBJECT_ID = None


def reset_robot_to_custom_idle(robot_data):
    """
    Reset robot ke custom idle yang tidak melipat/stuck seperti pose original.
    """
    clear_active_grasp_for_custom_idle()

    robot_id = robot_data["robot_id"]
    arm_joints = robot_data["arm_joints"]
    finger_joints = robot_data.get("finger_joints", [])
    initial_finger_positions = robot_data.get("initial_finger_positions", [])

    arm_targets = get_custom_idle_arm_targets(robot_data)
    finger_targets = [
        initial_finger_positions[i] if i < len(initial_finger_positions) else 0.0
        for i, _ in enumerate(finger_joints)
    ]

    print("\n=== RESET ROBOT TO CUSTOM IDLE ===")
    print("CUSTOM_IDLE_PRESET:", CUSTOM_IDLE_PRESET)
    print("Arm targets:", [round(v, 5) for v in arm_targets])

    for i, joint_index in enumerate(arm_joints):
        target = arm_targets[i] if i < len(arm_targets) else 0.0
        p.resetJointState(robot_id, joint_index, target, 0.0)

    for i, joint_index in enumerate(finger_joints):
        target = finger_targets[i] if i < len(finger_targets) else 0.0
        p.resetJointState(robot_id, joint_index, target, 0.0)

    for _ in range(max(1, int(CUSTOM_IDLE_SETTLE_STEPS))):
        for i, joint_index in enumerate(arm_joints):
            target = arm_targets[i] if i < len(arm_targets) else 0.0
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target,
                force=CUSTOM_IDLE_ARM_FORCE,
                maxVelocity=CUSTOM_IDLE_ARM_MAX_VELOCITY
            )

        for i, joint_index in enumerate(finger_joints):
            target = finger_targets[i] if i < len(finger_targets) else 0.0
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target,
                force=CUSTOM_IDLE_FINGER_FORCE,
                maxVelocity=CUSTOM_IDLE_FINGER_MAX_VELOCITY
            )

        p.stepSimulation()
        custom_idle_sleep_step()

    # Zero velocity di akhir.
    for i, joint_index in enumerate(arm_joints):
        target = arm_targets[i] if i < len(arm_targets) else 0.0
        p.resetJointState(robot_id, joint_index, target, 0.0)

    for i, joint_index in enumerate(finger_joints):
        target = finger_targets[i] if i < len(finger_targets) else 0.0
        p.resetJointState(robot_id, joint_index, target, 0.0)

    try:
        ee_pos, _ = get_end_effector_position(robot_data)
        print("Custom idle done. EE pos:", [round(v, 5) for v in ee_pos])
    except Exception:
        print("Custom idle done.")


def reset_robot(robot_data):
    """
    Reset robot.
    Jika CUSTOM_IDLE_ENABLED True, robot kembali ke custom idle yang tidak melipat/stuck.
    """
    if CUSTOM_IDLE_ENABLED:
        reset_robot_to_custom_idle(robot_data)
        return

    robot_id = robot_data["robot_id"]
    arm_joints = robot_data["arm_joints"]
    finger_joints = robot_data.get("finger_joints", [])
    initial_arm_positions = robot_data.get("initial_arm_positions", [])
    initial_finger_positions = robot_data.get("initial_finger_positions", [])

    print("\nReset robot ke initial pose.")

    for i, joint_index in enumerate(arm_joints):
        target = initial_arm_positions[i] if i < len(initial_arm_positions) else 0.0
        p.resetJointState(robot_id, joint_index, target, 0.0)
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target,
            force=HOME_RETURN_FORCE,
            maxVelocity=HOME_RETURN_MAX_VELOCITY
        )

    for i, joint_index in enumerate(finger_joints):
        target = initial_finger_positions[i] if i < len(initial_finger_positions) else 0.0
        p.resetJointState(robot_id, joint_index, target, 0.0)
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target,
            force=60,
            maxVelocity=0.45
        )


# =========================================================
# RESET OBJEK
# =========================================================
def reset_objects(objects, initial_poses):
    for name, obj in objects.items():
        body_id = obj["id"]
        pos = initial_poses[name]["position"]
        orn = initial_poses[name]["orientation"]

        p.resetBasePositionAndOrientation(body_id, pos, orn)
        p.resetBaseVelocity(body_id, [0, 0, 0], [0, 0, 0])


def ensure_object_random_log_dir():
    os.makedirs(OBJECT_RANDOM_LOG_DIR, exist_ok=True)


def update_object_registry_pose(objects, name, pos, orn):
    """
    Update registry object supaya posisi terbaru tersimpan.
    Ini penting karena setelah object di-random, fungsi gerak/pick tetap bisa membaca posisi terbaru.
    """
    if name not in objects:
        return

    obj = objects[name]

    for key in ["position", "pos", "base_position", "initial_position"]:
        if key in obj:
            obj[key] = list(pos)

    obj["current_position"] = list(pos)
    obj["current_orientation"] = list(orn)
    obj["randomized_position"] = list(pos)
    obj["randomized_orientation"] = list(orn)


def log_randomized_object_poses(objects, reason="reset"):
    """
    Simpan posisi random ke CSV agar bisa dipakai sebagai bukti Bab 4/lampiran.
    """
    ensure_object_random_log_dir()

    path = os.path.join(
        OBJECT_RANDOM_LOG_DIR,
        "object_randomized_poses.csv"
    )

    file_exists = os.path.exists(path)

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow([
                "timestamp",
                "reason",
                "object_name",
                "body_id",
                "x",
                "y",
                "z",
                "roll",
                "pitch",
                "yaw"
            ])

        ts = time.strftime("%Y-%m-%d %H:%M:%S")

        for name, obj in objects.items():
            body_id = obj["id"]
            pos, orn = p.getBasePositionAndOrientation(body_id)
            eul = p.getEulerFromQuaternion(orn)

            writer.writerow([
                ts,
                reason,
                name,
                body_id,
                round(pos[0], 5),
                round(pos[1], 5),
                round(pos[2], 5),
                round(eul[0], 5),
                round(eul[1], 5),
                round(eul[2], 5),
            ])


def reset_objects_randomized(objects, initial_poses, reason="reset"):
    """
    Reset objek dengan posisi acak.

    Revisi multi-rack:
    - jika OBJECT_RANDOMIZE_KEEP_RACK_GROUP=True, object hanya ditukar slot
      di dalam rak asalnya, tidak loncat dari rak 1 ke rak 2.
    - Ini membuat demonstrasi fallback rak 1 -> rak 2 tetap konsisten.
    - Jitter posisi dan yaw tetap aktif agar pengujian tidak terlalu kaku.
    """
    names = list(objects.keys())

    def make_slot(name):
        return {
            "name": name,
            "position": list(initial_poses[name]["position"]),
            "orientation": list(initial_poses[name]["orientation"]),
            "rack_id": get_rack_id_from_y(initial_poses[name]["position"][1])
        }

    if any(name not in initial_poses for name in names):
        print("Random reset gagal: sebagian object tidak punya initial pose.")
        print("Fallback ke reset_objects normal.")
        reset_objects(objects, initial_poses)
        return

    slot_assignment = {}

    if OBJECT_RANDOMIZE_SHUFFLE_SLOTS and OBJECT_RANDOMIZE_KEEP_RACK_GROUP:
        rack_ids = sorted(set(make_slot(name)["rack_id"] for name in names))

        for rack_id in rack_ids:
            rack_names = [
                name for name in names
                if get_rack_id_from_y(initial_poses[name]["position"][1]) == rack_id
            ]

            rack_slots = [make_slot(name) for name in rack_names]
            random.shuffle(rack_slots)

            for name, slot in zip(rack_names, rack_slots):
                slot_assignment[name] = slot

        print("Object placement randomized: shuffle slot hanya di dalam rak yang sama.")

    else:
        base_slots = [make_slot(name) for name in names]

        if OBJECT_RANDOMIZE_SHUFFLE_SLOTS:
            random.shuffle(base_slots)

        if len(base_slots) != len(names):
            print("Random reset gagal: jumlah slot tidak sama dengan jumlah objek.")
            print("Fallback ke reset_objects normal.")
            reset_objects(objects, initial_poses)
            return

        for name, slot in zip(names, base_slots):
            slot_assignment[name] = slot

    for name in names:
        body_id = objects[name]["id"]
        slot = slot_assignment[name]

        base_pos = list(slot["position"])
        base_orn = list(slot["orientation"])

        new_pos = [
            base_pos[0] + random.uniform(-OBJECT_RANDOM_X_JITTER, OBJECT_RANDOM_X_JITTER),
            base_pos[1] + random.uniform(-OBJECT_RANDOM_Y_JITTER, OBJECT_RANDOM_Y_JITTER),
            base_pos[2] + random.uniform(-OBJECT_RANDOM_Z_JITTER, OBJECT_RANDOM_Z_JITTER),
        ]

        roll, pitch, yaw = p.getEulerFromQuaternion(base_orn)
        yaw += math.radians(random.uniform(-OBJECT_RANDOM_YAW_DEG, OBJECT_RANDOM_YAW_DEG))
        new_orn = p.getQuaternionFromEuler([roll, pitch, yaw])

        p.resetBasePositionAndOrientation(body_id, new_pos, new_orn)
        p.resetBaseVelocity(body_id, [0, 0, 0], [0, 0, 0])

        update_object_registry_pose(objects, name, new_pos, new_orn)

    for _ in range(OBJECT_RANDOM_SETTLE_STEPS):
        p.stepSimulation()

    log_randomized_object_poses(objects, reason=reason)

    print(f"Object placement randomized saat {reason}.")


def set_mobile_platform_y_offset(offset_y):
    """
    Menggeser wheelchair + robot + basket ke offset Y tertentu.
    Rak dan objek tidak ikut bergerak.
    """
    global CURRENT_PLATFORM_Y_OFFSET

    CURRENT_PLATFORM_Y_OFFSET = float(offset_y)

    # Update posisi referensi global supaya motion planning ikut station aktif.
    ROBOT_BASE_POS[1] = ROBOT_BASE_POS_ORIGINAL[1] + offset_y
    HANDOVER_POS[1] = HANDOVER_POS_ORIGINAL[1] + offset_y
    BASKET_CENTER[1] = BASKET_CENTER_ORIGINAL[1] + offset_y
    SAFE_STANDBY_POSITION[1] = ROBOT_BASE_POS[1]

    if BASKET_INFO is not None:
        BASKET_INFO["center"] = BASKET_CENTER[:]

    for body_id in MOBILE_PLATFORM_BODY_IDS:
        if body_id not in MOBILE_PLATFORM_ORIGINAL_POSES:
            continue

        base = MOBILE_PLATFORM_ORIGINAL_POSES[body_id]
        base_pos = base["position"]
        base_orn = base["orientation"]

        new_pos = [
            base_pos[0],
            base_pos[1] + offset_y,
            base_pos[2]
        ]

        try:
            p.resetBasePositionAndOrientation(body_id, new_pos, base_orn)
            p.resetBaseVelocity(body_id, [0, 0, 0], [0, 0, 0])
        except Exception as exc:
            print("WARNING: gagal geser body mobile:", body_id, exc)


def move_wheelchair_to_station(robot_data, station_id, duration=2.0):
    """
    Wheelchair bergerak maju/mundur ke station rak.
    Ini bukan autonomous navigation, hanya station-based movement.
    """
    global CURRENT_STATION_ID, ACTIVE_RACK_ID, ACTIVE_RACK_Y_OFFSET

    if station_id not in WHEELCHAIR_STATIONS:
        print("Station tidak dikenal:", station_id)
        return False

    station = WHEELCHAIR_STATIONS[station_id]
    target_offset = float(station["platform_y_offset"])
    start_offset = float(CURRENT_PLATFORM_Y_OFFSET)

    if CURRENT_STATION_ID == station_id and abs(start_offset - target_offset) < 1e-6:
        ACTIVE_RACK_ID = int(station["rack_id"])
        ACTIVE_RACK_Y_OFFSET = float(station["rack_y_offset"])
        print("\n=== MOVE WHEELCHAIR TO STATION ===")
        print("Wheelchair sudah di station:", station_id)
        print("Active rack:", ACTIVE_RACK_ID)
        print("Active rack y offset:", ACTIVE_RACK_Y_OFFSET)
        return True

    print("\n=== MOVE WHEELCHAIR TO STATION ===")
    print("From station :", CURRENT_STATION_ID)
    print("To station   :", station_id)
    print("Start offset :", round(start_offset, 4))
    print("Target offset:", round(target_offset, 4))

    # Arm dibuat aman dulu sebelum platform bergerak.
    reset_robot(robot_data)

    steps = max(1, int(duration * 240))

    for i in range(steps):
        alpha = (i + 1) / steps
        s = alpha * alpha * (3.0 - 2.0 * alpha)

        offset = (1.0 - s) * start_offset + s * target_offset
        set_mobile_platform_y_offset(offset)

        p.stepSimulation()
        time.sleep(1 / 240)

    set_mobile_platform_y_offset(target_offset)

    CURRENT_STATION_ID = station_id
    ACTIVE_RACK_ID = int(station["rack_id"])
    ACTIVE_RACK_Y_OFFSET = float(station["rack_y_offset"])

    print("Wheelchair sampai station:", CURRENT_STATION_ID)
    print("Active rack:", ACTIVE_RACK_ID)
    print("Active rack y offset:", ACTIVE_RACK_Y_OFFSET)

    return True


# =========================================================
# SYNTHETIC CAMERA
# =========================================================
def normalize_vector(vec, fallback=None):
    arr = np.array(vec, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-9:
        if fallback is None:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        return np.array(fallback, dtype=float)
    return arr / norm


def rotate_vector_by_quaternion(local_vec, quat):
    mat = np.array(p.getMatrixFromQuaternion(quat), dtype=float).reshape(3, 3)
    return mat.dot(np.array(local_vec, dtype=float))


def get_wrist_pov_mount_link_index(robot_data):
    """
    Link untuk POV kamera.

    Dipasang secara virtual pada wrist / link lengan terakhir agar POV terasa mounted,
    tetapi bukan di ujung jari/gripper.
    """
    if WRIST_POV_USE_WRIST_LINK:
        arm_joints = robot_data.get("arm_joints", [])
        if len(arm_joints) > 0:
            return arm_joints[-1]
    return robot_data["end_effector_link_index"]


def get_wrist_pov_camera_pose(robot_data, objects=None, selected_name=None):
    """
    Kamera POV virtual mounted dekat wrist/end-effector.

    Tidak membuat body kamera di PyBullet.
    Eye position:
    - mengikuti link wrist/last arm link + WRIST_POV_LOCAL_OFFSET.

    Target:
    - default look-at objek aktif/objek hasil ASR agar kamera seperti mencari target user.
    """
    # =====================================================
    # REALTIME FIXED CAMERA
    # =====================================================
    # Camera bergerak HANYA bersama wheelchair/base station.
    # Tidak mengikuti joint/wrist, sehingga angle tidak muter.
    if REALTIME_FIXED_CAMERA_MODE:
        eye = [
            ROBOT_BASE_POS[0] + REALTIME_FIXED_CAMERA_EYE_OFFSET[0],
            ROBOT_BASE_POS[1] + REALTIME_FIXED_CAMERA_EYE_OFFSET[1],
            ROBOT_BASE_POS[2] + REALTIME_FIXED_CAMERA_EYE_OFFSET[2],
        ]

        look_dir = normalize_vector(
            REALTIME_FIXED_CAMERA_LOOK_DIRECTION,
            fallback=[1.0, 0.0, 0.0]
        )

        target = (
            np.array(eye, dtype=float)
            + look_dir * float(REALTIME_FIXED_CAMERA_LOOK_DISTANCE)
        ).tolist()

        up = normalize_vector(
            REALTIME_FIXED_CAMERA_WORLD_UP,
            fallback=[0.0, 0.0, 1.0]
        ).tolist()

        return list(eye), list(target), list(up)

    robot_id = robot_data["robot_id"]
    mount_link = get_wrist_pov_mount_link_index(robot_data)

    link_state = p.getLinkState(robot_id, mount_link, computeForwardKinematics=True)
    mount_pos = link_state[4]
    mount_orn = link_state[5]

    camera_eye_position, _ = p.multiplyTransforms(
        mount_pos,
        mount_orn,
        WRIST_POV_LOCAL_OFFSET,
        [0, 0, 0, 1]
    )

    # =====================================================
    # WRIST-MOUNTED EYE + WORLD-LOCKED ORIENTATION
    # =====================================================
    # Eye tetap pada wrist/link terakhir dan bergerak realtime bersama arm.
    # Arah pandang tidak ikut rotasi wrist, sehingga window tidak spin.
    if WRIST_POV_LOCK_WORLD_ORIENTATION:
        fixed_dir = normalize_vector(
            WRIST_POV_WORLD_VIEW_DIRECTION,
            fallback=[1.0, 0.0, 0.0]
        )

        # v11: Forced target dipakai HANYA untuk elevasi shelf bawah.
        # Arah horizontal tetap +X (tidak yaw ke kanan/kiri), up world +Z
        # (tidak roll). Posisi eye tetap hasil kinematika wrist, tidak
        # dipindahkan ke base, objek, atau pose ground truth.
        if (
            VISUAL_SEARCH_LOW_SHELF_AIM_ENABLED
            and WRIST_POV_FORCED_TARGET is not None
            and float(WRIST_POV_FORCED_TARGET[2])
                < VISUAL_SEARCH_LOW_SHELF_TARGET_Z_LIMIT
        ):
            eye = np.asarray(camera_eye_position, dtype=float)
            low_shelf_target = np.asarray(WRIST_POV_FORCED_TARGET, dtype=float)
            forward_m = max(
                float(low_shelf_target[0] - eye[0]),
                VISUAL_SEARCH_LOW_SHELF_MIN_FORWARD_M,
            )
            pitch_rad = float(np.arctan2(
                float(low_shelf_target[2] - eye[2]), forward_m
            ))
            pitch_rad = float(np.clip(
                pitch_rad,
                -np.deg2rad(VISUAL_SEARCH_LOW_SHELF_MAX_DOWN_DEG),
                np.deg2rad(VISUAL_SEARCH_LOW_SHELF_MAX_UP_DEG),
            ))
            fixed_dir = np.array(
                [np.cos(pitch_rad), 0.0, np.sin(pitch_rad)], dtype=float
            )

        camera_target_position = (
            np.array(camera_eye_position, dtype=float)
            + fixed_dir * float(WRIST_POV_WORLD_VIEW_DISTANCE)
        ).tolist()

        up_world = normalize_vector(
            WRIST_POV_WORLD_UP,
            fallback=[0.0, 0.0, 1.0]
        )

        return (
            list(camera_eye_position),
            list(camera_target_position),
            up_world.tolist()
        )

    if WRIST_POV_FORCED_TARGET is not None:
        # Dipakai saat robot melakukan visual search.
        # Pada fixed-angle mode, eye tetap mengikuti wrist tetapi arah pandang
        # DIKUNCI ke arah dunia (+X), sehingga kamera tidak muter antar scan.
        if VISUAL_SEARCH_LOCK_WORLD_VIEW_DIRECTION:
            fixed_dir = normalize_vector(
                VISUAL_SEARCH_WORLD_VIEW_DIRECTION,
                fallback=[1.0, 0.0, 0.0]
            )
            camera_target_position = (
                np.array(camera_eye_position, dtype=float)
                + fixed_dir * float(VISUAL_SEARCH_WORLD_VIEW_DISTANCE)
            ).tolist()
        else:
            camera_target_position = list(WRIST_POV_FORCED_TARGET)

    elif (
        WRIST_POV_LOOK_AT_SELECTED_OBJECT
        and objects is not None
        and selected_name in objects
    ):
        object_id = objects[selected_name]["id"]
        obj_pos, _ = p.getBasePositionAndOrientation(object_id)
        camera_target_position = [
            obj_pos[0],
            obj_pos[1],
            obj_pos[2] + WRIST_POV_TARGET_Z_OFFSET
        ]
    else:
        forward_world = rotate_vector_by_quaternion(WRIST_POV_FORWARD_AXIS_LOCAL, mount_orn)
        forward_world = normalize_vector(forward_world, fallback=[1, 0, 0])
        camera_target_position = (
            np.array(camera_eye_position, dtype=float)
            + forward_world * WRIST_POV_LOOK_DISTANCE
        ).tolist()

    up_world = rotate_vector_by_quaternion(WRIST_POV_UP_AXIS_LOCAL, mount_orn)
    up_world = normalize_vector(up_world, fallback=[0, 0, 1])

    # Saat visual search overview, horizon dibuat tetap tegak terhadap dunia.
    # Eye tetap mengikuti wrist, tetapi gambar tidak ikut roll/berputar bolak-balik.
    if WRIST_POV_FORCED_TARGET is not None and VISUAL_SEARCH_STATIC_WORLD_UP:
        up_world = np.array([0.0, 0.0, 1.0], dtype=float)

    view_dir = normalize_vector(
        np.array(camera_target_position, dtype=float) - np.array(camera_eye_position, dtype=float),
        fallback=[1, 0, 0]
    )

    # Hindari up vector sejajar dengan arah pandang.
    if abs(float(np.dot(view_dir, up_world))) > 0.95:
        up_world = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(view_dir, up_world))) > 0.95:
            up_world = np.array([0.0, 1.0, 0.0], dtype=float)

    return list(camera_eye_position), list(camera_target_position), up_world.tolist()


def capture_synthetic_camera(width=950, height=700, robot_data=None, objects=None, selected_name=None):
    """
    Kamera utama simulasi.

    Jika WRIST_POV_CAMERA_ENABLED=True dan robot_data tersedia:
    - kamera berupa POV virtual dari wrist/end-effector.
    - kamera tidak tampak sebagai objek di simulasi.
    - tidak ada body, collision, atau constraint tambahan.
    """
    if WRIST_POV_CAMERA_ENABLED and robot_data is not None:
        camera_eye_position, camera_target_position, camera_up_vector = get_wrist_pov_camera_pose(
            robot_data=robot_data,
            objects=objects,
            selected_name=selected_name
        )
        fov = WRIST_POV_FOV
        near_val = WRIST_POV_NEAR
        far_val = WRIST_POV_FAR
    else:
        # Fallback third-person hanya kalau robot_data belum tersedia.
        camera_eye_position = [-0.25, -1.35, 1.15]
        camera_target_position = [0.72, 0.02, 0.88]
        camera_up_vector = [0, 0, 1]
        fov = 58
        near_val = 0.02
        far_val = 4.0

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=camera_eye_position,
        cameraTargetPosition=camera_target_position,
        cameraUpVector=camera_up_vector
    )

    projection_matrix = p.computeProjectionMatrixFOV(
        fov=fov,
        aspect=width / height,
        nearVal=near_val,
        farVal=far_val
    )

    image = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )

    rgb = np.reshape(image[2], (height, width, 4))[:, :, :3].astype(np.uint8)
    depth = np.reshape(image[3], (height, width))
    seg_raw = np.reshape(image[4], (height, width))
    seg_body = extract_body_id_from_segmentation(seg_raw)

    return rgb, depth, seg_body


def make_depth_visual(depth):
    depth_vis = cv2.normalize(depth, None, 0, 255, cv2.NORM_MINMAX)
    depth_vis = depth_vis.astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)
    return depth_vis


def make_segmentation_visual(seg_body, objects):
    h, w = seg_body.shape
    seg_vis = np.zeros((h, w, 3), dtype=np.uint8)

    palette = [
        [255,  60,  60],   # red
        [ 60, 255,  60],   # green
        [ 60,  60, 255],   # blue
        [255, 220,  40],   # yellow
        [255, 120,  20],   # orange
        [180,  80, 255],   # purple
        [255, 255, 255],   # white
        [120, 255, 255],   # cyan
        [255, 120, 220],   # pink
        [180, 180,  60],
        [120, 180, 255],
        [220, 120, 120],
    ]

    for idx, (name, data) in enumerate(objects.items()):
        body_id = data["id"]
        mask = (seg_body == body_id)
        seg_vis[mask] = palette[idx % len(palette)]

    return seg_vis


def draw_object_centroids(rgb, seg_body, objects, selected_name=None):
    rgb_draw = rgb.copy()

    for name, data in objects.items():
        body_id = data["id"]
        mask = (seg_body == body_id)

        ys, xs = np.where(mask)
        if len(xs) > 0 and len(ys) > 0:
            cx = int(np.mean(xs))
            cy = int(np.mean(ys))

            if name == selected_name:
                color = (0, 255, 255)
                radius = 8
                thickness = -1
            else:
                color = (255, 255, 255)
                radius = 5
                thickness = -1

            cv2.circle(rgb_draw, (cx, cy), radius, color, thickness)
            cv2.putText(
                rgb_draw,
                name,
                (cx + 8, cy - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2
            )

    return rgb_draw


def set_live_camera_context(robot_data=None, objects=None, selected_name=None):
    """
    Simpan konteks kamera agar POV wrist tetap update saat robot bergerak.
    """
    if robot_data is not None:
        LIVE_CAMERA_CONTEXT["robot_data"] = robot_data
    if objects is not None:
        LIVE_CAMERA_CONTEXT["objects"] = objects
    if selected_name is not None:
        LIVE_CAMERA_CONTEXT["selected_name"] = selected_name


def update_live_wrist_pov_camera_window():
    """
    Update RGB window dari POV kamera wrist saat robot sedang bergerak.
    Fungsi ini tidak membuat objek kamera di dunia simulasi.
    """
    if not SHOW_CAMERA_WINDOWS:
        return

    if not WRIST_POV_CAMERA_ENABLED or not WRIST_POV_LIVE_DURING_MOTION:
        return

    robot_data = LIVE_CAMERA_CONTEXT.get("robot_data")
    objects = LIVE_CAMERA_CONTEXT.get("objects")
    selected_name = LIVE_CAMERA_CONTEXT.get("selected_name")

    if robot_data is None or objects is None:
        return

    try:
        rgb, depth, seg_body = capture_synthetic_camera(
            width=960,
            height=720,
            robot_data=robot_data,
            objects=objects,
            selected_name=selected_name
        )

        if CAMERA_DRAW_OBJECT_LABELS:
            rgb_annotated = draw_object_centroids(
                rgb=rgb,
                seg_body=seg_body,
                objects=objects,
                selected_name=selected_name
            )
        else:
            rgb_annotated = rgb.copy()

        if not CAMERA_VIEW_ONLY_MODE:
            cv2.putText(
                rgb_annotated,
                f"Wrist-mounted POV | Target: {selected_name}",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 255),
                2
            )

            cv2.putText(
                rgb_annotated,
                "virtual camera only - no visible camera body",
                (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.58,
                (0, 255, 255),
                2
            )

            cv2.putText(
                rgb_annotated,
                f"offset: {[round(v, 3) for v in WRIST_POV_LOCAL_OFFSET]}",
                (20, 95),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 255),
                2
            )

        rgb_bgr = cv2.cvtColor(rgb_annotated, cv2.COLOR_RGB2BGR)
        safe_imshow(CAMERA_WINDOW_NAME, rgb_bgr)

    except Exception as exc:
        print("WARNING: wrist POV camera update gagal:", exc)



def ensure_wrist_pov_log_file():
    """
    Membuat file CSV log tuning kamera.
    """
    global WRIST_POV_LOG_FILE

    if WRIST_POV_LOG_FILE is not None:
        return WRIST_POV_LOG_FILE

    os.makedirs(WRIST_POV_LOG_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    WRIST_POV_LOG_FILE = os.path.join(
        WRIST_POV_LOG_DIR,
        f"wrist_pov_camera_tuning_{timestamp}.csv"
    )

    with open(WRIST_POV_LOG_FILE, "w", encoding="utf-8") as f:
        f.write(
            "timestamp,selected_name,"
            "local_offset_x,local_offset_y,local_offset_z,"
            "eye_x,eye_y,eye_z,"
            "target_x,target_y,target_z,"
            "up_x,up_y,up_z,"
            "mount_link_index\n"
        )

    print("Camera tuning log file:", WRIST_POV_LOG_FILE)
    return WRIST_POV_LOG_FILE


def get_current_wrist_pov_camera_state(robot_data, objects=None, selected_name=None):
    """
    Ambil state kamera saat ini:
    - local offset yang sedang dipakai
    - world eye position
    - world target position
    - up vector
    - mount link index
    """
    camera_eye, camera_target, camera_up = get_wrist_pov_camera_pose(
        robot_data=robot_data,
        objects=objects,
        selected_name=selected_name
    )

    mount_link = get_wrist_pov_mount_link_index(robot_data)

    return {
        "selected_name": selected_name,
        "local_offset": list(WRIST_POV_LOCAL_OFFSET),
        "eye": list(camera_eye),
        "target": list(camera_target),
        "up": list(camera_up),
        "mount_link_index": mount_link
    }


def print_wrist_pov_camera_state(robot_data, objects=None, selected_name=None, save_to_csv=True):
    """
    Print koordinat kamera ke terminal dan opsional simpan ke CSV.
    Kirim bagian print ini ke chat untuk saya permanenkan offset-nya.
    """
    state = get_current_wrist_pov_camera_state(
        robot_data=robot_data,
        objects=objects,
        selected_name=selected_name
    )

    print("\n=== WRIST POV CAMERA TUNING STATE ===")
    print("selected_name:", state["selected_name"])
    print("mount_link_index:", state["mount_link_index"])
    print("WRIST_POV_LOCAL_OFFSET =", [round(v, 6) for v in state["local_offset"]])
    print("camera_eye_world =", [round(v, 6) for v in state["eye"]])
    print("camera_target_world =", [round(v, 6) for v in state["target"]])
    print("camera_up_vector =", [round(v, 6) for v in state["up"]])

    if save_to_csv:
        log_file = ensure_wrist_pov_log_file()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")

        row = [
            timestamp,
            str(state["selected_name"]),
            *state["local_offset"],
            *state["eye"],
            *state["target"],
            *state["up"],
            state["mount_link_index"]
        ]

        with open(log_file, "a", encoding="utf-8") as f:
            f.write(",".join(str(x) for x in row) + "\n")

        print("Saved camera tuning row to:", log_file)

    return state


def tune_wrist_pov_camera_from_key(key, robot_data=None, objects=None, selected_name=None):
    """
    Geser posisi POV kamera secara realtime di frame lokal wrist/link akhir.

    Tombol:
    j/l = X -/+
    i/k = Y +/-
    u/o = Z +/-
    y   = print + simpan koordinat tanpa menggeser
    """
    if not WRIST_POV_TUNING_ENABLED:
        return False

    step = WRIST_POV_TUNE_STEP
    delta = None

    if key == ord('j'):
        delta = [-step, 0.0, 0.0]
    elif key == ord('l'):
        delta = [step, 0.0, 0.0]
    elif key == ord('i'):
        delta = [0.0, step, 0.0]
    elif key == ord('k'):
        delta = [0.0, -step, 0.0]
    elif key == ord('u'):
        delta = [0.0, 0.0, step]
    elif key == ord('o'):
        delta = [0.0, 0.0, -step]
    elif key == ord('y'):
        if robot_data is not None:
            print_wrist_pov_camera_state(
                robot_data=robot_data,
                objects=objects,
                selected_name=selected_name,
                save_to_csv=True
            )
            update_live_wrist_pov_camera_window()
        return True
    else:
        return False

    WRIST_POV_LOCAL_OFFSET[0] += delta[0]
    WRIST_POV_LOCAL_OFFSET[1] += delta[1]
    WRIST_POV_LOCAL_OFFSET[2] += delta[2]

    if robot_data is not None:
        print_wrist_pov_camera_state(
            robot_data=robot_data,
            objects=objects,
            selected_name=selected_name,
            save_to_csv=True
        )
        update_live_wrist_pov_camera_window()

    return True


# =========================================================
# KONTROL GERAK OBJEK
# =========================================================
def move_selected_object(object_id, key, step=0.03):
    """
    Kontrol:
    w/s -> +x / -x
    a/d -> +y / -y
    z/x -> -z / +z
    """
    pos, orn = p.getBasePositionAndOrientation(object_id)
    x, y, z = pos

    if key == ord('w'):
        x += step
    elif key == ord('s'):
        x -= step
    elif key == ord('a'):
        y += step
    elif key == ord('d'):
        y -= step
    elif key == ord('z'):
        z -= step
    elif key == ord('x'):
        z += step

    p.resetBasePositionAndOrientation(object_id, [x, y, z], orn)
    p.resetBaseVelocity(object_id, [0, 0, 0], [0, 0, 0])

# =========================================================
# GERAK ROBOT KE OBJEK TARGET / IK REACH TEST
# =========================================================
def get_end_effector_position(robot_data):
    robot_id = robot_data["robot_id"]
    ee_link = robot_data["end_effector_link_index"]

    link_state = p.getLinkState(robot_id, ee_link)
    ee_position = link_state[0]
    ee_orientation = link_state[1]

    return ee_position, ee_orientation

def rotate_joint1_to_face_object(robot_data, object_pos, duration=JOINT1_ALIGN_DURATION):
    """
    Pre-alignment sebelum pengambilan objek.

    Pada tahap ini hanya joint-1 lengan yang digerakkan.
    Joint lain ditahan pada posisi saat ini.

    Tujuan:
    - base/yaw lengan menghadap target dulu,
    - joint-2 tidak langsung mengambil kompensasi gerakan,
    - gerakan awal menuju objek terlihat lebih natural dan tidak menyilang.

    Catatan:
    - Ini bukan pengambilan objek full dengan satu joint.
    - Setelah joint-1 align, IK utama tetap diperlukan untuk mengatur reach, tinggi, dan gripper.
    """
    if not ENABLE_JOINT1_PREALIGN:
        return

    robot_id = robot_data["robot_id"]
    arm_joints = robot_data["arm_joints"]

    if len(arm_joints) < 1:
        print("Joint-1 pre-align dibatalkan: arm_joints kosong.")
        return

    joint1 = arm_joints[0]

    # Tahan joint lain supaya hanya joint-1 yang bergerak.
    hold_positions = {
        joint_index: p.getJointState(robot_id, joint_index)[0]
        for joint_index in arm_joints
    }

    base_x, base_y, _ = ROBOT_BASE_POS
    obj_x, obj_y, _ = object_pos

    dx = obj_x - base_x
    dy = obj_y - base_y

    # Estimasi yaw menuju objek.
    # Pada layout ini sumbu X mengarah ke rak, sehingga atan2(dy, dx) cukup untuk mengarahkan base joint.
    target_joint1 = math.atan2(dy, dx)

    info = p.getJointInfo(robot_id, joint1)
    lower = info[8]
    upper = info[9]

    # Clamp jika joint punya limit valid.
    if upper > lower:
        target_joint1 = max(lower, min(upper, target_joint1))

    print("\nStep joint-1 pre-align: hanya joint-1 yang bergerak")
    print("Joint-1 PyBullet index:", joint1)
    print("Target object pos:", object_pos)
    print("Target joint-1 yaw:", target_joint1)

    steps = max(1, int(duration * 240))

    for _ in range(steps):
        # Joint-1 bergerak ke yaw target.
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=joint1,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target_joint1,
            force=JOINT1_ALIGN_FORCE,
            maxVelocity=JOINT1_ALIGN_MAX_VELOCITY
        )

        # Joint lain ditahan agar tidak ikut bergerak di fase ini.
        for joint_index in arm_joints[1:]:
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=hold_positions[joint_index],
                force=180,
                maxVelocity=0.20
            )

        p.stepSimulation()
        time.sleep(1 / 240)

    final_joint1 = p.getJointState(robot_id, joint1)[0]
    print("Joint-1 final position:", final_joint1)


def move_robot_to_home_pose(robot_data, duration=HOME_RETURN_DURATION):
    """
    Kembali ke home/idle.
    Diarahkan ke reset_robot() supaya tombol m tidak memakai pose folded original.
    """
    print("\nTrigger manual: custom idle/home pose.")
    reset_robot(robot_data)


def set_gripper_open(robot_data):
    """
    Membuka finger secara visual. Nilai finger bisa berbeda antar URDF,
    jadi ini dibuat konservatif agar tidak memicu gerakan aneh.
    """
    robot_id = robot_data["robot_id"]
    finger_joints = robot_data.get("finger_joints", [])

    for joint_index in finger_joints:
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=0.0,
            force=60,
            maxVelocity=0.5
        )


def set_gripper_close(robot_data):
    """
    Menutup finger secara visual. Untuk JACO URDF ini hanya membantu tampilan;
    pengangkatan objek tetap memakai virtual constraint.
    """
    robot_id = robot_data["robot_id"]
    finger_joints = robot_data.get("finger_joints", [])

    for joint_index in finger_joints:
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=0.8,
            force=80,
            maxVelocity=0.6
        )

    for _ in range(80):
        p.stepSimulation()
        time.sleep(1 / 240)


def fast_duration(duration):
    """
    Skala durasi gerak agar E2E robot lebih cepat.
    """
    return max(0.25, float(duration) * E2E_SPEED_FACTOR)


def fast_velocity(max_velocity):
    """
    Skala max velocity dengan batas aman agar tidak terlalu kasar.
    """
    return min(0.42, float(max_velocity) * E2E_MAX_VELOCITY_SCALE)


def fast_force(motor_force):
    """
    Skala motor force.
    """
    return int(float(motor_force) * E2E_MOTOR_FORCE_SCALE)


def move_robot_to_position(
    robot_data,
    target_position,
    duration=3.0,
    attached_object_id=None,
    grasp_constraint_id=None,
    release_on_collision=False,
    monitor_start_delay_steps=40,
    motor_force=200,
    max_velocity=0.25,
    error_print_limit=0.20
):
    """
    Menggerakkan end-effector robot ke target_position menggunakan IK PyBullet.

    Jika attached_object_id dan grasp_constraint_id diberikan,
    fungsi ini juga mengecek apakah objek yang sedang dibawa
    menabrak benda lain. Jika iya, grasp dilepas.
    """
    robot_id = robot_data["robot_id"]
    arm_joints = robot_data["arm_joints"]
    ee_link = robot_data["end_effector_link_index"]

    # E2E speed tuning: semua motion yang lewat fungsi ini dipercepat.
    original_duration = duration
    original_motor_force = motor_force
    original_max_velocity = max_velocity
    duration = fast_duration(duration)
    motor_force = fast_force(motor_force)
    max_velocity = fast_velocity(max_velocity)

# Untuk JACO 6-DoF, position-only IK lebih stabil.
# Jangan paksa orientasi dulu, karena bisa bikin robot stuck.
    ik_solution = p.calculateInverseKinematics(
        bodyUniqueId=robot_id,
        endEffectorLinkIndex=ee_link,
        targetPosition=target_position,
        solver=p.IK_DLS,
        maxNumIterations=300,
        residualThreshold=1e-4
    )

    steps = int(duration * 240)
    dropped = False

    print("\n=== IK MOVE ===")
    print("Target position:", target_position)
    print("End-effector link:", ee_link)
    print("Arm joints:", arm_joints)
    print(
        "Speed tuning | duration:",
        original_duration, "->", round(duration, 3),
        "| motor_force:", original_motor_force, "->", motor_force,
        "| max_velocity:", original_max_velocity, "->", round(max_velocity, 3)
    )

    for step in range(steps):
        for i, joint_index in enumerate(arm_joints):
            if i < len(ik_solution):
                p.setJointMotorControl2(
                    bodyUniqueId=robot_id,
                    jointIndex=joint_index,
                    controlMode=p.POSITION_CONTROL,
                    targetPosition=ik_solution[i],
                    force=motor_force,
                    maxVelocity=max_velocity
                )

        p.stepSimulation()

        if (
            MANUAL_RELEASE_ENABLED
            and grasp_constraint_id is not None
            and manual_release_key_pressed()
        ):
            print("Manual release key detected during IK motion.")
            manual_release_active_grasp(
                robot_data=robot_data,
                reason="key 'o' during IK motion"
            )
            grasp_constraint_id = None
            dropped = True
            break

        collision_stop_requested = monitor_collision_during_motion(
            robot_data=robot_data,
            step=step,
            stage_name="IK_MOVE"
        )

        if collision_stop_requested:
            dropped = True
            break

        if (
            WRIST_POV_CAMERA_ENABLED
            and WRIST_POV_LIVE_DURING_MOTION
            and step % WRIST_POV_UPDATE_EVERY_N_STEPS == 0
        ):
            update_live_wrist_pov_camera_window()

        # Cek collision objek yang sedang dibawa
        if (
            release_on_collision
            and attached_object_id is not None
            and grasp_constraint_id is not None
            and step > monitor_start_delay_steps
        ):
            collision, other_body = carried_object_has_bad_contact(
                robot_id=robot_id,
                object_id=attached_object_id
            )

            if collision:
                print("WARNING: Objek yang dibawa menabrak body ID:", other_body)
                print("Virtual grasp dilepas karena collision.")
                p.removeConstraint(grasp_constraint_id)
                grasp_constraint_id = None
                dropped = True
                break

        time.sleep(1 / 240)

    ee_pos, _ = get_end_effector_position(robot_data)
    error = np.linalg.norm(np.array(ee_pos) - np.array(target_position))
    print("End-effector akhir:", ee_pos)
    print("Position error:", error)

    if error > error_print_limit:
        print("IK WARNING: target belum tercapai sesuai limit tahap ini.")

    return error, grasp_constraint_id, dropped

def carried_object_has_bad_contact(robot_id, object_id):
    """
    Mengecek apakah objek yang sedang dibawa menabrak benda lain.

    Kontak yang diizinkan:
    - objek dengan robot/gripper

    Kontak yang tidak diizinkan:
    - objek dengan rak
    - objek dengan distractor
    - objek dengan lantai
    - objek dengan base kursi roda
    """
    contacts = p.getContactPoints(bodyA=object_id)

    for contact in contacts:
        body_a = contact[1]
        body_b = contact[2]

        if body_a == object_id:
            other_body = body_b
        else:
            other_body = body_a

        # Kontak dengan robot masih dianggap wajar
        if other_body == robot_id:
            continue

        # Selain robot, dianggap collision tidak diizinkan
        return True, other_body

    return False, None


def edge_aware_y(obj_y, stage="ready"):
    """
    Edge-aware Y berbasis posisi robot saat ini.
    Penting untuk multi-rack karena robot/wheelchair bisa pindah station.
    """
    if not EDGE_REACH_TUNING_ENABLED:
        return obj_y

    base_y = float(ROBOT_BASE_POS[1])
    y_local = float(obj_y) - base_y

    if abs(y_local) <= EDGE_Y_THRESHOLD:
        return obj_y

    sign = 1.0 if y_local > 0 else -1.0

    if stage == "ready":
        limit = EDGE_READY_Y_LIMIT
    elif stage == "pre_grasp":
        limit = EDGE_PREGRASP_Y_LIMIT
    elif stage == "approach":
        limit = EDGE_APPROACH_Y_LIMIT
    elif stage == "retreat":
        limit = EDGE_RETREAT_Y_LIMIT
    elif stage == "lift":
        limit = EDGE_LIFT_Y_LIMIT
    else:
        limit = EDGE_PREGRASP_Y_LIMIT

    y_limited_local = sign * min(abs(y_local), abs(limit))
    return base_y + y_limited_local


def get_shelf_ready_position(object_pos):
    """
    Posisi transisi sebelum masuk rak.

    Edge tuning:
    Untuk objek di ujung kiri/kanan rak, y ready tidak dipaksa sama dengan y objek.
    Ready hanya titik aman, bukan titik grasp. Ini mencegah IK batal terlalu awal.
    """
    obj_x, obj_y, obj_z = object_pos

    shelf_index = min(
        range(len(SHELF_Z_LEVELS)),
        key=lambda i: abs(obj_z - SHELF_Z_LEVELS[i])
    )

    ready_y = edge_aware_y(obj_y, stage="ready")

    if shelf_index == 0:
        # Rak bawah: jangan terlalu memaksa y ekstrem pada ready.
        return [ROBOT_BASE_POS[0] + 0.36, ready_y, ROBOT_BASE_POS[2] + 0.23]
    elif shelf_index == 1:
        # Rak tengah: ready lebih dekat ke tengah agar reachable.
        return [ROBOT_BASE_POS[0] + 0.30, ready_y, ROBOT_BASE_POS[2] + 0.35]
    else:
        # Rak atas tetap tinggi, tapi y juga dibuat edge-aware.
        return [ROBOT_BASE_POS[0] + 0.24, ready_y, ROBOT_BASE_POS[2] + 0.62]


def get_shelf_aware_motion_points(object_pos):
    """
    Titik gerak berbeda untuk rak bawah, tengah, dan atas.

    Revisi edge-aware:
    - Untuk objek di ujung kiri/kanan rak, y target dibuat bertahap.
    - ready paling konservatif.
    - pre_grasp mendekat.
    - approach paling dekat ke y objek.
    - retreat/lift kembali sedikit ke tengah agar objek keluar rak lebih aman.
    """
    obj_x, obj_y, obj_z = object_pos

    shelf_index = min(
        range(len(SHELF_Z_LEVELS)),
        key=lambda i: abs(obj_z - SHELF_Z_LEVELS[i])
    )

    y_ready = edge_aware_y(obj_y, stage="ready")
    y_pre = edge_aware_y(obj_y, stage="pre_grasp")
    y_app = edge_aware_y(obj_y, stage="approach")
    y_ret = edge_aware_y(obj_y, stage="retreat")
    y_lift = edge_aware_y(obj_y, stage="lift")

    if shelf_index == 0:
        # Rak bawah:
        # y dibuat bertahap agar tidak gagal pada objek paling ujung.
        pre_grasp = [obj_x - 0.19, y_pre, obj_z + 0.165]
        approach  = [obj_x - 0.070, y_app, obj_z + 0.100]
        retreat   = [obj_x - 0.27, y_ret, obj_z + 0.215]
        lift      = [obj_x - 0.27, y_lift, obj_z + 0.365]

    elif shelf_index == 1:
        # Rak tengah:
        # problem utama user ada pada objek ujung kiri.
        # y pre/approach dibuat tidak langsung ekstrem.
        pre_grasp = [obj_x - 0.19, y_pre, obj_z + 0.145]
        approach  = [obj_x - 0.080, y_app, obj_z + 0.060]
        retreat   = [obj_x - 0.26, y_ret, obj_z + 0.150]
        lift      = [obj_x - 0.26, y_lift, obj_z + 0.320]

    else:
        # Rak atas relatif lebih reachable, tetap pakai edge-aware agar konsisten.
        pre_grasp = [obj_x - 0.23, y_pre, obj_z + 0.105]
        approach  = [obj_x - 0.100, y_app, obj_z + 0.055]
        retreat   = [obj_x - 0.30, y_ret, obj_z + 0.135]
        lift      = [obj_x - 0.30, y_lift, obj_z + 0.245]

    return {
        "shelf_index": shelf_index,
        "ready": get_shelf_ready_position(object_pos),
        "pre_grasp": pre_grasp,
        "approach": approach,
        "retreat": retreat,
        "lift": lift
    }


def move_robot_to_selected_object(robot_data, objects, selected_name):
    """
    Tes reach ke objek aktif.
    Robot selalu mulai dari home pose agar tidak stuck.
    """
    if selected_name not in objects:
        print(f"Objek {selected_name} tidak ditemukan.")
        return

    object_id = objects[selected_name]["id"]

    if PURE_CV_MODE and "visual_pick_position" in objects[selected_name]:
        object_pos = list(objects[selected_name]["visual_pick_position"])
        _, object_orn = p.getBasePositionAndOrientation(object_id)

        print("\n=== PURE CV PICK TARGET ===")
        print("Selected body       :", selected_name)
        print("Pick position source: depth camera / visual contour")
        print("Visual pick position:", [round(v, 5) for v in object_pos])
        print("Metadata position   : NOT USED as pick target")
    else:
        object_pos, object_orn = p.getBasePositionAndOrientation(object_id)
        print("\nWARNING: fallback memakai PyBullet base position karena visual_pick_position belum tersedia.")

    points = get_shelf_aware_motion_points(object_pos)

    print(f"\nRobot reach ke objek: {selected_name}")
    print("Object position:", object_pos)
    print("Shelf index:", points["shelf_index"])
    print("Pre-grasp position:", points["pre_grasp"])

    print("Auto-home dimatikan. Tekan m untuk kembali ke home pose.")

    # Pre-align: hanya joint-1 yang mutar ke arah objek.
    rotate_joint1_to_face_object(robot_data, object_pos, duration=JOINT1_ALIGN_DURATION)

    print("Move to shelf-ready position:", points["ready"])
    ready_error, _, _ = move_robot_to_position(
        robot_data=robot_data,
        target_position=points["ready"],
        duration=1.8
    )

    error, _, _ = move_robot_to_position(
        robot_data=robot_data,
        target_position=points["pre_grasp"],
        duration=2.5
    )

    if error > 0.20:
        print("Reach gagal. Target ini belum reachable untuk konfigurasi JACO 6-DoF sekarang.")
    else:
        print("Reach berhasil / cukup dekat.")

def get_grasp_slot_world_pose(robot_data, object_id=None):
    """
    Hitung posisi slot genggam di world frame.

    Slot ini adalah titik visual di dekat sela gripper/palm.
    Bukan di udara jauh dari finger.
    """
    robot_id = robot_data["robot_id"]
    ee_link = robot_data["end_effector_link_index"]

    ee_state = p.getLinkState(robot_id, ee_link, computeForwardKinematics=True)
    ee_pos = ee_state[4]
    ee_orn = ee_state[5]

    slot_pos, slot_orn = p.multiplyTransforms(
        ee_pos,
        ee_orn,
        GRASP_LOCAL_OFFSET,
        [0, 0, 0, 1]
    )

    return list(slot_pos), list(slot_orn), list(ee_pos), list(ee_orn)


def get_relative_pose_from_ee_to_object(robot_data, object_id):
    """
    Ambil transform relatif object terhadap EE.
    Ini dipakai agar orientasi objek tidak mendadak berubah saat constraint dibuat.
    """
    robot_id = robot_data["robot_id"]
    ee_link = robot_data["end_effector_link_index"]

    ee_state = p.getLinkState(robot_id, ee_link, computeForwardKinematics=True)
    ee_pos = ee_state[4]
    ee_orn = ee_state[5]

    obj_pos, obj_orn = p.getBasePositionAndOrientation(object_id)

    inv_ee_pos, inv_ee_orn = p.invertTransform(ee_pos, ee_orn)
    rel_pos, rel_orn = p.multiplyTransforms(
        inv_ee_pos,
        inv_ee_orn,
        obj_pos,
        obj_orn
    )

    return list(rel_pos), list(rel_orn)


def seat_object_into_gripper_slot(robot_data, object_id, duration=GRASP_SEAT_DURATION):
    """
    Visual seating sebelum constraint:
    objek digeser halus dari posisi sekarang ke slot genggam.
    Ini mengurangi kesan objek melayang jauh dan membuat attach terlihat lebih natural.

    Catatan:
    masih virtual grasp, bukan simulasi force-closure jari penuh.
    Tetapi tampilan visual jauh lebih dekat ke objek benar-benar digenggam.
    """
    if not GRASP_SEAT_OBJECT_IN_GRIPPER:
        return

    slot_pos, slot_orn, ee_pos, ee_orn = get_grasp_slot_world_pose(robot_data, object_id)

    start_pos, start_orn = p.getBasePositionAndOrientation(object_id)
    start = np.array(start_pos, dtype=float)
    target = np.array(slot_pos, dtype=float)

    dist = float(np.linalg.norm(target - start))
    print("\n=== NATURAL GRASP SEAT OBJECT INTO FINGERS ===")
    print("Object start:", [round(v, 5) for v in start.tolist()])
    print("Grasp slot target:", [round(v, 5) for v in target.tolist()])
    print("Seat distance:", round(dist, 5))
    print("GRASP_LOCAL_OFFSET:", GRASP_LOCAL_OFFSET)

    shelf_index = min(
        range(len(SHELF_Z_LEVELS)),
        key=lambda i: abs(start[2] - SHELF_Z_LEVELS[i])
    )
    max_seat_distance = (
        BOTTOM_GRASP_SEAT_MAX_DISTANCE if shelf_index == 0 else GRASP_SEAT_MAX_DISTANCE
    )

    print("Shelf index for seating:", shelf_index)
    print("Max seat distance allowed:", max_seat_distance)

    if dist > max_seat_distance:
        print(
            "Seat skipped: object terlalu jauh dari slot jari. "
            "Robot harus align gripper slot dulu, bukan menarik object jauh."
        )
        return

    steps = max(1, int(duration * 240))

    for i in range(steps):
        alpha = (i + 1) / steps
        s = alpha * alpha * (3.0 - 2.0 * alpha)

        pos = (1.0 - s) * start + s * target
        pos[2] += GRASP_SEAT_LIFT_ARC * math.sin(math.pi * alpha)

        p.resetBasePositionAndOrientation(
            object_id,
            pos.tolist(),
            start_orn
        )
        p.resetBaseVelocity(object_id, [0, 0, 0], [0, 0, 0])

        p.stepSimulation()
        time.sleep(1 / 240)

    for _ in range(GRASP_SEAT_STEPS_AFTER):
        p.resetBaseVelocity(object_id, [0, 0, 0], [0, 0, 0])
        p.stepSimulation()
        time.sleep(1 / 240)

    final_pos, _ = p.getBasePositionAndOrientation(object_id)
    print("Object after seating:", [round(v, 5) for v in final_pos])


def align_gripper_slot_to_object_before_attach(robot_data, object_id):
    """
    FIX3:
    Sebelum attach, yang digerakkan adalah robot/gripper slot mendekati objek.
    Bukan objek yang ditarik jauh ke gripper.

    Ini membuat grasp lebih realistis:
    - gripper mendekat ke object
    - object hanya di-seat kecil jika sudah dekat
    - constraint tidak membuat object melayang dari jari
    """
    if not GRASP_SLOT_ALIGN_BEFORE_ATTACH:
        return

    for iteration in range(GRASP_SLOT_ALIGN_MAX_ITERS):
        slot_pos, _, ee_pos, _ = get_grasp_slot_world_pose(robot_data, object_id)
        obj_pos, _ = p.getBasePositionAndOrientation(object_id)

        slot_np = np.array(slot_pos, dtype=float)
        obj_np = np.array(obj_pos, dtype=float)

        correction = obj_np - slot_np
        dist = float(np.linalg.norm(correction))

        print(f"\n=== GRASP SLOT ALIGN ITER {iteration + 1}/{GRASP_SLOT_ALIGN_MAX_ITERS} ===")
        print("slot_pos:", [round(v, 5) for v in slot_pos])
        print("object_pos:", [round(v, 5) for v in obj_pos])
        print("slot_to_object_dist:", round(dist, 5))

        if dist <= GRASP_SLOT_ALIGN_TOLERANCE:
            print("Gripper slot sudah cukup dekat dengan objek.")
            return

        # Limit correction agar robot tidak melakukan gerakan besar mendadak di rak.
        max_corr = 0.085
        corr_norm = float(np.linalg.norm(correction))
        if corr_norm > max_corr:
            correction = correction / corr_norm * max_corr

        ee_target = (np.array(ee_pos, dtype=float) + correction).tolist()

        print("EE target correction:", [round(v, 5) for v in correction.tolist()])
        print("EE target for slot align:", [round(v, 5) for v in ee_target])

        move_robot_to_position(
            robot_data=robot_data,
            target_position=ee_target,
            duration=GRASP_SLOT_ALIGN_DURATION,
            motor_force=190,
            max_velocity=0.16,
            error_print_limit=0.25
        )

    slot_pos, _, _, _ = get_grasp_slot_world_pose(robot_data, object_id)
    obj_pos, _ = p.getBasePositionAndOrientation(object_id)
    final_dist = float(np.linalg.norm(np.array(slot_pos) - np.array(obj_pos)))
    print("Final slot-object distance after align:", round(final_dist, 5))


def set_active_grasp_state(constraint_id=None, object_id=None):
    """
    Simpan constraint/object yang sedang digenggam agar bisa dilepas manual.
    """
    global ACTIVE_GRASP_CONSTRAINT_ID, ACTIVE_GRASP_OBJECT_ID
    ACTIVE_GRASP_CONSTRAINT_ID = constraint_id
    ACTIVE_GRASP_OBJECT_ID = object_id


def clear_active_grasp_state(constraint_id=None):
    """
    Hapus state grasp aktif.
    Jika constraint_id diberikan, hanya clear jika cocok.
    """
    global ACTIVE_GRASP_CONSTRAINT_ID, ACTIVE_GRASP_OBJECT_ID

    if constraint_id is None or ACTIVE_GRASP_CONSTRAINT_ID == constraint_id:
        ACTIVE_GRASP_CONSTRAINT_ID = None
        ACTIVE_GRASP_OBJECT_ID = None


def manual_release_active_grasp(robot_data=None, reason="manual key"):
    """
    Melepas object yang masih dalam grip robot.

    Dipanggil dari:
    - tombol keyboard 'o'
    - terminal command: lepas / release / drop

    Catatan:
    object akan jatuh sesuai posisi saat tombol ditekan.
    Jadi pakai sebagai manual/emergency release.
    """
    global ACTIVE_GRASP_CONSTRAINT_ID, ACTIVE_GRASP_OBJECT_ID

    if not MANUAL_RELEASE_ENABLED:
        print("Manual release disabled.")
        return False

    if ACTIVE_GRASP_CONSTRAINT_ID is None:
        print("Manual release: tidak ada object yang sedang attached di gripper.")
        return False

    print("\n=== MANUAL RELEASE ACTIVE GRASP ===")
    print("Reason:", reason)
    print("Constraint ID:", ACTIVE_GRASP_CONSTRAINT_ID)
    print("Object ID:", ACTIVE_GRASP_OBJECT_ID)

    if ACTIVE_GRASP_OBJECT_ID is not None:
        try:
            if "enable_object_collision_for_natural_drop" in globals():
                enable_object_collision_for_natural_drop(ACTIVE_GRASP_OBJECT_ID)
            else:
                set_object_transport_mode(ACTIVE_GRASP_OBJECT_ID, enabled=False)
                p.resetBaseVelocity(ACTIVE_GRASP_OBJECT_ID, [0, 0, 0], [0, 0, 0])
        except Exception as exc:
            print("WARNING: gagal aktifkan object natural/drop mode:", exc)

    release_object(ACTIVE_GRASP_CONSTRAINT_ID)

    if robot_data is not None:
        try:
            set_gripper_open(robot_data)
        except Exception as exc:
            print("WARNING: gagal open gripper setelah manual release:", exc)

    for _ in range(MANUAL_RELEASE_SETTLE_STEPS):
        p.stepSimulation()
        time.sleep(1 / 240)

    print("Manual release selesai.")
    return True


def manual_release_key_pressed():
    """
    Cek tombol release manual dari OpenCV/PyBullet.
    Dipakai saat robot sedang bergerak/menunggu, bukan hanya di main loop.
    """
    try:
        key = cv2.waitKey(1) & 0xFF
        if key == MANUAL_RELEASE_KEY:
            return True
    except Exception:
        pass

    try:
        events = p.getKeyboardEvents()
        state = events.get(MANUAL_RELEASE_KEY, 0)
        if state & p.KEY_WAS_TRIGGERED:
            return True
    except Exception:
        pass

    return False


def attach_object_to_end_effector(robot_data, object_id):
    """
    Virtual grasp versi stabil:
    - tidak memakai relative pose liar dari posisi objek saat approach.
    - objek dikunci pada fixed slot GRASP_LOCAL_OFFSET di dekat palm/gripper.
    - collision objek dimatikan sementara saat transport agar tidak terpental.
    """
    robot_id = robot_data["robot_id"]
    ee_link = robot_data["end_effector_link_index"]

    # FIX3: Align slot gripper ke object terlebih dahulu.
    # Ini mengurangi kebutuhan "menarik" object ke gripper.
    align_gripper_slot_to_object_before_attach(robot_data, object_id)

    ee_state = p.getLinkState(robot_id, ee_link)
    ee_pos = ee_state[4]
    ee_orn = ee_state[5]

    obj_pos, obj_orn = p.getBasePositionAndOrientation(object_id)

    # Grasp distance sekarang dinilai dari slot jari ke object, bukan EE origin ke object.
    slot_pos, _, _, _ = get_grasp_slot_world_pose(robot_data, object_id)
    distance = np.linalg.norm(np.array(slot_pos) - np.array(obj_pos))

    shelf_index = min(
        range(len(SHELF_Z_LEVELS)),
        key=lambda i: abs(obj_pos[2] - SHELF_Z_LEVELS[i])
    )

    max_grasp_distance = BOTTOM_GRASP_DISTANCE if shelf_index == 0 else MAX_GRASP_DISTANCE

    print("Jarak grasp-slot ke objek:", distance)
    print("Shelf index object:", shelf_index)
    print("Maksimal jarak grasp tahap ini:", max_grasp_distance)

    if distance > max_grasp_distance:
        print("Grasp gagal: end-effector terlalu jauh dari objek.")
        return None

    # Tutup gripper visual dulu.
    set_gripper_close(robot_data)

    # Khusus rak bawah, koreksi kecil supaya objek berada dekat palm.
    # Koreksi dibuat lebih halus agar tidak terlihat melayang jauh.
    if (
        shelf_index == 0
        and BOTTOM_GRASP_ASSIST_ENABLED
        and distance > MAX_GRASP_DISTANCE
    ):
        ee_state = p.getLinkState(robot_id, ee_link)
        ee_pos = ee_state[4]
        ee_orn = ee_state[5]

        assist_pos = [
            ee_pos[0] + BOTTOM_GRASP_ASSIST_OFFSET[0],
            ee_pos[1] + BOTTOM_GRASP_ASSIST_OFFSET[1],
            ee_pos[2] + BOTTOM_GRASP_ASSIST_OFFSET[2]
        ]

        min_z = SHELF_Z_LEVELS[0] + 0.070
        assist_pos[2] = max(assist_pos[2], min_z)

        print("Bottom grasp assist halus aktif.")
        print("Objek dikoreksi dekat gripper:", assist_pos)

        p.resetBasePositionAndOrientation(object_id, assist_pos, obj_orn)
        p.resetBaseVelocity(object_id, [0, 0, 0], [0, 0, 0])

        for _ in range(20):
            p.stepSimulation()
            time.sleep(1 / 240)

    # Natural grasp visual setelah bottom-assist:
    # objek baru dirapikan ke slot genggam dekat finger/palm sebelum fixed constraint dibuat.
    if NATURAL_GRASP_ENABLED:
        seat_object_into_gripper_slot(
            robot_data=robot_data,
            object_id=object_id,
            duration=GRASP_SEAT_DURATION
        )

    # Mode transport ON sebelum constraint agar objek tidak mental kena rak/handrest.
    set_object_transport_mode(object_id, enabled=True)

    # Fixed grasp natural FIX2:
    # Ambil transform relatif aktual object terhadap EE SETELAH seating.
    # Ini mencegah object snap/melayang karena constraint dipaksa ke offset yang beda.
    rel_pos, rel_orn = get_relative_pose_from_ee_to_object(
        robot_data=robot_data,
        object_id=object_id
    )

    slot_error = float(np.linalg.norm(np.array(rel_pos) - np.array(GRASP_LOCAL_OFFSET)))
    print("Relative object pose for grasp constraint:")
    print("  rel_pos actual:", [round(v, 5) for v in rel_pos])
    print("  rel_orn actual:", [round(v, 5) for v in rel_orn])
    print("  target GRASP_LOCAL_OFFSET:", GRASP_LOCAL_OFFSET)
    print("  slot_error:", round(slot_error, 5))

    if slot_error > GRASP_VERIFY_MAX_DISTANCE_FROM_SLOT:
        print("WARNING: object masih agak jauh dari slot jari, tetapi constraint memakai rel_pos aktual agar tidak snap.")

    parent_frame_pos = rel_pos if GRASP_USE_EXACT_RELATIVE_CONSTRAINT else GRASP_LOCAL_OFFSET

    constraint_id = p.createConstraint(
        parentBodyUniqueId=robot_id,
        parentLinkIndex=ee_link,
        childBodyUniqueId=object_id,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=[0, 0, 0],
        parentFramePosition=parent_frame_pos,
        childFramePosition=[0, 0, 0],
        parentFrameOrientation=rel_orn,
        childFrameOrientation=[0, 0, 0, 1]
    )

    p.changeConstraint(
        constraint_id,
        maxForce=GRASP_MAX_FORCE
    )

    # Settle sebentar supaya objek masuk ke slot genggaman.
    for _ in range(35):
        p.stepSimulation()
        time.sleep(1 / 240)

    obj_pos_after, _ = p.getBasePositionAndOrientation(object_id)
    print("Virtual grasp fixed-slot dibuat. Constraint ID:", constraint_id)
    print("GRASP_LOCAL_OFFSET:", GRASP_LOCAL_OFFSET)
    print("Object position after fixed-slot grasp:", obj_pos_after)
    print("Grasp max force:", GRASP_MAX_FORCE)

    set_active_grasp_state(
        constraint_id=constraint_id,
        object_id=object_id
    )
    print("Active grasp state registered. Press 'o' to manual release.")

    return constraint_id


def release_object(constraint_id):
    """
    Melepas objek dari end-effector.
    Aman dipanggil beberapa kali; jika constraint sudah hilang tidak membuat simulasi crash.
    """
    if constraint_id is not None:
        try:
            p.removeConstraint(constraint_id)
            print("Objek dilepas dari gripper.")
        except Exception as exc:
            print("WARNING: release_object gagal / constraint mungkin sudah hilang:", exc)
        finally:
            clear_active_grasp_state(constraint_id)


def get_handover_object_target_position(object_id):
    """
    Menghitung posisi pusat objek di area handover.
    Handover marker sekarang non-solid/tembus.
    Target ini hanya titik bantuan final-align sebelum objek dijatuhkan ke basket.
    Fungsi ini TIDAK memindahkan objek.
    """
    aabb_min, aabb_max = p.getAABB(object_id)
    half_height = max(0.035, (aabb_max[2] - aabb_min[2]) / 2.0)
    support_top_z = HANDOVER_POS[2] - 0.010 + 0.010 / 2.0

    return [
        HANDOVER_POS[0],
        HANDOVER_POS[1],
        support_top_z + half_height + 0.012
    ]


def get_basket_natural_drop_object_target_position(object_id):
    """
    Posisi pusat objek sebelum release natural.
    Target berada tepat di atas basket, bukan di marker handover.
    Setelah release, objek dibiarkan jatuh karena gravity.
    """
    if BASKET_INFO is None:
        # fallback ke area handover lama jika basket belum dibuat
        return get_handover_object_target_position(object_id)

    aabb_min, aabb_max = p.getAABB(object_id)
    half_height = max(0.035, (aabb_max[2] - aabb_min[2]) / 2.0)

    return [
        BASKET_CENTER[0],
        BASKET_CENTER[1],
        BASKET_INFO["floor_top_z"] + half_height + BASKET_DROP_HEIGHT
    ]


def move_carried_object_over_basket_for_gravity_drop(
    robot_data,
    object_id,
    grasp_constraint,
    duration=1.25
):
    """
    Bawa objek yang masih attached ke atas basket.
    Tidak memindahkan objek langsung.
    Yang digerakkan adalah EE agar pusat objek berada di atas basket.
    """
    ee_pos, _ = get_end_effector_position(robot_data)
    obj_pos, _ = p.getBasePositionAndOrientation(object_id)
    target_obj_pos = get_basket_natural_drop_object_target_position(object_id)

    correction = np.array(target_obj_pos) - np.array(obj_pos)
    ee_target = (np.array(ee_pos) + correction).tolist()

    print("\n=== MOVE CARRIED OBJECT OVER BASKET FOR GRAVITY DROP ===")
    print("Object current position:", [round(v, 5) for v in obj_pos])
    print("Object target above basket:", [round(v, 5) for v in target_obj_pos])
    print("EE correction:", [round(v, 5) for v in correction.tolist()])
    print("EE target:", [round(v, 5) for v in ee_target])

    error, grasp_constraint, dropped = move_robot_to_position(
        robot_data=robot_data,
        target_position=ee_target,
        duration=duration,
        attached_object_id=object_id,
        grasp_constraint_id=grasp_constraint,
        release_on_collision=False,
        monitor_start_delay_steps=50,
        motor_force=150,
        max_velocity=0.12,
        error_print_limit=HANDOVER_ERROR_LIMIT
    )

    obj_pos_after, _ = p.getBasePositionAndOrientation(object_id)
    dist_after = np.linalg.norm(np.array(obj_pos_after) - np.array(target_obj_pos))
    print("Object position above basket after move:", [round(v, 5) for v in obj_pos_after])
    print("Distance object to basket drop target:", round(float(dist_after), 5))

    return grasp_constraint, dropped


def move_gripper_slot_over_basket_for_vertical_drop(
    robot_data,
    object_id,
    grasp_constraint,
    duration=1.15
):
    """
    FIX3:
    Yang disejajarkan ke atas basket adalah SLOT GRIPPER,
    bukan posisi object secara langsung.

    Kalau object sudah benar-benar berada di slot jari,
    maka saat gripper slot ada di atas basket, object juga berada di atas basket
    dan jatuhnya lebih vertikal.
    """
    if BASKET_INFO is None:
        print("Basket info belum ada, skip slot-over-basket align.")
        return grasp_constraint, False

    slot_pos, _, ee_pos, _ = get_grasp_slot_world_pose(robot_data, object_id)
    aabb_min, aabb_max = p.getAABB(object_id)
    half_height = max(0.035, (aabb_max[2] - aabb_min[2]) / 2.0)

    target_slot_pos = [
        BASKET_CENTER[0],
        BASKET_CENTER[1],
        BASKET_INFO["floor_top_z"] + half_height + BASKET_DROP_HEIGHT
    ]

    correction = np.array(target_slot_pos) - np.array(slot_pos)
    ee_target = (np.array(ee_pos) + correction).tolist()

    print("\n=== MOVE GRIPPER SLOT OVER BASKET FOR VERTICAL DROP ===")
    print("slot current:", [round(v, 5) for v in slot_pos])
    print("target slot above basket:", [round(v, 5) for v in target_slot_pos])
    print("EE correction:", [round(v, 5) for v in correction.tolist()])
    print("EE target:", [round(v, 5) for v in ee_target])

    error, grasp_constraint, dropped = move_robot_to_position(
        robot_data=robot_data,
        target_position=ee_target,
        duration=duration,
        attached_object_id=object_id,
        grasp_constraint_id=grasp_constraint,
        release_on_collision=False,
        monitor_start_delay_steps=50,
        motor_force=145,
        max_velocity=0.10,
        error_print_limit=HANDOVER_ERROR_LIMIT
    )

    slot_after, _, _, _ = get_grasp_slot_world_pose(robot_data, object_id)
    obj_after, _ = p.getBasePositionAndOrientation(object_id)
    print("slot after:", [round(v, 5) for v in slot_after])
    print("object after:", [round(v, 5) for v in obj_after])
    print("slot distance to target:", round(float(np.linalg.norm(np.array(slot_after) - np.array(target_slot_pos))), 5))

    return grasp_constraint, dropped


def get_object_half_height(object_id):
    aabb_min, aabb_max = p.getAABB(object_id)
    return max(0.030, (aabb_max[2] - aabb_min[2]) / 2.0)


def get_basket_release_target_for_object(object_id):
    """
    Target center object ketika masih attached sebelum release.
    Center object harus berada di atas center basket.
    """
    if BASKET_INFO is None:
        return get_basket_natural_drop_object_target_position(object_id)

    half_height = get_object_half_height(object_id)

    return [
        BASKET_CENTER[0],
        BASKET_CENTER[1],
        BASKET_INFO["floor_top_z"] + half_height + BASKET_DROP_HEIGHT
    ]


def get_basket_inner_half_size_for_release():
    """
    Half-size area aman di dalam basket.
    Dipakai sebagai guard supaya object tidak dilepas di luar basket.
    """
    inner_x = max(0.05, BASKET_OUTER_SIZE[0] - 2.0 * BASKET_WALL_THICKNESS - 2.0 * BASKET_SUPPORT_MARGIN)
    inner_y = max(0.05, BASKET_OUTER_SIZE[1] - 2.0 * BASKET_WALL_THICKNESS - 2.0 * BASKET_SUPPORT_MARGIN)
    return [inner_x / 2.0, inner_y / 2.0]


def check_object_ready_to_release_into_basket(object_id, verbose=True):
    """
    Cek apakah object sudah aman dilepas.
    Jika belum ready, release tidak boleh dilakukan.
    """
    if BASKET_INFO is None:
        if verbose:
            print("Release guard: BASKET_INFO None, tidak bisa validasi basket.")
        return False, {"reason": "basket_info_none"}

    obj_pos, _ = p.getBasePositionAndOrientation(object_id)
    target = get_basket_release_target_for_object(object_id)

    dx = float(obj_pos[0] - BASKET_CENTER[0])
    dy = float(obj_pos[1] - BASKET_CENTER[1])
    dz = float(obj_pos[2] - target[2])
    xy_error = float(math.sqrt(dx * dx + dy * dy))
    z_error = abs(dz)

    half_x, half_y = get_basket_inner_half_size_for_release()

    inside_x = abs(dx) <= min(half_x, RELEASE_GUARD_XY_TOLERANCE)
    inside_y = abs(dy) <= min(half_y, RELEASE_GUARD_XY_TOLERANCE)
    z_ok = z_error <= RELEASE_GUARD_Z_TOLERANCE

    ready = inside_x and inside_y and z_ok

    info = {
        "ready": ready,
        "object_pos": obj_pos,
        "target": target,
        "dx": dx,
        "dy": dy,
        "dz": dz,
        "xy_error": xy_error,
        "z_error": z_error,
        "inside_x": inside_x,
        "inside_y": inside_y,
        "z_ok": z_ok,
        "inner_half_size": [half_x, half_y]
    }

    if verbose:
        print("\n=== RELEASE GUARD CHECK ===")
        print("object_pos:", [round(v, 5) for v in obj_pos])
        print("basket_center:", [round(v, 5) for v in BASKET_CENTER])
        print("release_target:", [round(v, 5) for v in target])
        print("dx/dy/dz:", round(dx, 5), round(dy, 5), round(dz, 5))
        print("xy_error:", round(xy_error, 5), "| z_error:", round(z_error, 5))
        print("inner_half_size:", [round(v, 5) for v in [half_x, half_y]])
        print("inside_x:", inside_x, "| inside_y:", inside_y, "| z_ok:", z_ok)
        print("release_ready:", ready)

    return ready, info


def align_attached_object_over_basket_release_guard(
    robot_data,
    object_id,
    grasp_constraint,
    max_iters=RELEASE_GUARD_MAX_ALIGN_ITERS
):
    """
    FIX5:
    Menahan object tetap attached, lalu menggerakkan EE sampai object center
    benar-benar berada di atas basket. Release hanya boleh setelah guard ready.

    Ini bukan teleport object.
    Yang dipindah adalah robot/EE, sementara object tetap dipegang constraint.
    """
    if not RELEASE_GUARD_ENABLED:
        return grasp_constraint, False, True

    print("\n=== RELEASE GUARD ALIGN ATTACHED OBJECT OVER BASKET ===")
    print("Rule: object tidak dilepas sampai center object berada di atas basket.")

    for i in range(max_iters):
        ready, info = check_object_ready_to_release_into_basket(
            object_id,
            verbose=RELEASE_GUARD_PRINT_EVERY_CHECK
        )

        if ready:
            print("Release guard PASSED before align iteration:", i)
            return grasp_constraint, False, True

        obj_pos = np.array(info["object_pos"], dtype=float)
        target = np.array(info["target"], dtype=float)
        correction = target - obj_pos

        # Limit correction supaya gerakan robot tidak mendadak/keras.
        max_corr = 0.115
        corr_norm = float(np.linalg.norm(correction))
        if corr_norm > max_corr:
            correction = correction / corr_norm * max_corr

        ee_pos, _ = get_end_effector_position(robot_data)
        ee_target = (np.array(ee_pos, dtype=float) + correction).tolist()

        print(f"\nRelease guard align iter {i + 1}/{max_iters}")
        print("correction applied:", [round(v, 5) for v in correction.tolist()])
        print("EE target:", [round(v, 5) for v in ee_target])

        error, grasp_constraint, dropped = move_robot_to_position(
            robot_data=robot_data,
            target_position=ee_target,
            duration=RELEASE_GUARD_ALIGN_DURATION,
            attached_object_id=object_id,
            grasp_constraint_id=grasp_constraint,
            release_on_collision=False,
            monitor_start_delay_steps=40,
            motor_force=RELEASE_GUARD_ALIGN_FORCE,
            max_velocity=RELEASE_GUARD_ALIGN_MAX_VELOCITY,
            error_print_limit=HANDOVER_ERROR_LIMIT
        )

        if dropped:
            print("Release guard align gagal: object dropped saat masih menuju basket.")
            return grasp_constraint, True, False

        # settle kecil supaya posisi attached object stabil sebelum check ulang.
        for _ in range(10):
            if manual_release_key_pressed():
                print("Manual release key detected during release-guard settle.")
                manual_release_active_grasp(
                    robot_data=robot_data,
                    reason="key 'o' during release-guard settle"
                )
                return None, True, False

            p.resetBaseVelocity(object_id, [0, 0, 0], [0, 0, 0])
            p.stepSimulation()
            time.sleep(1 / 240)

    ready, info = check_object_ready_to_release_into_basket(
        object_id,
        verbose=True
    )

    if ready:
        print("Release guard PASSED after max align.")
        return grasp_constraint, False, True

    print("\nRELEASE GUARD FAILED: object belum aman dilepas ke basket.")
    print("Release dibatalkan agar object tidak jatuh sebelum masuk basket.")
    return grasp_constraint, False, False


def enable_object_collision_for_natural_drop(object_id):
    """
    Aktifkan collision dan dynamics natural sebelum object dilepas.
    Dibuat terpisah dari set_object_transport_mode(False) karena fungsi lama memberi damping besar.
    """
    p.setCollisionFilterGroupMask(object_id, -1, 1, 1)
    p.changeDynamics(
        object_id,
        -1,
        lateralFriction=1.05,
        spinningFriction=0.02,
        rollingFriction=0.01,
        restitution=0.02,
        linearDamping=0.03,
        angularDamping=0.03
    )
    p.resetBaseVelocity(object_id, [0, 0, 0], [0, 0, 0])
    print("Object collision/dynamics: NATURAL DROP mode ON.")


def natural_gravity_drop_to_basket(object_id, steps=BASKET_DROP_SETTLE_STEPS):
    """
    Setelah constraint dilepas:
    - tidak ada resetBasePositionAndOrientation
    - tidak ada interpolasi diagonal
    - objek jatuh karena gravity ke basket
    """
    print("\n=== NATURAL GRAVITY DROP TO BASKET ===")
    print("No smooth_place_object_to_handover.")
    print("No resetBasePositionAndOrientation during drop.")
    print("Let PyBullet gravity handle the fall.")

    # Hilangkan velocity horizontal agar jatuhnya tidak diagonal akibat sisa gerak robot.
    if BASKET_DROP_MAX_HORIZONTAL_SPEED == 0.0:
        lin_vel, ang_vel = p.getBaseVelocity(object_id)
        p.resetBaseVelocity(
            object_id,
            [0.0, 0.0, min(0.0, lin_vel[2])],
            [0.0, 0.0, 0.0]
        )

    for i in range(steps):
        p.stepSimulation()
        time.sleep(1 / 240)

    final_pos, _ = p.getBasePositionAndOrientation(object_id)
    final_vel, _ = p.getBaseVelocity(object_id)
    print("Final object position after gravity drop:", [round(v, 5) for v in final_pos])
    print("Final object linear velocity:", [round(v, 5) for v in final_vel])


def move_carried_object_to_handover_target(
    robot_data,
    object_id,
    grasp_constraint,
    duration=2.0
):
    """
    Handover tanpa teleport:
    posisi objek diarahkan ke handover target saat objek masih attached pada gripper.
    """
    ee_pos, _ = get_end_effector_position(robot_data)
    obj_pos, _ = p.getBasePositionAndOrientation(object_id)
    target_obj_pos = get_handover_object_target_position(object_id)

    correction = np.array(target_obj_pos) - np.array(obj_pos)
    ee_target = (np.array(ee_pos) + correction).tolist()

    print("Step 6e: Align carried object to handover target tanpa teleport")
    print("Object current position:", obj_pos)
    print("Object target position:", target_obj_pos)
    print("EE correction:", correction.tolist())
    print("EE target for object alignment:", ee_target)

    error, grasp_constraint, dropped = move_robot_to_position(
        robot_data=robot_data,
        target_position=ee_target,
        duration=duration,
        attached_object_id=object_id,
        grasp_constraint_id=grasp_constraint,
        release_on_collision=False,
        monitor_start_delay_steps=80,
        motor_force=150,
        max_velocity=0.10,
        error_print_limit=HANDOVER_ERROR_LIMIT
    )

    obj_pos_after, _ = p.getBasePositionAndOrientation(object_id)
    dist_after = np.linalg.norm(np.array(obj_pos_after) - np.array(target_obj_pos))
    print("Distance object to handover target after align:", dist_after)

    if (not dropped) and dist_after > 0.08:
        ee_pos2, _ = get_end_effector_position(robot_data)
        correction2 = np.array(target_obj_pos) - np.array(obj_pos_after)
        ee_target2 = (np.array(ee_pos2) + correction2).tolist()

        print("Step 6f: Fine align carried object")
        print("Fine EE correction:", correction2.tolist())

        error, grasp_constraint, dropped = move_robot_to_position(
            robot_data=robot_data,
            target_position=ee_target2,
            duration=1.2,
            attached_object_id=object_id,
            grasp_constraint_id=grasp_constraint,
            release_on_collision=False,
            monitor_start_delay_steps=40,
            motor_force=130,
            max_velocity=0.08,
            error_print_limit=HANDOVER_ERROR_LIMIT
        )

    return grasp_constraint, dropped




def delay_after_robot_stop_before_release(robot_data, object_id=None, seconds=RELEASE_DELAY_AFTER_STOP_SEC):
    """
    Extra delay sebelum release dimatikan.
    Fungsi dipertahankan agar struktur code lama tetap kompatibel,
    tapi tidak print dan tidak menunggu 1.5 detik lagi.
    """
    if not RELEASE_EXTRA_DELAY_ENABLED or seconds <= 0:
        return

    robot_id = robot_data["robot_id"]
    arm_joints = robot_data["arm_joints"]
    hold_positions = {
        joint_index: p.getJointState(robot_id, joint_index)[0]
        for joint_index in arm_joints
    }

    steps = max(1, int(seconds * 240))
    for _ in range(steps):
        for joint_index, target in hold_positions.items():
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target,
                force=ROBOT_STOP_HOLD_FORCE,
                maxVelocity=ROBOT_STOP_HOLD_MAX_VELOCITY
            )
        if object_id is not None:
            p.resetBaseVelocity(object_id, [0, 0, 0], [0, 0, 0])
        p.stepSimulation()
        time.sleep(1 / 240)



def wait_until_robot_stops_before_release(robot_data, object_id=None):
    """
    Menunggu sampai robot benar-benar berhenti sebelum objek dilepas.

    Revisi:
    - setelah sampai titik handover, joint target dikunci pada posisi terakhir.
    - maxVelocity hold dibuat kecil agar joint velocity turun.
    - extra delay 1.5 detik dihapus; stabilisasi hanya lewat fungsi wait ini.
    - jumlah stable step diperbanyak dan semua parameter bisa dituning di konstanta atas.
    """
    if not WAIT_ROBOT_STOP_BEFORE_RELEASE:
        print("Wait robot stop sebelum release: disabled.")
        return True

    robot_id = robot_data["robot_id"]
    arm_joints = robot_data["arm_joints"]

    hold_positions = {
        joint_index: p.getJointState(robot_id, joint_index)[0]
        for joint_index in arm_joints
    }

    # Command motor untuk hold pose terakhir, supaya velocity benar-benar turun.
    for joint_index, target in hold_positions.items():
        p.setJointMotorControl2(
            bodyUniqueId=robot_id,
            jointIndex=joint_index,
            controlMode=p.POSITION_CONTROL,
            targetPosition=target,
            force=ROBOT_STOP_HOLD_FORCE,
            maxVelocity=ROBOT_STOP_HOLD_MAX_VELOCITY
        )

    stable_steps = 0
    last_ee_pos, _ = get_end_effector_position(robot_data)
    last_ee_pos = np.array(last_ee_pos, dtype=float)

    print("\n=== WAIT UNTIL ROBOT FULL STOP BEFORE RELEASE ===")
    print("Joint velocity threshold:", ROBOT_STOP_JOINT_VELOCITY_THRESHOLD)
    print("EE delta threshold:", ROBOT_STOP_EE_DELTA_THRESHOLD)
    print("Min stable steps:", ROBOT_STOP_MIN_STABLE_STEPS)
    print("Max wait steps:", ROBOT_STOP_MAX_WAIT_STEPS)
    print("Min total wait steps:", ROBOT_STOP_MIN_TOTAL_WAIT_STEPS)
    print("Hold force:", ROBOT_STOP_HOLD_FORCE)
    print("Hold max velocity:", ROBOT_STOP_HOLD_MAX_VELOCITY)

    max_joint_vel_seen = 0.0
    max_ee_delta_seen = 0.0

    for step in range(ROBOT_STOP_MAX_WAIT_STEPS):
        # Tahan joint pada posisi terakhir di setiap step.
        for joint_index, target in hold_positions.items():
            p.setJointMotorControl2(
                bodyUniqueId=robot_id,
                jointIndex=joint_index,
                controlMode=p.POSITION_CONTROL,
                targetPosition=target,
                force=ROBOT_STOP_HOLD_FORCE,
                maxVelocity=ROBOT_STOP_HOLD_MAX_VELOCITY
            )

        joint_velocities = [
            abs(p.getJointState(robot_id, joint_index)[1])
            for joint_index in arm_joints
        ]
        max_joint_vel = max(joint_velocities) if joint_velocities else 0.0

        ee_pos, _ = get_end_effector_position(robot_data)
        ee_pos_np = np.array(ee_pos, dtype=float)
        ee_delta = float(np.linalg.norm(ee_pos_np - last_ee_pos))
        last_ee_pos = ee_pos_np

        max_joint_vel_seen = max(max_joint_vel_seen, float(max_joint_vel))
        max_ee_delta_seen = max(max_ee_delta_seen, float(ee_delta))

        is_stable = (
            max_joint_vel <= ROBOT_STOP_JOINT_VELOCITY_THRESHOLD
            and ee_delta <= ROBOT_STOP_EE_DELTA_THRESHOLD
        )

        if is_stable:
            stable_steps += 1
        else:
            stable_steps = 0

        if step % ROBOT_STOP_PRINT_EVERY_STEPS == 0:
            print(
                "waiting stop | step:",
                step,
                "| max_joint_vel:",
                float(max_joint_vel),
                "| ee_delta:",
                ee_delta,
                "| stable_steps:",
                stable_steps
            )

        if object_id is not None:
            # Objek masih attached; velocity dinolkan agar sebelum release tidak ada energi liar.
            p.resetBaseVelocity(object_id, [0, 0, 0], [0, 0, 0])

        if manual_release_key_pressed():
            print("Manual release key detected during wait-stop.")
            manual_release_active_grasp(
                robot_data=robot_data,
                reason="key 'o' during wait-stop"
            )
            return False

        p.stepSimulation()
        time.sleep(1 / 240)

        if (
            stable_steps >= ROBOT_STOP_MIN_STABLE_STEPS
            and (step + 1) >= ROBOT_STOP_MIN_TOTAL_WAIT_STEPS
        ):
            print("Robot sudah stop/stabil sebelum release.")
            print("Wait steps:", step + 1)
            print("Final stable steps:", stable_steps)
            print("Max joint velocity seen:", max_joint_vel_seen)
            print("Max EE delta seen:", max_ee_delta_seen)
            return True

        if (
            stable_steps == ROBOT_STOP_MIN_STABLE_STEPS
            and (step + 1) < ROBOT_STOP_MIN_TOTAL_WAIT_STEPS
        ):
            print(
                "Robot sudah stabil, tapi release ditahan sampai min total wait:",
                ROBOT_STOP_MIN_TOTAL_WAIT_STEPS
            )

    print("WARNING: robot belum memenuhi kriteria stop sampai max wait.")
    print("Release tetap dilanjutkan setelah max wait agar simulasi tidak stuck.")
    print("Last stable steps:", stable_steps)
    print("Max joint velocity seen:", max_joint_vel_seen)
    print("Max EE delta seen:", max_ee_delta_seen)
    return False


def stabilize_object_after_release(object_id, steps=90):
    """
    Stabilize setelah release tanpa teleport.
    Karena handover marker dibuat tembus/non-solid, velocity object JANGAN di-reset terus.
    Biarkan gravity menjatuhkan objek ke basket.
    """
    for _ in range(steps):
        p.stepSimulation()
        time.sleep(1 / 240)


def smooth_place_object_to_handover(object_id, duration=1.6):
    """
    Final align ke handover tanpa teleport satu-frame.

    Kenapa perlu:
    IK JACO kadang tidak bisa menurunkan EE cukup rendah ke area handover.
    Kalau langsung resetBasePositionAndOrientation, objek terlihat teleport.
    Fungsi ini menggerakkan objek bertahap dari posisi terakhir ke atas support
    handover, sehingga terlihat seperti objek diserahkan/diletakkan secara halus.
    """
    start_pos, start_orn = p.getBasePositionAndOrientation(object_id)
    target_pos = get_handover_object_target_position(object_id)

    # Buat target sedikit mengikuti posisi akhir objek jika error X terlalu besar,
    # tetapi tetap menuju marker handover.
    start = np.array(start_pos, dtype=float)
    target = np.array(target_pos, dtype=float)

    steps = max(1, int(duration * 240))

    print("Step 7b: Smooth final align object to handover support")
    print("Smooth start object position:", start.tolist())
    print("Smooth target handover position:", target.tolist())

    for i in range(steps):
        alpha = (i + 1) / steps

        # Ease-in/ease-out agar tidak terlihat patah.
        s = alpha * alpha * (3.0 - 2.0 * alpha)

        # Jalur sedikit melengkung:
        # awalnya turun pelan, akhir benar-benar di atas support.
        pos = (1.0 - s) * start + s * target

        # Arc kecil agar tidak menembus support saat bergerak.
        arc = 0.035 * math.sin(math.pi * alpha)
        pos[2] += arc

        p.resetBasePositionAndOrientation(
            object_id,
            pos.tolist(),
            start_orn
        )
        p.resetBaseVelocity(object_id, [0, 0, 0], [0, 0, 0])

        p.stepSimulation()
        time.sleep(1 / 240)

    # Pastikan velocity nol di akhir, tetapi posisi bukan teleport karena sudah diinterpolasi.
    p.resetBaseVelocity(object_id, [0, 0, 0], [0, 0, 0])

    final_pos, _ = p.getBasePositionAndOrientation(object_id)
    dist = np.linalg.norm(np.array(final_pos) - np.array(target_pos))
    print("Smooth final object position:", final_pos)
    print("Distance object to handover after smooth align:", dist)




def pick_lift_handover_selected_object(robot_data, objects, selected_name):
    """
    Pick-lift-handover versi lebih aman untuk JACO 6-DoF.

    Urutan:
    1. home pose
    2. pre-grasp dari depan rak
    3. approach
    4. attach
    5. retreat keluar rak
    6. lift
    7. video-style handover ke area pengguna
    8. release
    9. kembali home pose
    """
    if selected_name not in objects:
        print(f"Objek {selected_name} tidak ditemukan.")
        return

    object_id = objects[selected_name]["id"]
    object_pos, object_orn = p.getBasePositionAndOrientation(object_id)

    points = get_shelf_aware_motion_points(object_pos)

    print("\n=== PICK-LIFT-HANDOVER TEST ===")
    print("Target:", selected_name)
    print("Object position:", object_pos)
    print("Shelf index:", points["shelf_index"])

    # Handover trajectory stabilized:
    # Jangan langsung memaksa EE ke marker handover kiri, karena dari video/log titik itu
    # sering tidak reachable. Jalur dibuat bertahap lewat area tengah yang lebih reachable.
    handover_bridge_position = [
        ROBOT_BASE_POS[0] + 0.30,
        ROBOT_BASE_POS[1] + 0.02,
        HANDOVER_POS[2] + 0.16
    ]

    handover_present_position = [
        HANDOVER_POS[0] + 0.12,
        HANDOVER_POS[1] - 0.10,
        HANDOVER_POS[2] + 0.12
    ]

    handover_release_position = [
        HANDOVER_POS[0] + 0.06,
        HANDOVER_POS[1] - 0.04,
        HANDOVER_POS[2] + 0.07
    ]

    print("WAIT STOP + 1.5s DELAY + HANDOVER TEMBUS aktif. Marker hijau non-solid, basket yang menahan objek.")
    print("HANDOVER_POS:", HANDOVER_POS)
    print("Handover present position:", handover_present_position)
    print("Handover release position:", handover_release_position)

    # 0. Hold posisi sekarang, lalu pindah ke shelf-ready pose.
    print("Step 0: Hold current pose")
    print("Auto-home dimatikan. Tekan m untuk kembali ke home pose.")

    print("Step 0a: Joint-1 pre-align to object")
    rotate_joint1_to_face_object(robot_data, object_pos, duration=JOINT1_ALIGN_DURATION)

    print("Step 0b: Shelf-ready transition")
    ready_error, _, _ = move_robot_to_position(
        robot_data=robot_data,
        target_position=points["ready"],
        duration=1.8
    )

    if ready_error > READY_ERROR_LIMIT:
        print("Pick dibatalkan: shelf-ready position belum reachable.")
        print("Auto-home dimatikan. Tekan m untuk kembali ke home pose.")

        return

    # 1. Pre-grasp
    print("Step 1: Pre-grasp")
    set_gripper_open(robot_data)
    pre_error, _, _ = move_robot_to_position(
        robot_data=robot_data,
        target_position=points["pre_grasp"],
        duration=2.5,
        motor_force=190,
        max_velocity=0.18
    )

    if pre_error > REACH_ERROR_LIMIT:
        print("Pick dibatalkan: pre-grasp tidak reachable.")
        print("Auto-home dimatikan. Tekan m untuk kembali ke home pose.")

        return

    # 2. Approach
    print("Step 2: Approach")
    approach_error, _, _ = move_robot_to_position(
        robot_data=robot_data,
        target_position=points["approach"],
        duration=2.2,
        motor_force=170,
        max_velocity=0.12,
        error_print_limit=HANDOVER_ERROR_LIMIT
    )

    approach_limit = BOTTOM_APPROACH_ERROR_LIMIT if points["shelf_index"] == 0 else APPROACH_ERROR_LIMIT

    if approach_error > approach_limit:
        print("Pick dibatalkan: approach terlalu jauh dari target, jangan attach.")
        print("Auto-home dimatikan. Tekan m untuk kembali ke home pose.")

        return

    # 3. Virtual grasp
    print("Step 3: Virtual grasp / attach object")
    grasp_constraint = attach_object_to_end_effector(
        robot_data=robot_data,
        object_id=object_id
    )

    if grasp_constraint is None:
        print("Pick gagal: objek tidak berhasil di-grasp.")
        print("Auto-home dimatikan. Tekan m untuk kembali ke home pose.")

        return

    for _ in range(60):
        p.stepSimulation()
        time.sleep(1 / 240)

    # 4. Retreat keluar dari rak dulu
    print("Step 4: Retreat keluar rak")
    error, grasp_constraint, dropped = move_robot_to_position(
        robot_data=robot_data,
        target_position=points["retreat"],
        duration=2.0,
        attached_object_id=object_id,
        grasp_constraint_id=grasp_constraint,
        # SAFE rollback:
        # jangan langsung lepas saat retreat, karena kontak kecil dengan papan rak sering false positive.
        release_on_collision=False,
        monitor_start_delay_steps=120
    )

    if dropped:
        print("Grasp failed: objek lepas saat retreat.")
        print("Auto-home dimatikan. Tekan m untuk kembali ke home pose.")

        return

    # 5. Lift setelah objek keluar rak
    print("Step 5: Lift")
    error, grasp_constraint, dropped = move_robot_to_position(
        robot_data=robot_data,
        target_position=points["lift"],
        duration=2.5,
        attached_object_id=object_id,
        grasp_constraint_id=grasp_constraint,
        # Untuk rak bawah kontak kecil sering false-positive.
        # Jangan langsung lepas saat lift; evaluasi visual/log setelah gerakan.
        release_on_collision=False,
        monitor_start_delay_steps=40
    )

    if dropped:
        print("Grasp failed: objek lepas saat lift.")
        print("Auto-home dimatikan. Tekan m untuk kembali ke home pose.")

        return

    lifted_pos, _ = p.getBasePositionAndOrientation(object_id)
    lift_height = lifted_pos[2] - object_pos[2]

    print("Lift height:", lift_height)

    if lift_height >= 0.05:
        print("Grasp success: objek terangkat >= 5 cm")
    else:
        print("Grasp failed: objek belum terangkat cukup tinggi.")
        release_object(grasp_constraint)
        set_object_transport_mode(object_id, enabled=False)
        print("Auto-home dimatikan. Tekan m untuk kembali ke home pose.")

        return

    # 6. Handover video-style:
    # Fase 6a: bridge/carry high.
    # Robot membawa objek keluar dari rak ke titik aman, mirip gerakan awal pada video.
    print("Step 6a: Video-style handover - carry high / bridge")
    error, grasp_constraint, dropped = move_robot_to_position(
        robot_data=robot_data,
        target_position=handover_bridge_position,
        duration=1.80,
        attached_object_id=object_id,
        grasp_constraint_id=grasp_constraint,
        release_on_collision=False,
        monitor_start_delay_steps=80,
        motor_force=180,
        max_velocity=0.20,
        error_print_limit=HANDOVER_ERROR_LIMIT
    )

    if dropped:
        print("Task failed: objek lepas saat handover bridge.")
        print("Auto-home dimatikan. Tekan m untuk kembali ke home pose.")

        return

    if error > HANDOVER_ERROR_LIMIT:
        print("WARNING: handover bridge belum presisi, lanjut ke present/fallback tanpa melepas objek.")

    # Fase 6b: present object.
    # Ini titik utama yang membuat handover terlihat seperti disodorkan ke user.
    print("Step 6b: Video-style handover - present object to user area")
    error, grasp_constraint, dropped = move_robot_to_position(
        robot_data=robot_data,
        target_position=handover_present_position,
        duration=1.90,
        attached_object_id=object_id,
        grasp_constraint_id=grasp_constraint,
        release_on_collision=False,
        monitor_start_delay_steps=80,
        motor_force=175,
        max_velocity=0.16,
        error_print_limit=HANDOVER_ERROR_LIMIT
    )

    if dropped:
        print("Task failed: objek lepas saat present object.")
        print("Auto-home dimatikan. Tekan m untuk kembali ke home pose.")

        return

    if error > HANDOVER_ERROR_LIMIT:
        print("WARNING: handover present-position belum presisi, lanjut ke release/fallback tanpa melepas objek.")

    # Hold sebentar seperti pada demonstrasi real robot, agar user punya waktu menerima objek.
    print("Step 6c: Hold object in front of user")
    for _ in range(60):
        p.stepSimulation()
        time.sleep(1 / 240)

    # Fase 6d: turunkan sedikit ke titik serah-terima.
    # Turun hanya sedikit supaya tidak terlihat seperti menjatuhkan objek.
    print("Step 6d: Lower slightly to handover release point")
    error, grasp_constraint, dropped = move_robot_to_position(
        robot_data=robot_data,
        target_position=handover_release_position,
        duration=1.20,
        attached_object_id=object_id,
        grasp_constraint_id=grasp_constraint,
        release_on_collision=False,
        monitor_start_delay_steps=45,
        motor_force=155,
        max_velocity=0.13,
        error_print_limit=HANDOVER_ERROR_LIMIT
    )

    if dropped:
        print("Task failed: objek lepas saat turun ke release point.")
        print("Auto-home dimatikan. Tekan m untuk kembali ke home pose.")

        return

    if error > HANDOVER_ERROR_LIMIT:
        print("WARNING: handover release-position belum presisi, objek akan tetap diletakkan terkontrol di HANDOVER_POS.")

    # FIX5:
    # Release guard: object harus benar-benar berada di atas basket sebelum release.
    # Kalau belum, robot/EE akan align ulang sambil object tetap attached.
    grasp_constraint, dropped, release_ready = align_attached_object_over_basket_release_guard(
        robot_data=robot_data,
        object_id=object_id,
        grasp_constraint=grasp_constraint,
        max_iters=RELEASE_GUARD_MAX_ALIGN_ITERS
    )

    if dropped:
        print("Task failed: objek lepas saat release-guard align.")
        print("Auto-home dimatikan. Tekan m untuk kembali ke home pose.")
        return

    if not release_ready and RELEASE_GUARD_ABORT_IF_NOT_READY:
        print("Task aborted: release dibatalkan karena object belum di atas basket.")
        print("Gripper tetap menahan object supaya tidak jatuh sebelum basket.")
        print("Auto-home dimatikan. Tekan m untuk kembali ke home pose.")
        return

    # Tunggu sampai robot benar-benar stop sebelum release objek.
    # Ini mengganti pause fixed 70 step, karena yang kita butuhkan bukan sekadar delay,
    # tapi kondisi robot sudah tidak bergerak lagi.
    wait_until_robot_stops_before_release(
        robot_data=robot_data,
        object_id=object_id
    )

    # Extra delay 1.5 detik dimatikan; release hanya setelah wait-stop.
    delay_after_robot_stop_before_release(
        robot_data=robot_data,
        object_id=object_id,
        seconds=RELEASE_DELAY_AFTER_STOP_SEC
    )

    # 7. Final release guard check.
    print("Step 7: Final release guard check before natural drop")
    final_ready, final_info = check_object_ready_to_release_into_basket(
        object_id,
        verbose=True
    )

    if not final_ready and RELEASE_GUARD_ABORT_IF_NOT_READY:
        print("RELEASE CANCELLED: object belum aman di atas basket.")
        print("Gripper tetap menahan object. Tidak ada drop sebelum basket.")
        return

    print("Step 7: Natural release above basket, object falls by gravity")

    # Collision objek diaktifkan kembali dalam mode drop natural sebelum constraint dilepas.
    # Velocity dinolkan agar jatuhnya vertikal, bukan diagonal akibat sisa gerak robot.
    enable_object_collision_for_natural_drop(object_id)

    release_object(grasp_constraint)
    set_gripper_open(robot_data)

    # Tidak ada smooth_place_object_to_handover.
    # Tidak ada teleport/interpolasi object.
    # Objek dibiarkan jatuh natural karena gravity ke basket.
    natural_gravity_drop_to_basket(object_id, steps=BASKET_DROP_SETTLE_STEPS)

    final_pos, _ = p.getBasePositionAndOrientation(object_id)
    print("Final object position:", final_pos)

    dist_to_basket_xy = math.sqrt(
        (final_pos[0] - BASKET_CENTER[0]) ** 2
        + (final_pos[1] - BASKET_CENTER[1]) ** 2
    )

    print("Distance XY to basket center:", dist_to_basket_xy)

    if dist_to_basket_xy <= 0.16:
        print("Task completion: SUCCESS")
    else:
        print("Task completion: FAILED")

    # 8. Home pose manual
    print("Step 8: Auto-home disabled. Press m to return to home pose.")

    print("Pick-lift-handover selesai.")


# =========================================================
# COLLISION DETECTION HELPERS
# =========================================================
def ensure_collision_log_file():
    """
    Membuat CSV log collision/proximity.
    """
    global COLLISION_LOG_FILE

    if COLLISION_LOG_FILE is not None:
        return COLLISION_LOG_FILE

    os.makedirs(COLLISION_LOG_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    COLLISION_LOG_FILE = os.path.join(
        COLLISION_LOG_DIR,
        f"collision_monitor_{timestamp}.csv"
    )

    with open(COLLISION_LOG_FILE, "w", encoding="utf-8") as f:
        f.write(
            "timestamp,stage,event_type,severity,"
            "robot_link,robot_link_index,"
            "other_body,other_body_id,other_role,"
            "distance,normal_force,"
            "position_on_robot_x,position_on_robot_y,position_on_robot_z,"
            "position_on_other_x,position_on_other_y,position_on_other_z,"
            "selected_name,allowed_reason\\n"
        )

    print("Collision log file:", COLLISION_LOG_FILE)
    return COLLISION_LOG_FILE


def get_selected_object_id_from_context(objects=None, selected_name=None):
    if objects is None or selected_name is None:
        return None
    if selected_name not in objects:
        return None
    return objects[selected_name]["id"]


def classify_collision_event(other_body_id, distance, selected_object_id=None):
    """
    Severity:
    - CONTACT: sudah kontak/penetration
    - NEAR: jarak dekat sebelum kontak
    - CLEAR: tidak ada event
    Allowed reason dipakai agar target object touch tidak dianggap tabrakan buruk.
    """
    if should_ignore_collision_body(other_body_id):
        return "IGNORE", "ignored_body"

    if selected_object_id is not None and other_body_id == selected_object_id:
        if distance <= COLLISION_HARD_CONTACT_DISTANCE:
            return "CONTACT_ALLOWED", "selected_target_object"
        return "NEAR_ALLOWED", "selected_target_object"

    role = get_body_role(other_body_id)

    if role == "floor":
        return "IGNORE", "floor_ignored"

    if distance <= COLLISION_HARD_CONTACT_DISTANCE:
        return "CONTACT", "bad_environment_contact"

    if distance <= COLLISION_PROXIMITY_DISTANCE:
        return "NEAR", "proximity_warning"

    return "CLEAR", ""


def write_collision_event_to_csv(event):
    log_file = ensure_collision_log_file()

    row = [
        time.strftime("%Y-%m-%d %H:%M:%S"),
        event.get("stage", ""),
        event.get("event_type", ""),
        event.get("severity", ""),
        event.get("robot_link", ""),
        event.get("robot_link_index", ""),
        event.get("other_body", ""),
        event.get("other_body_id", ""),
        event.get("other_role", ""),
        event.get("distance", ""),
        event.get("normal_force", ""),
        *event.get("position_on_robot", ["", "", ""]),
        *event.get("position_on_other", ["", "", ""]),
        event.get("selected_name", ""),
        event.get("allowed_reason", "")
    ]

    with open(log_file, "a", encoding="utf-8") as f:
        f.write(",".join(str(x) for x in row) + "\\n")


def draw_collision_debug_event(event):
    """
    Visualisasi collision/proximity di PyBullet GUI.
    Merah = contact, kuning = near/proximity.
    """
    if not COLLISION_DEBUG_DRAW_ENABLED:
        return

    pos_a = event.get("position_on_robot")
    pos_b = event.get("position_on_other")
    if not pos_a or not pos_b:
        return

    severity = event.get("severity", "")
    if "CONTACT" in severity:
        color = [1.0, 0.0, 0.0]
    else:
        color = [1.0, 1.0, 0.0]

    try:
        p.addUserDebugLine(
            pos_a,
            pos_b,
            lineColorRGB=color,
            lineWidth=3.0,
            lifeTime=COLLISION_DEBUG_LIFETIME
        )
        p.addUserDebugText(
            f"{severity}: {event.get('other_body', '')}",
            textPosition=pos_b,
            textColorRGB=color,
            textSize=0.9,
            lifeTime=COLLISION_DEBUG_LIFETIME
        )
    except Exception:
        pass


def collect_robot_collision_events(robot_data, objects=None, selected_name=None, stage="snapshot"):
    """
    Mengambil event collision/proximity robot terhadap semua body environment.

    Teknik:
    - getClosestPoints untuk proximity dan penetrasi dengan threshold.
    - normalForce dari hasil getContactPoints ketika contact sudah terjadi.
    """
    robot_id = robot_data["robot_id"]
    selected_object_id = get_selected_object_id_from_context(objects, selected_name)

    events = []
    body_ids = get_all_body_ids()

    for body_id in body_ids:
        if body_id == robot_id:
            continue

        if should_ignore_collision_body(body_id):
            continue

        try:
            closest_points = p.getClosestPoints(
                bodyA=robot_id,
                bodyB=body_id,
                distance=COLLISION_PROXIMITY_DISTANCE
            )
        except Exception:
            continue

        if not closest_points:
            continue

        for cp in closest_points:
            link_a = cp[3]
            # link_b = cp[4]
            pos_on_a = list(cp[5])
            pos_on_b = list(cp[6])
            distance = float(cp[8])
            normal_force = float(cp[9]) if len(cp) > 9 else 0.0

            severity, allowed_reason = classify_collision_event(
                other_body_id=body_id,
                distance=distance,
                selected_object_id=selected_object_id
            )

            if severity in ["CLEAR", "IGNORE"]:
                continue

            event_type = "contact" if distance <= COLLISION_HARD_CONTACT_DISTANCE else "proximity"

            event = {
                "stage": stage,
                "event_type": event_type,
                "severity": severity,
                "robot_link": get_robot_link_name(robot_id, link_a),
                "robot_link_index": link_a,
                "other_body": get_body_name(body_id),
                "other_body_id": body_id,
                "other_role": get_body_role(body_id),
                "distance": round(distance, 6),
                "normal_force": round(normal_force, 6),
                "position_on_robot": [round(v, 6) for v in pos_on_a],
                "position_on_other": [round(v, 6) for v in pos_on_b],
                "selected_name": selected_name,
                "allowed_reason": allowed_reason
            }
            events.append(event)

    # Urutkan dari jarak terkecil/penetrasi terdalam.
    events.sort(key=lambda e: float(e.get("distance", 999)))
    return events


def summarize_collision_events(events, title="COLLISION EVENTS", max_rows=20, save_to_csv=True, draw_debug=True):
    print(f"\\n=== {title} ===")
    print("Jumlah event:", len(events))

    if len(events) == 0:
        print("Tidak ada contact/proximity robot yang melewati threshold.")
        return

    for i, event in enumerate(events[:max_rows]):
        print(
            f"{i+1:02d}. {event['severity']} | "
            f"stage={event['stage']} | "
            f"link={event['robot_link']}({event['robot_link_index']}) | "
            f"body={event['other_body']}({event['other_role']}) | "
            f"distance={event['distance']} | "
            f"force={event['normal_force']} | "
            f"reason={event['allowed_reason']}"
        )

        if save_to_csv:
            write_collision_event_to_csv(event)
        if draw_debug:
            draw_collision_debug_event(event)

    if len(events) > max_rows:
        print("... event lain disembunyikan:", len(events) - max_rows)


def print_collision_snapshot(robot_data, objects=None, selected_name=None):
    """
    Snapshot manual dari tombol E.
    """
    events = collect_robot_collision_events(
        robot_data=robot_data,
        objects=objects,
        selected_name=selected_name,
        stage="manual_snapshot"
    )
    summarize_collision_events(
        events,
        title="MANUAL COLLISION / PROXIMITY SNAPSHOT",
        max_rows=30,
        save_to_csv=True,
        draw_debug=True
    )


def monitor_collision_during_motion(robot_data, step, stage_name="IK_MOVE"):
    """
    Monitor collision saat robot bergerak.
    Default hanya log/print jika COLLISION_VERBOSE_DURING_MOTION=True.
    """
    if not COLLISION_MONITOR_ENABLED:
        return False

    objects = LIVE_CAMERA_CONTEXT.get("objects") if "LIVE_CAMERA_CONTEXT" in globals() else None
    selected_name = LIVE_CAMERA_CONTEXT.get("selected_name") if "LIVE_CAMERA_CONTEXT" in globals() else None

    events = collect_robot_collision_events(
        robot_data=robot_data,
        objects=objects,
        selected_name=selected_name,
        stage=stage_name
    )

    bad_events = [
        e for e in events
        if e["severity"] in ["CONTACT", "NEAR"]
    ]

    if not bad_events:
        return False

    # Selalu simpan event bad pertama ke CSV, tapi print hanya periodik/verbose.
    worst = bad_events[0]
    signature = (
        worst.get("severity"),
        worst.get("robot_link_index"),
        worst.get("other_body_id")
    )

    last_step = COLLISION_LAST_PRINT_SIGNATURE.get(signature, -999999)
    should_print = (
        COLLISION_VERBOSE_DURING_MOTION
        and (step - last_step) >= COLLISION_PRINT_INTERVAL_STEPS
    )

    write_collision_event_to_csv(worst)
    draw_collision_debug_event(worst)

    if should_print:
        COLLISION_LAST_PRINT_SIGNATURE[signature] = step
        summarize_collision_events(
            bad_events,
            title=f"COLLISION MONITOR DURING {stage_name} step {step}",
            max_rows=8,
            save_to_csv=False,
            draw_debug=False
        )

    if COLLISION_AUTO_STOP_ON_BAD_CONTACT and worst["severity"] == "CONTACT":
        print("COLLISION AUTO STOP: bad contact detected.")
        return True

    return False


def toggle_collision_verbose():
    global COLLISION_VERBOSE_DURING_MOTION
    COLLISION_VERBOSE_DURING_MOTION = not COLLISION_VERBOSE_DURING_MOTION
    print("\\nCollision verbose during motion:", "ON" if COLLISION_VERBOSE_DURING_MOTION else "OFF")
    print("COLLISION_PROXIMITY_DISTANCE:", COLLISION_PROXIMITY_DISTANCE)
    print("Collision CSV:", ensure_collision_log_file())



# =========================================================
# PURE VISUAL OBJECT SEARCH FROM WRIST POV
# =========================================================
def set_wrist_pov_forced_target(target_position):
    """
    Set target pandang kamera untuk visual search.
    None = kembali ke behavior normal.
    """
    global WRIST_POV_FORCED_TARGET
    WRIST_POV_FORCED_TARGET = None if target_position is None else list(target_position)


def capture_wrist_pov_for_visual_search(width=960, height=720, robot_data=None):
    """
    Capture internal untuk visual search.
    Tidak menampilkan window.
    Return termasuk view/projection matrix untuk estimasi depth->world jika dibutuhkan.
    """
    camera_eye_position, camera_target_position, camera_up_vector = get_wrist_pov_camera_pose(
        robot_data=robot_data,
        objects=None,
        selected_name=None
    )

    view_matrix = p.computeViewMatrix(
        cameraEyePosition=camera_eye_position,
        cameraTargetPosition=camera_target_position,
        cameraUpVector=camera_up_vector
    )

    projection_matrix = p.computeProjectionMatrixFOV(
        fov=WRIST_POV_FOV,
        aspect=width / height,
        nearVal=WRIST_POV_NEAR,
        farVal=WRIST_POV_FAR
    )

    image = p.getCameraImage(
        width=width,
        height=height,
        viewMatrix=view_matrix,
        projectionMatrix=projection_matrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )

    rgb = np.reshape(image[2], (height, width, 4))[:, :, :3].astype(np.uint8)
    depth = np.reshape(image[3], (height, width))
    seg_raw = np.reshape(image[4], (height, width))
    seg_body = extract_body_id_from_segmentation(seg_raw)

    return {
        "rgb": rgb,
        "depth": depth,
        "seg_body": seg_body,
        "view_matrix": view_matrix,
        "projection_matrix": projection_matrix,
        "camera_eye": camera_eye_position,
        "camera_target": camera_target_position,
        "camera_up": camera_up_vector,
        "width": width,
        "height": height
    }


def parse_terminal_target_command(raw_text):
    """
    Parse command terminal:
    contoh:
    - ambil balok merah
    - cari silinder hijau
    - scan segitiga biru
    """
    text = raw_text.strip().lower()
    if not text:
        return None

    if text in ["q", "quit", "exit", "keluar"]:
        return {"action": "quit"}

    if text in ["lepas", "release", "drop", "buka gripper", "buka"]:
        return {"action": "manual_release"}

    if text in ["station 1", "rak 1", "wheelchair 1", "kursi 1"]:
        return {"action": "move_station", "station_id": 1}

    if text in ["station 2", "rak 2", "wheelchair 2", "kursi 2"]:
        return {"action": "move_station", "station_id": 2}

    if text in ["reset", "ulang", "reset robot", "reset scene", "home"]:
        return {"action": "reset_scene"}

    if text in ["home low", "idle low"]:
        return {"action": "set_idle_preset", "preset": "low"}

    if text in ["home mid", "idle mid"]:
        return {"action": "set_idle_preset", "preset": "mid"}

    if text in ["home high", "idle high"]:
        return {"action": "set_idle_preset", "preset": "high"}

    if text in ["help", "bantuan", "?"]:
        return {"action": "help"}

    if text in ["asr off", "asr manual", "auto asr off", "matikan asr", "asr mati"]:
        return {"action": "asr_off"}

    if text in ["asr on", "asr auto", "auto asr on", "nyalakan asr", "asr hidup"]:
        return {"action": "asr_on"}

    if text in ["asr clear", "clear asr", "hapus asr", "abaikan asr", "ignore asr"]:
        return {"action": "asr_clear"}

    if text in ["asr status", "status asr"]:
        return {"action": "asr_status"}

    if text in ["asr run", "run asr", "jalankan asr"]:
        return {"action": "asr_run"}

    shape = None
    color = None

    if "balok" in text or "kotak" in text or "box" in text:
        shape = "box"
    elif "silinder" in text or "cylinder" in text or "tabung" in text:
        shape = "cylinder"
    elif "segitiga" in text or "prisma" in text or "triangle" in text:
        shape = "triangle"

    if "merah" in text or "red" in text:
        color = "red"
    elif "hijau" in text or "green" in text:
        color = "green"
    elif "biru" in text or "blue" in text:
        color = "blue"
    elif "kuning" in text or "yellow" in text:
        color = "yellow"

    if shape is None or color is None:
        return {
            "action": "invalid",
            "raw": raw_text,
            "reason": "command harus berisi bentuk dan warna, contoh: ambil balok merah"
        }

    return {
        "action": "visual_search_pick",
        "shape": shape,
        "color": color,
        "raw": raw_text
    }


def target_shape_to_indonesian(shape):
    if shape == "box":
        return "balok"
    if shape == "cylinder":
        return "silinder"
    if shape == "triangle":
        return "segitiga"
    return shape


def target_color_to_indonesian(color):
    return {
        "red": "merah",
        "green": "hijau",
        "blue": "biru",
        "yellow": "kuning"
    }.get(color, color)


def make_color_mask_from_rgb(rgb, target_color):
    """
    Segmentasi warna sederhana dari RGB camera.
    Ini sengaja memakai RGB image, bukan metadata object list.
    """
    r = rgb[:, :, 0].astype(np.int16)
    g = rgb[:, :, 1].astype(np.int16)
    b = rgb[:, :, 2].astype(np.int16)

    if target_color == "red":
        mask = (r > 135) & (g < 115) & (b < 115) & ((r - np.maximum(g, b)) > 45)
    elif target_color == "green":
        mask = (g > 120) & (r < 130) & (b < 130) & ((g - np.maximum(r, b)) > 35)
    elif target_color == "blue":
        mask = (b > 120) & (r < 135) & (g < 170) & ((b - r) > 35)
    elif target_color == "yellow":
        mask = (r > 145) & (g > 115) & (b < 100) & ((np.minimum(r, g) - b) > 45)
    else:
        mask = np.zeros(rgb.shape[:2], dtype=bool)

    mask_u8 = (mask.astype(np.uint8) * 255)
    kernel = np.ones((5, 5), np.uint8)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, kernel)
    mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)
    return mask_u8


def classify_shape_from_contour(contour):
    """
    Klasifikasi visual kasar dari contour RGB.
    Tidak sempurna, tapi cukup untuk eksperimen apakah robot bisa mencari dari kamera.
    """
    area = float(cv2.contourArea(contour))
    if area <= 1.0:
        return "unknown", 0.0, {}

    perimeter = float(cv2.arcLength(contour, True))
    if perimeter <= 1e-6:
        return "unknown", 0.0, {}

    x, y, w, h = cv2.boundingRect(contour)
    bbox_area = max(1.0, float(w * h))
    rectangularity = area / bbox_area
    circularity = 4.0 * math.pi * area / (perimeter * perimeter)
    approx = cv2.approxPolyDP(contour, 0.04 * perimeter, True)
    vertices = len(approx)

    # Scores intentionally loose because camera viewpoint can distort objects.
    box_score = max(0.0, min(1.0, rectangularity))
    cylinder_score = max(0.0, min(1.0, circularity * 1.25))
    triangle_score = 0.0

    if vertices <= 4:
        triangle_score += 0.55
    if rectangularity < 0.72:
        triangle_score += 0.30
    if circularity < 0.70:
        triangle_score += 0.15
    triangle_score = max(0.0, min(1.0, triangle_score))

    scores = {
        "box": box_score,
        "cylinder": cylinder_score,
        "triangle": triangle_score
    }

    best_shape = max(scores, key=scores.get)
    best_score = scores[best_shape]

    metrics = {
        "area": area,
        "bbox": [int(x), int(y), int(w), int(h)],
        "rectangularity": round(rectangularity, 4),
        "circularity": round(circularity, 4),
        "vertices": int(vertices),
        "shape_scores": scores
    }
    return best_shape, best_score, metrics


def get_majority_body_id_from_mask(seg_body, mask_u8):
    ids = seg_body[mask_u8 > 0]
    if ids.size == 0:
        return None

    unique, counts = np.unique(ids, return_counts=True)

    filtered = []
    for body_id, count in zip(unique, counts):
        body_id_int = int(body_id)
        if body_id_int < 0:
            continue
        # ignore plane/shelf if registry is available
        role = get_body_role(body_id_int) if "get_body_role" in globals() else "unknown"
        if role in ["floor", "shelf", "handover_area", "wheelchair_or_mount", "robot"]:
            continue
        filtered.append((body_id_int, int(count)))

    if not filtered:
        # fallback: return majority nonnegative
        valid = [(int(b), int(c)) for b, c in zip(unique, counts) if int(b) >= 0]
        if not valid:
            return None
        return max(valid, key=lambda x: x[1])[0]

    return max(filtered, key=lambda x: x[1])[0]


def pixel_depth_to_world(u, v, depth_value, view_matrix, projection_matrix, width, height):
    """
    Unproject pixel + depth PyBullet ke world coordinate.
    """
    x = (2.0 * float(u) / float(width)) - 1.0
    y = 1.0 - (2.0 * float(v) / float(height))
    z = 2.0 * float(depth_value) - 1.0

    clip = np.array([x, y, z, 1.0], dtype=float)

    view = np.array(view_matrix, dtype=float).reshape((4, 4)).T
    proj = np.array(projection_matrix, dtype=float).reshape((4, 4)).T

    inv = np.linalg.inv(proj @ view)
    world = inv @ clip
    if abs(world[3]) > 1e-9:
        world = world / world[3]

    return world[:3].tolist()


def detect_target_from_visual_frame(frame, target_shape, target_color):
    """
    Deteksi target dari RGB camera:
    - threshold warna dari RGB
    - contour shape score
    - body id hanya dipakai setelah objek tampak, untuk virtual grasp PyBullet
    """
    rgb = frame["rgb"]
    depth = frame["depth"]
    seg_body = frame["seg_body"]

    mask_color = make_color_mask_from_rgb(rgb, target_color)
    contours, _ = cv2.findContours(mask_color, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < VISUAL_SEARCH_MIN_AREA:
            continue

        shape_label, shape_score, metrics = classify_shape_from_contour(contour)

        # Kalau shape score tidak cocok, tetap simpan sebagai low confidence untuk debug.
        target_score = metrics.get("shape_scores", {}).get(target_shape, 0.0)
        color_score = 1.0
        score = 0.45 * color_score + 0.55 * target_score

        x, y, w, h = metrics["bbox"]
        cx = int(x + w / 2)
        cy = int(y + h / 2)

        contour_mask = np.zeros(mask_color.shape, dtype=np.uint8)
        cv2.drawContours(contour_mask, [contour], -1, 255, -1)

        body_id = get_majority_body_id_from_mask(seg_body, contour_mask)

        depth_values = depth[contour_mask > 0]
        depth_values = depth_values[np.isfinite(depth_values)]
        if depth_values.size > 0:
            depth_median = float(np.median(depth_values))
        else:
            depth_median = float(depth[cy, cx])

        try:
            world_pos = pixel_depth_to_world(
                u=cx,
                v=cy,
                depth_value=depth_median,
                view_matrix=frame["view_matrix"],
                projection_matrix=frame["projection_matrix"],
                width=frame["width"],
                height=frame["height"]
            )
        except Exception:
            world_pos = [None, None, None]

        candidate = {
            "score": float(score),
            "target_shape_score": float(target_score),
            "shape_label": shape_label,
            "shape_score": float(shape_score),
            "target_shape": target_shape,
            "target_color": target_color,
            "bbox": metrics["bbox"],
            "centroid_px": [cx, cy],
            "area": area,
            "metrics": metrics,
            "body_id": body_id,
            "body_name": get_body_name(body_id) if body_id is not None else None,
            "body_role": get_body_role(body_id) if body_id is not None else None,
            "depth_median": depth_median,
            "world_pos_from_depth": world_pos
        }

        # Target final harus sesuai warna dan cukup masuk akal untuk shape.
        # Shape scoring dibuat loose; jika target score sangat kecil, reject.
        if target_score >= 0.20 and score >= VISUAL_SEARCH_MIN_SCORE:
            candidates.append(candidate)

    candidates.sort(key=lambda c: c["score"], reverse=True)
    return candidates, mask_color


def save_visual_search_debug_image(frame, mask_u8, candidate, target_shape, target_color, scan_index):
    if not VISUAL_SEARCH_SAVE_DEBUG_IMAGES:
        return None

    os.makedirs(VISUAL_SEARCH_LOG_DIR, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(
        VISUAL_SEARCH_LOG_DIR,
        f"{VISUAL_SEARCH_DEBUG_PREFIX}_{timestamp}_scan{scan_index:02d}_{target_shape}_{target_color}.png"
    )

    rgb = frame["rgb"].copy()

    if candidate is not None:
        x, y, w, h = candidate["bbox"]
        cv2.rectangle(rgb, (x, y), (x + w, y + h), (255, 255, 0), 2)
        cv2.putText(
            rgb,
            f"{target_shape}_{target_color} score={candidate['score']:.2f}",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2
        )

    mask_rgb = cv2.cvtColor(mask_u8, cv2.COLOR_GRAY2RGB)
    combined = np.hstack([rgb, mask_rgb])
    combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, combined_bgr)
    return filename


def build_visual_search_scan_plan():
    """
    Visual search rak aktif:
    - 3 shelf
    - 2 posisi per shelf
    - total 6 scan per rak

    Kamera TIDAK lagi mengubah arah pandang mengikuti slot.
    Eye/wrist berpindah ke posisi scan, tetapi view direction dikunci lurus +X.
    Live camera tetap ON, kamera tetap pada wrist. Shelf bawah dapat pitch
    menurun secara terkontrol, tanpa roll/yaw mengikuti setiap slot.
    """
    plan = []
    scan_y_values = [-0.22, 0.22]

    for shelf_index, shelf_z in enumerate(SHELF_Z_LEVELS):
        for y in scan_y_values:
            scan_position = [
                ROBOT_BASE_POS[0] + 0.30,
                ROBOT_BASE_POS[1] + y * 0.55,
                shelf_z + 0.24
            ]

            # Tetap disimpan untuk log/debug dan sebagai penanda search mode.
            # Rendering kamera fixed-angle akan mengabaikan arah ke titik ini.
            view_target = [
                0.54,
                ACTIVE_RACK_Y_OFFSET + y,
                shelf_z + 0.08
            ]

            plan.append({
                "rack_id": ACTIVE_RACK_ID,
                "rack_y_offset": ACTIVE_RACK_Y_OFFSET,
                "shelf_index": shelf_index,
                "slot_y": y,
                "scan_position": scan_position,
                "view_target": view_target
            })

    return plan

def find_object_name_by_body_id(objects, body_id):
    if body_id is None:
        return None

    for name, data in objects.items():
        if int(data["id"]) == int(body_id):
            return name
    return None


def visual_target_expected_name(target_shape, target_color):
    """
    Nama target ideal berdasarkan command terminal.
    """
    return f"{target_shape_to_indonesian(target_shape)}_{target_color_to_indonesian(target_color)}"


def object_name_matches_visual_command(objects, object_name, target_shape, target_color):
    """
    Validasi kandidat visual.

    Ini bukan memilih target dari metadata di awal.
    Validasi dilakukan setelah kamera melihat contour warna dan mendapatkan body_id.
    Tujuannya mencegah salah pick, misalnya:
    command = silinder hijau
    yang terlihat besar = balok_hijau
    maka balok_hijau ditolak dan robot lanjut scan.
    """
    if object_name is None:
        return False, "body_id_not_in_object_registry"

    if object_name not in objects:
        return False, "object_name_missing"

    object_shape = objects[object_name].get("shape")
    object_color = normalize_object_color_from_name(object_name)
    target_color_id = target_color_to_indonesian(target_color)

    if object_shape != target_shape:
        return False, f"shape_mismatch_detected_{object_shape}_expected_{target_shape}"

    if object_color != target_color_id:
        return False, f"color_mismatch_detected_{object_color}_expected_{target_color_id}"

    return True, "match"


def clamp_pure_cv_value(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def is_valid_world_position_from_depth(pos):
    if pos is None:
        return False

    if len(pos) != 3:
        return False

    for v in pos:
        if v is None:
            return False

    try:
        x, y, z = [float(v) for v in pos]
    except Exception:
        return False

    if not (PURE_CV_PICK_X_MIN <= x <= PURE_CV_PICK_X_MAX):
        return False
    if not (PURE_CV_PICK_Y_MIN <= y <= PURE_CV_PICK_Y_MAX):
        return False
    if not (PURE_CV_PICK_Z_MIN <= z <= PURE_CV_PICK_Z_MAX):
        return False

    return True


def make_pure_cv_pick_position(candidate):
    pos = candidate.get("world_pos_from_depth")

    if not is_valid_world_position_from_depth(pos):
        return None

    x, y, z = [float(v) for v in pos]

    x += PURE_CV_PICK_X_OFFSET
    y += PURE_CV_PICK_Y_OFFSET
    z += PURE_CV_PICK_Z_OFFSET

    x = clamp_pure_cv_value(x, PURE_CV_PICK_X_MIN, PURE_CV_PICK_X_MAX)
    y = clamp_pure_cv_value(y, PURE_CV_PICK_Y_MIN, PURE_CV_PICK_Y_MAX)
    z = clamp_pure_cv_value(z, PURE_CV_PICK_Z_MIN, PURE_CV_PICK_Z_MAX)

    return [x, y, z]


def evaluate_candidate_pure_cv(candidate, target_shape):
    area = float(candidate.get("area", 0.0))
    score = float(candidate.get("score", 0.0))
    target_shape_score = float(candidate.get("target_shape_score", 0.0))

    metrics = candidate.get("metrics", {})
    shape_scores = metrics.get("shape_scores", {})

    other_scores = [
        float(v)
        for k, v in shape_scores.items()
        if k != target_shape
    ]

    best_other = max(other_scores) if other_scores else 0.0
    margin = target_shape_score - best_other

    pick_pos = make_pure_cv_pick_position(candidate)

    area_ok = area >= PURE_CV_MIN_AREA
    score_ok = score >= PURE_CV_MIN_SCORE
    shape_ok = target_shape_score >= PURE_CV_MIN_TARGET_SHAPE_SCORE
    margin_ok = margin >= PURE_CV_MIN_SHAPE_MARGIN
    depth_ok = pick_pos is not None

    ok = area_ok and score_ok and shape_ok and margin_ok and depth_ok

    info = {
        "ok": ok,
        "area": area,
        "score": score,
        "target_shape_score": target_shape_score,
        "best_other_shape_score": best_other,
        "shape_margin": margin,
        "area_ok": area_ok,
        "score_ok": score_ok,
        "shape_ok": shape_ok,
        "margin_ok": margin_ok,
        "depth_ok": depth_ok,
        "visual_pick_position": pick_pos
    }

    candidate["pure_cv_eval"] = info
    candidate["visual_pick_position"] = pick_pos

    return ok, info


def print_pure_cv_candidate(candidate, rank, target_shape, target_color, prefix=""):
    info = candidate.get("pure_cv_eval", {})
    print(
        f"{prefix}PURE CV candidate rank {rank} | "
        f"target={target_shape}_{target_color} | "
        f"shape_visual={candidate.get('shape_label')} | "
        f"area={float(candidate.get('area', 0.0)):.1f} | "
        f"score={float(candidate.get('score', 0.0)):.3f} | "
        f"target_shape_score={float(candidate.get('target_shape_score', 0.0)):.3f} | "
        f"margin={float(info.get('shape_margin', 0.0)):.3f} | "
        f"body_id={candidate.get('body_id')} | "
        f"pick_pos={candidate.get('visual_pick_position')}"
    )

def choose_valid_visual_candidate(candidates, objects, target_shape, target_color):
    """
    PURE VISUAL DECISION.

    Kandidat dipilih berdasarkan:
    1. luas contour hasil kamera,
    2. total visual score,
    3. shape score target,
    4. margin terhadap shape alternatif.

    Metadata object TIDAK dipakai untuk menentukan apakah kandidat benar.

    body_id -> object name hanya digunakan SETELAH keputusan visual
    untuk:
    - mengambil object fisik di PyBullet,
    - audit ground truth,
    - evaluasi benar/salah.
    """

    for rank, candidate in enumerate(candidates, start=1):

        area = float(candidate.get("area", 0.0))
        score = float(candidate.get("score", 0.0))
        target_shape_score = float(candidate.get("target_shape_score", 0.0))

        metrics = candidate.get("metrics", {})
        shape_scores = metrics.get("shape_scores", {})

        # Score shape selain target.
        alternative_scores = [
            float(v)
            for k, v in shape_scores.items()
            if k != target_shape
        ]

        best_alternative_score = (
            max(alternative_scores)
            if alternative_scores
            else 0.0
        )

        shape_margin = target_shape_score - best_alternative_score

        area_ok = area >= VISUAL_SEARCH_STRICT_MIN_AREA
        score_ok = score >= VISUAL_SEARCH_STRICT_MIN_SCORE
        shape_ok = target_shape_score >= VISUAL_SEARCH_STRICT_SHAPE_SCORE
        margin_ok = shape_margin >= VISUAL_SEARCH_SHAPE_MARGIN

        candidate["visual_area_ok"] = area_ok
        candidate["visual_score_ok"] = score_ok
        candidate["visual_shape_ok"] = shape_ok
        candidate["visual_shape_margin"] = shape_margin
        candidate["visual_margin_ok"] = margin_ok

        print(
            f"\nVISUAL candidate rank {rank}"
            f" | target={target_shape}_{target_color}"
            f" | shape_visual={candidate.get('shape_label')}"
            f" | area={area:.1f}"
            f" | score={score:.3f}"
            f" | target_shape_score={target_shape_score:.3f}"
            f" | best_other_shape={best_alternative_score:.3f}"
            f" | margin={shape_margin:.3f}"
        )

        if not area_ok:
            print("  REJECT: area terlalu kecil.")
            continue

        if not score_ok:
            print("  REJECT: total visual score rendah.")
            continue

        if not shape_ok:
            print("  REJECT: target shape score rendah.")
            continue

        if not margin_ok:
            print("  REJECT: bentuk ambigu terhadap shape lain.")
            continue

        # =================================================
        # SAMPAI TITIK INI KEPUTUSAN SUDAH MURNI VISUAL
        # =================================================

        body_id = candidate.get("body_id")

        if body_id is None:
            print("  REJECT: visual ditemukan tetapi body_id segmentation tidak tersedia.")
            continue

        detected_name = find_object_name_by_body_id(
            objects,
            body_id
        )

        if detected_name is None:
            print("  REJECT: body hasil segmentation tidak termasuk object manipulasi.")
            continue

        candidate["detected_name"] = detected_name

        # =================================================
        # STRICT BODY MATCH GATE — DIPAKAI SETELAH KANDIDAT LOLOS VISUAL
        # =================================================
        # Visual search tetap memulai dari RGB contour + color/shape score.
        # Tetapi setelah body_id dari segmentation diketahui, body yang akan diambil
        # divalidasi agar tidak mengambil objek warna sama tetapi bentuk berbeda.
        # Ini penting untuk multi-rack: kalau di rak 1 hanya ada prisma kuning,
        # command silinder kuning harus lanjut ke rak 2, bukan langsung mengambil prisma.
        if VISUAL_SEARCH_STRICT_OBJECT_MATCH:
            strict_ok, strict_reason = object_name_matches_visual_command(
                objects=objects,
                object_name=detected_name,
                target_shape=target_shape,
                target_color=target_color
            )
            candidate["strict_match"] = bool(strict_ok)
            candidate["strict_reject_reason"] = strict_reason

            if not strict_ok:
                print("  REJECT: objek terlihat punya warna target, tetapi body tidak cocok command.")
                print("  Reason:", strict_reason)
                print("  Detected body name:", detected_name)
                continue
        else:
            candidate["strict_match"] = True
            candidate["strict_reject_reason"] = "accepted_by_visual_only"

        # =================================================
        # ACTIVE RACK GATE — HUBUNGKAN COMVIS DENGAN WHEELCHAIR STATION
        # =================================================
        if ACTIVE_RACK_FILTER_ENABLED:
            rack_ok, rack_info = candidate_belongs_to_active_rack(candidate)
            candidate["active_rack_ok"] = bool(rack_ok)
            candidate["active_rack_info"] = rack_info

            if not rack_ok:
                print("  REJECT: kandidat visual berada di luar rak aktif.")
                print("  Active rack        :", rack_info.get("active_rack_id"))
                print("  Candidate rack     :", rack_info.get("used_rack_id"))
                print("  Candidate y source :", rack_info.get("source"))
                print("  Candidate y        :", rack_info.get("used_y"))
                print("  Body y/rack        :", rack_info.get("body_y"), rack_info.get("body_rack_id"))
                print("  Depth y/rack       :", rack_info.get("depth_y"), rack_info.get("depth_rack_id"))
                print("  Station belum mengambil object ini; lanjut scan / pindah station.")
                continue

        # =================================================
        # AUDIT GROUND TRUTH — SETELAH KEPUTUSAN
        # =================================================
        if VISUAL_SEARCH_AUDIT_ENABLED:

            gt_shape = objects[detected_name].get("shape")
            gt_color = normalize_object_color_from_name(detected_name)
            expected_color = target_color_to_indonesian(target_color)

            gt_correct = (
                gt_shape == target_shape
                and gt_color == expected_color
            )

            candidate["audit_gt_name"] = detected_name
            candidate["audit_gt_shape"] = gt_shape
            candidate["audit_gt_color"] = gt_color
            candidate["audit_visual_correct"] = gt_correct

            print("  === VISUAL AUDIT ===")
            print("  Predicted target :", target_shape, target_color)
            print("  Body from camera :", body_id)
            print("  GT object name   :", detected_name)
            print("  GT shape         :", gt_shape)
            print("  GT color         :", gt_color)
            print("  Visual correct   :", gt_correct)

        print("  ACCEPTED BY VISUAL.")

        return candidate, detected_name

    return None, None

def run_visual_search_for_target(robot_data, objects, target_shape, target_color):
    """
    Robot scan rak sampai target tampak pada kamera.
    Return nama objek jika terdeteksi.
    """
    global WRIST_POV_FORCED_TARGET

    print("\n=== PURE VISUAL SEARCH START ===")
    print("Target command:", target_shape_to_indonesian(target_shape), target_color_to_indonesian(target_color))
    print("Prinsip: robot tidak langsung ambil dari metadata; harus terlihat di RGB wrist POV dulu.")

    plan = build_visual_search_scan_plan()

    best_candidate = None
    best_detected_name = None

    for idx, item in enumerate(plan, start=1):
        print(f"\n--- Visual scan {idx}/{len(plan)} | rack={item.get('rack_id', ACTIVE_RACK_ID)} | view={item['shelf_index']} | y={item['slot_y']} ---")
        print("scan_position:", item["scan_position"])
        print("view_target:", item["view_target"])

        set_wrist_pov_forced_target(item["view_target"])

        error, _, _ = move_robot_to_position(
            robot_data=robot_data,
            target_position=item["scan_position"],
            duration=VISUAL_SEARCH_SCAN_DURATION,
            motor_force=VISUAL_SEARCH_SCAN_FORCE,
            max_velocity=VISUAL_SEARCH_SCAN_MAX_VELOCITY,
            error_print_limit=0.45
        )

        # settle kecil agar image stabil
        for _ in range(20):
            p.stepSimulation()
            time.sleep(1 / 240)

        # Diagnostic: pastikan kamera wrist benar-benar mampu memandang ke
        # shelf bawah (termasuk bila IK berhenti sebelum mencapai pose scan).
        scan_eye, scan_look, _scan_up = get_wrist_pov_camera_pose(robot_data)
        scan_dir = np.asarray(scan_look, dtype=float) - np.asarray(scan_eye, dtype=float)
        scan_pitch_deg = float(np.degrees(np.arctan2(
            float(scan_dir[2]), max(float(scan_dir[0]), 1e-9)
        )))
        print(
            "P05 SCAN CAMERA DIAGNOSTIC:",
            "eye_xyz=", [round(float(v), 4) for v in scan_eye],
            "pitch_deg=", round(scan_pitch_deg, 2),
            "IK_position_error_m=", round(float(error), 4)
        )

        frame = capture_wrist_pov_for_visual_search(
            width=960,
            height=720,
            robot_data=robot_data
        )

        # Diagnostik saja (TIDAK digunakan untuk meloloskan kandidat CV):
        # berapa persen area tengah-bawah frame ditutup body robot sendiri.
        # Segmentation PyBullet dipakai untuk audit occlusion, bukan untuk
        # mengganti hasil deteksi warna/bentuk atau menaikkan skor pengujian.
        frame_seg = np.asarray(frame["seg_body"])
        frame_height, frame_width = frame_seg.shape[:2]
        cy0, cy1 = int(frame_height * 0.28), int(frame_height * 0.88)
        cx0, cx1 = int(frame_width * 0.30), int(frame_width * 0.70)
        center_roi = frame_seg[cy0:cy1, cx0:cx1]
        center_robot_percent = (
            100.0 * float(np.mean(center_roi == int(robot_data["robot_id"])))
            if center_roi.size else float("nan")
        )
        print(
            "P05 WRIST SELF-OCCLUSION DIAGNOSTIC:",
            "center_robot_percent=", round(center_robot_percent, 2),
            "mount_offset_local_m=", [round(float(v), 3) for v in WRIST_POV_LOCAL_OFFSET],
            "(diagnostic_only=True)"
        )

        # Saat GUI aktif, tampilkan frame setelah scan settle; realtime
        # selama pergerakan tetap memakai WRIST_POV_LIVE_DURING_MOTION=True.
        if SHOW_CAMERA_WINDOWS:
            try:
                static_rgb_bgr = cv2.cvtColor(frame["rgb"], cv2.COLOR_RGB2BGR)
                safe_imshow(CAMERA_WINDOW_NAME, static_rgb_bgr)
            except Exception as exc:
                print("WARNING: static overview camera display gagal:", exc)

        candidates, mask = detect_target_from_visual_frame(
            frame=frame,
            target_shape=target_shape,
            target_color=target_color
        )

        candidate, detected_name = choose_valid_visual_candidate(
            candidates=candidates,
            objects=objects,
            target_shape=target_shape,
            target_color=target_color
        )

        # Debug image memakai kandidat valid jika ada.
        # Kalau belum ada kandidat valid, pakai kandidat score tertinggi untuk dokumentasi false positive.
        debug_candidate = candidate if candidate is not None else (candidates[0] if candidates else None)

        debug_path = save_visual_search_debug_image(
            frame=frame,
            mask_u8=mask,
            candidate=debug_candidate,
            target_shape=target_shape,
            target_color=target_color,
            scan_index=idx
        )

        if debug_path:
            print("debug image:", debug_path)

        if candidate is None:
            if len(candidates) == 0:
                print("Target belum terdeteksi dari kamera pada titik scan ini.")
            else:
                print("Ada objek warna sama terdeteksi, tetapi tidak cocok dengan bentuk+warna command. Scan lanjut.")
            continue

        print("\nVISUAL DETECTION CANDIDATE FOUND AND VERIFIED")
        print("expected_name:", candidate.get("expected_name"))
        print("verified_detected_name:", detected_name)
        print("score:", round(candidate["score"], 4))
        print("shape_label_visual:", candidate["shape_label"])
        print("target_shape_score:", round(candidate["target_shape_score"], 4))
        print("bbox:", candidate["bbox"])
        print("centroid_px:", candidate["centroid_px"])
        print("body_id_from_camera_seg:", candidate["body_id"])
        print("body_name_from_detected_body_id:", detected_name)
        print("world_pos_from_depth:", [round(v, 5) if v is not None else None for v in candidate["world_pos_from_depth"]])

        if candidate["body_role"] not in ["store_object", "unknown"]:
            print("Candidate ditolak karena role bukan store_object:", candidate["body_role"])
            continue

        best_candidate = candidate
        best_detected_name = detected_name
        break

    set_wrist_pov_forced_target(None)

    if best_detected_name is None:
        print("\nVISUAL SEARCH RESULT: TARGET NOT FOUND")
        print("Robot tidak bergerak mengambil karena target tidak terdeteksi kamera.")
        return None, best_candidate

    print("\nVISUAL SEARCH RESULT: TARGET FOUND")
    print("Detected object:", best_detected_name)
    return best_detected_name, best_candidate


def run_visual_search_and_pick(robot_data, objects, target_shape, target_color):
    """
    Search visual dulu, baru pick.
    Jika ada posisi dari depth camera, simpan ke object dict.
    """
    detected_name, candidate = run_visual_search_for_target(
        robot_data=robot_data,
        objects=objects,
        target_shape=target_shape,
        target_color=target_color
    )

    if detected_name is None:
        return None

    if PURE_CV_MODE and candidate is not None:
        visual_pick_position = candidate.get("visual_pick_position")

        if visual_pick_position is None:
            visual_pick_position = make_pure_cv_pick_position(candidate)

        if visual_pick_position is not None:
            objects[detected_name]["visual_pick_position"] = list(visual_pick_position)
            objects[detected_name]["visual_detection_score"] = float(candidate.get("score", 0.0))
            objects[detected_name]["visual_detection_shape_score"] = float(candidate.get("target_shape_score", 0.0))
            objects[detected_name]["visual_detection_source"] = "rgb_contour_depth_unprojection"

            print("\n=== PURE CV POSITION LOCKED ===")
            print("Object body to pick :", detected_name)
            print("Visual pick position:", visual_pick_position)
            print("Source              : RGB contour + depth unprojection")
            print("Metadata position   : NOT USED for pick target")
        else:
            print("WARNING: kandidat visual tidak punya posisi depth valid. Fallback ke PyBullet body position.")

    print("\n=== VISUAL SEARCH PASSED → PICK START ===")
    print("Robot akan mengambil objek yang sudah terlihat:", detected_name)

    set_live_camera_context(
        robot_data=robot_data,
        objects=objects,
        selected_name=detected_name
    )

    pick_lift_handover_selected_object(
        robot_data=robot_data,
        objects=objects,
        selected_name=detected_name
    )

    return detected_name


def run_multi_station_visual_search_and_pick(robot_data, objects, target_shape, target_color):
    """
    Cari target di rak 1 dulu.
    Jika tidak ditemukan, wheelchair pindah ke rak 2 lalu scan ulang.
    """
    # Hindari posisi pick lama dari command sebelumnya ikut terpakai.
    for obj in objects.values():
        obj.pop("visual_pick_position", None)
        obj.pop("visual_detection_score", None)
        obj.pop("visual_detection_shape_score", None)
        obj.pop("visual_detection_source", None)

    station_order = [1, 2] if MULTI_RACK_MODE else [1]

    for station_id in station_order:
        move_wheelchair_to_station(
            robot_data=robot_data,
            station_id=station_id,
            duration=2.8
        )

        print("\n=== MULTI-RACK VISUAL SEARCH ===")
        print("Current station:", station_id)
        print("Searching rack :", ACTIVE_RACK_ID)

        detected_name = run_visual_search_and_pick(
            robot_data=robot_data,
            objects=objects,
            target_shape=target_shape,
            target_color=target_color
        )

        if detected_name is not None:
            print("Target ditemukan di station:", station_id)
            return detected_name

        print("Target tidak ditemukan di station:", station_id)

    print("\nTARGET TIDAK DITEMUKAN DI SEMUA RAK.")
    return None


def terminal_input_worker():
    """
    Thread input terminal.
    Ketik command tanpa perlu OpenCV window:
    sim> ambil balok merah
    """
    global TERMINAL_COMMAND_STOP_REQUESTED

    print("\n=== TERMINAL COMMAND MODE AKTIF ===")
    print("Ketik contoh command:")
    print("  ambil balok merah")
    print("  cari silinder hijau")
    print("  scan segitiga biru")
    print("  help")
    print("  lepas / release / drop")
    print("  reset / home")
    print("  home low / home mid / home high")
    print("  quit")
    print("Catatan: robot baru mengambil kalau target terdeteksi dan cocok shape+color dari kamera wrist POV.")
    print("sim> ", end="", flush=True)

    while not TERMINAL_COMMAND_STOP_REQUESTED:
        try:
            raw = input()
        except EOFError:
            break
        except Exception:
            break

        TERMINAL_COMMAND_QUEUE.put(raw)
        print("sim> ", end="", flush=True)


def start_terminal_input_thread_once():
    global TERMINAL_COMMAND_THREAD_STARTED

    if TERMINAL_COMMAND_THREAD_STARTED:
        return

    thread = threading.Thread(
        target=terminal_input_worker,
        daemon=True
    )
    thread.start()
    TERMINAL_COMMAND_THREAD_STARTED = True


def print_terminal_help():
    print("\n=== TERMINAL COMMAND HELP ===")
    print("Format:")
    print("  ambil <bentuk> <warna>")
    print("  cari <bentuk> <warna>")
    print("  scan <bentuk> <warna>")
    print("")
    print("Bentuk:")
    print("  balok / kotak")
    print("  silinder")
    print("  segitiga / prisma")
    print("")
    print("Warna:")
    print("  merah, hijau, biru, kuning")
    print("")
    print("Contoh:")
    print("  ambil balok merah")
    print("  cari silinder hijau")
    print("  scan segitiga biru")
    print("")
    print("lepas / release / drop = manual release object yang masih digenggam")
    print("station 1 / rak 1 = pindahkan wheelchair ke rak 1")
    print("station 2 / rak 2 = pindahkan wheelchair ke rak 2")
    print("home low / home mid / home high = ganti preset idle robot")
    print("")
    print("ASR hybrid:")
    print("  asr run    = jalankan latest_command.json sekali via visual search")
    print("  asr on     = nyalakan auto ASR untuk command BARU, command lama diabaikan")
    print("  asr off    = matikan auto ASR, prompt terminal tetap jalan")
    print("  asr clear  = abaikan latest_command.json saat ini")
    print("  asr status = cek status auto ASR")
    print("  asr run    = jalankan latest_command.json sekali via visual search")
    print("")
    print("quit / keluar = keluar simulasi")


def process_terminal_command_queue(
    robot_data,
    objects,
    initial_poses,
    current_selected_name
):
    """
    Dipanggil dari main loop.
    Return:
    {"selected_name": ..., "quit": bool}
    """
    global AUTO_ASR_VISUAL_SEARCH_ENABLED, AUTO_ASR_LAST_SIGNATURE, AUTO_ASR_PROCESSED_SIGNATURES

    result = {
        "selected_name": current_selected_name,
        "quit": False
    }

    while not TERMINAL_COMMAND_QUEUE.empty():
        raw = TERMINAL_COMMAND_QUEUE.get_nowait()
        parsed = parse_terminal_target_command(raw)

        if parsed is None:
            continue

        if parsed["action"] == "quit":
            result["quit"] = True
            return result

        if parsed["action"] == "help":
            print_terminal_help()
            continue

        if parsed["action"] == "move_station":
            station_id = int(parsed.get("station_id", 1))
            move_wheelchair_to_station(
                robot_data=robot_data,
                station_id=station_id,
                duration=2.0
            )
            set_live_camera_context(
                robot_data=robot_data,
                objects=objects,
                selected_name=current_selected_name
            )
            continue

        if parsed["action"] == "reset_scene":
            print("\n=== RESET SCENE ===")

            reset_robot(robot_data)

            if OBJECT_RANDOMIZE_ON_RESET:
                reset_objects_randomized(
                    objects=objects, 
                    initial_poses=initial_poses,
                    reason="terminal_reset"
                )

                print("Robot reset + object placement RANDOMIZED.")
            else:
                reset_objects(
                    objects=objects,
                    initial_poses=initial_poses
                )

                print("Robot reset + object kembali ke posisi original.")

            move_wheelchair_to_station(
                robot_data=robot_data,
                station_id=1,
                duration=1.2
            )

            set_live_camera_context(
                robot_data=robot_data,
                objects=objects,
                selected_name=current_selected_name
            )          

            continue

        if parsed["action"] == "set_idle_preset":
            global CUSTOM_IDLE_PRESET
            CUSTOM_IDLE_PRESET = parsed.get("preset", CUSTOM_IDLE_PRESET)
            print("CUSTOM_IDLE_PRESET sekarang:", CUSTOM_IDLE_PRESET)
            reset_robot(robot_data)
            continue

        if parsed["action"] == "asr_off":
            AUTO_ASR_VISUAL_SEARCH_ENABLED = False
            print("\nAUTO ASR: OFF. Prompt terminal simulasi aktif normal.")
            print("Ketik 'asr on' untuk mengaktifkan lagi.")
            continue

        if parsed["action"] == "asr_on":
            AUTO_ASR_VISUAL_SEARCH_ENABLED = True
            current_command = read_latest_asr_command_silent(ASR_COMMAND_JSON)
            current_signature = get_asr_command_signature(current_command, ASR_COMMAND_JSON)
            if current_signature is not None:
                AUTO_ASR_LAST_SIGNATURE = current_signature
                AUTO_ASR_PROCESSED_SIGNATURES.add(current_signature)
                print("\nAUTO ASR: ON. Command JSON yang sudah ada diabaikan; menunggu command ASR baru.")
            else:
                print("\nAUTO ASR: ON. Menunggu command ASR baru.")
            continue

        if parsed["action"] == "asr_clear":
            current_command = read_latest_asr_command_silent(ASR_COMMAND_JSON)
            current_signature = get_asr_command_signature(current_command, ASR_COMMAND_JSON)
            if current_signature is not None:
                AUTO_ASR_LAST_SIGNATURE = current_signature
                AUTO_ASR_PROCESSED_SIGNATURES.add(current_signature)
                print("\nAUTO ASR CLEAR: latest_command.json saat ini sudah ditandai processed/ignored.")
                print("Transcript ignored:", current_command.get("asr_transcript"))
            else:
                print("\nAUTO ASR CLEAR: tidak ada command JSON untuk diabaikan.")
            continue

        if parsed["action"] == "asr_status":
            print("\n=== AUTO ASR STATUS ===")
            print("AUTO_ASR_VISUAL_SEARCH_ENABLED:", AUTO_ASR_VISUAL_SEARCH_ENABLED)
            print("AUTO_ASR_BUSY:", AUTO_ASR_BUSY)
            print("Processed signatures:", len(AUTO_ASR_PROCESSED_SIGNATURES))
            print("Last signature:", AUTO_ASR_LAST_SIGNATURE)
            current_command = read_latest_asr_command_silent(ASR_COMMAND_JSON)
            if current_command is not None:
                print("Current latest transcript:", current_command.get("asr_transcript"))
                print("Current latest created_at:", current_command.get("created_at"))
                print("Current latest command_id:", current_command.get("command_id"))
            else:
                print("Current latest command: None")
            continue

        if parsed["action"] == "asr_run":
            command = read_latest_asr_command(ASR_COMMAND_JSON)
            detected_name, status = run_asr_command_via_visual_search(
                robot_data=robot_data,
                objects=objects,
                command=command
            )
            print("ASR terminal-run visual-search status:", status)
            if detected_name is not None:
                result["selected_name"] = detected_name
            continue

        if parsed["action"] == "invalid":
            print("\nCommand tidak valid:", parsed.get("raw", ""))
            print("Alasan:", parsed.get("reason", ""))
            print("Ketik help untuk contoh.")
            continue

        if parsed["action"] == "visual_search_pick":
            detected_name = run_multi_station_visual_search_and_pick(
                robot_data=robot_data,
                objects=objects,
                target_shape=parsed["shape"],
                target_color=parsed["color"]
            )

            if detected_name is not None:
                result["selected_name"] = detected_name

    return result


def get_keyboard_key_optional():
    """
    Karena mode utama terminal, keyboard GUI hanya opsional.
    """
    if SHOW_CAMERA_WINDOWS:
        return cv2.waitKey(1) & 0xFF

    try:
        events = p.getKeyboardEvents()
        for candidate in [ord('q'), ord('r'), ord('m'), MANUAL_RELEASE_KEY]:
            if events.get(candidate, 0) & p.KEY_WAS_TRIGGERED:
                return candidate
    except Exception:
        pass

    return 255



# =========================================================
# AUTO ASR JSON → VISUAL SEARCH
# =========================================================
def asr_shape_to_visual_shape(shape):
    """
    Konversi shape dari realtime_asr_command.py ke shape visual search simulasi.
    ASR:
      silinder, kotak, prisma_segitiga
    Visual search:
      cylinder, box, triangle
    """
    mapping = {
        "silinder": "cylinder",
        "cylinder": "cylinder",
        "tabung": "cylinder",

        "kotak": "box",
        "balok": "box",
        "box": "box",

        "prisma_segitiga": "triangle",
        "prisma": "triangle",
        "segitiga": "triangle",
        "triangle": "triangle",
    }
    return mapping.get(str(shape).lower().strip())


def asr_color_to_visual_color(color):
    """
    Konversi color dari realtime_asr_command.py ke color visual search simulasi.
    """
    mapping = {
        "merah": "red",
        "red": "red",

        "biru": "blue",
        "blue": "blue",

        "kuning": "yellow",
        "yellow": "yellow",

        "hijau": "green",
        "green": "green",
    }
    return mapping.get(str(color).lower().strip())


def get_asr_command_signature(command, json_path=ASR_COMMAND_JSON):
    """
    Signature dipakai agar satu command ASR tidak dieksekusi berulang-ulang.

    Penting:
    - Jangan selalu memasukkan mtime kalau command_id/created_at sudah ada.
    - Kalau mtime ikut signature, file JSON yang tersentuh ulang bisa dianggap command baru,
      padahal isi command masih sama.
    """
    if command is None:
        return None

    command_id = command.get("command_id")
    created_at = command.get("created_at")
    transcript = command.get("asr_transcript")
    predicted = command.get("predicted_command_class")
    shape = command.get("shape")
    color = command.get("color")

    if command_id is not None or created_at is not None:
        return (
            command_id,
            created_at,
            transcript,
            predicted,
            shape,
            color
        )

    # Fallback untuk file lama yang belum punya command_id/created_at.
    try:
        mtime = os.path.getmtime(json_path)
    except Exception:
        mtime = 0.0

    return (
        transcript,
        predicted,
        shape,
        color,
        round(float(mtime), 3)
    )


def run_asr_command_via_visual_search(robot_data, objects, command):
    """
    Jalur ASR final untuk Bab 4:
    ASR JSON -> parse shape/color -> VISUAL SEARCH -> pick/handover.

    Fungsi ini sengaja TIDAK memakai select_object_from_asr_command()
    karena fungsi tersebut memilih objek langsung dari metadata nama objek.
    """
    if command is None:
        return None, "command_file_missing"

    if not command.get("ready_for_simulation", False):
        return None, command.get("parse_status", "command_not_ready")

    if command.get("parse_status") != "valid":
        return None, command.get("parse_status", "invalid_command")

    visual_shape = asr_shape_to_visual_shape(command.get("shape"))
    visual_color = asr_color_to_visual_color(command.get("color"))

    if visual_shape is None or visual_color is None:
        print("ASR command valid, tetapi shape/color belum bisa dikonversi ke visual search.")
        print("shape ASR:", command.get("shape"), "->", visual_shape)
        print("color ASR:", command.get("color"), "->", visual_color)
        return None, "shape_or_color_conversion_failed"

    print("\n=== AUTO ASR COMMAND → VISUAL SEARCH ===")
    print("ASR transcript:", command.get("asr_transcript"))
    print("Normalized:", command.get("normalized_text"))
    print("ASR shape/color:", command.get("shape"), command.get("color"))
    print("Visual search shape/color:", visual_shape, visual_color)
    print("Robot TIDAK langsung memilih metadata objek.")
    print("Robot akan scan rak dan hanya pick jika target terlihat & strict-match dari kamera.")

    detected_name = run_multi_station_visual_search_and_pick(
        robot_data=robot_data,
        objects=objects,
        target_shape=visual_shape,
        target_color=visual_color
    )

    if detected_name is None:
        return None, "visual_search_target_not_found"

    return detected_name, "visual_search_pick_executed"



def read_latest_asr_command_silent(json_path=ASR_COMMAND_JSON):
    """
    Baca JSON ASR tanpa print.
    Dipakai untuk guard startup dan polling supaya prompt simulasi tidak penuh log.
    """
    if not os.path.exists(json_path):
        return None

    try:
        with open(json_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        print("AUTO ASR silent read gagal:", exc)
        return None


def initialize_auto_asr_startup_guard():
    """
    Dipanggil sekali setelah scene+robot siap.

    Fungsi:
    - menandai waktu simulasi mulai listen ASR
    - jika latest_command.json sudah ada dari run sebelumnya,
      signature-nya disimpan agar TIDAK dieksekusi otomatis
    - command baru tetap akan jalan karena signature/command_id/created_at berubah
    """
    global AUTO_ASR_LAST_SIGNATURE, AUTO_ASR_ARMED_AT_TIME, AUTO_ASR_PROCESSED_SIGNATURES

    AUTO_ASR_ARMED_AT_TIME = time.time()

    if not AUTO_ASR_IGNORE_EXISTING_COMMAND_ON_START:
        print("AUTO ASR startup guard disabled.")
        return

    old_command = read_latest_asr_command_silent(ASR_COMMAND_JSON)
    if old_command is None:
        print("AUTO ASR startup guard: tidak ada latest_command.json lama.")
        return

    AUTO_ASR_LAST_SIGNATURE = get_asr_command_signature(
        old_command,
        ASR_COMMAND_JSON
    )

    if AUTO_ASR_LAST_SIGNATURE is not None:
        AUTO_ASR_PROCESSED_SIGNATURES.add(AUTO_ASR_LAST_SIGNATURE)

    print("\n=== AUTO ASR STARTUP GUARD ===")
    print("latest_command.json lama ditemukan dan DIABAIKAN saat startup.")
    print("Ignored transcript:", old_command.get("asr_transcript"))
    print("Ignored created_at:", old_command.get("created_at"))
    print("Ignored command_id:", old_command.get("command_id"))
    print("Simulasi baru akan bergerak setelah ASR menulis command baru.")
    print("Prompt terminal tetap bisa dipakai.")



def poll_asr_json_and_run_visual_search(robot_data, objects, current_selected_name):
    """
    Dipanggil otomatis di main loop.

    Revisi HYBRID:
    - Terminal prompt tetap aktif dan tidak dikalahkan ASR.
    - latest_command.json yang sama tidak di-print / dieksekusi berulang.
    - command ASR baru tetap otomatis menjalankan visual search.
    - file lama saat startup tetap diabaikan.
    """
    global AUTO_ASR_LAST_SIGNATURE, AUTO_ASR_BUSY, AUTO_ASR_PROCESSED_SIGNATURES
    global AUTO_ASR_VISUAL_SEARCH_ENABLED

    result = {
        "selected_name": current_selected_name,
        "executed": False,
        "status": None,
    }

    if not AUTO_ASR_VISUAL_SEARCH_ENABLED:
        return result

    if AUTO_ASR_BUSY:
        return result

    if not os.path.exists(ASR_COMMAND_JSON):
        return result

    # Kalau user sedang mengetik command terminal, beri prioritas ke prompt simulasi.
    if AUTO_ASR_PROMPT_PRIORITY and not TERMINAL_COMMAND_QUEUE.empty():
        return result

    now = time.time()
    last_poll = getattr(poll_asr_json_and_run_visual_search, "_last_poll", 0.0)
    if now - last_poll < AUTO_ASR_POLL_INTERVAL_SEC:
        return result
    poll_asr_json_and_run_visual_search._last_poll = now

    command = read_latest_asr_command_silent(ASR_COMMAND_JSON)
    signature = get_asr_command_signature(command, ASR_COMMAND_JSON)

    if command is None or signature is None:
        return result

    # Jangan eksekusi command lama yang file-nya sudah ada sebelum simulasi mulai listen.
    if AUTO_ASR_REQUIRE_JSON_AFTER_SIM_START and AUTO_ASR_ARMED_AT_TIME is not None:
        try:
            json_mtime = os.path.getmtime(ASR_COMMAND_JSON)
        except Exception:
            json_mtime = 0.0

        if json_mtime <= AUTO_ASR_ARMED_AT_TIME:
            if signature != AUTO_ASR_LAST_SIGNATURE:
                AUTO_ASR_LAST_SIGNATURE = signature
                AUTO_ASR_PROCESSED_SIGNATURES.add(signature)
                print("AUTO ASR: latest_command.json lama diabaikan. Menunggu command ASR baru...")
            return result

    # Kalau signature sama, jangan print dan jangan jalan.
    if AUTO_ASR_RUN_ONLY_ON_NEW_COMMAND and signature == AUTO_ASR_LAST_SIGNATURE:
        return result

    # Kalau pernah diproses, jangan ulangi walaupun polling jalan terus.
    if signature in AUTO_ASR_PROCESSED_SIGNATURES:
        AUTO_ASR_LAST_SIGNATURE = signature
        return result

    AUTO_ASR_LAST_SIGNATURE = signature
    AUTO_ASR_PROCESSED_SIGNATURES.add(signature)

    if not command.get("ready_for_simulation", False):
        print("AUTO ASR: command baru terbaca tetapi belum ready, robot tidak bergerak.")
        print("parse_status:", command.get("parse_status"))
        result["status"] = command.get("parse_status")
        return result

    try:
        AUTO_ASR_BUSY = True
        detected_name, status = run_asr_command_via_visual_search(
            robot_data=robot_data,
            objects=objects,
            command=command
        )
        result["status"] = status

        if detected_name is not None:
            result["selected_name"] = detected_name
            result["executed"] = True

        print("AUTO ASR visual-search status:", status)

        if AUTO_ASR_DISABLE_AFTER_ONE_COMMAND:
            AUTO_ASR_VISUAL_SEARCH_ENABLED = False
            print("AUTO ASR disabled after one command. Terminal prompt tetap aktif.")

        return result

    finally:
        AUTO_ASR_BUSY = False


# =========================================================
# ASR COMMAND → TARGET SELECTION
# =========================================================
def normalize_object_shape_from_name(name):
    name = name.lower()

    if "silinder" in name:
        return "silinder"

    if "kotak" in name or "balok" in name:
        return "kotak"

    if "prisma" in name or "segitiga" in name:
        return "prisma_segitiga"

    return None


def normalize_object_color_from_name(name):
    name = name.lower()

    if "merah" in name:
        return "merah"

    if "biru" in name:
        return "biru"

    if "kuning" in name:
        return "kuning"

    # sementara kalau file lama masih pakai hijau
    if "hijau" in name:
        return "hijau"

    return None


def read_latest_asr_command(json_path=ASR_COMMAND_JSON):
    """
    Membaca hasil ASR dari latest_command.json.
    File ini dibuat oleh evaluate_asr_dataset.py mode --single.
    """
    if not os.path.exists(json_path):
        print("File command ASR tidak ditemukan:", json_path)
        return None

    with open(json_path, "r", encoding="utf-8") as f:
        command = json.load(f)

    print("\n=== ASR COMMAND LOADED ===")
    print(json.dumps(command, indent=2, ensure_ascii=False))

    return command


def select_object_from_asr_command(objects, command):
    """
    Memilih objek target berdasarkan shape + color dari hasil ASR.

    Aturan:
    - jika command tidak valid → robot tidak bergerak
    - jika target tidak ditemukan → robot tidak bergerak
    - jika target lebih dari satu → ambigu, robot tidak bergerak
    - jika target satu → objek dipilih
    """
    if command is None:
        return None, "command_file_missing"

    if not command.get("ready_for_simulation", False):
        return None, "command_not_ready"

    if command.get("parse_status") != "valid":
        return None, command.get("parse_status", "invalid_command")

    target_shape = command.get("shape")
    target_color = command.get("color")

    if target_shape is None or target_color is None:
        return None, "missing_shape_or_color"

    candidates = []

    for name, obj in objects.items():
        obj_shape = normalize_object_shape_from_name(name)
        obj_color = normalize_object_color_from_name(name)

        if obj_shape == target_shape and obj_color == target_color:
            candidates.append(name)

    if len(candidates) == 0:
        return None, "target_not_found"

    if len(candidates) > 1:
        print("Target ambigu. Kandidat:", candidates)
        return None, "target_ambiguous"

    return candidates[0], "target_selected"

# =========================================================
# MAIN
# =========================================================
def main():
    physics_client = p.connect(
        p.GUI,
        options="--width=1800 --height=1100"
    )

    if physics_client < 0:
        raise RuntimeError("Gagal membuka PyBullet GUI")

    # Bersihkan semua body lama agar robot/objek tidak dobel
    p.resetSimulation()

    p.setAdditionalSearchPath(pybullet_data.getDataPath())

    # Gravity normal bumi
    p.setGravity(0, 0, -9.81)
    p.setRealTimeSimulation(0)

    # Physics tuning agar jatuh lebih natural
    p.setPhysicsEngineParameter(
        fixedTimeStep=1.0 / 240.0,
        numSolverIterations=150,
        numSubSteps=2
    )

    # Scene
    objects, initial_poses = create_indonesian_store_scene()

    # Biarkan objek stabil dulu di atas rak sebelum robot dimuat/bergerak
    for _ in range(240):
        p.stepSimulation()
        time.sleep(1 / 240)

    # Simpan ulang pose awal setelah objek stabil
    for name, obj in objects.items():
        pos, orn = p.getBasePositionAndOrientation(obj["id"])
        initial_poses[name]["position"] = list(pos)
        initial_poses[name]["orientation"] = list(orn)

    if OBJECT_RANDOMIZE_ON_LOAD:
        reset_objects_randomized(objects, initial_poses, reason="load")

    # Robot
    robot_data = load_jaco_robot()
    register_mobile_platform_body(robot_data["robot_id"])
    reset_robot(robot_data)

    # Set konteks awal kamera POV wrist.
    # Tidak membuat model kamera fisik di simulasi.
    set_live_camera_context(
        robot_data=robot_data,
        objects=objects,
        selected_name=None
    )

    # Kamera GUI PyBullet
    p.resetDebugVisualizerCamera(
        cameraDistance=1.35,
        cameraYaw=35,
        cameraPitch=-22,
        cameraTargetPosition=[0.05, 0.00, 0.78]
    )

    # Teks di scene
    p.addUserDebugText(
        "Scriptis 4.0 - Store Shelf Scenario | ASR Manual-Gated Visual Search",
        textPosition=[-0.85, -0.65, 1.85],
        textColorRGB=[1, 1, 1],
        textSize=1.4
    )

    p.addUserDebugText(
        "ASR default OFF | prompt safe | visual search first | q quit",
        textPosition=[-0.85, -0.65, 1.70],
        textColorRGB=[1, 0.2, 0.2],
        textSize=1.2
    )

    object_names = list(objects.keys())
    selected_idx = 0
    selected_name = object_names[selected_idx]

    print("\n=== KONTROL ===")
    print("n : pilih objek berikutnya")
    print("p : pilih objek sebelumnya")
    print("w/s : gerak objek pada sumbu X")
    print("a/d : gerak objek pada sumbu Y")
    print("z/x : gerak objek pada sumbu Z")
    print("r : reset robot custom idle + reset semua objek")
    print("m : kembali ke custom idle/home pose")
    print("g : gerakkan robot ke objek aktif")
    print("e : collision/proximity snapshot robot")
    print("f : toggle collision verbose saat robot bergerak")
    print("h : pick-lift-handover objek aktif")
    print("AUTO ASR : DEFAULT OFF, ketik asr run untuk jalankan JSON sekali atau asr on untuk auto command baru")
    print("ASR flow : ASR -> shape/color -> visual search camera -> strict match -> pick")
    print("ASR controls terminal : asr run / asr on / asr off / asr clear / asr status")
    print("camera : POV end-effector/wrist ditampilkan, view-only, tanpa model kamera fisik")
    print("visual search : strict match aktif, objek warna sama tapi bentuk salah akan ditolak")
    print("speed : E2E dipercepat, extra release delay dimatikan, wait-stop tuning aktif")
    print("edge reach tuning : aktif untuk objek ujung kiri/kanan rak")
    print("grasp/drop : fix5 release guard aktif, object tidak dilepas sebelum di atas basket")
    print("manual release : tekan o atau ketik lepas/release/drop di terminal")
    print("camera tuning : disabled, kamera hanya visual/view-only")
    print("t : taruh objek aktif langsung ke basket")
    print("v : manual fallback baca ASR JSON, tetap lewat visual search")
    print("q : keluar")
    print(f"Objek aktif awal: {selected_name}")
    print("ASR DEFAULT MODE: OFF / manual-gated. Simulasi tidak akan auto ambil dari latest_command.json.")
    print("Gunakan: asr run = jalankan JSON sekali, asr on = auto untuk command baru, asr off = prompt manual.")

    if VISUAL_SEARCH_TERMINAL_MODE:
        start_terminal_input_thread_once()

    # OpenCV windows
    show_depth = False
    show_segmentation = False

    if SHOW_CAMERA_WINDOWS:
        safe_named_window(CAMERA_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(CAMERA_WINDOW_NAME, 560, 420) 

    try:
        while True:
            # Step beberapa kali supaya physics lebih stabil
            for _ in range(4):
                p.stepSimulation()

            set_live_camera_context(
                robot_data=robot_data,
                objects=objects,
                selected_name=selected_name
            )

            terminal_result = process_terminal_command_queue(
                robot_data=robot_data,
                objects=objects,
                initial_poses=initial_poses,
                current_selected_name=selected_name
            )

            selected_name = terminal_result.get("selected_name", selected_name)
            if selected_name in object_names:
                selected_idx = object_names.index(selected_name)

            if terminal_result.get("quit", False):
                print("Keluar dari simulasi via terminal command.")
                break

            # AUTO ASR:
            # realtime_asr_command.py menulis latest_command.json.
            # Main simulasi membaca otomatis file itu dan menjalankan visual search.
            auto_asr_result = poll_asr_json_and_run_visual_search(
                robot_data=robot_data,
                objects=objects,
                current_selected_name=selected_name
            )

            selected_name = auto_asr_result.get("selected_name", selected_name)
            if selected_name in object_names:
                selected_idx = object_names.index(selected_name)

            if SHOW_CAMERA_WINDOWS:
                rgb, depth, seg_body = capture_synthetic_camera(
                    width=960,
                    height=720,
                    robot_data=robot_data,
                    objects=objects,
                    selected_name=selected_name
                )

                if CAMERA_DRAW_OBJECT_LABELS:
                    rgb_annotated = draw_object_centroids(
                        rgb=rgb,
                        seg_body=seg_body,
                        objects=objects,
                        selected_name=selected_name
                    )
                else:
                    rgb_annotated = rgb.copy()

                depth_vis = make_depth_visual(depth)
                seg_vis = make_segmentation_visual(seg_body, objects)

                if CAMERA_DRAW_SELECTED_TEXT:
                    cv2.putText(
                        rgb_annotated,
                        f"Selected: {selected_name}",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 255),
                        2
                    )

                rgb_bgr = cv2.cvtColor(rgb_annotated, cv2.COLOR_RGB2BGR)
                cv2.imshow(CAMERA_WINDOW_NAME, rgb_bgr)

                if show_depth:
                    cv2.namedWindow("Depth Image", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Depth Image", 520, 390)
                    cv2.imshow("Depth Image", depth_vis)
                else:
                    try:
                        cv2.destroyWindow("Depth Image")
                    except Exception:
                        pass

                if show_segmentation:
                    cv2.namedWindow("Segmentation Mask", cv2.WINDOW_NORMAL)
                    cv2.resizeWindow("Segmentation Mask", 520, 390)
                    cv2.imshow("Segmentation Mask", seg_vis)
                else:
                    try:
                        cv2.destroyWindow("Segmentation Mask")
                    except Exception:
                        pass

            key = get_keyboard_key_optional()

            if key == ord('n'):
                selected_idx = (selected_idx + 1) % len(object_names)
                selected_name = object_names[selected_idx]
                set_live_camera_context(selected_name=selected_name)
                print(f"Objek aktif: {selected_name}")

            elif key == ord('p'):
                selected_idx = (selected_idx - 1) % len(object_names)
                selected_name = object_names[selected_idx]
                set_live_camera_context(selected_name=selected_name)
                print(f"Objek aktif: {selected_name}")

            elif key == ord('e'):
                print_collision_snapshot(
                    robot_data=robot_data,
                    objects=objects,
                    selected_name=selected_name
                )

            elif key == ord('f'):
                toggle_collision_verbose()

            elif False and tune_wrist_pov_camera_from_key(
                key=key,
                robot_data=robot_data,
                objects=objects,
                selected_name=selected_name
            ):
                print("Wrist POV camera tuning input processed.")

            elif key in [ord('w'), ord('s'), ord('a'), ord('d'), ord('z'), ord('x')]:
                selected_id = objects[selected_name]["id"]
                move_selected_object(selected_id, key, step=0.03)

            elif key == ord('g'):
                set_live_camera_context(robot_data=robot_data, objects=objects, selected_name=selected_name)
                move_robot_to_selected_object(
                    robot_data=robot_data,
                    objects=objects,
                    selected_name=selected_name
                )

            elif key == ord('h'):
                set_live_camera_context(robot_data=robot_data, objects=objects, selected_name=selected_name)
                pick_lift_handover_selected_object(
                    robot_data=robot_data,
                    objects=objects,
                    selected_name=selected_name
                )

            elif key == MANUAL_RELEASE_KEY:
                manual_release_active_grasp(
                    robot_data=robot_data,
                    reason="main loop key 'o'"
                )

            elif key == ord('t'):
                print("Trigger manual: taruh objek aktif langsung ke handover basket.")
                place_selected_object_in_handover_basket(objects, selected_name)
            elif key == ord('v'):
                # Manual fallback:
                # tombol v tetap boleh dipakai, tetapi sekarang juga lewat VISUAL SEARCH.
                # Tidak lagi langsung memilih object dari metadata.
                command = read_latest_asr_command(ASR_COMMAND_JSON)
                detected_name, status = run_asr_command_via_visual_search(
                    robot_data=robot_data,
                    objects=objects,
                    command=command
                )

                print("ASR manual visual-search status:", status)

                if detected_name is not None:
                    selected_name = detected_name
                    selected_idx = object_names.index(selected_name)
                    set_live_camera_context(
                        robot_data=robot_data,
                        objects=objects,
                        selected_name=selected_name
                    )
                else:
                    print("Robot tidak bergerak karena ASR tidak valid atau target tidak ditemukan secara visual.")

           
                print("Trigger manual: kembali ke home pose.")
                move_robot_to_home_pose(robot_data, duration=HOME_RETURN_DURATION)



            elif key == ord('r'):
                print("\n=== KEYBOARD RESET + RANDOMIZE ===")

                reset_robot(robot_data)

                if OBJECT_RANDOMIZE_ON_RESET:
                    reset_objects_randomized(
                        objects=objects,
                        initial_poses=initial_poses,
                        reason="keyboard_r_reset"
                    )

                    print("Robot reset.")
                    print("Object placement RANDOMIZED.")
                else:
                    reset_objects(
                        objects=objects,
                        initial_poses=initial_poses
                    )

                    print("Robot dan object kembali ke posisi original.")

                set_live_camera_context(
                    robot_data=robot_data,
                    objects=objects,
                    selected_name=selected_name
                )

            elif key == ord('m'):
                print("Trigger manual: kembali ke home pose.")
                move_robot_to_home_pose(
                    robot_data,
                    duration=HOME_RETURN_DURATION
                )

            elif key == ord('q'):
                print("Keluar dari simulasi.")
                break

            elif key == ord('c'):
                show_depth = not show_depth
                if not show_depth:
                    cv2.destroyWindow("Depth Image")
                print("Depth Image:", "ON" if show_depth else "OFF")

            elif key == ord('b'):
                show_segmentation = not show_segmentation
                if not show_segmentation:
                    cv2.destroyWindow("Segmentation Mask")
                print("Segmentation Mask:", "ON" if show_segmentation else "OFF")
            time.sleep(1 / 240.0)

    except KeyboardInterrupt:
        print("\nSimulasi dihentikan oleh user.")

    finally:
        cv2.destroyAllWindows()
        p.disconnect()
        print("PyBullet disconnected.")


if __name__ == "__main__":
    main()

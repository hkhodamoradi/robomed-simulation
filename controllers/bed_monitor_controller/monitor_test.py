# monitor_test.py
# Vital monitor Television controller
# - Uses the Television Display device
# - Waits briefly for the shared demo case json written by unified_controller_V3
# - Loads assets from normal/ or emergency/
# - Resizes static images and vital animation frames
# - Loop: scan -> vitals -> qr_dark -> qr_only -> repeat

from controller import Robot
from PIL import Image
import os
import json
import time


# ------------------------
# User timing (seconds)
# ------------------------
T_SCAN = 3.0
T_VITALS = 6.0
T_QR_DARK = 4.0
T_QR_ONLY = 6.0

VITAL_FRAMES = 24
ALLOWED_CASES = {"normal", "emergency"}

# Wait for shared case file from unified controller
CASE_WAIT_SECONDS = 3.0
CASE_POLL_SECONDS = 0.1


# ------------------------
# Helpers
# ------------------------
def find_display(robot):
    for name in ["display", "screen", "Display"]:
        try:
            dev = robot.getDevice(name)
            if dev:
                print(f"[monitor_controller] Using Display device: {name}")
                return dev
        except Exception:
            pass
    return None


def prepare_resized_png(src_path, out_path, w, h):
    image = Image.open(src_path).convert("RGB")
    image = image.resize((w, h), Image.Resampling.LANCZOS)
    image.save(out_path)


def wait_and_load_demo_case_from_shared_file(base_dir):
    shared_file = os.path.normpath(
        os.path.join(base_dir, "..", "unified_controller_V3", "runtime", "monitor_case.json")
    )

    print(f"[monitor_controller] Looking for shared case file: {shared_file}")

    deadline = time.time() + CASE_WAIT_SECONDS
    controller_start_time = time.time()

    while time.time() < deadline:
        if os.path.exists(shared_file):
            try:
                mtime = os.path.getmtime(shared_file)

                # Ignore stale file from previous run
                if mtime < (controller_start_time - 0.5):
                    time.sleep(CASE_POLL_SECONDS)
                    continue

                with open(shared_file, "r", encoding="utf-8") as f:
                    payload = json.load(f)

                case_name = str(payload.get("demo_patient_case", "normal")).strip().lower()

                if case_name not in ALLOWED_CASES:
                    print(f"[monitor_controller] WARNING: invalid case '{case_name}', using 'normal'")
                    return "normal"

                print(f"[monitor_controller] Fresh shared case file accepted: {case_name}")
                return case_name

            except Exception as exc:
                print(f"[monitor_controller] WARNING: shared case file exists but is not ready yet: {exc}")

        time.sleep(CASE_POLL_SECONDS)

    print("[monitor_controller] WARNING: fresh shared case file not found in time, using 'normal'")
    return "normal"


def resolve_case_folder(base_dir):
    case_name = wait_and_load_demo_case_from_shared_file(base_dir)
    case_dir = os.path.join(base_dir, case_name)

    if not os.path.isdir(case_dir):
        print(f"[monitor_controller] ERROR: case folder not found: {case_dir}")
        raise SystemExit(1)

    print(f"[monitor_controller] Selected case: {case_name}")
    print(f"[monitor_controller] Using assets from: {case_dir}")
    return case_name, case_dir


def validate_required_files(paths):
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        for path in missing:
            print(f"[monitor_controller] ERROR: missing file: {path}")
        raise SystemExit(1)


def safe_image_load(display, path):
    img = display.imageLoad(path)
    if img is None:
        print(f"[monitor_controller] ERROR: failed to load image into Webots Display: {path}")
        raise SystemExit(1)
    return img


# ------------------------
# Init
# ------------------------
robot = Robot()
ts = int(robot.getBasicTimeStep())

display = find_display(robot)
if display is None:
    print("[monitor_controller] ERROR: No Display device found")
    raise SystemExit(1)

DW, DH = display.getWidth(), display.getHeight()
print(f"[monitor_controller] Display resolution: {DW} x {DH}")

BASE = os.path.dirname(os.path.abspath(__file__))
CASE_NAME, CASE_DIR = resolve_case_folder(BASE)


# ------------------------
# Source files
# ------------------------
SRC_SCAN = os.path.join(CASE_DIR, "scanning_patient.png")
SRC_QR_DARK = os.path.join(CASE_DIR, "qr_dark_screen.png")
SRC_QR_ONLY = os.path.join(CASE_DIR, "qr_only_screen.png")

SRC_VITALS = [
    os.path.join(CASE_DIR, f"vitals_{i:03d}.png")
    for i in range(VITAL_FRAMES)
]

validate_required_files([SRC_SCAN, SRC_QR_DARK, SRC_QR_ONLY] + SRC_VITALS)


# ------------------------
# Resize static images
# ------------------------
OUT_SCAN = os.path.join(CASE_DIR, f"_scan_{DW}x{DH}.png")
OUT_QR_DARK = os.path.join(CASE_DIR, f"_qr_dark_{DW}x{DH}.png")
OUT_QR_ONLY = os.path.join(CASE_DIR, f"_qr_only_{DW}x{DH}.png")

prepare_resized_png(SRC_SCAN, OUT_SCAN, DW, DH)
prepare_resized_png(SRC_QR_DARK, OUT_QR_DARK, DW, DH)
prepare_resized_png(SRC_QR_ONLY, OUT_QR_ONLY, DW, DH)


# ------------------------
# Resize vital frames
# ------------------------
OUT_VITALS = []
for i, src in enumerate(SRC_VITALS):
    out = os.path.join(CASE_DIR, f"_vitals_{i:03d}_{DW}x{DH}.png")
    prepare_resized_png(src, out, DW, DH)
    OUT_VITALS.append(out)


# ------------------------
# Load images
# ------------------------
img_scan = safe_image_load(display, OUT_SCAN)
img_qr_dark = safe_image_load(display, OUT_QR_DARK)
img_qr_only = safe_image_load(display, OUT_QR_ONLY)
img_vitals = [safe_image_load(display, path) for path in OUT_VITALS]


# ------------------------
# Display helper
# ------------------------
def show(img_id):
    display.setColor(0x000000)
    display.fillRectangle(0, 0, DW, DH)
    display.imagePaste(img_id, 0, 0, False)


# ------------------------
# Main loop
# ------------------------
start = robot.getTime()
cycle = T_SCAN + T_VITALS + T_QR_DARK + T_QR_ONLY

print("[monitor_controller] Running loop: scan -> vitals -> qr_dark -> qr_only -> repeat")

while robot.step(ts) != -1:
    t = robot.getTime() - start
    tc = t % cycle

    if tc < T_SCAN:
        show(img_scan)

    elif tc < (T_SCAN + T_VITALS):
        vit_t = tc - T_SCAN
        frame = int((vit_t / T_VITALS) * VITAL_FRAMES)
        frame = max(0, min(frame, VITAL_FRAMES - 1))
        show(img_vitals[frame])

    elif tc < (T_SCAN + T_VITALS + T_QR_DARK):
        show(img_qr_dark)

    else:
        show(img_qr_only)
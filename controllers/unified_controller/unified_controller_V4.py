from controller import Robot
import math
import statistics
import os
import json
from datetime import datetime
from enum import Enum, auto
from dataclasses import dataclass

import cv2
import numpy as np

from speech_announcer import SpeechAnnouncer
from qr_decoder import decode_qr_from_bgr
from demo_patient_status import get_demo_patient_assessment

# =========================================================
# SHARED CONFIG
# =========================================================
TIME_STEP = 32

# Demo timing / speech
STARTUP_WAIT_SECONDS = 4.0
ENABLE_SPEECH = True
SPEECH_RATE = 160

# Spoken pauses
PAUSE_ENTER_ROOM = 1.6
PAUSE_PATIENT_FOUND = 1.2
PAUSE_APPROACHING = 0.9
PAUSE_INSPECTING = 2.5
PAUSE_BED_RESULT = 1.6
PAUSE_MONITOR_FOUND = 1.0
PAUSE_WAIT_QR = 0.8
PAUSE_QR_CAPTURED = 1.5
PAUSE_PROCESSING = 2.0
PAUSE_DONE = 1.0

# Demo enrichment switches
ENABLE_QR_DECODE = True
ENABLE_DEMO_PATIENT_ASSESSMENT = True
CONTROLLER_NAME = "unified_controller_V3"
SOFTWARE_VERSION = "demo_v3"

# Hardcoded bed label for demo
HARD_CODED_BED_LABEL = "BED 1"

# Devices
RGB_CAMERA_NAME = "Astra rgb"
DEPTH_CAMERA_NAME = "Astra depth"
HEAD_YAW_MOTOR_NAME = "head_1_joint"
HEAD_PITCH_MOTOR_NAME = "head_2_joint"
LEFT_WHEEL_NAME = "wheel_left_joint"
RIGHT_WHEEL_NAME = "wheel_right_joint"

# Geometry
WHEEL_RADIUS = 0.0985
AXLE_LENGTH = 0.404

# Shared head limits
HEAD_YAW_MIN = -1.20
HEAD_YAW_MAX = 1.20
HEAD_PITCH_MIN = -0.95
HEAD_PITCH_MAX = 0.35

# Runtime folders
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RUNTIME_DIR = os.path.join(BASE_DIR, "runtime")
LOG_DIR = os.path.join(RUNTIME_DIR, "logs")
SNAPSHOT_DIR = os.path.join(RUNTIME_DIR, "snapshots")
EXPORT_DIR = os.path.join(RUNTIME_DIR, "exports")
SHARED_MONITOR_CASE_FILE = os.path.join(RUNTIME_DIR, "monitor_case.json")

# Demo patient case selection
DEMO_PATIENT_CASE = "normal"
# Allowed values: "normal", "emergency"

for d in [LOG_DIR, SNAPSHOT_DIR, EXPORT_DIR]:
    os.makedirs(d, exist_ok=True)


# =========================================================
# HELPERS
# =========================================================
def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def bgra_to_bgr(image_bytes, width, height):
    arr = np.frombuffer(image_bytes, dtype=np.uint8).reshape((height, width, 4))
    return arr[:, :, :3].copy()


def normalize_bed_label(value):
    if value is None:
        return None

    text = str(value).strip().upper()
    mapping = {
        "1A": "BED 1",
        "2A": "BED 2",
        "3A": "BED 3",
        "BED 1": "BED 1",
        "BED 2": "BED 2",
        "BED 3": "BED 3",
    }
    return mapping.get(text, text)


def write_monitor_case_file(case_name: str):
    payload = {
        "demo_patient_case": str(case_name).strip().lower(),
        "written_at": datetime.now().isoformat()
    }
    with open(SHARED_MONITOR_CASE_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


# =========================================================
# MISSION STATE
# =========================================================
class MissionState(Enum):
    INIT = auto()
    FIND_PATIENT = auto()
    READ_BED_LABEL = auto()
    FIND_MONITOR = auto()
    SAVE_RESULTS = auto()
    DONE = auto()


# =========================================================
# RUNTIME LOGGER
# =========================================================
class MissionLogger:
    def __init__(self):
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.run_id = f"mission_{stamp}"
        self.text_log_path = os.path.join(LOG_DIR, f"{self.run_id}.log")
        self.summary_path = os.path.join(EXPORT_DIR, f"{self.run_id}_summary.json")

        self.summary = {
            "run_id": self.run_id,
            "controller_name": CONTROLLER_NAME,
            "software_version": SOFTWARE_VERSION,
            "start_time": datetime.now().isoformat(),
            "end_time": None,
            "start_sim_time_s": 0.0,
            "end_sim_time_s": None,
            "mission_duration_s": None,
            "result": "IN_PROGRESS",
            "basic_time_step_ms": TIME_STEP,
            "patient_found": False,
            "patient_centered": False,
            "approach_done": False,
            "bed_label": None,
            "bed_label_source": "hardcoded_demo",
            "monitor_found": False,
            "monitor_snapshot": None,
            "full_snapshot": None,
            "qr_decode_success": False,
            "qr_payload": None,
            "qr_points": None,
            "qr_source_image": None,
            "qr_decoder": None,
            "qr_payload_json": None,
            "qr_patient_id": None,
            "qr_bed_raw": None,
            "qr_bed": None,
            "qr_vitals": None,
            "qr_timestamp": None,
            "profile_name": None,
            "urgency_level": None,
            "patient_assessment": None,
            "human_review_required": False,
            "status_light": "green",
            "mission_status_text": "NORMAL",
            "states": [],
            "notes": [],
        }

        self.log(0.0, f"[MISSION_START] run_id={self.run_id} summary_path={self.summary_path}")

    def log(self, sim_time, msg):
        wall_time = datetime.now().isoformat(timespec="milliseconds")
        line = f"[sim={sim_time:8.3f}s | wall={wall_time}] {msg}"
        print(line)
        with open(self.text_log_path, "a", encoding="utf-8") as f:
            f.write(line + "\n")

    def transition(self, sim_time, old_state, new_state, reason=""):
        self.summary["states"].append({
            "sim_time_s": sim_time,
            "wall_time": datetime.now().isoformat(),
            "from": old_state.name if hasattr(old_state, "name") else str(old_state),
            "to": new_state.name if hasattr(new_state, "name") else str(new_state),
            "reason": reason,
        })
        self.log(
            sim_time,
            f"[MISSION] {old_state.name if hasattr(old_state, 'name') else old_state} -> "
            f"{new_state.name if hasattr(new_state, 'name') else new_state} ({reason})"
        )

    def save_summary(self, sim_time):
        self.summary["end_time"] = datetime.now().isoformat()
        self.summary["end_sim_time_s"] = sim_time
        self.summary["mission_duration_s"] = sim_time - float(self.summary.get("start_sim_time_s", 0.0))

        self.log(
            sim_time,
            f"[MISSION_END] run_id={self.run_id} result={self.summary['result']} "
            f"duration_s={self.summary['mission_duration_s']:.3f}"
        )

        with open(self.summary_path, "w", encoding="utf-8") as f:
            json.dump(self.summary, f, indent=2)

        self.log(sim_time, f"[SAVE] summary json: {self.summary_path}")


# =========================================================
# SHARED ROBOT DEVICES
# =========================================================
class RobotContext:
    def __init__(self, robot: Robot):
        self.robot = robot
        self.ts = int(robot.getBasicTimeStep()) or TIME_STEP

        self.left_motor = robot.getDevice(LEFT_WHEEL_NAME)
        self.right_motor = robot.getDevice(RIGHT_WHEEL_NAME)
        self.head_yaw = robot.getDevice(HEAD_YAW_MOTOR_NAME)
        self.head_pitch = robot.getDevice(HEAD_PITCH_MOTOR_NAME)
        self.cam = robot.getDevice(RGB_CAMERA_NAME)
        self.depth = robot.getDevice(DEPTH_CAMERA_NAME)

        self.left_motor.setPosition(float("inf"))
        self.right_motor.setPosition(float("inf"))
        self.left_motor.setVelocity(0.0)
        self.right_motor.setVelocity(0.0)

        self.cam.enable(self.ts)
        if self.depth is not None:
            self.depth.enable(self.ts)

        self.width = self.cam.getWidth()
        self.height = self.cam.getHeight()

        self.head_cmd_yaw = 0.0
        self.head_cmd_pitch = 0.0

    def drive(self, v, w):
        omega_l = (v - 0.5 * AXLE_LENGTH * w) / WHEEL_RADIUS
        omega_r = (v + 0.5 * AXLE_LENGTH * w) / WHEEL_RADIUS
        self.left_motor.setVelocity(omega_l)
        self.right_motor.setVelocity(omega_r)

    def stop_base(self):
        self.drive(0.0, 0.0)

    def set_head(self, yaw, pitch):
        yaw = clamp(yaw, HEAD_YAW_MIN, HEAD_YAW_MAX)
        pitch = clamp(pitch, HEAD_PITCH_MIN, HEAD_PITCH_MAX)
        self.head_cmd_yaw = yaw
        self.head_cmd_pitch = pitch
        self.head_yaw.setPosition(yaw)
        self.head_pitch.setPosition(pitch)

    def move_head_smooth(self, target_yaw, target_pitch, max_yaw_step, max_pitch_step):
        dy = clamp(target_yaw - self.head_cmd_yaw, -max_yaw_step, max_yaw_step)
        dp = clamp(target_pitch - self.head_cmd_pitch, -max_pitch_step, max_pitch_step)
        self.set_head(self.head_cmd_yaw + dy, self.head_cmd_pitch + dp)

    def rgb_image(self):
        return self.cam.getImage()

    def depth_image(self):
        return self.depth.getRangeImage() if self.depth is not None else None

    def frame_bgr(self):
        img = self.rgb_image()
        if img is None:
            return None
        return bgra_to_bgr(img, self.width, self.height)

    def save_snapshot(self, logger: MissionLogger, name, img):
        sim_time = self.robot.getTime()
        stem, ext = os.path.splitext(name)
        stamped_name = f"{stem}_t{sim_time:07.3f}_{logger.run_id}{ext}"
        path = os.path.join(SNAPSHOT_DIR, stamped_name)
        cv2.imwrite(path, img)
        logger.log(sim_time, f"[SNAPSHOT] {path}")
        return path


# =========================================================
# STAGE 1: PATIENT FIND + SAFE APPROACH + FACE CENTER
# =========================================================
class PatientStage:
    TURN_SIGN = -1.0
    FACE_YAW_SIGN = 1.0

    SEARCH_V = 0.02
    SEARCH_W = 0.25

    K_TURN = 1.2
    W_MAX = 0.85
    CENTER_TOL = 0.08
    CENTER_HOLD = 6

    APPROACH_V = 0.10
    APPROACH_W_MAX = 0.40

    LOST_GRACE_STEPS = 18
    SAFE_DIST = 1.20
    DEPTH_MAX_VALID = 8.0

    PITCH_APPROACH = 0.0
    PITCH_FACE = -0.55
    YAW_NEUTRAL = 0.0

    SAMPLE_STRIDE = 6

    G_MIN = 120
    GR_MIN = 35
    GB_MIN = 35
    MIN_GREEN_PIXELS = 10

    MIN_FACE_PIXELS = 8

    def __init__(self, ctx: RobotContext, logger: MissionLogger):
        self.ctx = ctx
        self.logger = logger

        self.state = "SEARCH_TORSO"
        self.center_ok = 0
        self.lost_steps = 0
        self.face_setup_wait = 0
        self.done = False

        self.face_search_steps = 0
        self.MAX_FACE_SEARCH_STEPS = 180

    def detect_green_center_and_depth(self, rgb_bytes, depth_img):
        xs = 0
        cnt = 0
        depths = []

        W, H = self.ctx.width, self.ctx.height

        if self.ctx.depth is not None:
            DW = self.ctx.depth.getWidth()
            DH = self.ctx.depth.getHeight()
        else:
            DW, DH = W, H

        for y in range(0, H, self.SAMPLE_STRIDE):
            row = y * W * 4
            for x in range(0, W, self.SAMPLE_STRIDE):
                i = row + x * 4
                b = rgb_bytes[i + 0]
                g = rgb_bytes[i + 1]
                r = rgb_bytes[i + 2]

                if g >= self.G_MIN and (g - r) >= self.GR_MIN and (g - b) >= self.GB_MIN:
                    xs += x
                    cnt += 1

                    dx = int(x * DW / W)
                    dy = int(y * DH / H)
                    dx = max(0, min(DW - 1, dx))
                    dy = max(0, min(DH - 1, dy))

                    d = depth_img[dy * DW + dx]
                    if math.isfinite(d) and 0.05 < d < self.DEPTH_MAX_VALID:
                        depths.append(d)

        if cnt < self.MIN_GREEN_PIXELS:
            return False, 0.0, cnt, float("inf")

        x_mean = xs / cnt
        x_norm = (x_mean - (W / 2.0)) / (W / 2.0)
        med_d = statistics.median(depths) if len(depths) >= 6 else float("inf")

        return True, x_norm, cnt, med_d

    def detect_face_center(self, rgb_bytes):
        xs = 0
        cnt = 0
        W = self.ctx.width
        H = self.ctx.height

        for y in range(0, H, self.SAMPLE_STRIDE):
            row = y * W * 4
            for x in range(0, W, self.SAMPLE_STRIDE):
                i = row + x * 4
                b = rgb_bytes[i + 0]
                g = rgb_bytes[i + 1]
                r = rgb_bytes[i + 2]

                if (r > 170 and g > 120 and b > 95) and (r - g > 10) and (r - b > 20):
                    xs += x
                    cnt += 1

        if cnt < self.MIN_FACE_PIXELS:
            return False, 0.0, cnt

        x_mean = xs / cnt
        x_norm = (x_mean - (W / 2.0)) / (W / 2.0)
        return True, x_norm, cnt

    def step(self):
        if self.done:
            return True

        rgb = self.ctx.rgb_image()
        dep = self.ctx.depth_image()

        if rgb is None or dep is None:
            return False

        if self.state in ["SEARCH_TORSO", "CENTER_TORSO", "APPROACH_TORSO"]:
            self.ctx.set_head(self.YAW_NEUTRAL, self.PITCH_APPROACH)
            found, x, npx, dmed = self.detect_green_center_and_depth(rgb, dep)

            if not found:
                self.lost_steps += 1

                if self.lost_steps < self.LOST_GRACE_STEPS and self.state == "APPROACH_TORSO":
                    self.ctx.drive(0.06, 0.0)
                    return False

                self.state = "SEARCH_TORSO"
                self.center_ok = 0
                self.ctx.drive(self.SEARCH_V, self.SEARCH_W)
                return False

            self.lost_steps = 0

            if self.state == "SEARCH_TORSO":
                self.state = "CENTER_TORSO"
                self.center_ok = 0
                self.logger.log(
                    self.ctx.robot.getTime(),
                    f"[PATIENT][TORSO] detect green={npx} x={x:.3f} dmed={dmed:.2f}"
                )

            if self.state == "CENTER_TORSO":
                if abs(x) < self.CENTER_TOL:
                    self.center_ok += 1
                    self.ctx.stop_base()
                    if self.center_ok >= self.CENTER_HOLD:
                        self.state = "APPROACH_TORSO"
                        self.logger.log(self.ctx.robot.getTime(), "[PATIENT][TORSO] centered -> APPROACH_TORSO")
                        self.logger.summary["patient_found"] = True
                        self.logger.summary["patient_centered"] = True
                else:
                    self.center_ok = 0
                    w = clamp(self.TURN_SIGN * self.K_TURN * x, -self.W_MAX, self.W_MAX)
                    self.ctx.drive(0.0, w)
                return False

            if self.state == "APPROACH_TORSO":
                if math.isfinite(dmed) and dmed <= self.SAFE_DIST:
                    self.ctx.stop_base()
                    self.state = "FACE_SETUP"
                    self.face_setup_wait = 8
                    self.logger.log(
                        self.ctx.robot.getTime(),
                        f"[PATIENT][SAFE] reached safe distance dmed={dmed:.2f}"
                    )
                    self.logger.summary["approach_done"] = True
                    return False

                if abs(x) > 0.22:
                    self.state = "CENTER_TORSO"
                    self.center_ok = 0
                    return False

                w = clamp(self.TURN_SIGN * 0.9 * x, -self.APPROACH_W_MAX, self.APPROACH_W_MAX)
                self.ctx.drive(self.APPROACH_V, w)
                return False

        if self.state == "FACE_SETUP":
            self.ctx.set_head(0.0, self.PITCH_FACE)
            self.ctx.stop_base()
            self.face_setup_wait -= 1
            if self.face_setup_wait <= 0:
                self.state = "SEARCH_FACE"
                self.face_search_steps = 0
                self.logger.log(self.ctx.robot.getTime(), "[PATIENT][FACE] setup done -> SEARCH_FACE")
            return False

        if self.state in ["SEARCH_FACE", "CENTER_FACE"]:
            self.face_search_steps += 1
            if self.face_search_steps > self.MAX_FACE_SEARCH_STEPS:
                self.logger.log(self.ctx.robot.getTime(), "[PATIENT][FACE] timeout -> STOP")
                self.state = "STOP"
                self.done = True
                return True

            self.ctx.set_head(self.ctx.head_cmd_yaw, self.PITCH_FACE)

            found, x, npx = self.detect_face_center(rgb)
            self.logger.log(self.ctx.robot.getTime(), f"[PATIENT][FACE] tracking x={x:.3f} pixels={npx}")

            if not found:
                t = self.ctx.robot.getTime()
                yaw_cmd = 0.6 * math.sin(0.8 * t)
                self.ctx.set_head(yaw_cmd, self.PITCH_FACE)
                self.ctx.stop_base()
                return False

            if self.state == "SEARCH_FACE":
                self.state = "CENTER_FACE"
                self.center_ok = 0
                self.logger.log(self.ctx.robot.getTime(), f"[PATIENT][FACE] detect n={npx} x={x:.3f}")

            if abs(x) < 0.20:
                self.center_ok += 1
                self.ctx.stop_base()
                if self.center_ok >= 4:
                    self.state = "STOP"
                    self.done = True
                    self.logger.log(self.ctx.robot.getTime(), "[PATIENT][FACE] centered -> STOP")
            else:
                self.center_ok = 0
                yaw = clamp(self.ctx.head_cmd_yaw + self.FACE_YAW_SIGN * 0.9 * x, -0.60, 0.60)
                self.ctx.set_head(yaw, self.PITCH_FACE)
                self.ctx.stop_base()

            return False

        if self.state == "STOP":
            self.ctx.set_head(0.0, self.PITCH_FACE)
            self.ctx.stop_base()
            self.done = True
            return True

        return self.done


# =========================================================
# STAGE 2: BED LABEL
# =========================================================
class BedState(Enum):
    SEARCH = auto()
    CONFIRM = auto()
    TRACK = auto()
    READ = auto()
    HOLD_SUCCESS = auto()
    LOST = auto()


@dataclass
class BedDetection:
    valid: bool = False
    area: int = 0
    bbox: tuple = (0, 0, 0, 0)
    center: tuple = (0, 0)
    fill_ratio: float = 0.0
    aspect_ratio: float = 0.0
    solidity: float = 0.0


class BedLabelStage:
    CONTROL_DT_MS = 32
    DEBUG_PRINT_EVERY_N_FRAMES = 8

    BLUE_LOWER = np.array([92, 100, 45], dtype=np.uint8)
    BLUE_UPPER = np.array([132, 255, 255], dtype=np.uint8)

    DETECTION_ROI_X_MIN = 0.20
    DETECTION_ROI_X_MAX = 0.85
    DETECTION_ROI_Y_MIN = 0.45
    DETECTION_ROI_Y_MAX = 0.98

    MORPH_OPEN_KERNEL = 3
    MORPH_CLOSE_KERNEL = 5

    MIN_LABEL_AREA_PX = 1300
    MIN_LABEL_W_PX = 42
    MIN_LABEL_H_PX = 24
    MIN_LABEL_FILL_RATIO = 0.45
    MIN_LABEL_ASPECT = 0.8
    MAX_LABEL_ASPECT = 4.2
    MIN_LABEL_SOLIDITY = 0.80

    CONFIRM_REQUIRED_FRAMES = 2
    READ_STABLE_REQUIRED_FRAMES = 6
    MAX_MISSED_IN_CONFIRM = 3
    MAX_MISSED_IN_TRACK = 10

    REACQUIRE_YAW = 0.0
    REACQUIRE_PITCH = -0.45

    SEARCH_YAW_MIN = -0.65
    SEARCH_YAW_MAX = 0.65
    SEARCH_YAW_STEP = 0.006
    SEARCH_PITCH_BASE = -0.48
    SEARCH_PITCH_AMPLITUDE = 0.02
    SEARCH_PITCH_PERIOD_FRAMES = 120

    TARGET_ZONE_X_MIN = 0.38
    TARGET_ZONE_X_MAX = 0.65
    TARGET_ZONE_Y_MIN = 0.60
    TARGET_ZONE_Y_MAX = 0.90

    DEADBAND_PX_X = 10
    DEADBAND_PX_Y = 10

    MAX_YAW_STEP = 0.028
    MAX_PITCH_STEP = 0.018
    YAW_GAIN = 0.00045
    PITCH_GAIN = 0.00035
    YAW_SIGN = 1.0
    PITCH_SIGN = 1.0

    SAVE_DEBUG_OVERLAY = False
    DEBUG_OVERLAY_EVERY_N_FRAMES = 10

    def __init__(self, ctx: RobotContext, logger: MissionLogger):
        self.ctx = ctx
        self.logger = logger

        self.frame_count = 0
        self.state = BedState.SEARCH
        self.confirm_count = 0
        self.stable_track_count = 0
        self.missed_count = 0
        self.last_detection = BedDetection()

        self.search_yaw = self.REACQUIRE_YAW
        self.search_dir = 1
        self.done = False

        self.ctx.set_head(self.REACQUIRE_YAW, self.REACQUIRE_PITCH)
        self.logger.log(self.ctx.robot.getTime(), "[BED] initialized")

    @staticmethod
    def roi_bounds(width, height):
        x0 = int(BedLabelStage.DETECTION_ROI_X_MIN * width)
        x1 = int(BedLabelStage.DETECTION_ROI_X_MAX * width)
        y0 = int(BedLabelStage.DETECTION_ROI_Y_MIN * height)
        y1 = int(BedLabelStage.DETECTION_ROI_Y_MAX * height)
        x0 = max(0, min(width - 1, x0))
        x1 = max(x0 + 1, min(width, x1))
        y0 = max(0, min(height - 1, y0))
        y1 = max(y0 + 1, min(height, y1))
        return x0, y0, x1, y1

    def detect_blue_label(self, bgr):
        h, w = bgr.shape[:2]
        rx0, ry0, rx1, ry1 = self.roi_bounds(w, h)
        roi = bgr[ry0:ry1, rx0:rx1]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        mask_roi = cv2.inRange(hsv, self.BLUE_LOWER, self.BLUE_UPPER)

        k_open = np.ones((self.MORPH_OPEN_KERNEL, self.MORPH_OPEN_KERNEL), np.uint8)
        k_close = np.ones((self.MORPH_CLOSE_KERNEL, self.MORPH_CLOSE_KERNEL), np.uint8)
        mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_OPEN, k_open)
        mask_roi = cv2.morphologyEx(mask_roi, cv2.MORPH_CLOSE, k_close)

        contours, _ = cv2.findContours(mask_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best = BedDetection(valid=False)

        for c in contours:
            area = int(cv2.contourArea(c))
            if area < self.MIN_LABEL_AREA_PX:
                continue

            x, y, bw, bh = cv2.boundingRect(c)
            if bw < self.MIN_LABEL_W_PX or bh < self.MIN_LABEL_H_PX:
                continue

            aspect = bw / float(max(bh, 1))
            if not (self.MIN_LABEL_ASPECT <= aspect <= self.MAX_LABEL_ASPECT):
                continue

            bbox_area = float(max(bw * bh, 1))
            fill_ratio = area / bbox_area
            if fill_ratio < self.MIN_LABEL_FILL_RATIO:
                continue

            hull = cv2.convexHull(c)
            hull_area = float(max(cv2.contourArea(hull), 1.0))
            solidity = area / hull_area
            if solidity < self.MIN_LABEL_SOLIDITY:
                continue

            fx, fy = x + rx0, y + ry0
            cx = fx + bw // 2
            cy = fy + bh // 2

            score = area * (0.6 + 0.2 * fill_ratio + 0.2 * solidity)
            best_score = (
                best.area * (0.6 + 0.2 * best.fill_ratio + 0.2 * best.solidity)
                if best.valid else -1
            )

            if score > best_score:
                best = BedDetection(
                    valid=True,
                    area=area,
                    bbox=(fx, fy, bw, bh),
                    center=(cx, cy),
                    fill_ratio=float(fill_ratio),
                    aspect_ratio=float(aspect),
                    solidity=float(solidity),
                )

        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[ry0:ry1, rx0:rx1] = mask_roi

        return best, full_mask, (rx0, ry0, rx1, ry1)

    def in_target_zone(self, det, width, height):
        cx, cy = det.center
        x_min = int(self.TARGET_ZONE_X_MIN * width)
        x_max = int(self.TARGET_ZONE_X_MAX * width)
        y_min = int(self.TARGET_ZONE_Y_MIN * height)
        y_max = int(self.TARGET_ZONE_Y_MAX * height)
        return (x_min <= cx <= x_max) and (y_min <= cy <= y_max)

    def track_adjust(self, det, width, height):
        cx, cy = det.center

        zone_x_min = int(self.TARGET_ZONE_X_MIN * width)
        zone_x_max = int(self.TARGET_ZONE_X_MAX * width)
        zone_y_min = int(self.TARGET_ZONE_Y_MIN * height)
        zone_y_max = int(self.TARGET_ZONE_Y_MAX * height)

        yaw_delta = 0.0
        pitch_delta = 0.0

        if cx < zone_x_min - self.DEADBAND_PX_X:
            yaw_delta = -self.YAW_SIGN * self.YAW_GAIN * (zone_x_min - cx)
        elif cx > zone_x_max + self.DEADBAND_PX_X:
            yaw_delta = self.YAW_SIGN * self.YAW_GAIN * (cx - zone_x_max)

        if cy < zone_y_min - self.DEADBAND_PX_Y:
            pitch_delta = -self.PITCH_SIGN * self.PITCH_GAIN * (zone_y_min - cy)
        elif cy > zone_y_max + self.DEADBAND_PX_Y:
            pitch_delta = self.PITCH_SIGN * self.PITCH_GAIN * (cy - zone_y_max)

        self.ctx.move_head_smooth(
            self.ctx.head_cmd_yaw + yaw_delta,
            self.ctx.head_cmd_pitch + pitch_delta,
            self.MAX_YAW_STEP,
            self.MAX_PITCH_STEP,
        )

    def search_step(self):
        self.search_yaw += self.search_dir * self.SEARCH_YAW_STEP
        if self.search_yaw >= self.SEARCH_YAW_MAX:
            self.search_yaw = self.SEARCH_YAW_MAX
            self.search_dir = -1
        elif self.search_yaw <= self.SEARCH_YAW_MIN:
            self.search_yaw = self.SEARCH_YAW_MIN
            self.search_dir = 1

        phase = (self.frame_count % self.SEARCH_PITCH_PERIOD_FRAMES) / float(max(self.SEARCH_PITCH_PERIOD_FRAMES, 1))
        pitch_wave = math.sin(2.0 * math.pi * phase)
        search_pitch = self.SEARCH_PITCH_BASE + self.SEARCH_PITCH_AMPLITUDE * pitch_wave

        self.ctx.move_head_smooth(
            self.search_yaw,
            search_pitch,
            self.MAX_YAW_STEP,
            self.MAX_PITCH_STEP,
        )

    def step(self):
        if self.done:
            return True

        self.frame_count += 1
        frame = self.ctx.frame_bgr()
        if frame is None:
            return False

        h, w = frame.shape[:2]
        det, _, roi_bounds = self.detect_blue_label(frame)
        self.last_detection = det

        if self.state == BedState.SEARCH:
            self.search_step()
            if det.valid:
                self.confirm_count = 1
                self.missed_count = 0
                self.state = BedState.CONFIRM
                self.logger.log(self.ctx.robot.getTime(), "[BED][SEARCH] candidate found")

        elif self.state == BedState.CONFIRM:
            if det.valid:
                self.confirm_count += 1
                self.missed_count = 0
                self.track_adjust(det, w, h)
                if self.confirm_count >= self.CONFIRM_REQUIRED_FRAMES:
                    self.stable_track_count = 0
                    self.state = BedState.TRACK
                    self.logger.log(self.ctx.robot.getTime(), "[BED][CONFIRM] confirmed")
            else:
                self.missed_count += 1
                if self.missed_count > self.MAX_MISSED_IN_CONFIRM:
                    self.confirm_count = 0
                    self.state = BedState.SEARCH

        elif self.state == BedState.TRACK:
            if det.valid:
                self.missed_count = 0
                self.track_adjust(det, w, h)
                if self.in_target_zone(det, w, h):
                    self.stable_track_count += 1
                else:
                    self.stable_track_count = max(0, self.stable_track_count - 1)

                if self.stable_track_count >= self.READ_STABLE_REQUIRED_FRAMES:
                    self.state = BedState.READ
            else:
                self.missed_count += 1
                if self.missed_count > self.MAX_MISSED_IN_TRACK:
                    self.state = BedState.LOST

        elif self.state == BedState.READ:
            self.logger.log(self.ctx.robot.getTime(), f"[BED][READ] label={HARD_CODED_BED_LABEL}")
            self.logger.summary["bed_label"] = HARD_CODED_BED_LABEL
            self.state = BedState.HOLD_SUCCESS

        elif self.state == BedState.HOLD_SUCCESS:
            if det.valid:
                self.track_adjust(det, w, h)
            self.done = True
            return True

        elif self.state == BedState.LOST:
            self.ctx.move_head_smooth(
                self.REACQUIRE_YAW,
                self.REACQUIRE_PITCH,
                self.MAX_YAW_STEP,
                self.MAX_PITCH_STEP,
            )
            if abs(self.ctx.head_cmd_yaw - self.REACQUIRE_YAW) < 0.03 and abs(self.ctx.head_cmd_pitch - self.REACQUIRE_PITCH) < 0.03:
                self.confirm_count = 0
                self.stable_track_count = 0
                self.missed_count = 0
                self.search_yaw = self.REACQUIRE_YAW
                self.search_dir = 1
                self.state = BedState.SEARCH

        if self.SAVE_DEBUG_OVERLAY and self.frame_count % self.DEBUG_OVERLAY_EVERY_N_FRAMES == 0:
            overlay = frame.copy()
            rx0, ry0, rx1, ry1 = roi_bounds
            cv2.rectangle(overlay, (rx0, ry0), (rx1, ry1), (255, 0, 255), 2)
            if det.valid:
                x, y, bw, bh = det.bbox
                cv2.rectangle(overlay, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
            cv2.putText(
                overlay,
                f"BED_STATE: {self.state.name}",
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            self.ctx.save_snapshot(self.logger, "bed_overlay_latest.png", overlay)

        return False


# =========================================================
# STAGE 3: TV DETECTION + SNAPSHOT
# =========================================================
class TVStageState(Enum):
    SEARCH_TV = auto()
    CONFIRM_TV = auto()
    WAIT_QR = auto()
    SAVE_TV = auto()
    DONE = auto()


class TVStage:
    MONITOR_SEARCH_YAW = 0.0
    MONITOR_SEARCH_PITCH = -0.08

    MAX_YAW_STEP = 0.012
    MAX_PITCH_STEP = 0.010

    TV_CONFIRM_REQUIRED = 4

    BASE_SEARCH_DELAY = 60
    BASE_ROT_SPEED = 0.22

    TV_ROI_X_FRAC = 0.10
    TV_ROI_Y_FRAC = 0.03
    TV_ROI_W_FRAC = 0.78
    TV_ROI_H_FRAC = 0.58

    TV_MAX_MEAN_GRAY = 95
    TV_MIN_AREA = 2500
    TV_MIN_W = 60
    TV_MIN_H = 35
    TV_MIN_ASPECT = 1.1
    TV_MAX_ASPECT = 3.2
    TV_MIN_RECTANGULARITY = 0.45

    TV_PAD_X_FRAC = 0.35
    TV_PAD_Y_FRAC = 0.30

    SAVE_DEBUG_OVERLAY = False
    DEBUG_OVERLAY_EVERY_N_FRAMES = 10

    MONITOR_T_SCAN = 3.0
    MONITOR_T_VITALS = 6.0
    MONITOR_T_QR_DARK = 5.0
    MONITOR_T_QR_ONLY = 5.0

    QR_ONLY_ENTRY_MARGIN = 0.4
    QR_ONLY_EXIT_MARGIN = 0.4
    QR_CONFIRM_REQUIRED = 6
    QR_MAX_WAIT_CYCLES = 2

    QR_CAPTURE_EVERY_N_FRAMES = 2
    QR_MAX_CANDIDATES = 12

    def __init__(self, ctx: RobotContext, logger: MissionLogger):
        self.ctx = ctx
        self.logger = logger

        self.state = TVStageState.SEARCH_TV
        self.tv_confirm_count = 0
        self.last_tv_bbox = None
        self.saved_once = False
        self.frame_count = 0
        self.search_counter = 0
        self.done = False

        self.qr_wait_counter = 0
        self.qr_confirm_counter = 0
        self.best_qr_like_crop = None
        self.qr_candidate_crops = []
        self.qr_candidate_times = []

        self.ctx.set_head(self.MONITOR_SEARCH_YAW, self.MONITOR_SEARCH_PITCH)

    def compute_tv_roi(self, width, height):
        x = int(width * self.TV_ROI_X_FRAC)
        y = int(height * self.TV_ROI_Y_FRAC)
        w = int(width * self.TV_ROI_W_FRAC)
        h = int(height * self.TV_ROI_H_FRAC)
        return x, y, w, h

    def detect_tv_frame(self, frame_bgr):
        H, W, _ = frame_bgr.shape
        rx, ry, rw, rh = self.compute_tv_roi(W, H)
        roi = frame_bgr[ry:ry + rh, rx:rx + rw]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        _, dark = cv2.threshold(gray, self.TV_MAX_MEAN_GRAY, 255, cv2.THRESH_BINARY_INV)

        kernel = np.ones((5, 5), np.uint8)
        dark = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
        dark = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        best_score = -1e9
        best_bbox = None
        best_debug = None

        roi_cx = rw / 2.0
        roi_cy = rh / 2.0

        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            area = w * h

            if area < self.TV_MIN_AREA:
                continue
            if w < self.TV_MIN_W or h < self.TV_MIN_H:
                continue

            aspect = w / float(h + 1e-6)
            if aspect < self.TV_MIN_ASPECT or aspect > self.TV_MAX_ASPECT:
                continue

            patch = gray[y:y + h, x:x + w]
            mean_gray = float(np.mean(patch))
            if mean_gray > self.TV_MAX_MEAN_GRAY:
                continue

            contour_area = cv2.contourArea(cnt)
            rectangularity = contour_area / float(area + 1e-6)
            if rectangularity < self.TV_MIN_RECTANGULARITY:
                continue

            cx = x + w / 2.0
            cy = y + h / 2.0
            dist = math.hypot(cx - roi_cx, cy - roi_cy)

            score = (
                0.002 * area
                + 1.5 * rectangularity
                - 0.008 * dist
                - 0.012 * mean_gray
            )

            if score > best_score:
                best_score = score
                best_bbox = (rx + x, ry + y, w, h)
                best_debug = {
                    "area": area,
                    "aspect": aspect,
                    "mean_gray": mean_gray,
                    "rectangularity": rectangularity,
                    "score": score,
                }

        return best_bbox is not None, best_bbox, best_debug, (rx, ry, rw, rh)

    def expand_bbox(self, bbox, frame_shape, pad_x_frac=0.35, pad_y_frac=0.30):
        x, y, w, h = bbox
        H, W = frame_shape[:2]

        pad_x = int(w * pad_x_frac)
        pad_y = int(h * pad_y_frac)

        x0 = max(0, x - pad_x)
        y0 = max(0, y - pad_y)
        x1 = min(W, x + w + pad_x)
        y1 = min(H, y + h + pad_y)

        return (x0, y0, x1 - x0, y1 - y0)

    def crop_bbox(self, frame_bgr, bbox):
        x, y, w, h = bbox
        H, W, _ = frame_bgr.shape
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(W, x + w)
        y1 = min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            return None
        return frame_bgr[y0:y1, x0:x1].copy()

    def in_qr_only_window(self):
        cycle = (
            self.MONITOR_T_SCAN
            + self.MONITOR_T_VITALS
            + self.MONITOR_T_QR_DARK
            + self.MONITOR_T_QR_ONLY
        )

        tc = self.ctx.robot.getTime() % cycle

        qr_only_start = self.MONITOR_T_SCAN + self.MONITOR_T_VITALS + self.MONITOR_T_QR_DARK
        qr_only_end = qr_only_start + self.MONITOR_T_QR_ONLY

        start_ok = qr_only_start + self.QR_ONLY_ENTRY_MARGIN
        end_ok = qr_only_end - self.QR_ONLY_EXIT_MARGIN

        return (start_ok <= tc <= end_ok), tc

    def draw_overlay(self, frame_bgr, state, tv_roi=None, tv_bbox=None, status_text=""):
        out = frame_bgr.copy()

        if state in [TVStageState.SEARCH_TV, TVStageState.CONFIRM_TV] and tv_roi is not None:
            rx, ry, rw, rh = tv_roi
            cv2.rectangle(out, (rx, ry), (rx + rw, ry + rh), (0, 255, 0), 2)
            cv2.putText(
                out, "TV SEARCH ROI", (rx, max(20, ry - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
            )

        if tv_bbox is not None:
            x, y, w, h = tv_bbox
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cx = x + w // 2
            cy = y + h // 2
            cv2.drawMarker(
                out, (cx, cy), (0, 255, 0),
                markerType=cv2.MARKER_CROSS, markerSize=12, thickness=2
            )
            cv2.putText(
                out, "TV LOCKED", (x, max(20, y - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA
            )

        cv2.putText(
            out, f"STATE: {state.name}", (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA
        )

        if status_text:
            cv2.putText(
                out, status_text, (12, 56),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA
            )

        return out

    def _store_qr_candidate(self, crop):
        if crop is None:
            return
        if len(self.qr_candidate_crops) >= self.QR_MAX_CANDIDATES:
            return

        self.qr_candidate_crops.append(crop.copy())
        self.qr_candidate_times.append(self.ctx.robot.getTime())

    def step(self):
        if self.done:
            return True

        self.frame_count += 1
        frame_bgr = self.ctx.frame_bgr()
        if frame_bgr is None:
            return False

        tv_found, tv_bbox, _, tv_roi = self.detect_tv_frame(frame_bgr)
        status_text = ""

        if tv_found and tv_bbox is not None:
            self.last_tv_bbox = tv_bbox
            self.logger.summary["monitor_found"] = True

        locked_bbox = (
            self.expand_bbox(self.last_tv_bbox, frame_bgr.shape, self.TV_PAD_X_FRAC, self.TV_PAD_Y_FRAC)
            if self.last_tv_bbox is not None else None
        )

        if self.state == TVStageState.SEARCH_TV:
            self.search_counter += 1

            sweep = 0.25 * math.sin(self.frame_count * 0.02)
            self.ctx.move_head_smooth(
                sweep,
                self.MONITOR_SEARCH_PITCH,
                self.MAX_YAW_STEP,
                self.MAX_PITCH_STEP
            )

            if tv_found:
                self.ctx.stop_base()
                self.search_counter = 0
                self.tv_confirm_count = 1
                self.state = TVStageState.CONFIRM_TV
                self.logger.log(self.ctx.robot.getTime(), f"[TV][SEARCH] candidate found bbox={tv_bbox}")
                return False

            if self.search_counter > self.BASE_SEARCH_DELAY:
                self.ctx.drive(0.0, self.BASE_ROT_SPEED)
            else:
                self.ctx.stop_base()

        elif self.state == TVStageState.CONFIRM_TV:
            self.ctx.stop_base()

            if tv_found:
                self.tv_confirm_count += 1
                self.logger.log(
                    self.ctx.robot.getTime(),
                    f"[TV][CONFIRM] {self.tv_confirm_count}/{self.TV_CONFIRM_REQUIRED}"
                )
                if self.tv_confirm_count >= self.TV_CONFIRM_REQUIRED:
                    self.last_tv_bbox = tv_bbox
                    self.state = TVStageState.WAIT_QR
                    self.qr_wait_counter = 0
                    self.qr_confirm_counter = 0
                    self.best_qr_like_crop = None
                    self.qr_candidate_crops = []
                    self.qr_candidate_times = []
                    self.logger.log(
                        self.ctx.robot.getTime(),
                        f"[TV][CONFIRM] TV confirmed and locked bbox={self.last_tv_bbox}"
                    )
            else:
                self.tv_confirm_count = 0
                self.state = TVStageState.SEARCH_TV
                self.logger.log(self.ctx.robot.getTime(), "[TV][CONFIRM] lost candidate, back to SEARCH")

        elif self.state == TVStageState.WAIT_QR:
            self.ctx.stop_base()

            if locked_bbox is None:
                self.state = TVStageState.SEARCH_TV
                self.logger.log(self.ctx.robot.getTime(), "[TV][WAIT_QR] lost TV, back to SEARCH")
                return False

            self.qr_wait_counter += 1
            tv_crop = self.crop_bbox(frame_bgr, locked_bbox)

            in_qr_window, tc = self.in_qr_only_window()

            if in_qr_window:
                self.qr_confirm_counter += 1

                if tv_crop is not None:
                    self.best_qr_like_crop = tv_crop.copy()
                    if self.frame_count % self.QR_CAPTURE_EVERY_N_FRAMES == 0:
                        self._store_qr_candidate(tv_crop)

                status_text = f"qr_only window {self.qr_confirm_counter}/{self.QR_CONFIRM_REQUIRED}"

                if self.qr_confirm_counter == 1:
                    self.logger.log(
                        self.ctx.robot.getTime(),
                        f"[TV][WAIT_QR] entered qr_only window at cycle t={tc:.2f}s"
                    )

                if self.qr_confirm_counter >= self.QR_CONFIRM_REQUIRED:
                    self.state = TVStageState.SAVE_TV
                    self.logger.log(
                        self.ctx.robot.getTime(),
                        f"[TV][WAIT_QR] qr_only confirmed -> SAVE_TV with {len(self.qr_candidate_crops)} candidates"
                    )

            cycle = (
                self.MONITOR_T_SCAN
                + self.MONITOR_T_VITALS
                + self.MONITOR_T_QR_DARK
                + self.MONITOR_T_QR_ONLY
            )
            max_wait_frames = int((self.QR_MAX_WAIT_CYCLES * cycle * 1000.0) / TIME_STEP)

            if self.qr_wait_counter > max_wait_frames:
                self.state = TVStageState.SAVE_TV
                self.logger.log(self.ctx.robot.getTime(), "[TV][WAIT_QR] timeout after full cycles -> SAVE_TV")

        elif self.state == TVStageState.SAVE_TV:
            self.ctx.stop_base()

            if not self.saved_once and locked_bbox is not None:
                full_path = self.ctx.save_snapshot(self.logger, "full_camera_snapshot.png", frame_bgr)
                self.logger.summary["full_snapshot"] = full_path

                chosen_crop = None
                chosen_result = None
                chosen_path = None

                for i, candidate in enumerate(self.qr_candidate_crops):
                    result = decode_qr_from_bgr(candidate)
                    if result.get("qr_decode_success", False):
                        chosen_crop = candidate
                        chosen_result = result
                        chosen_path = self.ctx.save_snapshot(
                            self.logger,
                            f"tv_crop_qr_success_{i:02d}.png",
                            candidate
                        )
                        self.logger.log(
                            self.ctx.robot.getTime(),
                            f"[TV][SAVE_TV] QR success from candidate index={i}"
                        )
                        break

                if chosen_crop is None:
                    fallback_crop = self.best_qr_like_crop
                    if fallback_crop is None:
                        fallback_crop = self.crop_bbox(frame_bgr, locked_bbox)

                    if fallback_crop is not None:
                        chosen_crop = fallback_crop
                        chosen_path = self.ctx.save_snapshot(
                            self.logger,
                            "tv_crop_full_for_agent.png",
                            fallback_crop
                        )
                        chosen_result = decode_qr_from_bgr(fallback_crop)
                        self.logger.log(
                            self.ctx.robot.getTime(),
                            "[TV][SAVE_TV] fallback crop used for decoding"
                        )

                if chosen_path is not None:
                    self.logger.summary["monitor_snapshot"] = chosen_path

                if chosen_result is not None:
                    self.logger.summary["qr_decode_success"] = chosen_result.get("qr_decode_success", False)
                    self.logger.summary["qr_payload"] = chosen_result.get("qr_payload")
                    self.logger.summary["qr_points"] = chosen_result.get("qr_points")
                    self.logger.summary["qr_source_image"] = chosen_path
                    self.logger.summary["qr_decoder"] = chosen_result.get("qr_decoder")
                    self.logger.summary["qr_payload_json"] = chosen_result.get("qr_payload_json")

                self.saved_once = True
                self.done = True
                self.state = TVStageState.DONE
                return True

        elif self.state == TVStageState.DONE:
            self.ctx.stop_base()
            self.done = True
            return True

        if self.SAVE_DEBUG_OVERLAY and self.frame_count % self.DEBUG_OVERLAY_EVERY_N_FRAMES == 0:
            overlay = self.draw_overlay(
                frame_bgr,
                self.state,
                tv_roi=tv_roi,
                tv_bbox=locked_bbox,
                status_text=status_text
            )
            self.ctx.save_snapshot(self.logger, "tv_overlay_latest.png", overlay)

        return False


# =========================================================
# UNIFIED CONTROLLER
# =========================================================
class UnifiedMedRobotController:
    def __init__(self):
        self.robot = Robot()
        self.ctx = RobotContext(self.robot)
        self.logger = MissionLogger()

        write_monitor_case_file(DEMO_PATIENT_CASE)
        self.logger.log(
            self.robot.getTime(),
            f"[MONITOR_CASE] wrote shared case file: {SHARED_MONITOR_CASE_FILE} case={DEMO_PATIENT_CASE}"
        )

        self.state = MissionState.INIT
        self.patient_stage = PatientStage(self.ctx, self.logger)
        self.bed_stage = BedLabelStage(self.ctx, self.logger)
        self.tv_stage = TVStage(self.ctx, self.logger)

        self.mission_done = False

        self.speech = SpeechAnnouncer(enabled=ENABLE_SPEECH, rate=SPEECH_RATE)
        self.init_wait_steps = int((STARTUP_WAIT_SECONDS * 1000.0) / self.ctx.ts)
        self.init_announced = False
        self.init_logged = False
        self.init_speech_wait_steps = int((3.0 * 1000.0) / self.ctx.ts)

        self.pause_until_time = 0.0
        self.pending_transition = None

        self.monitor_pose_initialized = False
        self.results_enriched = False
        self.results_finalized = False

        self.prev_patient_state = self.patient_stage.state
        self.prev_bed_state = self.bed_stage.state
        self.prev_tv_state = self.tv_stage.state

        self.announced = {
            "enter_room": False,
            "search_patient": False,
            "patient_found": False,
            "approaching_patient": False,
            "inspecting_patient": False,
            "bed_check": False,
            "bed_result": False,
            "monitor_search": False,
            "monitor_found": False,
            "waiting_qr": False,
            "qr_captured": False,
            "qr_decoded": False,
            "processing_data": False,
            "mission_complete": False,
        }

    def transition(self, new_state, reason=""):
        old = self.state
        self.state = new_state
        self.monitor_pose_initialized = False

        if new_state == MissionState.SAVE_RESULTS:
            self.results_enriched = False
            self.results_finalized = False

        self.logger.transition(self.robot.getTime(), old, new_state, reason)

    def say_once(self, key, text, pause_sec=0.0, force=False):
        if not self.announced.get(key, False):
            self.speech.say(text, force=force)
            self.announced[key] = True
            if pause_sec > 0.0:
                self.pause_until_time = max(self.pause_until_time, self.robot.getTime() + pause_sec)

    def is_paused(self):
        return self.robot.getTime() < self.pause_until_time

    def queue_transition(self, new_state, reason):
        self.pending_transition = (new_state, reason)

    def flush_pending_transition_if_ready(self):
        if self.pending_transition is not None and not self.is_paused():
            new_state, reason = self.pending_transition
            self.pending_transition = None
            self.transition(new_state, reason)
            return True
        return False

    def handle_patient_stage_events(self):
        current = self.patient_stage.state
        prev = self.prev_patient_state

        if prev != current:
            self.logger.log(self.robot.getTime(), f"[PATIENT][STATE] {prev} -> {current}")

        if prev == "SEARCH_TORSO" and current == "CENTER_TORSO":
            self.say_once("patient_found", "Patient found", pause_sec=PAUSE_PATIENT_FOUND)

        if prev == "CENTER_TORSO" and current == "APPROACH_TORSO":
            self.say_once("approaching_patient", "Approaching patient", pause_sec=PAUSE_APPROACHING)

        if prev == "APPROACH_TORSO" and current == "FACE_SETUP":
            self.say_once("inspecting_patient", "Inspecting patient condition", pause_sec=PAUSE_INSPECTING)

        self.prev_patient_state = current

    def handle_bed_stage_events(self):
        current = self.bed_stage.state
        prev = self.prev_bed_state

        if prev != current:
            self.logger.log(self.robot.getTime(), f"[BED][STATE] {prev.name} -> {current.name}")

        if prev == BedState.SEARCH and current == BedState.CONFIRM:
            self.say_once("bed_check", "Checking bed number", pause_sec=2.0)

        if self.logger.summary["bed_label"] is not None and not self.announced["bed_result"]:
            label = str(self.logger.summary["bed_label"]).strip().upper()
            spoken = (
                label.replace("BED 1", "bed number one")
                     .replace("BED 2", "bed number two")
                     .replace("BED 3", "bed number three")
            )
            if spoken == label:
                spoken = label.lower()

            self.say_once(
                "bed_result",
                f"Patient on {spoken}",
                pause_sec=PAUSE_BED_RESULT
            )

        self.prev_bed_state = current

    def handle_tv_stage_events(self):
        current = self.tv_stage.state
        prev = self.prev_tv_state

        if prev != current:
            self.logger.log(self.robot.getTime(), f"[TV][STATE] {prev.name} -> {current.name}")

        if prev == TVStageState.SEARCH_TV and current == TVStageState.CONFIRM_TV:
            self.say_once("monitor_found", "Vital screen found", pause_sec=2.0)

        if prev == TVStageState.CONFIRM_TV and current == TVStageState.WAIT_QR:
            self.say_once("waiting_qr", "Waiting for Q R code", pause_sec=PAUSE_WAIT_QR)

        if prev == TVStageState.WAIT_QR and current == TVStageState.SAVE_TV:
            self.say_once("qr_captured", "Q R image captured", pause_sec=PAUSE_QR_CAPTURED)

        self.prev_tv_state = current

    def enrich_summary_with_qr_and_demo_assessment(self):
        sim_time = self.robot.getTime()

        if ENABLE_DEMO_PATIENT_ASSESSMENT:
            assessment = get_demo_patient_assessment(DEMO_PATIENT_CASE)

            self.logger.summary["patient_assessment"] = assessment
            self.logger.summary["profile_name"] = assessment.get("profile_name")
            self.logger.summary["urgency_level"] = assessment.get("urgency_level")

            self.logger.log(sim_time, f"[DEMO][PATIENT] loaded demo case={assessment['profile_name']}")
            self.logger.log(
                sim_time,
                "[DEMO][PATIENT] "
                f"face_temp={assessment['face_temperature_c']:.1f}C, "
                f"chest_temp={assessment['chest_temperature_c']:.1f}C, "
                f"resp_rate={assessment['respiration_rate_bpm']:.1f} bpm, "
                f"face_color={assessment['face_color_label']}, "
                f"urgency={assessment['urgency_level']}"
            )

            review_needed = bool(assessment.get("human_review_recommended", False))
            self.logger.summary["human_review_required"] = review_needed
            self.logger.summary["status_light"] = "red" if review_needed else "green"
            self.logger.summary["mission_status_text"] = "REVIEW REQUIRED" if review_needed else "NORMAL"

        if ENABLE_QR_DECODE:
            if self.logger.summary.get("qr_decode_success"):
                self.logger.log(sim_time, f"[QR] decoded via {self.logger.summary.get('qr_decoder')}")
                self.logger.log(sim_time, f"[QR] payload={self.logger.summary.get('qr_payload')}")
            else:
                crop_path = self.logger.summary.get("monitor_snapshot")
                if crop_path and os.path.exists(crop_path):
                    image_bgr = cv2.imread(crop_path)
                    qr_result = decode_qr_from_bgr(image_bgr)

                    self.logger.summary["qr_decode_success"] = qr_result["qr_decode_success"]
                    self.logger.summary["qr_payload"] = qr_result["qr_payload"]
                    self.logger.summary["qr_points"] = qr_result["qr_points"]
                    self.logger.summary["qr_source_image"] = crop_path
                    self.logger.summary["qr_decoder"] = qr_result["qr_decoder"]
                    self.logger.summary["qr_payload_json"] = qr_result.get("qr_payload_json")

                    if qr_result["qr_decode_success"]:
                        self.logger.log(sim_time, f"[QR] decoded via {qr_result['qr_decoder']}")
                        self.logger.log(sim_time, f"[QR] payload={qr_result['qr_payload']}")
                    else:
                        self.logger.log(sim_time, "[QR] decode failed on monitor crop")
                else:
                    self.logger.log(sim_time, "[QR] monitor crop not available for decoding")

        payload_json = self.logger.summary.get("qr_payload_json")
        if isinstance(payload_json, dict):
            self.logger.summary["qr_patient_id"] = payload_json.get("patient_id")
            raw_qr_bed = payload_json.get("bed")
            self.logger.summary["qr_bed_raw"] = raw_qr_bed
            self.logger.summary["qr_bed"] = normalize_bed_label(raw_qr_bed)
            self.logger.summary["qr_vitals"] = payload_json.get("vitals")
            self.logger.summary["qr_timestamp"] = payload_json.get("ts")

        if self.logger.summary.get("bed_label") is None and self.logger.summary.get("qr_bed") is not None:
            self.logger.summary["bed_label"] = self.logger.summary["qr_bed"]

    def step(self):
        if self.flush_pending_transition_if_ready():
            return

        if self.state == MissionState.INIT:
            if not self.init_logged:
                self.logger.log(
                    self.robot.getTime(),
                    f"[INIT] rgb={self.ctx.width}x{self.ctx.height} depth={'ok' if self.ctx.depth is not None else 'missing'}"
                )
                self.ctx.set_head(0.0, 0.0)
                self.init_logged = True

            if not self.init_announced:
                if self.init_speech_wait_steps > 0:
                    self.init_speech_wait_steps -= 1
                    self.ctx.stop_base()
                    return

                self.say_once(
                    "enter_room",
                    "Entering patient room. Initializing mission.",
                    pause_sec=PAUSE_ENTER_ROOM,
                    force=True
                )
                self.init_announced = True

            self.ctx.stop_base()

            if self.init_wait_steps > 0:
                self.init_wait_steps -= 1
                return

            if self.is_paused():
                return

            self.transition(MissionState.FIND_PATIENT, "start mission")
            return

        if self.state == MissionState.FIND_PATIENT:
            if self.is_paused():
                self.ctx.stop_base()
                return

            self.say_once("search_patient", "Searching for patient", force=True)

            done = self.patient_stage.step()
            self.handle_patient_stage_events()

            if done:
                self.pause_until_time = max(self.pause_until_time, self.robot.getTime() + 1.5)
                self.queue_transition(MissionState.READ_BED_LABEL, "patient found and approached")
            return

        if self.state == MissionState.READ_BED_LABEL:
            if self.is_paused():
                self.ctx.stop_base()
                return

            self.ctx.stop_base()
            done = self.bed_stage.step()
            self.handle_bed_stage_events()

            if done:
                self.pause_until_time = max(self.pause_until_time, self.robot.getTime() + 1.2)
                self.queue_transition(MissionState.FIND_MONITOR, "bed label read")
            return

        if self.state == MissionState.FIND_MONITOR:
            if self.is_paused():
                self.ctx.stop_base()
                return

            self.ctx.stop_base()

            if not self.monitor_pose_initialized:
                self.ctx.set_head(TVStage.MONITOR_SEARCH_YAW, TVStage.MONITOR_SEARCH_PITCH)
                self.monitor_pose_initialized = True

            self.say_once("monitor_search", "Looking for vital screen", force=True, pause_sec=2.0)

            done = self.tv_stage.step()
            self.handle_tv_stage_events()

            if done:
                self.queue_transition(MissionState.SAVE_RESULTS, "monitor snapshot saved")
            return

        if self.state == MissionState.SAVE_RESULTS:
            if self.is_paused():
                self.ctx.stop_base()
                return

            self.ctx.stop_base()

            if not self.announced["processing_data"]:
                self.say_once(
                    "processing_data",
                    "Processing patient data",
                    pause_sec=PAUSE_PROCESSING,
                    force=True
                )
                return

            if not self.results_enriched:
                self.enrich_summary_with_qr_and_demo_assessment()
                self.results_enriched = True

                if self.logger.summary.get("qr_decode_success"):
                    self.say_once(
                        "qr_decoded",
                        "Q R code decoded",
                        pause_sec=0.6,
                        force=True
                    )
                    return
                else:
                    self.announced["qr_decoded"] = True

            if self.is_paused():
                self.ctx.stop_base()
                return

            if not self.results_finalized:
                self.logger.summary["result"] = "SUCCESS"
                self.logger.save_summary(self.robot.getTime())
                self.results_finalized = True
                self.transition(MissionState.DONE, "results saved")
            return

        if self.state == MissionState.DONE:
            self.ctx.stop_base()
            self.say_once("mission_complete", "Mission complete", pause_sec=PAUSE_DONE, force=True)
            self.mission_done = True
            return


# =========================================================
# MAIN
# =========================================================
def main():
    controller = UnifiedMedRobotController()

    for _ in range(10):
        if controller.robot.step(controller.ctx.ts) == -1:
            return

    while controller.robot.step(controller.ctx.ts) != -1:
        controller.step()
        if controller.mission_done:
            break


if __name__ == "__main__":
    main()
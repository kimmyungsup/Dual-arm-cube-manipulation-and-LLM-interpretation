import argparse
import csv
import json
import os
import select
import socket
import sys
import termios
import threading
import time
import tty
from datetime import datetime
from typing import Optional
import re
import tempfile

import numpy as np
import pandas as pd
try:
    import zmq
except Exception:
    zmq = None

# =============================================================================
# Optional CubeNet integration
# =============================================================================
# This controller can optionally start CubeNet detection when scenario mode is
# entered. CubeNet is started as a background daemon thread so that the current
# keyboard UI remains responsive.
#
# Assumptions
# - cubenet_with_face_guide.py exists in the same project root and exposes:
#       main(detector_path: str, classifier_path: str)
# - assets are located relative to this file:
#       assets/detector/model.tflite
#       assets/classifier/model.knn
#
# Limitation
# - The current cubenet.py implementation does not expose a stop API, so once
#   started it continues running until its own loop exits (for example after a
#   solve or process termination).
# =============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CUBENET_DETECTOR_PATH = os.path.join(SCRIPT_DIR, "assets", "detector", "model.tflite")
CUBENET_CLASSIFIER_PATH = os.path.join(SCRIPT_DIR, "assets", "classifier", "model.knn")

cubenet_thread = None
cubenet_lock = threading.Lock()
cubenet_face_position_lock = threading.Lock()
cubenet_latest_face_position = None
cubenet_face_position_records = []

# =============================================================================
# Optional OpenAI chat-mode integration
# =============================================================================
try:
    from openai import OpenAI
except Exception:
    OpenAI = None

try:
    import sounddevice as sd
except Exception:
    sd = None

try:
    import soundfile as sf
except Exception:
    sf = None

OPENAI_MODEL = os.environ.get("OPENAI_MODEL", "gpt-5.2")
OPENAI_STT_MODEL = os.environ.get("OPENAI_STT_MODEL", "gpt-4o-mini-transcribe")
VOICE_SAMPLE_RATE = int(os.environ.get("VOICE_SAMPLE_RATE", "16000"))
VOICE_CHANNELS = int(os.environ.get("VOICE_CHANNELS", "1"))
VOICE_RECORD_SECONDS = float(os.environ.get("VOICE_RECORD_SECONDS", "4.0"))
VOICE_DEVICE = os.environ.get("VOICE_DEVICE")
VOICE_LANGUAGE_MODE = os.environ.get("VOICE_LANGUAGE_MODE", "en").lower()  # "en" | "ko" | "auto"
VOICE_CONTINUOUS_MODE = False
VOICE_CONFIRM_MODE = False
SEQUENCE_COMMAND_DELAY = float(os.environ.get("SEQUENCE_COMMAND_DELAY", "2.0"))
LLM_SPEC_PATH = os.path.join(SCRIPT_DIR, "robot_command_llm_brief_with_grasp.txt")
CHAT_CAM_PREFIX = "[cam]"
CHAT_CAM_IMAGE_PATH = os.path.join(SCRIPT_DIR, "chat_camera_latest.jpg")
CHAT_CAM_PREVIEW_WINDOW = "chat-camera-preview"



def _run_cubenet_worker(
    detector_path: str,
    classifier_path: str,
    on_face_registered=None,
    on_capture_completed=None,
):
    """Background worker that imports and runs CubeNet face-guide mode."""
    try:
        import cubenet_with_face_guide as cubenet_module
    except Exception as e:
        print(f"[ERR] failed to import cubenet_with_face_guide module: {e}")
        return

    try:
        print("[INFO] CubeNet face-guide detection thread started")
        print(f"[INFO] detector_path   = {detector_path}")
        print(f"[INFO] classifier_path = {classifier_path}")
        cubenet_module.main(
            detector_path,
            classifier_path,
            on_face_registered=on_face_registered,
            on_capture_completed=on_capture_completed,
        )
        reference_pose = getattr(
            cubenet_module,
            "CUBE_SOLVE_REFERENCE_POSE",
            "Hold the cube with WHITE center on top (U) and GREEN center facing front (F); RED is right (R).",
        )
        print(f"[INFO] CubeNet face-guide detection thread finished; solve reference pose: {reference_pose}")
    except Exception as e:
        print(f"[ERR] CubeNet face-guide runtime error: {e}")


def store_cubenet_face_position(face_position: dict):
    """Store the latest CubeNet face/cube position from scenario mode callbacks."""
    global cubenet_latest_face_position
    if not face_position:
        return

    record = dict(face_position)
    with cubenet_face_position_lock:
        cubenet_latest_face_position = record
        cubenet_face_position_records.append(record)


def get_latest_cubenet_face_position():
    """Return a copy of the latest CubeNet camera-frame face position, if any."""
    with cubenet_face_position_lock:
        if cubenet_latest_face_position is None:
            return None
        return dict(cubenet_latest_face_position)


def get_cubenet_face_position_records():
    """Return copies of all CubeNet face position records captured during this process."""
    with cubenet_face_position_lock:
        return [dict(record) for record in cubenet_face_position_records]


def format_cubenet_face_position(face_position: dict) -> str:
    if not face_position:
        return "N/A"
    return (
        f"face={face_position.get('face_name', 'N/A')} "
        f"progress={face_position.get('progress', 'N/A')}/{face_position.get('total_faces', 'N/A')} "
        "camera_xyz_m=("
        f"x={float(face_position.get('camera_x_m', 0.0)):.4f}, "
        f"y={float(face_position.get('camera_y_m', 0.0)):.4f}, "
        f"z={float(face_position.get('camera_z_m', 0.0)):.4f}"
        ") "
        "roi_px=("
        f"top={face_position.get('roi_top_px', 'N/A')}, "
        f"left={face_position.get('roi_left_px', 'N/A')}, "
        f"bottom={face_position.get('roi_bottom_px', 'N/A')}, "
        f"right={face_position.get('roi_right_px', 'N/A')}"
        ")"
    )


def start_cubenet_detection_if_needed(
    detector_path: str = CUBENET_DETECTOR_PATH,
    classifier_path: str = CUBENET_CLASSIFIER_PATH,
    on_face_registered=None,
    on_capture_completed=None,
):
    """Start CubeNet only once. If it is already running, do nothing."""
    global cubenet_thread

    with cubenet_lock:
        if cubenet_thread is not None and cubenet_thread.is_alive():
            print("[INFO] CubeNet face-guide detection is already running")
            return

        if not os.path.exists(detector_path):
            print(f"[ERR] CubeNet detector model not found: {detector_path}")
            return
        if not os.path.exists(classifier_path):
            print(f"[ERR] CubeNet classifier model not found: {classifier_path}")
            return

        cubenet_thread = threading.Thread(
            target=_run_cubenet_worker,
            args=(detector_path, classifier_path, on_face_registered, on_capture_completed),
            daemon=True,
            name="CubeNetThread",
        )
        cubenet_thread.start()


# =============================================================================
# Overview
# =============================================================================
# This script is a keyboard-based UDP controller for the KIDA dual-arm + DG5F
# hand system.
#
# Supported features
# - Motion feedback reception on UDP 6601
# - Motion command transmission on UDP 6600
# - Dual-arm task-space teleoperation
# - Single-hand teleoperation with active-hand switching
# - Scenario snapshot recording to TXT / CSV
# - Scenario mode for writing scenarios and exporting command examples
# - Optional CubeNet face-guide detection trigger when scenario mode starts
# - UDP receive test mode for inspecting packets from a specific source IP
#
# Assumptions based on the provided sample/manual
# - Task command format:
#     task <Lx Ly Lz Lroll Lpitch Lyaw Rx Ry Rz Rroll Rpitch Ryaw>
# - Hand command format:
#     none, joint <20 joints>, joint <20 joints>
#   or the active hand only can be controlled by updating the corresponding side
#   while sending "none" for arm.
# - Motion feedback layout: 162 float32 values
# =============================================================================


# =============================================================================
# Global state and configuration
# =============================================================================
MOTION_FEEDBACK_SIZE = 162

# Current motion feedback from UDP 6601.
v = np.zeros(MOTION_FEEDBACK_SIZE, dtype=np.float32)
v_lock = threading.Lock()

# Current task-space command targets.
# Format: [x, y, z, roll, pitch, yaw]
left_task = np.array([0.30, 0.25, -0.40, 0.0, 0.0, 0.0], dtype=np.float32)
right_task = np.array([0.30, -0.25, -0.40, 0.0, 0.0, 0.0], dtype=np.float32)

# Current hand joint command targets.
left_hand_target = np.zeros(20, dtype=np.float32)
right_hand_target = np.zeros(20, dtype=np.float32)

# Active hand/finger for single-hand teleoperation.
active_hand = "left"
active_finger = "thumb"

# UDP addresses.
DEFAULT_UDP_HOST = "127.0.0.1"
XMODE_UDP_HOST = "192.168.0.2"

RCV_ADDR = (DEFAULT_UDP_HOST, 6601)
SRV_ADDR = (DEFAULT_UDP_HOST, 6600)
TRANSPORT_MODE = "udp"
ZMQ_CMD_ENDPOINT = f"tcp://{DEFAULT_UDP_HOST}:6600"
ZMQ_FEEDBACK_ENDPOINT = f"tcp://{DEFAULT_UDP_HOST}:6601"

# Task-space teleoperation step sizes.
pos_step = 0.01   # [m]
rpy_step = 0.05   # [rad]

# Arm teleoperation rotation frame.
# "tool" preserves the legacy behavior by incrementing RPY components directly.
# "base" applies roll/pitch/yaw increments around the robot base X/Y/Z axes.
arm_rotation_frame = "tool"

# Hand teleoperation step sizes.
hand_step = 0.08       # [rad] for grouped finger motion
thumb_joint_step = 0.05  # [rad] for per-selected-finger joint motion
READY_GRASP_GROUPED_FLEX_COUNT = 3

# Default command speed scaling.
# Each callable command can override this through its own speed_scale argument.
DEFAULT_SPEED_SCALE = 0.7

# Rate limiting to avoid sending too many UDP packets during key-repeat.
teleop_send_min_interval = 0.02  # 50 Hz max
last_task_send_time = 0.0
last_hand_send_time = 0.0

# Scenario record output files.
SNAPSHOT_TXT_PATH = "scenario_records.txt"
SNAPSHOT_CSV_PATH = "scenario_records.csv"
CUSTOM_MOTION_CSV_PATH = os.path.join(SCRIPT_DIR, "custom_motion.csv")
SCENARIO_EXAMPLE_PATH = "scenario_command_examples.txt"
READY_CSV_PATH = os.path.join(SCRIPT_DIR, "scenario_records_ready.csv")

CUSTOM_MOTION_METADATA_COLUMNS = [
    "motion_name",
    "motion_alias",
    "motion_description",
    "motion_use_arm",
    "motion_use_hand",
    "motion_tags",
    "require",
]

# Scenario mode state.
scenario_step_counter = 1
last_executed_motion_identifier = ""

# Finger mapping assumption.
# 20 hand joints are grouped as 5 fingers x 4 joints.
FINGER_SLICES = {
    "thumb": slice(0, 4),
    "index": slice(4, 8),
    "middle": slice(8, 12),
    "ring": slice(12, 16),
    "little": slice(16, 20),
}
FINGER_SELECT_KEYS = {
    "1": "thumb",
    "2": "index",
    "3": "middle",
    "4": "ring",
    "5": "little",
}
FINGER_ORDER = tuple(FINGER_SELECT_KEYS.values())


# =============================================================================
# Motion feedback receiver
# =============================================================================
def motion_recv_task():
    """Background thread: receive motion feedback from UDP or ZeroMQ."""
    global v
    if TRANSPORT_MODE == "zmq":
        if zmq is None:
            print("[ERR] pyzmq is not installed; cannot use ZeroMQ feedback transport.")
            return
        ctx = zmq.Context.instance()
        rcv_sock = ctx.socket(zmq.SUB)
        rcv_sock.connect(ZMQ_FEEDBACK_ENDPOINT)
        rcv_sock.setsockopt(zmq.SUBSCRIBE, b"")
        while True:
            data = rcv_sock.recv()
            new_v = np.frombuffer(data, dtype=np.float32)
            if new_v.size != MOTION_FEEDBACK_SIZE:
                print(f"[WARN] unexpected motion data size: {new_v.size} (expected {MOTION_FEEDBACK_SIZE})")
                continue
            with v_lock:
                v = new_v.copy()
        return

    rcv_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rcv_sock.bind(RCV_ADDR)
    while True:
        data, _ = rcv_sock.recvfrom(2048)
        new_v = np.frombuffer(data, dtype=np.float32)

        if new_v.size != MOTION_FEEDBACK_SIZE:
            print(f"[WARN] unexpected motion data size: {new_v.size} (expected {MOTION_FEEDBACK_SIZE})")
            continue

        with v_lock:
            v = new_v.copy()


# =============================================================================
# Basic utilities
# =============================================================================
def snapshot_v():
    """Return a thread-safe copy of the latest feedback vector."""
    with v_lock:
        return v.copy()


def _normalize_speed_scale(speed_scale: float = DEFAULT_SPEED_SCALE) -> float:
    """Return a safe positive speed scale."""
    try:
        value = float(speed_scale)
    except (TypeError, ValueError):
        value = DEFAULT_SPEED_SCALE
    return max(value, 1e-3)


def scaled_sleep(duration: float, speed_scale: float = DEFAULT_SPEED_SCALE):
    """Sleep less when speed_scale is larger, and more when it is smaller."""
    time.sleep(float(duration) / _normalize_speed_scale(speed_scale))


def parse_optional_speed_scale(tokens, start_idx: int = 1, default: float = DEFAULT_SPEED_SCALE) -> float:
    """Parse an optional trailing speed_scale token without breaking existing commands."""
    if len(tokens) <= start_idx:
        return default
    return float(tokens[start_idx])


def set_sequence_command_delay(delay_sec: float):
    """Set the fixed delay inserted between sequential LLM commands."""
    global SEQUENCE_COMMAND_DELAY
    SEQUENCE_COMMAND_DELAY = max(0.0, float(delay_sec))


def send_cmd(sock, cmd: str, verbose: bool = True):
    """Send a raw command string to the slave controller."""
    if TRANSPORT_MODE == "zmq":
        sock.send_string(cmd)
    else:
        sock.sendto(cmd.encode(), SRV_ADDR)
    if verbose:
        print(f"[TX] {cmd}")


def split_motion_data(vec: np.ndarray):
    """Split the 162-value motion feedback vector into labeled blocks."""
    if vec.size != MOTION_FEEDBACK_SIZE:
        raise ValueError(f"Expected {MOTION_FEEDBACK_SIZE} values, got {vec.size}")

    idx = 0
    arm_pos = vec[idx:idx + 14]; idx += 14
    arm_vel = vec[idx:idx + 14]; idx += 14
    arm_cur = vec[idx:idx + 14]; idx += 14

    lhand_pos = vec[idx:idx + 20]; idx += 20
    lhand_vel = vec[idx:idx + 20]; idx += 20
    lhand_cur = vec[idx:idx + 20]; idx += 20

    rhand_pos = vec[idx:idx + 20]; idx += 20
    rhand_vel = vec[idx:idx + 20]; idx += 20
    rhand_cur = vec[idx:idx + 20]; idx += 20

    return {
        "arm": {
            "left": {"pos": arm_pos[:7], "vel": arm_vel[:7], "cur": arm_cur[:7]},
            "right": {"pos": arm_pos[7:], "vel": arm_vel[7:], "cur": arm_cur[7:]},
        },
        "left_hand": {"pos": lhand_pos, "vel": lhand_vel, "cur": lhand_cur},
        "right_hand": {"pos": rhand_pos, "vel": rhand_vel, "cur": rhand_cur},
    }


def format_named_array(values, names, precision=4):
    return "\n".join(f"    {name:<8}: {val: .{precision}f}" for name, val in zip(names, values))


def format_motion_log(vec: np.ndarray):
    """Convert the latest feedback into a readable multi-block log."""
    data = split_motion_data(vec)

    arm_joint_names_left = [f"LJ{i+1}" for i in range(7)]
    arm_joint_names_right = [f"RJ{i+1}" for i in range(7)]
    hand_joint_names_left = [f"LH{i+1}" for i in range(20)]
    hand_joint_names_right = [f"RH{i+1}" for i in range(20)]

    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    out = ["=" * 72, f"[STATE LOG] {now}", "=" * 72]

    out += ["[Dual Arm - Left]", "  pos:", format_named_array(data["arm"]["left"]["pos"], arm_joint_names_left)]
    out += ["  vel:", format_named_array(data["arm"]["left"]["vel"], arm_joint_names_left)]
    out += ["  cur:", format_named_array(data["arm"]["left"]["cur"], arm_joint_names_left), ""]

    out += ["[Dual Arm - Right]", "  pos:", format_named_array(data["arm"]["right"]["pos"], arm_joint_names_right)]
    out += ["  vel:", format_named_array(data["arm"]["right"]["vel"], arm_joint_names_right)]
    out += ["  cur:", format_named_array(data["arm"]["right"]["cur"], arm_joint_names_right), ""]

    out += ["[Left Hand]", "  pos:", format_named_array(data["left_hand"]["pos"], hand_joint_names_left)]
    out += ["  vel:", format_named_array(data["left_hand"]["vel"], hand_joint_names_left)]
    out += ["  cur:", format_named_array(data["left_hand"]["cur"], hand_joint_names_left), ""]

    out += ["[Right Hand]", "  pos:", format_named_array(data["right_hand"]["pos"], hand_joint_names_right)]
    out += ["  vel:", format_named_array(data["right_hand"]["vel"], hand_joint_names_right)]
    out += ["  cur:", format_named_array(data["right_hand"]["cur"], hand_joint_names_right)]

    return "\n".join(out)


def print_motion_log():
    print(format_motion_log(snapshot_v()))


def save_motion_log(path="motion_log.txt"):
    with open(path, "a", encoding="utf-8") as f:
        f.write(format_motion_log(snapshot_v()) + "\n\n")
    print(f"[INFO] motion log saved to: {path}")


# =============================================================================
# Task-space command helpers
# =============================================================================
def build_task_cmd():
    """Build the task-space command string for both arms."""
    vals = np.concatenate([left_task, right_task])
    return "task " + " ".join(f"{x:.5f}" for x in vals)


def send_current_task(sock, verbose=False):
    send_cmd(sock, build_task_cmd(), verbose=verbose)


def send_current_task_rate_limited(sock, verbose=False):
    global last_task_send_time
    now = time.time()
    if now - last_task_send_time >= teleop_send_min_interval:
        send_current_task(sock, verbose=verbose)
        last_task_send_time = now


def set_task_from_values(values):
    """Set both-arm task target from 12 float values."""
    global left_task, right_task
    left_task = np.array(values[:6], dtype=np.float32)
    right_task = np.array(values[6:], dtype=np.float32)


def move_task(arm: str, axis: str, delta: float):
    """Increment one axis of the left/right task-space target."""
    axis_map = {
        "x": 0, "y": 1, "z": 2,
        "roll": 3, "pitch": 4, "yaw": 5,
        "rx": 3, "ry": 4, "rz": 5,
    }

    if arm not in ("l", "r"):
        raise ValueError("arm must be 'l' or 'r'")
    if axis not in axis_map:
        raise ValueError("axis must be one of x,y,z,roll,pitch,yaw (or rx,ry,rz)")

    idx = axis_map[axis]
    if arm == "l":
        left_task[idx] += delta
    else:
        right_task[idx] += delta


def rotation_matrix_from_rpy(roll: float, pitch: float, yaw: float):
    """Build a rotation matrix from roll/pitch/yaw using Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    rx = np.array([
        [1.0, 0.0, 0.0],
        [0.0, cr, -sr],
        [0.0, sr, cr],
    ], dtype=np.float64)
    ry = np.array([
        [cp, 0.0, sp],
        [0.0, 1.0, 0.0],
        [-sp, 0.0, cp],
    ], dtype=np.float64)
    rz = np.array([
        [cy, -sy, 0.0],
        [sy, cy, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)
    return rz @ ry @ rx


def rpy_from_rotation_matrix(rot):
    """Extract roll/pitch/yaw from a rotation matrix using Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    pitch = np.arcsin(np.clip(-rot[2, 0], -1.0, 1.0))
    cp = np.cos(pitch)

    if abs(cp) > 1e-6:
        roll = np.arctan2(rot[2, 1], rot[2, 2])
        yaw = np.arctan2(rot[1, 0], rot[0, 0])
    else:
        roll = 0.0
        yaw = np.arctan2(-rot[0, 1], rot[1, 1])

    return np.array([roll, pitch, yaw], dtype=np.float32)


def axis_rotation_matrix(axis: str, delta: float):
    """Build a base-axis rotation matrix for roll(X), pitch(Y), or yaw(Z)."""
    c, s = np.cos(delta), np.sin(delta)
    if axis in ("roll", "rx"):
        return np.array([
            [1.0, 0.0, 0.0],
            [0.0, c, -s],
            [0.0, s, c],
        ], dtype=np.float64)
    if axis in ("pitch", "ry"):
        return np.array([
            [c, 0.0, s],
            [0.0, 1.0, 0.0],
            [-s, 0.0, c],
        ], dtype=np.float64)
    if axis in ("yaw", "rz"):
        return np.array([
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
    raise ValueError("rotation axis must be one of roll,pitch,yaw (or rx,ry,rz)")


def move_task_rotation(arm: str, axis: str, delta: float):
    """Increment task rotation in either legacy tool-frame mode or base-frame mode."""
    if arm_rotation_frame == "tool":
        move_task(arm, axis, delta)
        return
    if arm_rotation_frame != "base":
        raise ValueError("arm_rotation_frame must be 'tool' or 'base'")
    if arm not in ("l", "r"):
        raise ValueError("arm must be 'l' or 'r'")

    target = left_task if arm == "l" else right_task
    current_rot = rotation_matrix_from_rpy(float(target[3]), float(target[4]), float(target[5]))
    base_delta_rot = axis_rotation_matrix(axis, delta)
    target[3:6] = rpy_from_rotation_matrix(base_delta_rot @ current_rot)


def set_arm_rotation_frame(frame: str):
    global arm_rotation_frame
    if frame not in ("tool", "base"):
        raise ValueError("rotation frame must be 'tool' or 'base'")
    arm_rotation_frame = frame


def toggle_arm_rotation_frame():
    global arm_rotation_frame
    arm_rotation_frame = "base" if arm_rotation_frame == "tool" else "tool"


def print_task_target():
    print("[Current task target]")
    print("  Left :", " ".join(f"{x:.5f}" for x in left_task))
    print("  Right:", " ".join(f"{x:.5f}" for x in right_task))
    print(f"  step(pos)={pos_step:.5f} m, step(rpy)={rpy_step:.5f} rad, rotation_frame={arm_rotation_frame}")


# =============================================================================
# Hand command helpers
# =============================================================================
def get_active_hand_array():
    return left_hand_target if active_hand == "left" else right_hand_target


def get_feedback_hand_array(side: str):
    data = split_motion_data(snapshot_v())
    if side == "left":
        return data["left_hand"]["pos"].copy()
    return data["right_hand"]["pos"].copy()


def build_hand_cmd():
    """Build a command string that updates both hands and ignores the arm."""
    left_str = "joint " + " ".join(f"{x:.5f}" for x in left_hand_target)
    right_str = "joint " + " ".join(f"{x:.5f}" for x in right_hand_target)
    return f"none, {left_str}, {right_str}"


def send_current_hand(sock, verbose=False):
    send_cmd(sock, build_hand_cmd(), verbose=verbose)


def send_current_hand_rate_limited(sock, verbose=False):
    global last_hand_send_time
    now = time.time()
    if now - last_hand_send_time >= teleop_send_min_interval:
        send_current_hand(sock, verbose=verbose)
        last_hand_send_time = now


def set_active_hand(side: str):
    global active_hand
    if side not in ("left", "right"):
        raise ValueError("hand must be 'left' or 'right'")
    active_hand = side


def toggle_active_hand():
    global active_hand
    active_hand = "right" if active_hand == "left" else "left"


def set_active_finger(finger: str):
    global active_finger
    if finger not in FINGER_SLICES:
        raise ValueError(f"finger must be one of {', '.join(FINGER_ORDER)}")
    active_finger = finger


def select_active_finger_by_key(key: str):
    set_active_finger(FINGER_SELECT_KEYS[key])


def sync_both_hands_from_feedback():
    global left_hand_target, right_hand_target
    left_hand_target = get_feedback_hand_array("left")
    right_hand_target = get_feedback_hand_array("right")


def sync_active_hand_from_feedback():
    global left_hand_target, right_hand_target
    if active_hand == "left":
        left_hand_target = get_feedback_hand_array("left")
    else:
        right_hand_target = get_feedback_hand_array("right")


def move_active_finger_block(finger: str, delta: float):
    """Increment all 4 joints of one finger block for the active hand."""
    target = get_active_hand_array()
    target[FINGER_SLICES[finger]] += delta


def move_active_finger_joint(finger: str, joint_idx_1to4: int, delta: float):
    """Increment one of the 4 joints of the selected finger on the active hand."""
    if finger not in FINGER_SLICES:
        raise ValueError(f"finger must be one of {', '.join(FINGER_ORDER)}")
    if joint_idx_1to4 < 1 or joint_idx_1to4 > 4:
        raise ValueError("finger joint index must be 1..4")
    target = get_active_hand_array()
    finger_base = FINGER_SLICES[finger].start
    target[finger_base + (joint_idx_1to4 - 1)] += delta


def move_active_selected_finger_joint(joint_idx_1to4: int, delta: float):
    """Increment one of the 4 joints of the active finger on the active hand."""
    move_active_finger_joint(active_finger, joint_idx_1to4, delta)


def move_active_thumb_joint(joint_idx_1to4: int, delta: float):
    """Increment one of the 4 thumb joints of the active hand."""
    move_active_finger_joint("thumb", joint_idx_1to4, delta)


def move_active_all_fingers(delta: float):
    """
    Increment the active hand in a grouped way.

    Updated behavior:
    - Non-thumb fingers (index/middle/ring/little): all 16 joints move by delta.
    - Thumb: only j3 and j4 move.
    - For the left hand, thumb j3/j4 direction is reversed relative to delta.
    - Thumb j1 and j2 do not move in whole-hand mode.
    """
    target = get_active_hand_array()

    # index/middle/ring/little -> normal direction
    for finger in ("index", "middle", "ring", "little"):
        target[FINGER_SLICES[finger]] += delta

    # thumb -> only j3, j4 move
    thumb_sign = -1.0 if active_hand == "left" else 1.0
    thumb_delta = thumb_sign * delta

    thumb_base = FINGER_SLICES["thumb"].start
    target[thumb_base + 2] += thumb_delta  # j3
    target[thumb_base + 3] += thumb_delta  # j4


def print_hand_target(side=None):
    """Print the current hand target(s)."""
    if side is None:
        print(f"[Current hand target] active_hand={active_hand}")
        print("  Left :", " ".join(f"{x:.5f}" for x in left_hand_target))
        print("  Right:", " ".join(f"{x:.5f}" for x in right_hand_target))
    elif side == "left":
        print("[Left hand target]", " ".join(f"{x:.5f}" for x in left_hand_target))
    elif side == "right":
        print("[Right hand target]", " ".join(f"{x:.5f}" for x in right_hand_target))
    print(f"  active_hand={active_hand}, active_finger={active_finger}, hand_step={hand_step:.5f}, joint_step={thumb_joint_step:.5f}")


# =============================================================================
# Grasp preset helpers
# =============================================================================
# Distal 2 joints (within each finger's 4 joints):
# - thumb : j3, j4 -> indices 2, 3
# - index : j3, j4 -> indices 6, 7
# - middle: j3, j4 -> indices 10, 11
# - ring  : j3, j4 -> indices 14, 15
# - little: j3, j4 -> indices 18, 19
DISTAL_PAIR_INDICES = [2, 3, 6, 7, 10, 11, 14, 15, 18, 19]

# Closing magnitudes for distal-only grasp presets.
# Left thumb distal joints use negative sign, matching the existing left-thumb
# direction convention already used elsewhere in this controller.
LEFT_DISTAL_GRASP_ON = np.array(
    [-0.80, -0.80,  0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85],
    dtype=np.float32,
)
RIGHT_DISTAL_GRASP_ON = np.array(
    [0.80, 0.80, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85, 0.85],
    dtype=np.float32,
)
DISTAL_GRASP_PRESET_ACTIONS = {
    "left_grasp_on": ("left", True),
    "left_grasp_off": ("left", False),
    "right_grasp_on": ("right", True),
    "right_grasp_off": ("right", False),
}


def set_hand_distal_grasp_preset(side: str, on: bool):
    """
    Apply a distal-only grasp preset to one hand.

    Behavior:
    - Only the distal 2 joints of each finger are overwritten.
    - All other joints are left unchanged.
    - off preset sets the same distal joints to 0.0.
    """
    global left_hand_target, right_hand_target

    if side == "left":
        target = left_hand_target
        values = LEFT_DISTAL_GRASP_ON if on else np.zeros(len(DISTAL_PAIR_INDICES), dtype=np.float32)
    elif side == "right":
        target = right_hand_target
        values = RIGHT_DISTAL_GRASP_ON if on else np.zeros(len(DISTAL_PAIR_INDICES), dtype=np.float32)
    else:
        raise ValueError("side must be 'left' or 'right'")

    for idx, val in zip(DISTAL_PAIR_INDICES, values):
        target[idx] = val


def run_named_grasp_preset(sock, name: str, verbose: bool = True, speed_scale: float = DEFAULT_SPEED_SCALE):
    """
    Execute one of the predefined distal-only grasp presets.

    Supported names:
    - left_grasp_on
    - left_grasp_off
    - right_grasp_on
    - right_grasp_off
    """
    if name not in DISTAL_GRASP_PRESET_ACTIONS:
        raise ValueError(f"unknown grasp preset: {name}")

    side, on = DISTAL_GRASP_PRESET_ACTIONS[name]
    set_hand_distal_grasp_preset(side, on)

    if verbose:
        print(f"[INFO] applied grasp preset -> {name}")

    send_current_hand(sock, verbose=verbose)
    scaled_sleep(0.02, speed_scale)


READY_HAND_ACTIONS = {
    "lg": ("left", "grasp", "left grasp"),
    "lr": ("left", "release", "left release"),
    "le": ("left", "extend", "left extend"),
    "rg": ("right", "grasp", "right grasp"),
    "rr": ("right", "release", "right release"),
    "re": ("right", "extend", "right extend"),
}


def apply_grouped_flex_to_hand_pose(side: str, hand_pose, step: float, count: int = READY_GRASP_GROUPED_FLEX_COUNT):
    """Return a hand pose after applying z-style grouped flex several times."""
    if side not in ("left", "right"):
        raise ValueError("side must be 'left' or 'right'")

    target = np.array(hand_pose, dtype=np.float32).copy()
    delta = float(step) * int(count)

    for finger in ("index", "middle", "ring", "little"):
        target[FINGER_SLICES[finger]] += delta

    thumb_delta = (-delta if side == "left" else delta)
    thumb_base = FINGER_SLICES["thumb"].start
    target[thumb_base + 2] += thumb_delta
    target[thumb_base + 3] += thumb_delta
    return target


def set_ready_hand_action_target(
    side: str,
    mode: str,
    path: str = READY_CSV_PATH,
    step: float = None,
    count: int = READY_GRASP_GROUPED_FLEX_COUNT,
):
    """Set one hand to ready release, grasp, or extend while leaving arms untouched."""
    global left_hand_target, right_hand_target

    state = load_ready_state_from_csv(path)
    if state is None:
        return False

    ready_key = f"{side}_hand_target"
    ready_hand = state[ready_key].copy()
    action_step = hand_step if step is None else step
    if mode == "grasp":
        next_hand = apply_grouped_flex_to_hand_pose(
            side,
            ready_hand,
            action_step,
            count=count,
        )
    elif mode == "extend":
        next_hand = apply_grouped_flex_to_hand_pose(
            side,
            ready_hand,
            -action_step,
            count=count,
        )
    elif mode == "release":
        next_hand = ready_hand
    else:
        raise ValueError("mode must be 'grasp', 'release', or 'extend'")

    if side == "left":
        left_hand_target = next_hand
    elif side == "right":
        right_hand_target = next_hand
    else:
        raise ValueError("side must be 'left' or 'right'")

    return True


def run_ready_hand_action(sock, action: str, verbose: bool = True, speed_scale: float = DEFAULT_SPEED_SCALE):
    """Run ready-based hand-only grasp/release/extend actions."""
    if action not in READY_HAND_ACTIONS:
        raise ValueError(f"unknown ready hand action: {action}")

    side, mode, label = READY_HAND_ACTIONS[action]
    ok = set_ready_hand_action_target(side, mode)
    if not ok:
        return False

    if verbose:
        mode_text = {
            "grasp": "ready + grouped flex x3",
            "release": "ready hand pose",
            "extend": "ready + grouped extend x3",
        }[mode]
        print(f"[INFO] {label} ({action}) -> {mode_text}; arm targets unchanged")

    send_current_hand(sock, verbose=verbose)
    scaled_sleep(0.02, speed_scale)
    return True


# =============================================================================
# Scenario snapshot recording helpers
# =============================================================================
def build_snapshot_record(label=""):
    """Build one scenario snapshot row from current targets and feedback."""
    global scenario_step_counter

    vec = snapshot_v()
    data = split_motion_data(vec)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    record = {
        "step_index": scenario_step_counter,
        "timestamp": ts,
        "label": label,
        "active_hand": active_hand,
        "left_task_x": float(left_task[0]),
        "left_task_y": float(left_task[1]),
        "left_task_z": float(left_task[2]),
        "left_task_roll": float(left_task[3]),
        "left_task_pitch": float(left_task[4]),
        "left_task_yaw": float(left_task[5]),
        "right_task_x": float(right_task[0]),
        "right_task_y": float(right_task[1]),
        "right_task_z": float(right_task[2]),
        "right_task_roll": float(right_task[3]),
        "right_task_pitch": float(right_task[4]),
        "right_task_yaw": float(right_task[5]),
    }

    for i, val in enumerate(left_hand_target, start=1):
        record[f"left_hand_target_j{i}"] = float(val)
    for i, val in enumerate(right_hand_target, start=1):
        record[f"right_hand_target_j{i}"] = float(val)

    for i, val in enumerate(data["left_hand"]["pos"], start=1):
        record[f"left_hand_feedback_j{i}"] = float(val)
    for i, val in enumerate(data["right_hand"]["pos"], start=1):
        record[f"right_hand_feedback_j{i}"] = float(val)

    return record


def format_snapshot_text(record):
    lines = [
        "=" * 88,
        f"[SCENARIO SNAPSHOT] step={record['step_index']}  time={record['timestamp']}",
        f"label={record['label']}  active_hand={record['active_hand']}",
        "-" * 88,
        (
            "Left Task  : "
            f"x={record['left_task_x']:.5f}, y={record['left_task_y']:.5f}, z={record['left_task_z']:.5f}, "
            f"roll={record['left_task_roll']:.5f}, pitch={record['left_task_pitch']:.5f}, yaw={record['left_task_yaw']:.5f}"
        ),
        (
            "Right Task : "
            f"x={record['right_task_x']:.5f}, y={record['right_task_y']:.5f}, z={record['right_task_z']:.5f}, "
            f"roll={record['right_task_roll']:.5f}, pitch={record['right_task_pitch']:.5f}, yaw={record['right_task_yaw']:.5f}"
        ),
        "Left Hand Target : " + " ".join(f"{record[f'left_hand_target_j{i}']:.5f}" for i in range(1, 21)),
        "Right Hand Target: " + " ".join(f"{record[f'right_hand_target_j{i}']:.5f}" for i in range(1, 21)),
    ]
    return "\n".join(lines)


def append_snapshot_to_txt(record, path=SNAPSHOT_TXT_PATH):
    with open(path, "a", encoding="utf-8") as f:
        f.write(format_snapshot_text(record) + "\n\n")


def append_snapshot_to_csv(record, path=SNAPSHOT_CSV_PATH):
    file_exists = os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(record.keys()))
        if not file_exists:
            writer.writeheader()
        writer.writerow(record)


def _read_csv_rows_and_fieldnames(path: str):
    if not os.path.exists(path):
        return [], []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def append_dict_to_csv_with_header_union(record: dict, path: str, preferred_fieldnames=None):
    """Append a row while preserving editable CSV headers and adding missing columns if needed."""
    rows, existing_fieldnames = _read_csv_rows_and_fieldnames(path)
    preferred_fieldnames = list(preferred_fieldnames or [])

    fieldnames = []
    for name in preferred_fieldnames + existing_fieldnames + list(record.keys()):
        if name not in fieldnames:
            fieldnames.append(name)

    file_needs_rewrite = bool(existing_fieldnames) and fieldnames != existing_fieldnames
    mode = "w" if file_needs_rewrite or not existing_fieldnames else "a"
    with open(path, mode, newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if mode == "w":
            writer.writeheader()
            for row in rows:
                writer.writerow(row)
        writer.writerow(record)


def build_custom_motion_record(record: dict, label: str = ""):
    """Add editable custom-motion metadata columns to a snapshot record."""
    motion_name = label.strip() if label else record.get("label", "")
    custom_record = {
        "motion_name": motion_name,
        "motion_alias": "",
        "motion_description": "",
        "motion_use_arm": "TRUE",
        "motion_use_hand": "TRUE",
        "motion_tags": "",
        "require": "",
    }
    custom_record.update(record)
    return custom_record


def append_custom_motion_to_csv(record: dict, label: str = "", path: str = CUSTOM_MOTION_CSV_PATH):
    custom_record = build_custom_motion_record(record, label=label)
    preferred_fieldnames = CUSTOM_MOTION_METADATA_COLUMNS + [k for k in record.keys() if k not in CUSTOM_MOTION_METADATA_COLUMNS]
    append_dict_to_csv_with_header_union(custom_record, path, preferred_fieldnames=preferred_fieldnames)


def record_snapshot(label=""):
    """Record one scenario snapshot to TXT, scenario CSV, and editable custom-motion CSV."""
    global scenario_step_counter
    record = build_snapshot_record(label=label)
    append_snapshot_to_txt(record)
    append_snapshot_to_csv(record)
    append_custom_motion_to_csv(record, label=label)
    print(
        f"[REC] scenario snapshot saved -> {SNAPSHOT_TXT_PATH} / {SNAPSHOT_CSV_PATH} "
        f"and custom motion -> {CUSTOM_MOTION_CSV_PATH} (step={scenario_step_counter})"
    )
    scenario_step_counter += 1


def _csv_bool(value, default: bool = True) -> bool:
    if value is None or str(value).strip() == "":
        return default
    return str(value).strip().lower() in ("1", "true", "yes", "y", "on", "arm", "hand")


def load_custom_motion_row(identifier: str, path: str = CUSTOM_MOTION_CSV_PATH):
    """Load the most recent custom motion row matching name, alias, or label."""
    ident = str(identifier).strip()
    if not ident:
        return None
    rows, _ = _read_csv_rows_and_fieldnames(path)
    for row in reversed(rows):
        candidates = [
            row.get("motion_alias", ""),
            row.get("motion_name", ""),
            row.get("label", ""),
        ]
        if any(ident == str(candidate).strip() for candidate in candidates if candidate is not None):
            return row
    return None


def custom_motion_exists(identifier: str) -> bool:
    return load_custom_motion_row(identifier) is not None


def set_task_from_snapshot_row(row):
    values = [
        float(row["left_task_x"]),
        float(row["left_task_y"]),
        float(row["left_task_z"]),
        float(row["left_task_roll"]),
        float(row["left_task_pitch"]),
        float(row["left_task_yaw"]),
        float(row["right_task_x"]),
        float(row["right_task_y"]),
        float(row["right_task_z"]),
        float(row["right_task_roll"]),
        float(row["right_task_pitch"]),
        float(row["right_task_yaw"]),
    ]
    set_task_from_values(values)


def set_hand_from_snapshot_row(row):
    global left_hand_target, right_hand_target
    left_hand_target = np.array([float(row[f"left_hand_target_j{i}"]) for i in range(1, 21)], dtype=np.float32)
    right_hand_target = np.array([float(row[f"right_hand_target_j{i}"]) for i in range(1, 21)], dtype=np.float32)


def _parse_motion_requirements(require_value: str):
    """Parse custom-motion `require` cell into a list of prerequisite motion identifiers."""
    raw = str(require_value or "").strip()
    if not raw:
        return []
    normalized = raw.replace(';', ',').replace('/', ',').replace('|', ',')
    return [token.strip() for token in normalized.split(',') if token.strip()]


def _motion_identifier_candidates(row: dict, identifier: str):
    return [
        str(identifier or "").strip(),
        str(row.get("motion_alias", "")).strip(),
        str(row.get("motion_name", "")).strip(),
        str(row.get("label", "")).strip(),
    ]


def run_custom_motion(sock, identifier: str, verbose: bool = True, speed_scale: float = DEFAULT_SPEED_SCALE):
    """Run one editable custom motion by motion_name, motion_alias, or label."""
    global last_executed_motion_identifier

    row = load_custom_motion_row(identifier)
    if row is None:
        print(f"[ERR] custom motion not found: {identifier}")
        return False

    use_arm = _csv_bool(row.get("motion_use_arm"), default=True)
    use_hand = _csv_bool(row.get("motion_use_hand"), default=True)
    motion_name = row.get("motion_name") or row.get("label") or identifier
    motion_alias = row.get("motion_alias", "")
    require_tokens = _parse_motion_requirements(row.get("require", ""))

    if require_tokens:
        previous_id = (last_executed_motion_identifier or "").strip()
        if previous_id and previous_id in require_tokens:
            pass
        else:
            print(
                f"[WARN] custom motion '{motion_name}' requires previous motion in {require_tokens}, "
                f"but previous was '{previous_id or 'N/A'}'"
            )
            return False

    if verbose:
        alias_text = f" alias={motion_alias}" if motion_alias else ""
        require_text = ",".join(require_tokens) if require_tokens else "any"
        print(
            f"[CUSTOM] running motion '{motion_name}'{alias_text} "
            f"use_arm={use_arm} use_hand={use_hand} require={require_text}"
        )

    if use_arm:
        set_task_from_snapshot_row(row)
        send_current_task(sock, verbose=verbose)
        scaled_sleep(0.02, speed_scale)
    if use_hand:
        set_hand_from_snapshot_row(row)
        send_current_hand(sock, verbose=verbose)
        scaled_sleep(0.02, speed_scale)
    if not use_arm and not use_hand:
        print(f"[WARN] custom motion '{motion_name}' has both motion_use_arm and motion_use_hand disabled")

    candidates = _motion_identifier_candidates(row, identifier)
    for candidate in candidates:
        if candidate:
            last_executed_motion_identifier = candidate
            break
    return True


# =============================================================================
# Ready pose helpers
# =============================================================================
def load_ready_state_from_csv(path: str = READY_CSV_PATH):
    """
    Load the most recent ready state from scenario_records.csv.

    Returns:
        dict with keys:
            left_task, right_task, left_hand_target, right_hand_target
        or None if the CSV is missing / empty / invalid.
    """
    if not os.path.exists(path):
        print(f"[ERR] ready CSV not found: {path}")
        return None

    try:
        with open(path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        print(f"[ERR] failed to read ready CSV: {e}")
        return None

    if not rows:
        print(f"[ERR] ready CSV is empty: {path}")
        return None

    row = rows[-1]

    try:
        ready_left_task = np.array([
            float(row["left_task_x"]),
            float(row["left_task_y"]),
            float(row["left_task_z"]),
            float(row["left_task_roll"]),
            float(row["left_task_pitch"]),
            float(row["left_task_yaw"]),
        ], dtype=np.float32)

        ready_right_task = np.array([
            float(row["right_task_x"]),
            float(row["right_task_y"]),
            float(row["right_task_z"]),
            float(row["right_task_roll"]),
            float(row["right_task_pitch"]),
            float(row["right_task_yaw"]),
        ], dtype=np.float32)

        ready_left_hand = np.array(
            [float(row[f"left_hand_target_j{i}"]) for i in range(1, 21)],
            dtype=np.float32,
        )
        ready_right_hand = np.array(
            [float(row[f"right_hand_target_j{i}"]) for i in range(1, 21)],
            dtype=np.float32,
        )
    except KeyError as e:
        print(f"[ERR] ready CSV is missing required column: {e}")
        return None
    except ValueError as e:
        print(f"[ERR] ready CSV has invalid numeric value: {e}")
        return None

    return {
        "left_task": ready_left_task,
        "right_task": ready_right_task,
        "left_hand_target": ready_left_hand,
        "right_hand_target": ready_right_hand,
    }


def apply_ready_state(state: dict, apply_hand: bool = True):
    """Apply a previously loaded ready state to current task targets (and optionally hand targets)."""
    global left_task, right_task, left_hand_target, right_hand_target
    left_task = state["left_task"].copy()
    right_task = state["right_task"].copy()
    if apply_hand:
        left_hand_target = state["left_hand_target"].copy()
        right_hand_target = state["right_hand_target"].copy()


def send_ready_from_csv(sock, path: str = READY_CSV_PATH, verbose: bool = True, speed_scale: float = DEFAULT_SPEED_SCALE):
    """
    Load the latest state from scenario_records.csv and send it as a ready pose.

    Behavior (default):
    - update both arm task targets
    - keep current hand targets unchanged
    - send task command only
    """
    state = load_ready_state_from_csv(path)
    if state is None:
        return False

    apply_ready_state(state, apply_hand=False)

    if verbose:
        print(f"[INFO] ready pose loaded from: {path} (arm-only; hand targets preserved)")

    send_current_task(sock, verbose=verbose)

# =============================================================================
# Scenario command example exporters
# =============================================================================
def build_combined_command_example():
    """Return command examples that can be reused in future programs/sessions."""
    task_cmd = build_task_cmd()
    hand_cmd = build_hand_cmd()

    py_lines = [
        "# Python example: send current scenario commands over UDP",
        "import socket",
        f"srv_addr = {SRV_ADDR!r}",
        "sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)",
        f"task_cmd = {task_cmd!r}",
        f"hand_cmd = {hand_cmd!r}",
        "sock.sendto(task_cmd.encode(), srv_addr)",
        "sock.sendto(hand_cmd.encode(), srv_addr)",
    ]

    cli_lines = [
        "# Command examples",
        f"TASK: {task_cmd}",
        f"HAND: {hand_cmd}",
    ]

    return "\n".join(cli_lines + [""] + py_lines)


def export_scenario_command_example(path=SCENARIO_EXAMPLE_PATH):
    """Append the current task/hand command example block to a text file."""
    block = []
    block.append("=" * 88)
    block.append(f"[SCENARIO COMMAND EXAMPLE] {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    block.append(build_combined_command_example())
    block.append("")

    with open(path, "a", encoding="utf-8") as f:
        f.write("\n".join(block))
    print(f"[INFO] scenario command example exported -> {path}")


def print_scenario_command_example():
    print(build_combined_command_example())


# =============================================================================
# Raw keyboard input helper (Linux terminal)
# =============================================================================
class RawKeyReader:
    """Context manager for non-blocking single-key terminal input."""
    def __enter__(self):
        self.fd = sys.stdin.fileno()
        self.old_settings = termios.tcgetattr(self.fd)
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old_settings)

    def read_key(self, timeout=0.05):
        rlist, _, _ = select.select([sys.stdin], [], [], timeout)
        if not rlist:
            return None

        ch = sys.stdin.read(1)
        if ch == "\x1b":
            rlist, _, _ = select.select([sys.stdin], [], [], 0.001)
            if rlist:
                ch2 = sys.stdin.read(1)
                rlist, _, _ = select.select([sys.stdin], [], [], 0.001)
                if rlist:
                    ch3 = sys.stdin.read(1)
                    return ch + ch2 + ch3
            return ch
        return ch


# =============================================================================
# Dual-arm teleoperation mode
# =============================================================================
def print_arm_teleop_help():
    print(
        f"""
[ARM TELEOP MODE]  (press 'm' again to exit)
rotation_frame = {arm_rotation_frame}

Left arm translation
  w/s : +x / -x
  a/d : +y / -y
  r/f : +z / -z

Left arm rotation
  t/g : +roll / -roll
  y/h : +pitch / -pitch
  u/j : +yaw / -yaw

Right arm translation
  i/k : +x / -x
  o/l : +y / -y
  p/; : +z / -z

Right arm rotation
  7/4 : +roll / -roll
  8/5 : +pitch / -pitch
  9/6 : +yaw / -yaw

Other controls
  z/x : decrease/increase position step ({pos_step:.5f} m current)
  c/v : decrease/increase rotation step ({rpy_step:.5f} rad current)
  \\  : toggle rotation frame (tool/base)
  L   : lg, left ready-based grasp (hand only)
  O   : lr, left ready-based release (hand only)
  R   : rg, right ready-based grasp (hand only)
  P   : rr, right ready-based release (hand only)
  b   : record current scenario snapshot
  1   : send init
  2   : send rest
  3   : send home
  0   : print task target
  m   : exit arm teleop mode
  Q   : send quit and terminate program
"""
    )


def teleop_key_action(sock, key: str):
    global pos_step, rpy_step

    if key == "w":
        move_task("l", "x", +pos_step)
    elif key == "s":
        move_task("l", "x", -pos_step)
    elif key == "a":
        move_task("l", "y", +pos_step)
    elif key == "d":
        move_task("l", "y", -pos_step)
    elif key == "r":
        move_task("l", "z", +pos_step)
    elif key == "f":
        move_task("l", "z", -pos_step)
    elif key == "t":
        move_task_rotation("l", "roll", +rpy_step)
    elif key == "g":
        move_task_rotation("l", "roll", -rpy_step)
    elif key == "y":
        move_task_rotation("l", "pitch", +rpy_step)
    elif key == "h":
        move_task_rotation("l", "pitch", -rpy_step)
    elif key == "u":
        move_task_rotation("l", "yaw", +rpy_step)
    elif key == "j":
        move_task_rotation("l", "yaw", -rpy_step)
    elif key == "i":
        move_task("r", "x", +pos_step)
    elif key == "k":
        move_task("r", "x", -pos_step)
    elif key == "o":
        move_task("r", "y", +pos_step)
    elif key == "l":
        move_task("r", "y", -pos_step)
    elif key == "p":
        move_task("r", "z", +pos_step)
    elif key == ";":
        move_task("r", "z", -pos_step)
    elif key == "7":
        move_task_rotation("r", "roll", +rpy_step)
    elif key == "4":
        move_task_rotation("r", "roll", -rpy_step)
    elif key == "8":
        move_task_rotation("r", "pitch", +rpy_step)
    elif key == "5":
        move_task_rotation("r", "pitch", -rpy_step)
    elif key == "9":
        move_task_rotation("r", "yaw", +rpy_step)
    elif key == "6":
        move_task_rotation("r", "yaw", -rpy_step)
    elif key == "z":
        pos_step = max(0.001, pos_step * 0.5)
        print(f"\n[INFO] pos_step -> {pos_step:.5f} m")
        return None
    elif key == "x":
        pos_step = min(0.10, pos_step * 2.0)
        print(f"\n[INFO] pos_step -> {pos_step:.5f} m")
        return None
    elif key == "c":
        rpy_step = max(0.005, rpy_step * 0.5)
        print(f"\n[INFO] rpy_step -> {rpy_step:.5f} rad")
        return None
    elif key == "v":
        rpy_step = min(1.0, rpy_step * 2.0)
        print(f"\n[INFO] rpy_step -> {rpy_step:.5f} rad")
        return None
    elif key == "\\":
        toggle_arm_rotation_frame()
        print(f"\n[INFO] arm_rotation_frame -> {arm_rotation_frame}")
        return None
    elif key == "L":
        if run_ready_hand_action(sock, "lg", verbose=True):
            print_hand_target("left")
        return None
    elif key == "O":
        if run_ready_hand_action(sock, "lr", verbose=True):
            print_hand_target("left")
        return None
    elif key == "R":
        if run_ready_hand_action(sock, "rg", verbose=True):
            print_hand_target("right")
        return None
    elif key == "P":
        if run_ready_hand_action(sock, "rr", verbose=True):
            print_hand_target("right")
        return None
    elif key == "b":
        record_snapshot("arm_teleop")
        return None
    elif key == "1":
        send_cmd(sock, "init")
        return None
    elif key == "2":
        send_cmd(sock, "rest")
        return None
    elif key == "3":
        send_cmd(sock, "home")
        return None
    elif key == "0":
        print_task_target()
        return None
    elif key == "m":
        return "exit_arm_teleop"
    elif key == "Q":
        send_cmd(sock, "quit")
        return "quit_program"
    else:
        return None

    send_current_task_rate_limited(sock, verbose=False)
    sys.stdout.write(
        f"\r[rot={arm_rotation_frame}] "
        f"[L] xyz=({left_task[0]: .3f}, {left_task[1]: .3f}, {left_task[2]: .3f}) "
        f"rpy=({left_task[3]: .3f}, {left_task[4]: .3f}, {left_task[5]: .3f}) | "
        f"[R] xyz=({right_task[0]: .3f}, {right_task[1]: .3f}, {right_task[2]: .3f}) "
        f"rpy=({right_task[3]: .3f}, {right_task[4]: .3f}, {right_task[5]: .3f})   "
    )
    sys.stdout.flush()
    return None


def teleop_mode(sock):
    print_arm_teleop_help()
    print_task_target()
    print("[INFO] entering raw keyboard arm teleop mode...")

    with RawKeyReader() as reader:
        while True:
            key = reader.read_key(timeout=0.05)
            if key is None:
                continue
            result = teleop_key_action(sock, key)
            if result == "exit_arm_teleop":
                print("\n[INFO] arm teleop mode exited")
                return True
            if result == "quit_program":
                print("\n[INFO] program terminated by teleop key 'q'")
                return False

# =============================================================================
# Hand teleoperation mode (single active hand)
# =============================================================================
def print_hand_teleop_help():
    print(
        f"""
[HAND TELEOP MODE]  (press 'n' again to exit)
active_hand = {active_hand}
active_finger = {active_finger}

Finger select
  1 : select thumb  (joints 1-4)
  2 : select index  (joints 5-8)
  3 : select middle (joints 9-12)
  4 : select ring   (joints 13-16)
  5 : select little (joints 17-20)

Selected finger individual joints
  q/a : selected finger j1 + / -
  w/s : selected finger j2 + / -
  e/d : selected finger j3 + / -
  r/f : selected finger j4 + / -

Optional 4-joint block control
  t/g : index  flex / extend
  y/h : middle flex / extend
  u/j : ring   flex / extend
  i/k : little flex / extend

Whole active hand
  z/x : grouped flex / extend
        - thumb: only j3 and j4 move
        - left thumb j3/j4 direction is reversed

Hand select
  [   : select left hand
  ]   : select right hand
  \\  : toggle left/right

Other controls
  ,/. : decrease/increase grouped finger step ({hand_step:.5f} rad current)
  ;/' : decrease/increase selected-finger joint step ({thumb_joint_step:.5f} rad current)
  p   : print hand target
  0   : sync both hands from feedback
  6   : sync active hand from feedback
  b   : record current scenario snapshot
  I   : send init
  R   : send rest
  H   : send home
  n   : exit hand teleop mode
  Q   : send quit and terminate program
"""
    )


def hand_teleop_key_action(sock, key: str):
    global hand_step, thumb_joint_step

    if key in FINGER_SELECT_KEYS:
        select_active_finger_by_key(key)
        print(f"\n[INFO] active_finger -> {active_finger}")
        return None
    elif key == "q":
        move_active_selected_finger_joint(1, +thumb_joint_step)
    elif key == "a":
        move_active_selected_finger_joint(1, -thumb_joint_step)
    elif key == "w":
        move_active_selected_finger_joint(2, +thumb_joint_step)
    elif key == "s":
        move_active_selected_finger_joint(2, -thumb_joint_step)
    elif key == "e":
        move_active_selected_finger_joint(3, +thumb_joint_step)
    elif key == "d":
        move_active_selected_finger_joint(3, -thumb_joint_step)
    elif key == "r":
        move_active_selected_finger_joint(4, +thumb_joint_step)
    elif key == "f":
        move_active_selected_finger_joint(4, -thumb_joint_step)
    elif key == "t":
        move_active_finger_block("index", +hand_step)
    elif key == "g":
        move_active_finger_block("index", -hand_step)
    elif key == "y":
        move_active_finger_block("middle", +hand_step)
    elif key == "h":
        move_active_finger_block("middle", -hand_step)
    elif key == "u":
        move_active_finger_block("ring", +hand_step)
    elif key == "j":
        move_active_finger_block("ring", -hand_step)
    elif key == "i":
        move_active_finger_block("little", +hand_step)
    elif key == "k":
        move_active_finger_block("little", -hand_step)
    elif key == "z":
        move_active_all_fingers(+hand_step)
    elif key == "x":
        move_active_all_fingers(-hand_step)
    elif key == "[":
        set_active_hand("left")
        print(f"\n[INFO] active_hand -> {active_hand}")
        return None
    elif key == "]":
        set_active_hand("right")
        print(f"\n[INFO] active_hand -> {active_hand}")
        return None
    elif key == "\\":
        toggle_active_hand()
        print(f"\n[INFO] active_hand -> {active_hand}")
        return None
    elif key == ",":
        hand_step = max(0.005, hand_step * 0.5)
        print(f"\n[INFO] hand_step -> {hand_step:.5f} rad")
        return None
    elif key == ".":
        hand_step = min(1.0, hand_step * 2.0)
        print(f"\n[INFO] hand_step -> {hand_step:.5f} rad")
        return None
    elif key == ";":
        thumb_joint_step = max(0.002, thumb_joint_step * 0.5)
        print(f"\n[INFO] selected-finger joint_step -> {thumb_joint_step:.5f} rad")
        return None
    elif key == "'":
        thumb_joint_step = min(1.0, thumb_joint_step * 2.0)
        print(f"\n[INFO] selected-finger joint_step -> {thumb_joint_step:.5f} rad")
        return None
    elif key == "p":
        print_hand_target()
        return None
    elif key == "0":
        sync_both_hands_from_feedback()
        print("\n[INFO] both hand targets synced from feedback")
        return None
    elif key == "6":
        sync_active_hand_from_feedback()
        print(f"\n[INFO] active hand target synced from feedback -> {active_hand}")
        return None
    elif key == "b":
        record_snapshot("hand_teleop")
        return None
    elif key == "I":
        send_cmd(sock, "init")
        return None
    elif key == "R":
        send_cmd(sock, "rest")
        return None
    elif key == "H":
        send_cmd(sock, "home")
        return None
    elif key == "n":
        return "exit_hand_teleop"
    elif key == "Q":
        send_cmd(sock, "quit")
        return "quit_program"
    else:
        return None

    send_current_hand_rate_limited(sock, verbose=False)
    target = get_active_hand_array()
    selected = target[FINGER_SLICES[active_finger]]
    sys.stdout.write(
        f"\r[{active_hand}/{active_finger}] "
        f"selected=({selected[0]: .3f},{selected[1]: .3f},{selected[2]: .3f},{selected[3]: .3f}) "
        f"thumb=({target[0]: .3f},{target[1]: .3f},{target[2]: .3f},{target[3]: .3f}) "
        f"index=({target[4]: .3f},{target[5]: .3f},{target[6]: .3f},{target[7]: .3f}) "
        f"middle=({target[8]: .3f},{target[9]: .3f},{target[10]: .3f},{target[11]: .3f})   "
    )
    sys.stdout.flush()
    return None


def hand_teleop_mode(sock):
    print_hand_teleop_help()
    print_hand_target()
    print("[INFO] entering raw keyboard hand teleop mode...")

    with RawKeyReader() as reader:
        while True:
            key = reader.read_key(timeout=0.05)
            if key is None:
                continue
            result = hand_teleop_key_action(sock, key)
            if result == "exit_hand_teleop":
                print("\n[INFO] hand teleop mode exited")
                return True
            if result == "quit_program":
                print("\n[INFO] program terminated by hand teleop key 'Q'")
                return False

# =============================================================================
# Scenario mode
# =============================================================================
def print_scenario_mode_help():
    print(
        """
[SCENARIO MODE]  (press 'v' again to exit)
This mode is intended for scenario writing and future program integration.

Keys
  r : record current snapshot (task target + hand target)
  p : print current task/hand target summary
  c : print current command examples to terminal
  e : export current command examples to scenario_command_examples.txt
  t : print task command only
  h : print hand command only
  m : enter arm teleop mode temporarily, then return to command mode
  n : enter hand teleop mode temporarily, then return to command mode
  1 : send init
  2 : send rest
  3 : send home
  v : exit scenario mode
  q : send quit and terminate program
"""
    )


def print_current_summary_for_scenario():
    print_task_target()
    print_hand_target()


def scenario_mode(sock):
    """Keyboard mode for scenario authoring and command example export."""
    face_scan_order = ["F", "B", "L", "R", "U", "D"]
    face_view_motion = {
        "F": {"motion": "dual", "desc": "front face check"},
        "B": {"motion": "check_b", "desc": "back face check"},
        "L": {"motion": "check_l", "desc": "left face check"},
        "R": {"motion": "check_r", "desc": "right face check"},
        "U": {"motion": "check_u", "desc": "up face check"},
        "D": {"motion": "check_d", "desc": "down face check"},
    }

    def on_face_registered(face_idx, face_name, color_names, progress, total_faces, face_position=None):
        if face_position is not None:
            store_cubenet_face_position(face_position)
        print(
            f"[SCENARIO][CUBENET] registered face {face_name} (idx={face_idx}) "
            f"{progress}/{total_faces} -> {color_names}"
        )
        if face_position is not None:
            print(f"[SCENARIO][CUBENET] stored face position: {format_cubenet_face_position(face_position)}")
        expected_idx = min(max(int(progress) - 1, 0), len(face_scan_order) - 1)
        expected_face = face_scan_order[expected_idx]
        print(
            f"[SCENARIO][CUBENET] fixed scan order: {' -> '.join(face_scan_order)} | "
            f"expected now: {expected_face}"
        )
        if face_name != expected_face:
            print(
                f"[SCENARIO][CUBENET][WARN] observed face={face_name}, but fixed order expects {expected_face}. "
                "Rotation plan still follows fixed order."
            )

        view_plan = face_view_motion.get(expected_face)
        if view_plan:
            print(
                f"[SCENARIO][CUBENET] face-view action: {view_plan['desc']} -> motion {view_plan['motion']}"
            )
            ok = run_cube_custom_motion(
                sock,
                view_plan["motion"],
                row_delay=1.0,
                speed_scale=DEFAULT_SPEED_SCALE,
            )
            if not ok:
                print("[SCENARIO][CUBENET][ERR] face-view motion failed.")
                return

        if int(progress) >= int(total_faces):
            print("[SCENARIO][CUBENET] all faces acquired; no further rotation needed.")
            return

        next_face = face_scan_order[expected_idx + 1]
        next_plan = face_view_motion.get(next_face)
        if not next_plan:
            print(f"[SCENARIO][CUBENET][WARN] no next-face plan for {next_face}.")
            return
        print(
            f"[SCENARIO][CUBENET] next target face: {next_face} | "
            f"rotation/view motion: {next_plan['motion']}"
        )
        ok = run_cube_custom_motion(
            sock,
            next_plan["motion"],
            row_delay=1.0,
            speed_scale=DEFAULT_SPEED_SCALE,
        )
        if not ok:
            print("[SCENARIO][CUBENET][ERR] next-face transition motion failed; operator intervention may be required.")
            return
        print(
            f"[SCENARIO][CUBENET] transition complete. Please present face {next_face} to the camera."
        )

    def on_capture_completed(face_data_map, solution):
        print("\n[SCENARIO][CUBENET] all 6 faces captured.")
        print("[SCENARIO][CUBENET] captured cube face data:")
        for face_name in ["U", "R", "F", "D", "L", "B"]:
            colors = face_data_map.get(face_name)
            print(f"  - {face_name}: {colors if colors else 'N/A'}")
        print(f"[SCENARIO][CUBENET] cube manipulation sequence: {solution}")
        try:
            import cubenet_with_face_guide as cubenet_module
            cubenet_module.describe_cube_solution(solution)
        except Exception as e:
            print(f"[WARN] failed to print cube solution guide: {e}")

    start_cubenet_detection_if_needed(
        on_face_registered=on_face_registered,
        on_capture_completed=on_capture_completed,
    )

    print_scenario_mode_help()
    print_current_summary_for_scenario()
    print("[INFO] entering scenario mode...")

    with RawKeyReader() as reader:
        while True:
            key = reader.read_key(timeout=0.05)
            if key is None:
                continue

            if key == "r":
                record_snapshot("scenario_mode")
            elif key == "p":
                print("\n")
                print_current_summary_for_scenario()
            elif key == "c":
                print("\n")
                print_scenario_command_example()
            elif key == "e":
                export_scenario_command_example()
            elif key == "t":
                print("\n" + build_task_cmd())
            elif key == "h":
                print("\n" + build_hand_cmd())
            elif key == "1":
                send_cmd(sock, "init")
            elif key == "2":
                send_cmd(sock, "rest")
            elif key == "3":
                send_cmd(sock, "home")
            elif key == "m":
                print("\n[INFO] entering arm teleop from scenario mode...")
                ok = teleop_mode(sock)
                if not ok:
                    return False
                print_scenario_mode_help()
            elif key == "n":
                print("\n[INFO] entering hand teleop from scenario mode...")
                ok = hand_teleop_mode(sock)
                if not ok:
                    return False
                print_scenario_mode_help()
            elif key == "v":
                print("\n[INFO] scenario mode exited")
                return True
            elif key == "q":
                send_cmd(sock, "quit")
                print("\n[INFO] program terminated by scenario key 'q'")
                return False

# =============================================================================
# Receive test mode
# =============================================================================
def receive_test_mode(bind_ip: str, port: int, allowed_source_ip: Optional[str] = None, bufsize: int = 4096):
    """
    Listen for UDP packets on the given bind_ip:port and print received packets.

    Args:
        bind_ip: local interface IP to bind (use 0.0.0.0 to listen on all interfaces)
        port: local UDP port to bind
        allowed_source_ip: if provided, only packets from this source IP are shown
        bufsize: UDP receive buffer size
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((bind_ip, port))
    sock.settimeout(0.5)

    print("[INFO] receive test mode started")
    print(f"[INFO] listening on {bind_ip}:{port}")
    if allowed_source_ip:
        print(f"[INFO] filtering source IP = {allowed_source_ip}")
    else:
        print("[INFO] source IP filter disabled")
    print("[INFO] press Ctrl+C to exit receive test mode")

    try:
        while True:
            try:
                data, addr = sock.recvfrom(bufsize)
            except socket.timeout:
                continue

            src_ip, src_port = addr
            if allowed_source_ip and src_ip != allowed_source_ip:
                continue

            print("=" * 72)
            print(f"[RX] from {src_ip}:{src_port}  bytes={len(data)}")
            try:
                text_payload = data.decode("utf-8")
                print(text_payload)
            except UnicodeDecodeError:
                print(data.hex(" "))
    except KeyboardInterrupt:
        print("\n[INFO] receive test mode exited")
    finally:
        sock.close()


# =============================================================================
# LLM chat mode helpers
# =============================================================================
def load_llm_spec_text(path: str = LLM_SPEC_PATH) -> str:
    """Load the concise robot-command spec used by chat mode."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"LLM spec file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_llm_state_text() -> str:
    """Build a compact state snapshot for the LLM."""
    left_task_str = " ".join(f"{x:.5f}" for x in left_task)
    right_task_str = " ".join(f"{x:.5f}" for x in right_task)
    left_hand_str = " ".join(f"{x:.5f}" for x in left_hand_target)
    right_hand_str = " ".join(f"{x:.5f}" for x in right_hand_target)
    return (
        f"left_task: {left_task_str}\n"
        f"right_task: {right_task_str}\n"
        f"left_hand_target: {left_hand_str}\n"
        f"right_hand_target: {right_hand_str}\n"
        f"active_hand: {active_hand}\n"
    )


def extract_quoted_commands(text: str):
    """Extract all double-quoted command strings from LLM output."""
    return [m.strip() for m in re.findall(r'"([^"\n]+)"', text)]


def _parse_llm_json_commands(text: str):
    """Parse the preferred JSON LLM output format, if present."""
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return None

    if isinstance(parsed, str):
        return [parsed]
    if isinstance(parsed, list):
        commands = parsed
    elif isinstance(parsed, dict):
        commands = parsed.get("commands")
    else:
        return None

    if not isinstance(commands, list) or not all(isinstance(cmd, str) for cmd in commands):
        raise ValueError('JSON LLM output must be a string, a string list, or an object with "commands": [strings]')
    return commands


def parse_llm_robot_commands(text: str):
    """Parse and validate one or more robot commands from LLM output."""
    raw = text.strip()
    commands = _parse_llm_json_commands(raw)

    if commands is None:
        commands = extract_quoted_commands(raw)
        if commands:
            count_match = re.match(r"^\s*(\d+)\b", raw)
            if count_match:
                expected_count = int(count_match.group(1))
                if expected_count != len(commands):
                    raise ValueError(
                        f"LLM command count mismatch: declared {expected_count}, found {len(commands)} quoted commands"
                    )

    if not commands:
        raise ValueError(f"LLM output does not contain robot commands: {text}")

    return [validate_robot_command(cmd) for cmd in commands]


def _parse_task_command(cmd: str):
    parts = cmd.strip().replace(",", " ").split()
    if len(parts) != 13 or parts[0] != "task":
        raise ValueError("task command must be: task <12 floats>")
    values = [float(x) for x in parts[1:]]
    return values


def _parse_hand_command(cmd: str):
    s = cmd.strip()
    m = re.fullmatch(r"none,\s*joint\s+(.+?),\s*joint\s+(.+)", s)
    if not m:
        raise ValueError("hand command must be: none, joint <20 floats>, joint <20 floats>")
    left_vals = [float(x) for x in m.group(1).split()]
    right_vals = [float(x) for x in m.group(2).split()]
    if len(left_vals) != 20 or len(right_vals) != 20:
        raise ValueError("hand command must contain 20 left-hand floats and 20 right-hand floats")
    return left_vals, right_vals


def validate_robot_command(cmd: str) -> str:
    """Return a normalized valid robot command or raise ValueError."""
    s = cmd.strip()
    if s in ("init", "rest", "home", "ready", "quit") or s in READY_HAND_ACTIONS or s in DISTAL_GRASP_PRESET_ACTIONS:
        return s
    if s.startswith("motion "):
        identifier = s[len("motion "):].strip()
        if custom_motion_exists(identifier):
            return f"motion {identifier}"
        raise ValueError(f"custom motion not found: {identifier}")
    if custom_motion_exists(s):
        return f"motion {s}"
    if s.startswith("task "):
        values = _parse_task_command(s)
        return "task " + " ".join(f"{x:.5f}" for x in values)
    if s.startswith("none,"):
        left_vals, right_vals = _parse_hand_command(s)
        left_str = "joint " + " ".join(f"{x:.5f}" for x in left_vals)
        right_str = "joint " + " ".join(f"{x:.5f}" for x in right_vals)
        return f"none, {left_str}, {right_str}"
    raise ValueError("unsupported command format")


def apply_robot_command_to_state(cmd: str):
    """Update local task/hand state to match a validated command string."""
    global left_task, right_task, left_hand_target, right_hand_target
    if (
        cmd in ("init", "rest", "home", "ready", "quit")
        or cmd in READY_HAND_ACTIONS
        or cmd in DISTAL_GRASP_PRESET_ACTIONS
        or cmd.startswith("motion ")
    ):
        return
    if cmd.startswith("task "):
        values = _parse_task_command(cmd)
        set_task_from_values(values)
        return
    if cmd.startswith("none,"):
        left_vals, right_vals = _parse_hand_command(cmd)
        left_hand_target = np.array(left_vals, dtype=np.float32)
        right_hand_target = np.array(right_vals, dtype=np.float32)
        return
    raise ValueError("unsupported command format")


def dispatch_robot_command_sequence(
    sock,
    commands,
    verbose: bool = True,
    speed_scale: float = DEFAULT_SPEED_SCALE,
    inter_command_delay: float = None,
):
    """Dispatch multiple robot commands with a fixed delay between unit actions."""
    if isinstance(commands, str):
        commands = [commands]

    delay = SEQUENCE_COMMAND_DELAY if inter_command_delay is None else max(0.0, float(inter_command_delay))
    for idx, cmd in enumerate(commands, start=1):
        if verbose and len(commands) > 1:
            print(f"[SEQ] {idx}/{len(commands)} -> {cmd}")
        ok = dispatch_robot_command(sock, cmd, verbose=verbose, speed_scale=speed_scale)
        if not ok:
            return False
        if cmd == "quit":
            return True
        if idx < len(commands):
            if verbose:
                print(f"[SEQ] waiting {delay:.2f} sec before next command")
            time.sleep(delay)
    return True


def dispatch_robot_command(sock, cmd: str, verbose: bool = True, speed_scale: float = DEFAULT_SPEED_SCALE):
    """
    Dispatch one validated robot command immediately.
    - ready: load CSV state and send task only (preserve hand targets)
    - task: update local state and send task
    - hand: update local state and send hand
    - lg/lr/le/rg/rr/re: ready-based hand-only grasp/release/extend
    - init/rest/home/quit: forward as raw command
    """
    cmd = validate_robot_command(cmd)

    if cmd == "ready":
        return send_ready_from_csv(sock, verbose=verbose, speed_scale=speed_scale)

    if cmd in READY_HAND_ACTIONS:
        return run_ready_hand_action(sock, cmd, verbose=verbose, speed_scale=speed_scale)

    if cmd in DISTAL_GRASP_PRESET_ACTIONS:
        run_named_grasp_preset(sock, cmd, verbose=verbose, speed_scale=speed_scale)
        return True

    if cmd.startswith("motion "):
        return run_custom_motion(sock, cmd[len("motion "):].strip(), verbose=verbose, speed_scale=speed_scale)

    apply_robot_command_to_state(cmd)

    if cmd.startswith("task "):
        send_current_task(sock, verbose=verbose)
        scaled_sleep(0.02, speed_scale)
        return True

    if cmd.startswith("none,"):
        send_current_hand(sock, verbose=verbose)
        scaled_sleep(0.02, speed_scale)
        return True

    send_cmd(sock, cmd, verbose=verbose)
    scaled_sleep(0.02, speed_scale)
    return True



def _extract_chat_completion_text(response) -> str:
    """Extract text from an OpenAI chat-completions response object."""
    choices = getattr(response, "choices", None)
    if not choices:
        return ""

    first = choices[0]
    if isinstance(first, dict):
        message = first.get("message")
    else:
        message = getattr(first, "message", None)
    if message is None:
        return ""

    if isinstance(message, dict):
        content = message.get("content", "")
    else:
        content = getattr(message, "content", "")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict):
                text = item.get("text")
                if text:
                    parts.append(text)
            else:
                text = getattr(item, "text", None)
                if text:
                    parts.append(text)
        return "".join(parts)

    return str(content) if content else ""


def _build_openai_user_input(prompt: str, cam_image_b64: Optional[str] = None):
    if not cam_image_b64:
        return prompt
    return [
        {"type": "input_text", "text": prompt},
        {"type": "input_image", "image_url": f"data:image/jpeg;base64,{cam_image_b64}"},
    ]


def _build_chat_completion_user_content(prompt: str, cam_image_b64: Optional[str] = None):
    if not cam_image_b64:
        return prompt
    return [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{cam_image_b64}"}},
    ]


def capture_chat_camera_image(image_path: str = CHAT_CAM_IMAGE_PATH):
    """Capture one RealSense color frame for [cam] chat requests, save preview, and return base64 JPEG."""
    import base64
    import cv2
    import pyrealsense2 as rs

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    preview_opened = False
    try:
        pipeline.start(config)
        for _ in range(15):
            pipeline.wait_for_frames()
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            raise RuntimeError("failed to get color frame from camera")

        image = np.asanyarray(color_frame.get_data())
        ok, encoded = cv2.imencode('.jpg', image)
        if not ok:
            raise RuntimeError("failed to encode camera frame as jpeg")

        with open(image_path, 'wb') as f:
            f.write(encoded.tobytes())

        cv2.imshow(CHAT_CAM_PREVIEW_WINDOW, image)
        cv2.waitKey(1)
        preview_opened = True

        image_b64 = base64.b64encode(encoded.tobytes()).decode('utf-8')
        print(f"[CHAT][CAM] captured image saved: {image_path}")
        print(f"[CHAT][CAM] preview window: {CHAT_CAM_PREVIEW_WINDOW}")
        print("[CHAT][CAM] close the preview window (or press q/ESC in the window) to continue.")

        while cv2.getWindowProperty(CHAT_CAM_PREVIEW_WINDOW, cv2.WND_PROP_VISIBLE) >= 1:
            key = cv2.waitKey(50) & 0xFF
            if key in (27, ord('q')):
                cv2.destroyWindow(CHAT_CAM_PREVIEW_WINDOW)
                break

        return image_b64, image_path
    finally:
        pipeline.stop()
        if preview_opened:
            cv2.destroyWindow(CHAT_CAM_PREVIEW_WINDOW)


def parse_chat_camera_request(user_text: str):
    s = user_text.strip()
    if not s.lower().startswith(CHAT_CAM_PREFIX):
        return False, s
    return True, s[len(CHAT_CAM_PREFIX):].strip()


def call_openai_text_generation(prompt: str, cam_image_b64: Optional[str] = None) -> str:
    """Call the installed OpenAI SDK using Responses API when available, otherwise Chat Completions."""
    client = OpenAI()

    responses_api = getattr(client, "responses", None)
    if responses_api is not None:
        response = responses_api.create(
            model=OPENAI_MODEL,
            input=_build_openai_user_input(prompt, cam_image_b64=cam_image_b64),
        )
        return getattr(response, "output_text", "")

    chat_api = getattr(client, "chat", None)
    chat_completions = getattr(chat_api, "completions", None) if chat_api is not None else None
    if chat_completions is not None:
        response = chat_completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Return only the robot command JSON or legacy command format specified by the prompt.",
                },
                {"role": "user", "content": _build_chat_completion_user_content(prompt, cam_image_b64=cam_image_b64)},
            ],
        )
        return _extract_chat_completion_text(response)

    raise RuntimeError(
        "installed openai package supports neither client.responses nor client.chat.completions. "
        "Upgrade OpenAI SDK or install a compatible version."
    )


def request_llm_robot_command(user_text: str, cam_image_b64: Optional[str] = None):
    """
    Call OpenAI API and return (validated_robot_commands, raw_llm_output).
    """
    if OpenAI is None:
        raise RuntimeError("openai package is not installed. Run: pip install openai")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")

    spec = load_llm_spec_text()
    state_text = build_llm_state_text()

    prompt = (
        "Robot command specification:\n"
        f"{spec}\n\n"
        "Current robot state:\n"
        f"{state_text}\n"
        "User request:\n"
        f"{user_text}\n"
    )

    raw_output = call_openai_text_generation(prompt, cam_image_b64=cam_image_b64).strip()
    validated = parse_llm_robot_commands(raw_output)
    return validated, raw_output


def chat_mode(sock):
    """
    Interactive LLM chat mode.
    Each user message is sent to the OpenAI API.
    The LLM may return one command or a JSON command sequence.
    The extracted command(s) are validated and dispatched to the robot.
    """
    print("[INFO] entering chat mode...")
    print(f"[INFO] model = {OPENAI_MODEL}")
    print(f"[INFO] spec  = {LLM_SPEC_PATH}")
    print('[INFO] type natural-language commands. type "exit" to leave chat mode.')
    print('[INFO] prefix with [cam] to attach a live camera image. example: [cam] move the left arm above the cube.')

    while True:
        user_text = input("chat> ").strip()
        if not user_text:
            continue
        if user_text.lower() in ("exit", "quit", "back"):
            print("[INFO] chat mode exited")
            return True

        try:
            use_cam, prompt_text = parse_chat_camera_request(user_text)
            cam_image_b64 = None
            if use_cam:
                if not prompt_text:
                    print('[ERR] [cam] request is empty. example: [cam] move the left arm above the cube.')
                    continue
                cam_image_b64, _ = capture_chat_camera_image()
                print('[CHAT][CAM] sending camera image + robot description to LLM...')
            else:
                prompt_text = user_text

            ok = execute_natural_language_command(sock, prompt_text, cam_image_b64=cam_image_b64)
            if not ok:
                return False
        except Exception as e:
            print(f"[ERR] chat mode: {e}")


# =============================================================================
# Voice mode helpers
# =============================================================================
def _resolve_voice_device():
    """
    Resolve optional input device setting for sounddevice.
    Returns None if default device should be used.
    """
    if not VOICE_DEVICE:
        return None
    try:
        return int(VOICE_DEVICE)
    except ValueError:
        return VOICE_DEVICE


def print_voice_mode_status():
    """Print current voice-mode toggle states."""
    print(
        "[VOICE STATUS] "
        f"language={VOICE_LANGUAGE_MODE.upper()} | "
        f"continuous={'ON' if VOICE_CONTINUOUS_MODE else 'OFF'} | "
        f"confirm={'ON' if VOICE_CONFIRM_MODE else 'OFF'} | "
        f"record_seconds={VOICE_RECORD_SECONDS:.1f}"
    )


def set_voice_language_mode(mode: str):
    """Set forced STT language mode: en, ko, or auto."""
    global VOICE_LANGUAGE_MODE
    mode = mode.lower()
    if mode not in ("en", "ko", "auto"):
        raise ValueError("voice language mode must be one of: en, ko, auto")
    VOICE_LANGUAGE_MODE = mode
    print(f"[INFO] voice language mode -> {VOICE_LANGUAGE_MODE.upper()}")


def toggle_voice_continuous_mode():
    global VOICE_CONTINUOUS_MODE
    VOICE_CONTINUOUS_MODE = not VOICE_CONTINUOUS_MODE
    print(f"[INFO] voice continuous mode -> {'ON' if VOICE_CONTINUOUS_MODE else 'OFF'}")


def toggle_voice_confirm_mode():
    global VOICE_CONFIRM_MODE
    VOICE_CONFIRM_MODE = not VOICE_CONFIRM_MODE
    print(f"[INFO] voice confirm mode -> {'ON' if VOICE_CONFIRM_MODE else 'OFF'}")


def confirm_voice_action(cmd: str) -> bool:
    """Ask the user whether the parsed voice action should be executed."""
    while True:
        ans = input(f'confirm execute {cmd!r}? [y/n]: ').strip().lower()
        if ans in ("y", "yes"):
            time.sleep(0.02 / max(0.7, 1e-3))
            return True
        if ans in ("n", "no"):
            return False
        print("[INFO] please answer y or n")


def record_voice_audio(seconds: float = VOICE_RECORD_SECONDS, sample_rate: int = VOICE_SAMPLE_RATE):
    """
    Record mono audio from the system microphone using sounddevice.
    """
    if sd is None:
        raise RuntimeError("sounddevice is not installed. Run: pip install sounddevice")
    if sf is None:
        raise RuntimeError("soundfile is not installed. Run: pip install soundfile")

    device = _resolve_voice_device()
    frames = int(seconds * sample_rate)

    print(f"[VOICE] recording for {seconds:.1f} sec...")
    audio = sd.rec(
        frames,
        samplerate=sample_rate,
        channels=VOICE_CHANNELS,
        dtype="float32",
        device=device,
    )
    sd.wait()
    print("[VOICE] recording finished")
    return audio, sample_rate


def transcribe_voice_audio(audio, sample_rate: int) -> str:
    """
    Send recorded wav audio to OpenAI STT and return transcribed text.
    """
    if OpenAI is None:
        raise RuntimeError("openai package is not installed. Run: pip install openai")
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set")
    if sf is None:
        raise RuntimeError("soundfile is not installed. Run: pip install soundfile")

    client = OpenAI()

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        sf.write(tmp.name, audio, sample_rate)
        with open(tmp.name, "rb") as f:
            kwargs = {}
            if VOICE_LANGUAGE_MODE != "auto":
                kwargs["language"] = VOICE_LANGUAGE_MODE

            transcript = client.audio.transcriptions.create(
                model=OPENAI_STT_MODEL,
                file=f,
                **kwargs,
            )

    text = getattr(transcript, "text", "")
    if not text:
        raise RuntimeError("STT returned empty text")
    return text.strip()


def execute_natural_language_command(sock, user_text: str, require_confirm: bool = False, speed_scale: float = DEFAULT_SPEED_SCALE, cam_image_b64: Optional[str] = None):
    """
    Shared helper used by text chat mode and voice mode.
    Prints the interpreted sentence and action command before execution.
    """
    if not user_text:
        print("[WARN] empty user input")
        scaled_sleep(0.02, speed_scale)
        return True

    print(f"[USER TEXT] {user_text}")

    commands, raw = request_llm_robot_command(user_text, cam_image_b64=cam_image_b64)
    print(f'[LLM RAW] {raw}')
    print(f'[ACTION COUNT] {len(commands)}')
    for idx, cmd in enumerate(commands, start=1):
        print(f'[ACTION {idx}] "{cmd}"')

    if require_confirm:
        if not confirm_voice_action(commands):
            print("[INFO] action canceled")
            scaled_sleep(0.02, speed_scale)
            return True

    ok = dispatch_robot_command_sequence(sock, commands, verbose=True, speed_scale=speed_scale)
    if ok:
        if any(cmd == "ready" or cmd.startswith("task ") or cmd.startswith("motion ") for cmd in commands):
            print_task_target()
        if any(
            cmd == "ready"
            or cmd in READY_HAND_ACTIONS
            or cmd in DISTAL_GRASP_PRESET_ACTIONS
            or cmd.startswith("none,")
            or cmd.startswith("motion ")
            for cmd in commands
        ):
            print_hand_target()
        if any(cmd == "quit" for cmd in commands):
            print("[INFO] program terminated by LLM command")
            return False
        scaled_sleep(0.02, speed_scale)
    return True


def voice_mode(sock):
    """
    Interactive voice command mode.
    """
    print("[INFO] entering voice mode...")
    print(f"[INFO] STT model            = {OPENAI_STT_MODEL}")
    print(f"[INFO] voice sample rate   = {VOICE_SAMPLE_RATE}")
    print(f"[INFO] voice seconds       = {VOICE_RECORD_SECONDS}")
    print(f"[INFO] voice device        = {VOICE_DEVICE if VOICE_DEVICE else 'default'}")
    print("[INFO] voice controls:")
    print("       Enter : record one utterance")
    print("       e     : set English-only STT mode")
    print("       k     : set Korean-only STT mode")
    print("       a     : set automatic-language STT mode")
    print("       c     : toggle continuous listening mode")
    print("       y     : toggle confirm-before-execute mode")
    print("       s     : print current voice-mode status")
    print('       exit  : leave voice mode')
    print_voice_mode_status()

    while True:
        line = input("voice> ").strip().lower()
        if line in ("exit", "quit", "back"):
            print("[INFO] voice mode exited")
            return True
        if line == "e":
            set_voice_language_mode("en")
            print_voice_mode_status()
            continue
        if line == "k":
            set_voice_language_mode("ko")
            print_voice_mode_status()
            continue
        if line == "a":
            set_voice_language_mode("auto")
            print_voice_mode_status()
            continue
        if line == "c":
            toggle_voice_continuous_mode()
            print_voice_mode_status()
            continue
        if line == "y":
            toggle_voice_confirm_mode()
            print_voice_mode_status()
            continue
        if line == "s":
            print_voice_mode_status()
            continue

        try:
            while True:
                audio, sample_rate = record_voice_audio()
                stt_text = transcribe_voice_audio(audio, sample_rate)
                print(f"[STT] {stt_text}")

                ok = execute_natural_language_command(
                    sock,
                    stt_text,
                    require_confirm=VOICE_CONFIRM_MODE,
                )
                if not ok:
                    return False

                if not VOICE_CONTINUOUS_MODE:
                    break

                print("[INFO] continuous listening: next utterance...")
        except KeyboardInterrupt:
            if VOICE_CONTINUOUS_MODE:
                print("\n[INFO] continuous listening stopped; returning to voice prompt")
                continue
            print("\n[INFO] voice mode interrupted")
            return True
        except Exception as e:
            print(f"[ERR] voice mode: {e}")


# =============================================================================
# Snapshot CSV motion helpers
# =============================================================================
SNAPSHOT_REQUIRED_TASK_COLUMNS = [
    "left_task_x", "left_task_y", "left_task_z",
    "left_task_roll", "left_task_pitch", "left_task_yaw",
    "right_task_x", "right_task_y", "right_task_z",
    "right_task_roll", "right_task_pitch", "right_task_yaw",
]

SNAPSHOT_REQUIRED_HAND_COLUMNS = (
    [f"left_hand_target_j{i}" for i in range(1, 21)] +
    [f"right_hand_target_j{i}" for i in range(1, 21)]
)


def load_snapshot_csv(csv_file: str):
    """Load one snapshot CSV from SCRIPT_DIR and return (path, dataframe)."""
    path = os.path.join(SCRIPT_DIR, csv_file)
    print(f"[INFO] loading cube scenario file: {path}")

    if not os.path.exists(path):
        raise FileNotFoundError(f"cube scenario file not found: {path}")

    try:
        df = pd.read_csv(path)
    except Exception as e:
        raise RuntimeError(f"failed to read cube scenario file {path}: {e}")

    return path, df


def validate_snapshot_dataframe(df, csv_path: str, use_arm: bool = True, use_hand: bool = True):
    """Validate required columns depending on whether arm/hand will be used."""
    required_columns = []
    if use_arm:
        required_columns.extend(SNAPSHOT_REQUIRED_TASK_COLUMNS)
    if use_hand:
        required_columns.extend(SNAPSHOT_REQUIRED_HAND_COLUMNS)

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(
            f'cube scenario file is missing required columns: {csv_path}\n'
            f'missing columns: {missing}'
        )


def run_snapshot_row_motion(
    sock,
    row,
    csv_name: str,
    step_idx: int,
    use_arm: bool = True,
    use_hand: bool = True,
    arm_delay: float = 0.5,
    row_delay: float = 2.0,
    speed_scale: float = DEFAULT_SPEED_SCALE,
):
    """
    Execute one snapshot CSV row.

    Args:
        use_arm: send task command from this row
        use_hand: send hand command from this row
        arm_delay: delay inserted after arm command before hand command
        row_delay: delay inserted after the row is finished
    """
    if not use_arm and not use_hand:
        print(f"[WARN] {csv_name} step={step_idx}: both use_arm and use_hand are False, skipping row")

    apply_snapshot_row_to_local_state(row)

    if use_arm:
        task_cmd = build_task_cmd_from_snapshot_row(row)
        print(f'[CUBE] {csv_name} step={step_idx} TASK -> "{task_cmd}"')
        ok = dispatch_robot_command(sock, task_cmd, verbose=True, speed_scale=speed_scale)
        if not ok:
            print("[ERR] cube sequence stopped during task dispatch")
            return False

    if use_arm and use_hand:
        scaled_sleep(arm_delay, speed_scale)

    if use_hand:
        hand_cmd = build_hand_cmd_from_snapshot_row(row)
        print(f'[CUBE] {csv_name} step={step_idx} HAND -> "{hand_cmd}"')
        ok = dispatch_robot_command(sock, hand_cmd, verbose=True, speed_scale=speed_scale)
        if not ok:
            print("[ERR] cube sequence stopped during hand dispatch")
            return False

    scaled_sleep(row_delay, speed_scale)
    return True

def run_snapshot_csv_motion(
    sock,
    csv_file: str,
    use_arm: bool = True,
    use_hand: bool = True,
    arm_delay: float = 0.5,
    row_delay: float = 2.0,
    speed_scale: float = DEFAULT_SPEED_SCALE,
):
    """
    Execute one snapshot CSV file.

    Example:
        run_snapshot_csv_motion(sock, "scenario_left_grasp.csv", use_arm=True, use_hand=True)
        run_snapshot_csv_motion(sock, "scenario_right_rotate.csv", use_arm=True, use_hand=False)
        run_snapshot_csv_motion(sock, "some_hand_only.csv", use_arm=False, use_hand=True)
    """
    try:
        csv_path, df = load_snapshot_csv(csv_file)
        validate_snapshot_dataframe(df, csv_path, use_arm=use_arm, use_hand=use_hand)

        for idx, row in df.iterrows():
            ok = run_snapshot_row_motion(
                sock,
                row,
                csv_file,
                idx,
                use_arm=use_arm,
                use_hand=use_hand,
                arm_delay=arm_delay,
                row_delay=row_delay,
                speed_scale=speed_scale,
            )
            if not ok:
                return False
    except Exception as e:
        print(f"[ERR] snapshot CSV motion failed for {csv_file}: {e}")
        return False
    return True

# =============================================================================
# Cube scenario mode
# =============================================================================
def build_task_cmd_from_snapshot_row(row):
    """Build one task command from a snapshot CSV row."""
    vals = [
        float(row["left_task_x"]),
        float(row["left_task_y"]),
        float(row["left_task_z"]),
        float(row["left_task_roll"]),
        float(row["left_task_pitch"]),
        float(row["left_task_yaw"]),
        float(row["right_task_x"]),
        float(row["right_task_y"]),
        float(row["right_task_z"]),
        float(row["right_task_roll"]),
        float(row["right_task_pitch"]),
        float(row["right_task_yaw"]),
    ]
    return "task " + " ".join(f"{x:.5f}" for x in vals)


def build_hand_cmd_from_snapshot_row(row):
    """Build one hand command from a snapshot CSV row."""
    left_vals = [float(row[f"left_hand_target_j{i}"]) for i in range(1, 21)]
    right_vals = [float(row[f"right_hand_target_j{i}"]) for i in range(1, 21)]

    left_str = "joint " + " ".join(f"{x:.5f}" for x in left_vals)
    right_str = "joint " + " ".join(f"{x:.5f}" for x in right_vals)
    return f"none, {left_str}, {right_str}"


def apply_snapshot_row_to_local_state(row):
    """Update local task/hand state from one snapshot CSV row."""
    global left_task, right_task, left_hand_target, right_hand_target

    left_task = np.array([
        float(row["left_task_x"]),
        float(row["left_task_y"]),
        float(row["left_task_z"]),
        float(row["left_task_roll"]),
        float(row["left_task_pitch"]),
        float(row["left_task_yaw"]),
    ], dtype=np.float32)

    right_task = np.array([
        float(row["right_task_x"]),
        float(row["right_task_y"]),
        float(row["right_task_z"]),
        float(row["right_task_roll"]),
        float(row["right_task_pitch"]),
        float(row["right_task_yaw"]),
    ], dtype=np.float32)

    left_hand_target = np.array(
        [float(row[f"left_hand_target_j{i}"]) for i in range(1, 21)],
        dtype=np.float32,
    )
    right_hand_target = np.array(
        [float(row[f"right_hand_target_j{i}"]) for i in range(1, 21)],
        dtype=np.float32,
    )


def run_cube_custom_motion(
    sock,
    identifier: str,
    row_delay: float = 2.0,
    speed_scale: float = DEFAULT_SPEED_SCALE,
):
    """Run one custom_motion.csv row inside the cube sequence and wait after it."""
    print(f'[CUBE] custom_motion.csv -> motion "{identifier}"')
    ok = run_custom_motion(sock, identifier, verbose=True, speed_scale=speed_scale)
    if not ok:
        print(f'[ERR] cube sequence stopped during custom motion: {identifier}')
        return False
    scaled_sleep(row_delay, speed_scale)
    return True


def run_cube_custom_motion_sequence(
    sock,
    identifiers,
    row_delay: float = 2.0,
    speed_scale: float = DEFAULT_SPEED_SCALE,
):
    """Run a list of custom_motion.csv rows in order."""
    for idx, identifier in enumerate(identifiers, start=1):
        print(f'[CUBE] custom motion {idx}/{len(identifiers)}')
        ok = run_cube_custom_motion(
            sock,
            identifier,
            row_delay=row_delay,
            speed_scale=speed_scale,
        )
        if not ok:
            return False
    return True


def run_cube_sequence(sock, speed_scale: float = DEFAULT_SPEED_SCALE, custom_motion_names=None):
    """
    Execute predefined cube motions using reusable snapshot CSV helpers.

    Readable one-line styles:
        run_snapshot_csv_motion(sock, "file.csv", use_arm=True, use_hand=True)
        run_cube_custom_motion(sock, "my_motion_alias")

    Motion type options:
    - use_arm=True,  use_hand=True  -> execute both task and hand
    - use_arm=True,  use_hand=False -> execute arm only
    - use_arm=False, use_hand=True  -> execute hand only
    - custom_motion.csv rows use their own motion_use_arm / motion_use_hand flags

    Notes:
    - speed_scale is intentionally passed per called command.
    - In custom cube scenarios, each call can override speed_scale independently.
    - Passing custom_motion_names runs only those custom motions in order.
    """
    print("[INFO] entering cube mode...")

    if custom_motion_names:
        return run_cube_custom_motion_sequence(
            sock,
            custom_motion_names,
            row_delay=2.0,
            speed_scale=speed_scale,
        )

    # To mix custom_motion.csv rows into the predefined cube routine, insert a line
    # like this anywhere in the sequence and check the returned ok value:
    # ok = run_cube_custom_motion(sock, "my_motion_alias", row_delay=2.0, speed_scale=speed_scale)

    # 1. pick the cube
    ok = run_snapshot_csv_motion(
        sock,
        "scenario_left_grasp.csv",
        use_arm=True,
        use_hand=True,
        arm_delay=0.5,
        row_delay=1.0,
        speed_scale=speed_scale,
    )
    if not ok:
        return False

    run_named_grasp_preset(sock, "left_grasp_on", verbose=True, speed_scale=speed_scale)
    scaled_sleep(1.0, speed_scale)

    # 2. pass the cube (arm motion only)
    ok = run_snapshot_csv_motion(
        sock,
        "scenario_right_rotate.csv",
        use_arm=True,
        use_hand=False,
        arm_delay=0.5,
        row_delay=2.0,
        speed_scale=speed_scale,
    )
    if not ok:
        return False

    # narrow both arm
    dispatch_robot_command(
        sock,
        "task 0.31000 0.15000 -0.45000 1.15000 -0.45000 -1.05000 0.30000 -0.15000 -0.44000 -0.20000 0.00000 0.60000",
        speed_scale=speed_scale,
    )
    scaled_sleep(1.0, speed_scale)
    run_named_grasp_preset(sock, "left_grasp_off", verbose=True, speed_scale=speed_scale)
    run_named_grasp_preset(sock, "right_grasp_on", verbose=True, speed_scale=speed_scale)

    scaled_sleep(1.0, speed_scale)

    # 3. continue cube motion
    ok = run_snapshot_csv_motion(
        sock,
        "scenario_left_rotate.csv",
        use_arm=True,
        use_hand=True,
        arm_delay=0.5,
        row_delay=2.0,
        speed_scale=speed_scale,
    )
    if not ok:
        return False

    print("[INFO] cube mode finished")
    return True

# =============================================================================
# Hand pose estimator test mode
# =============================================================================
def hand_pose_test_mode(width: int = 640, height: int = 480, fps: int = 30):
    """Run the standalone RealSense + MediaPipe hand-pose preview loop."""
    from hand_pose_estimator import HandPoseEstimator

    print("[INFO] entering hand pose estimator test mode...")
    print("[INFO] RealSense RGB stream will open with MediaPipe hand landmarks overlay.")
    print("[INFO] Focus the preview window and press ESC or q to return to cmd>.")
    print(f"[INFO] hand pose stream config: width={width}, height={height}, fps={fps}")

    last_print_time = 0.0

    def on_hand_pose(hand_pose):
        nonlocal last_print_time
        now = time.time()
        if now - last_print_time < 1.0:
            return
        last_print_time = now
        joints = hand_pose.get("joints_2d_px") or []
        wrist = joints[0] if joints else [0.0, 0.0]
        print(
            "[HAND] "
            f"label={hand_pose.get('hand_label', 'N/A')} "
            f"score={float(hand_pose.get('score', 0.0)):.3f} "
            f"joints={len(joints)} "
            f"wrist_px=({float(wrist[0]):.1f}, {float(wrist[1]):.1f})"
        )

    estimator = HandPoseEstimator(width=width, height=height, fps=fps)
    try:
        estimator.run(on_hand_pose=on_hand_pose, preview=True)
    finally:
        estimator.stop()
    print("[INFO] hand pose estimator test mode exited")


# =============================================================================
# Command-line help text
# =============================================================================
def print_help():
    print(
        """
[Commands]
  help
      Show this help

  init / rest / home / ready / quit
      Send system motion commands
      ready [speed_scale] loads the latest pose from scenario_records.csv and sends task only (hand preserved)

  show
      Print labeled current feedback v data

  save
      Save labeled current feedback log to motion_log.txt

  raw
      Print raw v array

  target
      Print current task targets

  handtarget
      Print current hand targets and active hand

  left_grasp_on / left_grasp_off
  right_grasp_on / right_grasp_off
      Distal-only grasp presets for each hand
      Only the last 2 joints of each finger are modified

  lg / lr / le / rg / rr / re
      Ready-based hand-only actions: left/right grasp, release, extend
      Grasp = ready hand pose + grouped flex x3
      Release = ready hand pose
      Extend = ready hand pose + grouped extend x3 (opposite direction of grasp)
      Arm targets are not changed

  sendtask
      Send current task target without modifying it

  sendhand
      Send current hand target without modifying it

  motion <name_or_alias>
      Run a row from custom_motion.csv by motion_name, motion_alias, or label
      The CSV row can set motion_use_arm / motion_use_hand

  task <12 floats>
      Set full task pose and send immediately
      Format: left xyz rpy + right xyz rpy

  move <l|r> <x|y|z|roll|pitch|yaw> <delta>
      Increment one task-space axis and send immediately

  step <pos_step_m> <rpy_step_rad>
      Update task teleop step sizes

  rotframe [tool|base]
      Print or select the arm teleop rotation frame

  seqdelay [seconds]
      Print or set the fixed delay between sequential LLM commands

  handstep <group_step_rad> <selected_finger_joint_step_rad>
      Update hand teleop step sizes

  currenthand
      Print currently selected active hand/finger

  switchhand <left|right>
      Select active hand for hand teleop

  togglehand
      Toggle active hand left/right

  sync_hand
      Sync both hand targets from feedback

  sync_active_hand
      Sync only the active hand target from feedback

  record [label]
      Save current scenario snapshot to TXT/CSV and custom_motion.csv
      Edit custom_motion.csv columns: motion_name, motion_alias, motion_description, motion_use_arm, motion_use_hand, motion_tags, require

  scenarioexample
      Print current task/hand command example block

  exportscenario
      Append current command example block to scenario_command_examples.txt

  m / teleop
      Enter arm teleop mode

  n / handteleop
      Enter hand teleop mode

  hand [width height fps]
      Enter RealSense + MediaPipe hand pose test mode
      Streams camera preview with recognized hand landmarks overlay

  v / scenario
      Enter scenario mode
      Entering scenario mode also starts CubeNet detection once.

  recvtest <bind_ip> <port> [source_ip]
      Start UDP receive test mode
      Example:
      recvtest 0.0.0.0 7000
      recvtest 0.0.0.0 7000 192.168.0.10

  cube [speed_scale]
      Execute predefined cube scenario snapshot CSV files in sequence

  cube <motion_alias> [motion_alias ...] [speed_scale]
      Execute custom_motion.csv rows in the given order instead of the predefined cube sequence
      Prefer motion_alias values without spaces for this shorthand
      Current predefined order:
        scenario_left_grasp.csv
        scenario_right_rotate.csv
        scenario_left_rotate.csv
      For each CSV row:
        1) send task command
        2) wait 0.5 sec
        3) send hand command
        4) wait 2.0 sec

  chat
      Enter LLM chat mode
      The LLM must return one robot command inside double quotes.
      The extracted command is sent to the robot immediately.

  voice
      Enter voice mode
      Flow: microphone -> STT -> LLM -> robot
      The STT sentence and action command are printed before execution.
      Inside voice mode:
        Enter : record once
        e     : English-only STT mode
        k     : Korean-only STT mode
        a     : automatic-language STT mode
        c     : toggle continuous listening mode
        y     : toggle confirm-before-execute mode
        s     : print voice-mode status
"""
    )


# =============================================================================
# Launch argument helpers
# =============================================================================
def parse_launch_args():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "-x",
        action="store_true",
        dest="x_mode",
        help="use 192.168.0.2 instead of 127.0.0.1 for UDP addresses",
    )
    parser.add_argument(
        "-zmq",
        action="store_true",
        dest="zmq_mode",
        help="use ZeroMQ transport for local mode",
    )
    return parser.parse_args()


def configure_udp_addresses_from_args(args):
    """Set global transport addresses/endpoints from parsed launch arguments."""
    global RCV_ADDR, SRV_ADDR, TRANSPORT_MODE, ZMQ_CMD_ENDPOINT, ZMQ_FEEDBACK_ENDPOINT

    host = XMODE_UDP_HOST if args.x_mode else DEFAULT_UDP_HOST
    # Requested behavior:
    # - default run               -> UDP local
    # - -zmq                      -> ZeroMQ local
    # - -x                        -> ZeroMQ remote
    TRANSPORT_MODE = "zmq" if (args.x_mode or args.zmq_mode) else "udp"
    RCV_ADDR = (host, 6601)
    SRV_ADDR = (host, 6600)
    ZMQ_CMD_ENDPOINT = f"tcp://{host}:6600"
    ZMQ_FEEDBACK_ENDPOINT = f"tcp://{host}:6601"


# =============================================================================
# Main command loop
# =============================================================================
def main():
    args = parse_launch_args()
    configure_udp_addresses_from_args(args)

    receiver_thread = threading.Thread(target=motion_recv_task, daemon=True)
    receiver_thread.start()

    if TRANSPORT_MODE == "zmq":
        if zmq is None:
            print("[ERR] --transport zmq selected but pyzmq is not installed.")
            return
        ctx = zmq.Context.instance()
        snd_sock = ctx.socket(zmq.PUB)
        snd_sock.connect(ZMQ_CMD_ENDPOINT)
        time.sleep(0.1)
    else:
        snd_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    print("[INFO] motion receive thread started")
    print("[INFO] dual-arm keyboard controller started")
    print(f"[INFO] transport mode          : {TRANSPORT_MODE}")
    print(f"[INFO] RCV_ADDR                : {RCV_ADDR}")
    print(f"[INFO] SRV_ADDR                : {SRV_ADDR}")
    if TRANSPORT_MODE == "zmq":
        print(f"[INFO] ZMQ feedback endpoint   : {ZMQ_FEEDBACK_ENDPOINT}")
        print(f"[INFO] ZMQ command endpoint    : {ZMQ_CMD_ENDPOINT}")
    print(f"[INFO] CubeNet detector model   : {CUBENET_DETECTOR_PATH}")
    print(f"[INFO] CubeNet classifier model : {CUBENET_CLASSIFIER_PATH}")
    print(f"[INFO] LLM spec path            : {LLM_SPEC_PATH}")
    print(f"[INFO] OpenAI model            : {OPENAI_MODEL}")
    print(f"[INFO] OpenAI STT model        : {OPENAI_STT_MODEL}")
    print(f"[INFO] Voice language mode     : {VOICE_LANGUAGE_MODE}")
    print_help()

    while True:
        try:
            line = input("\ncmd> ").strip()
            if not line:
                continue

            tokens = line.split()
            cmd = tokens[0].lower()

            if cmd == "help":
                print_help()
            elif cmd == "init":
                send_cmd(snd_sock, "init")
            elif cmd == "rest":
                send_cmd(snd_sock, "rest")
            elif cmd == "home":
                send_cmd(snd_sock, "home")
            elif cmd == "ready":
                if len(tokens) > 2:
                    print("[ERR] ready command format: ready [speed_scale]")
                    continue
                speed_scale = parse_optional_speed_scale(tokens, 1)
                ok = send_ready_from_csv(snd_sock, verbose=True, speed_scale=speed_scale)
                if ok:
                    print_task_target()
                    print_hand_target()
            elif cmd == "quit":
                send_cmd(snd_sock, "quit")
                print("[INFO] exit")
                break
            elif cmd == "show":
                print_motion_log()
            elif cmd == "save":
                save_motion_log("motion_log.txt")
            elif cmd == "raw":
                print(snapshot_v())
            elif cmd == "target":
                print_task_target()
            elif cmd == "handtarget":
                print_hand_target()
            elif cmd in ("left_grasp_on", "left_grasp_off", "right_grasp_on", "right_grasp_off"):
                if len(tokens) > 2:
                    print(f"[ERR] {cmd} command format: {cmd} [speed_scale]")
                    continue
                speed_scale = parse_optional_speed_scale(tokens, 1)
                run_named_grasp_preset(snd_sock, cmd, verbose=True, speed_scale=speed_scale)
                print_hand_target()
            elif cmd in READY_HAND_ACTIONS:
                if len(tokens) > 2:
                    print(f"[ERR] {cmd} command format: {cmd} [speed_scale]")
                    continue
                speed_scale = parse_optional_speed_scale(tokens, 1)
                ok = run_ready_hand_action(snd_sock, cmd, verbose=True, speed_scale=speed_scale)
                if ok:
                    side = READY_HAND_ACTIONS[cmd][0]
                    print_hand_target(side)
            elif cmd == "motion":
                if len(tokens) < 2:
                    print("[ERR] motion command format: motion <name_or_alias> [speed_scale]")
                    continue
                speed_scale = DEFAULT_SPEED_SCALE
                motion_tokens = tokens[1:]
                try:
                    speed_scale = float(tokens[-1])
                    motion_tokens = tokens[1:-1]
                except ValueError:
                    pass
                identifier = " ".join(motion_tokens).strip()
                if not identifier:
                    print("[ERR] motion command format: motion <name_or_alias> [speed_scale]")
                    continue
                ok = run_custom_motion(snd_sock, identifier, verbose=True, speed_scale=speed_scale)
                if ok:
                    print_task_target()
                    print_hand_target()
            elif custom_motion_exists(cmd):
                if len(tokens) > 2:
                    print(f"[ERR] custom motion alias format: {cmd} [speed_scale]")
                    continue
                speed_scale = parse_optional_speed_scale(tokens, 1)
                ok = run_custom_motion(snd_sock, cmd, verbose=True, speed_scale=speed_scale)
                if ok:
                    print_task_target()
                    print_hand_target()
            elif cmd == "sendtask":
                send_current_task(snd_sock, verbose=True)
            elif cmd == "sendhand":
                send_current_hand(snd_sock, verbose=True)
            elif cmd == "task":
                if len(tokens) != 13:
                    print("[ERR] task command needs 12 numeric values")
                    continue
                values = [float(x) for x in tokens[1:13]]
                set_task_from_values(values)
                send_current_task(snd_sock, verbose=True)
                print_task_target()
            elif cmd == "move":
                if len(tokens) != 4:
                    print("[ERR] move command format: move <l|r> <axis> <delta>")
                    continue
                arm = tokens[1].lower()
                axis = tokens[2].lower()
                delta = float(tokens[3])
                move_task(arm, axis, delta)
                send_current_task(snd_sock, verbose=True)
                print_task_target()
            elif cmd == "step":
                global pos_step, rpy_step
                if len(tokens) != 3:
                    print("[ERR] step command format: step <pos_step_m> <rpy_step_rad>")
                    continue
                pos_step = float(tokens[1])
                rpy_step = float(tokens[2])
                print(f"[INFO] updated task step sizes -> pos: {pos_step:.5f} m, rpy: {rpy_step:.5f} rad")
            elif cmd == "rotframe":
                if len(tokens) == 1:
                    print(f"[INFO] arm_rotation_frame = {arm_rotation_frame}")
                    continue
                if len(tokens) != 2:
                    print("[ERR] rotframe command format: rotframe [tool|base]")
                    continue
                set_arm_rotation_frame(tokens[1].lower())
                print(f"[INFO] arm_rotation_frame -> {arm_rotation_frame}")
            elif cmd == "seqdelay":
                if len(tokens) == 1:
                    print(f"[INFO] sequence command delay = {SEQUENCE_COMMAND_DELAY:.2f} sec")
                    continue
                if len(tokens) != 2:
                    print("[ERR] seqdelay command format: seqdelay [seconds]")
                    continue
                set_sequence_command_delay(float(tokens[1]))
                print(f"[INFO] sequence command delay -> {SEQUENCE_COMMAND_DELAY:.2f} sec")
            elif cmd == "handstep":
                global hand_step, thumb_joint_step
                if len(tokens) != 3:
                    print("[ERR] handstep command format: handstep <group_step_rad> <selected_finger_joint_step_rad>")
                    continue
                hand_step = float(tokens[1])
                thumb_joint_step = float(tokens[2])
                print(f"[INFO] updated hand step sizes -> grouped: {hand_step:.5f} rad, selected-finger joint: {thumb_joint_step:.5f} rad")
            elif cmd == "currenthand":
                print(f"[INFO] active_hand = {active_hand}, active_finger = {active_finger}")
            elif cmd == "switchhand":
                if len(tokens) != 2:
                    print("[ERR] switchhand command format: switchhand <left|right>")
                    continue
                set_active_hand(tokens[1].lower())
                print(f"[INFO] active_hand -> {active_hand}")
            elif cmd == "togglehand":
                toggle_active_hand()
                print(f"[INFO] active_hand -> {active_hand}")
            elif cmd == "sync_hand":
                sync_both_hands_from_feedback()
                print("[INFO] both hands synced from feedback")
            elif cmd == "sync_active_hand":
                sync_active_hand_from_feedback()
                print(f"[INFO] active hand synced from feedback -> {active_hand}")
            elif cmd == "record":
                label = " ".join(tokens[1:]) if len(tokens) > 1 else "manual_command"
                record_snapshot(label)
            elif cmd == "scenarioexample":
                print_scenario_command_example()
            elif cmd == "exportscenario":
                export_scenario_command_example()
            elif cmd in ("m", "teleop"):
                ok = teleop_mode(snd_sock)
                if not ok:
                    break
            elif cmd in ("n", "handteleop"):
                ok = hand_teleop_mode(snd_sock)
                if not ok:
                    break
            elif cmd == "hand":
                if len(tokens) not in (1, 4):
                    print("[ERR] hand command format: hand [width height fps]")
                    continue
                width, height, fps = 640, 480, 30
                if len(tokens) == 4:
                    width = int(tokens[1])
                    height = int(tokens[2])
                    fps = int(tokens[3])
                hand_pose_test_mode(width=width, height=height, fps=fps)
            elif cmd in ("v", "scenario"):
                ok = scenario_mode(snd_sock)
                if not ok:
                    break
            elif cmd == "recvtest":
                if len(tokens) not in (3, 4):
                    print("[ERR] recvtest command format: recvtest <bind_ip> <port> [source_ip]")
                    continue
                bind_ip = tokens[1]
                port = int(tokens[2])
                allowed_source_ip = tokens[3] if len(tokens) == 4 else None
                receive_test_mode(bind_ip, port, allowed_source_ip=allowed_source_ip)
            elif cmd == "cube":
                speed_scale = DEFAULT_SPEED_SCALE
                custom_motion_names = tokens[1:]
                if len(tokens) >= 2:
                    try:
                        speed_scale = float(tokens[-1])
                        custom_motion_names = tokens[1:-1]
                    except ValueError:
                        pass
                ok = run_cube_sequence(
                    snd_sock,
                    speed_scale=speed_scale,
                    custom_motion_names=custom_motion_names,
                )
                if not ok:
                    print("[ERR] cube mode terminated with an error")
            elif cmd == "chat":
                ok = chat_mode(snd_sock)
                if not ok:
                    break
            elif cmd == "voice":
                ok = voice_mode(snd_sock)
                if not ok:
                    break
            else:
                print(f"[ERR] unknown command: {cmd}")
                print("      type 'help' for usage")

        except KeyboardInterrupt:
            print("\n[INFO] Ctrl+C detected")
            try:
                send_cmd(snd_sock, "quit")
            except Exception:
                pass
            break
        except Exception as e:
            print(f"[ERR] {e}")


if __name__ == "__main__":
    main()

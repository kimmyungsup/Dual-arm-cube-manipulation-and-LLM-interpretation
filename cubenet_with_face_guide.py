import argparse
import copy
import os
import time
import cv2
import numpy as np
import open3d as o3d

from app.cube import CubeInteractor
from app.models.color_classifier import KNNClassifier
from app.models.cube_detector import TFLiteDetector
from app.utils import *
from app.webcam import *
from fast_sdrsac import *

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DETECTOR_PATH = os.path.join(SCRIPT_DIR, "assets", "detector", "model.tflite")
DEFAULT_CLASSIFIER_PATH = os.path.join(SCRIPT_DIR, "assets", "classifier", "model.knn")

# 입력 순서: U, R, F, D, L, B
# 면 (Face)    센터 색상 (예시)    해당 면의 '위(Up)' 방향에 있어야 할 색상
# U (Up)       흰색               파란색 (Back)
# D (Down)     노란색             초록색 (Front)
# L (Left)     주황색             흰색 (Up)
# R (Right)    빨간색             흰색 (Up)
# F (Front)    초록색             흰색 (Up)
# B (Back)     파란색             흰색 (Up)

cube_size = 0.089  # 큐브 길이
s = cube_size / 2
cube_corners = np.array([
    [-s, -s, -s], [s, -s, -s], [s, s, -s], [-s, s, -s],  # 아래면
    [-s, -s, s], [s, -s, s], [s, s, s], [-s, s, s]       # 위면
])
diameter = (3 ** 0.5) * 2 * s

FACE_GUIDE_ORDER = [0, 1, 2, 3, 4, 5]  # U, R, F, D, L, B
FACE_GUIDE = {
    0: {"face": "U", "center": "WHITE",  "up": "BLUE (Back)"},
    1: {"face": "R", "center": "RED",    "up": "WHITE (Up)"},
    2: {"face": "F", "center": "GREEN",  "up": "WHITE (Up)"},
    3: {"face": "D", "center": "YELLOW", "up": "GREEN (Front)"},
    4: {"face": "L", "center": "ORANGE", "up": "WHITE (Up)"},
    5: {"face": "B", "center": "BLUE",   "up": "WHITE (Up)"},
}
FACE_INDEX_TO_NAME = {idx: info["face"] for idx, info in FACE_GUIDE.items()}
CUBE_SOLVE_REFERENCE_POSE = "Hold the cube with WHITE center on top (U) and GREEN center facing front (F); RED is right (R)."
CUBE_MOVE_NOTATION_HELP = {
    "U": "rotate the Up/top face 90° clockwise",
    "D": "rotate the Down/bottom face 90° clockwise",
    "L": "rotate the Left face 90° clockwise",
    "R": "rotate the Right face 90° clockwise",
    "F": "rotate the Front face 90° clockwise",
    "B": "rotate the Back face 90° clockwise",
}

CUBENET_DETECTOR_WARMUP_RUNS = int(os.environ.get("CUBENET_DETECTOR_WARMUP_RUNS", "2"))
CUBENET_POINT_COUNT = int(os.environ.get("CUBENET_POINT_COUNT", "800"))
CUBENET_SDRSAC_ITERS = int(os.environ.get("CUBENET_SDRSAC_ITERS", "90"))
CUBENET_SDRSAC_SUBSET_SIZE = int(os.environ.get("CUBENET_SDRSAC_SUBSET_SIZE", "5"))
CUBENET_ICP_EVERY_N_FRAMES = max(1, int(os.environ.get("CUBENET_ICP_EVERY_N_FRAMES", "1")))



def describe_cube_solution(solution: str):
    """Print a concise human guide for Kociemba cube move notation."""
    print(f"[GUIDE] Solve reference pose: {CUBE_SOLVE_REFERENCE_POSE}")
    if not solution:
        print("[GUIDE] No cube manipulation sequence was returned.")
        return

    print("[GUIDE] Move notation: no suffix=90° clockwise, '=90° counter-clockwise, 2=180°.")
    print("[GUIDE] Step-by-step manipulation guide:")
    for step_idx, move in enumerate(solution.split(), start=1):
        face = move[0]
        suffix = move[1:]
        base_text = CUBE_MOVE_NOTATION_HELP.get(face, f"rotate face {face}")
        if suffix == "'":
            action = base_text.replace("clockwise", "counter-clockwise")
        elif suffix == "2":
            action = base_text.replace("90° clockwise", "180°")
        else:
            action = base_text
        print(f"[GUIDE]   {step_idx}. {move}: {action}")


def get_next_face_index(cube: CubeInteractor):
    """Return the next face index to show in the intended capture order."""
    for face_idx in FACE_GUIDE_ORDER:
        if cube.faces[face_idx] is None:
            return face_idx
    return None


def draw_face_guide(frame: np.ndarray, cube: CubeInteractor) -> np.ndarray:
    """Overlay the next required face and progress information on the frame."""
    next_face_idx = get_next_face_index(cube)
    registered_faces = [FACE_GUIDE[idx]["face"] for idx in FACE_GUIDE_ORDER if cube.faces[idx] is not None]
    progress = f"Progress: {len(registered_faces)}/6"
    registered_text = "Registered: " + (", ".join(registered_faces) if registered_faces else "None")

    lines = [progress, registered_text]
    color = (0, 255, 255)

    if next_face_idx is None:
        lines += ["All faces captured. Solving..."]
        color = (0, 255, 0)
    else:
        info = FACE_GUIDE[next_face_idx]
        lines += [
            f"Show face: {info['face']} (center={info['center']})",
            f"Keep top direction toward: {info['up']}",
            "Hold the cube still until the face is registered.",
        ]

    x0, y0 = 25, 30
    line_h = 24
    pad = 10
    box_w = 650
    box_h = pad * 2 + line_h * len(lines)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0 - pad, y0 - 22), (x0 - pad + box_w, y0 - 22 + box_h), (20, 20, 20), -1)
    frame = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)

    for i, line in enumerate(lines):
        y = y0 + i * line_h
        cv2.putText(frame, line, (x0, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color if i >= 2 else (255, 255, 255), 2, cv2.LINE_AA)

    return frame


def _face_colors_to_names(face_colors):
    return [getattr(color, "name", str(color)) for color in face_colors]


def _collect_face_data_map(cube: CubeInteractor):
    face_data_map = {}
    for face_idx in FACE_GUIDE_ORDER:
        face_name = FACE_INDEX_TO_NAME.get(face_idx, str(face_idx))
        face_colors = cube.faces[face_idx]
        if face_colors is None:
            face_data_map[face_name] = None
        else:
            face_data_map[face_name] = _face_colors_to_names(face_colors)
    return face_data_map


def main(detector_path: str, classifier_path: str, on_face_registered=None, on_capture_completed=None) -> None:
    load_started = time.perf_counter()
    print("[CUBENET][LOAD] loading color classifier...")
    classifier = KNNClassifier(classifier_path)
    print("[CUBENET][LOAD] color classifier loaded")

    print("[CUBENET][LOAD] loading TFLite detector...")
    detector = TFLiteDetector(detector_path)
    print(
        f"[CUBENET][LOAD] TFLite detector loaded "
        f"(delegate={getattr(detector, 'delegate', 'unknown')}, threads={getattr(detector, 'num_threads', 'unknown')})"
    )

    if CUBENET_DETECTOR_WARMUP_RUNS > 0:
        warmup_started = time.perf_counter()
        print(f"[CUBENET][LOAD] warming up detector ({CUBENET_DETECTOR_WARMUP_RUNS} runs)...")
        detector.warmup(CUBENET_DETECTOR_WARMUP_RUNS)
        print(f"[CUBENET][LOAD] detector warmup completed in {time.perf_counter() - warmup_started:.2f}s")

    print("[CUBENET][LOAD] starting webcam...")
    cube = CubeInteractor()
    webcam = WebcamInteractor()
    intrinsic = webcam.intrinsic
    print("[CUBENET][LOAD] webcam started")

    print(f"[CUBENET][LOAD] preparing cube point cloud ({CUBENET_POINT_COUNT} points)...")
    mesh_cube = o3d.geometry.TriangleMesh.create_box(width=cube_size, height=cube_size, depth=cube_size)
    mesh_cube.translate(-np.array([cube_size / 2, cube_size / 2, cube_size / 2]))
    cube_pcd = mesh_cube.sample_points_uniformly(number_of_points=CUBENET_POINT_COUNT)
    cube_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

    print(f"[CUBENET][LOAD] CubeNet models/resources loaded in {time.perf_counter() - load_started:.2f}s")
    print("[CUBENET][LOAD] ready to process camera frames")

    do_init = True
    frame_idx = 0
    last_announced_face_idx = None

    while True:
        frame_idx += 1
        frame, depth = webcam.get_frame()
        if frame is None or depth is None:
            continue

        frame = color_preprocess(frame)

        next_face_idx = get_next_face_index(cube)
        if next_face_idx != last_announced_face_idx and next_face_idx is not None:
            info = FACE_GUIDE[next_face_idx]
            print(f"[GUIDE] Show face {info['face']} (center={info['center']}) / top direction -> {info['up']}")
            last_announced_face_idx = next_face_idx

        detection = detector.detect(frame)

        if detection.score < 0.5:
            frame = draw_face_guide(frame, cube)
            webcam.show_frame(frame)
            webcam.await_input()
            continue

        position = detection.get_position(frame)
        pcd = get_pcd(depth, intrinsic, position)

        if do_init:
            try:
                f_sdrsac = FAST_SDRSAC(
                    np.array(cube_pcd.points),
                    np.array(pcd.points),
                    np.array(cube_pcd.normals),
                    0.4,
                    diameter,
                    subset_size=CUBENET_SDRSAC_SUBSET_SIZE,
                    num_iters=CUBENET_SDRSAC_ITERS,
                )
                pose = f_sdrsac.run()
            except Exception:
                frame = draw_face_guide(frame, cube)
                webcam.show_frame(frame)
                webcam.await_input()
                continue

            if pose is None:
                print('pose is none')
                frame = draw_face_guide(frame, cube)
                webcam.show_frame(frame)
                webcam.await_input()
                continue

            do_init = False
        else:
            if frame_idx % CUBENET_ICP_EVERY_N_FRAMES == 0:
                result = o3d.pipelines.registration.registration_icp(
                    pcd,
                    cube_pcd,
                    max_correspondence_distance=0.015,
                    init=np.linalg.inv(pose),
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
                )
                result = o3d.pipelines.registration.registration_icp(
                    pcd,
                    cube_pcd,
                    max_correspondence_distance=0.01,
                    init=result.transformation,
                    estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
                )

                if result.fitness > 0.3 and result.inlier_rmse < 0.03:
                    pose = np.linalg.inv(result.transformation)
                    do_init = False
                else:
                    do_init = True

                cube_pcd_tr = copy.copy(cube_pcd).transform(pose)
                flip_error = np.mean(np.array(cube_pcd_tr.points), axis=0)[-1] - np.mean(np.array(pcd.points), axis=0)[-1]
                if flip_error < 0:
                    do_init = False

        colors_est = color_based_on_pose(frame, pose, intrinsic, cube_size)
        top, left, bot, right = position
        colors = classifier.my_get_colors(colors_est)

        previous_registered_count = sum(1 for f in cube.faces if f is not None)
        cube.register_face(colors, frame)
        current_registered_count = sum(1 for f in cube.faces if f is not None)
        if current_registered_count > previous_registered_count:
            registered_face_idx = int(colors[4].value)
            registered_face_name = FACE_INDEX_TO_NAME.get(registered_face_idx, str(registered_face_idx))
            registered_colors = _face_colors_to_names(cube.faces[registered_face_idx])
            print(
                f"[GUIDE] Face confirmed: {registered_face_name} "
                f"({current_registered_count}/6) -> {registered_colors}"
            )
            if callable(on_face_registered):
                try:
                    on_face_registered(
                        registered_face_idx,
                        registered_face_name,
                        registered_colors,
                        current_registered_count,
                        6,
                    )
                except Exception as callback_error:
                    print(f"[WARN] on_face_registered callback error: {callback_error}")

        if cube.is_solvable():
            solution = cube.solve()
            face_data_map = _collect_face_data_map(cube)
            frame = draw_face_guide(frame, cube)
            webcam.show_frame(frame)
            print("[GUIDE] All cube faces captured.")
            for face_name in ["U", "R", "F", "D", "L", "B"]:
                print(f"[GUIDE] {face_name}: {face_data_map.get(face_name)}")
            print(f"[GUIDE] Cube manipulation sequence: {solution}")
            describe_cube_solution(solution)
            if callable(on_capture_completed):
                try:
                    on_capture_completed(face_data_map, solution)
                except Exception as callback_error:
                    print(f"[WARN] on_capture_completed callback error: {callback_error}")
            break

        try:
            virtual_cube = get_virtual_cube(colors, (bot - top))
            frame[top:bot, left - (bot - top):left] = virtual_cube
            frame = visualize(copy.copy(frame), pose, intrinsic)
        except ValueError:
            pass  # Out of bounds

        msg_height = 0
        for msg_ in cube.last_registered_msg_list:
            msg = 'Registered face:' + str(msg_)
            cv2.putText(frame, msg, (50, 220 + msg_height), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1, cv2.LINE_AA)
            msg_height += 20

        frame = draw_face_guide(frame, cube)
        webcam.show_frame(frame)
        webcam.await_input()



def _validate_model_path(path: str, label: str) -> bool:
    if os.path.exists(path):
        return True
    print(f"[ERR] CubeNet {label} model not found: {path}")
    return False


def run_standalone(detector_path: str = DEFAULT_DETECTOR_PATH, classifier_path: str = DEFAULT_CLASSIFIER_PATH) -> int:
    """Run the face-guide detector with the same default assets/logs as dual_arm_main9.py v mode."""
    print("[INFO] CubeNet face-guide detection thread started")
    print(f"[INFO] detector_path   = {detector_path}")
    print(f"[INFO] classifier_path = {classifier_path}")

    if not _validate_model_path(detector_path, "detector"):
        return 1
    if not _validate_model_path(classifier_path, "classifier"):
        return 1

    try:
        main(detector_path, classifier_path)
    except KeyboardInterrupt:
        print("[INFO] CubeNet face-guide interrupted by user")
        return 130
    except Exception as e:
        print(f"[ERR] CubeNet face-guide runtime error: {e}")
        return 1

    print(f"[INFO] CubeNet face-guide detection thread finished; solve reference pose: {CUBE_SOLVE_REFERENCE_POSE}")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run CubeNet face-guide detection. If no model paths are provided, "
            "the same default assets used by dual_arm_main9.py scenario mode are used."
        )
    )
    parser.add_argument(
        '-d',
        '--detector_path',
        default=DEFAULT_DETECTOR_PATH,
        help=f"TFLite detector model path (default: {DEFAULT_DETECTOR_PATH})",
    )
    parser.add_argument(
        '-c',
        '--classifier_path',
        default=DEFAULT_CLASSIFIER_PATH,
        help=f"KNN color classifier path (default: {DEFAULT_CLASSIFIER_PATH})",
    )
    return parser


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    raise SystemExit(run_standalone(args.detector_path, args.classifier_path))

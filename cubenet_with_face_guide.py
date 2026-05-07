import argparse
import copy
import cv2
import numpy as np
import open3d as o3d

from app.cube import CubeInteractor
from app.models.color_classifier import KNNClassifier
from app.models.cube_detector import TFLiteDetector
from app.utils import *
from app.webcam import *
from fast_sdrsac import *

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
    classifier = KNNClassifier(classifier_path)
    detector = TFLiteDetector(detector_path)

    cube = CubeInteractor()
    webcam = WebcamInteractor()
    intrinsic = webcam.intrinsic

    mesh_cube = o3d.geometry.TriangleMesh.create_box(width=cube_size, height=cube_size, depth=cube_size)
    mesh_cube.translate(-np.array([cube_size / 2, cube_size / 2, cube_size / 2]))
    cube_pcd = mesh_cube.sample_points_uniformly(number_of_points=1000)
    cube_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

    do_init = True
    last_announced_face_idx = None

    while True:
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
                    subset_size=5,
                    num_iters=130,
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--detector_path', required=True)
    parser.add_argument('-c', '--classifier_path', required=True)

    args = parser.parse_args()
    main(args.detector_path, args.classifier_path)

import os
import time
import threading
from dataclasses import dataclass, asdict
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

try:
    import cv2
except Exception:
    cv2 = None

try:
    import pyrealsense2 as rs
except Exception:
    rs = None

try:
    import mediapipe as mp
except Exception:
    mp = None


@dataclass
class HandPoseResult:
    timestamp: float
    hand_label: str
    score: float
    joints_2d_px: List[List[float]]
    joints_3d_norm: List[List[float]]
    pose_rvec: Optional[List[float]]
    pose_tvec: Optional[List[float]]
    frame_size: Tuple[int, int]

    def to_dict(self) -> Dict:
        return asdict(self)


class HandPoseEstimator:
    """Realtime hand-joint and 6D pose estimator from RealSense RGB frames.

    Notes:
    - 2D joints are from MediaPipe Hands landmarks.
    - 6D pose is estimated with solvePnP against a canonical hand template and
      therefore is relative and model-dependent (good for simulation mapping).
    """

    def __init__(
        self,
        width: int = 640,
        height: int = 480,
        fps: int = 30,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ):
        if cv2 is None:
            raise ImportError("opencv-python is required for hand pose estimation.")
        if rs is None:
            raise ImportError("pyrealsense2 is required for RealSense input.")
        if mp is None:
            raise ImportError("mediapipe is required for hand landmark extraction.")

        self.width = width
        self.height = height
        self.fps = fps
        self.max_num_hands = max_num_hands
        self._running = False

        self._pipeline = rs.pipeline()
        self._config = rs.config()
        self._config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)

        self._mp_hands = mp.solutions.hands
        self._hands = self._mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=max_num_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self._lock = threading.Lock()
        self._latest_results: List[HandPoseResult] = []
        self._records: List[Dict] = []

    @staticmethod
    def _canonical_hand_model_m() -> np.ndarray:
        # Simple canonical model (meters), roughly centered around wrist.
        return np.array([
            [0.000, 0.000, 0.000],  # wrist
            [0.020, -0.015, 0.000], # thumb_cmc
            [0.045, -0.020, 0.000], # thumb_mcp
            [0.065, -0.018, 0.000], # thumb_ip
            [0.085, -0.015, 0.000], # thumb_tip
            [0.020, 0.000, 0.000],  # index_mcp
            [0.030, 0.020, 0.000],
            [0.035, 0.040, 0.000],
            [0.038, 0.060, 0.000],
            [0.000, 0.005, 0.000],  # middle_mcp
            [0.000, 0.028, 0.000],
            [0.000, 0.052, 0.000],
            [0.000, 0.074, 0.000],
            [-0.020, 0.000, 0.000], # ring_mcp
            [-0.028, 0.022, 0.000],
            [-0.034, 0.042, 0.000],
            [-0.038, 0.061, 0.000],
            [-0.040, -0.005, 0.000],# pinky_mcp
            [-0.052, 0.012, 0.000],
            [-0.060, 0.028, 0.000],
            [-0.066, 0.043, 0.000],
        ], dtype=np.float32)

    def _estimate_6d_pose(
        self,
        joints_2d_px: np.ndarray,
        camera_matrix: np.ndarray,
        dist_coeffs: np.ndarray,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        model_points = self._canonical_hand_model_m()
        if joints_2d_px.shape[0] != model_points.shape[0]:
            return None, None

        ok, rvec, tvec = cv2.solvePnP(
            model_points,
            joints_2d_px.astype(np.float32),
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return None, None
        return rvec.reshape(-1), tvec.reshape(-1)

    def get_latest_results(self) -> List[Dict]:
        with self._lock:
            return [r.to_dict() for r in self._latest_results]

    def get_records(self) -> List[Dict]:
        with self._lock:
            return [dict(r) for r in self._records]

    def run(
        self,
        on_hand_pose: Optional[Callable[[Dict], None]] = None,
        preview: bool = False,
        window_name: str = "hand-pose-preview",
    ):
        self._running = True
        profile = self._pipeline.start(self._config)
        color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_stream.get_intrinsics()

        camera_matrix = np.array([
            [intr.fx, 0.0, intr.ppx],
            [0.0, intr.fy, intr.ppy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        dist_coeffs = np.array(intr.coeffs[:5], dtype=np.float32)

        try:
            while self._running:
                frames = self._pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                frame = np.asanyarray(color_frame.get_data())
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_res = self._hands.process(rgb)

                frame_results: List[HandPoseResult] = []
                if mp_res.multi_hand_landmarks and mp_res.multi_handedness:
                    for hand_lm, handedness in zip(mp_res.multi_hand_landmarks, mp_res.multi_handedness):
                        label = handedness.classification[0].label
                        score = float(handedness.classification[0].score)

                        joints_2d = []
                        joints_3d = []
                        for lm in hand_lm.landmark:
                            joints_2d.append([lm.x * self.width, lm.y * self.height])
                            joints_3d.append([lm.x, lm.y, lm.z])

                        joints_2d_np = np.array(joints_2d, dtype=np.float32)
                        rvec, tvec = self._estimate_6d_pose(joints_2d_np, camera_matrix, dist_coeffs)

                        result = HandPoseResult(
                            timestamp=time.time(),
                            hand_label=label,
                            score=score,
                            joints_2d_px=joints_2d,
                            joints_3d_norm=joints_3d,
                            pose_rvec=rvec.tolist() if rvec is not None else None,
                            pose_tvec=tvec.tolist() if tvec is not None else None,
                            frame_size=(self.width, self.height),
                        )
                        frame_results.append(result)

                        if preview:
                            mp.solutions.drawing_utils.draw_landmarks(
                                frame,
                                hand_lm,
                                self._mp_hands.HAND_CONNECTIONS,
                            )

                        if on_hand_pose is not None:
                            on_hand_pose(result.to_dict())

                with self._lock:
                    self._latest_results = frame_results
                    for r in frame_results:
                        self._records.append(r.to_dict())

                if preview:
                    cv2.imshow(window_name, frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break
        finally:
            self._running = False
            self._pipeline.stop()
            self._hands.close()
            if preview:
                cv2.destroyWindow(window_name)

    def stop(self):
        self._running = False


def main(
    on_hand_pose: Optional[Callable[[Dict], None]] = None,
    preview: bool = True,
    width: int = 640,
    height: int = 480,
    fps: int = 30,
):
    estimator = HandPoseEstimator(width=width, height=height, fps=fps)
    estimator.run(on_hand_pose=on_hand_pose, preview=preview)


if __name__ == "__main__":
    main()

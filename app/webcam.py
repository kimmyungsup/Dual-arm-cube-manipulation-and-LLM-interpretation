import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as o3d

__appname__ = "CubeNet"

class WebcamInteractor:
    display_size = (1440, 820)


    def __init__(self):
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        self.pipeline.start(self.config)

        self.align = rs.align(rs.stream.color)

        profile = self.pipeline.get_active_profile()
        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()

        intr = profile.get_stream(rs.stream.depth) \
            .as_video_stream_profile() \
            .get_intrinsics()

        self.intrinsic = np.array([
            [intr.fx, 0, intr.ppx],
            [0, intr.fy, intr.ppy],
            [0, 0, 1]
        ])



    def get_frame(self) -> np.ndarray:
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)

        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        if not depth_frame or not color_frame:
            return None, None

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asarray(depth_image, dtype=np.float32) * self.depth_scale
        return color_image, depth_image

    def show_frame(self, frame: np.ndarray) -> None:
        if frame is None:
            return
        frame = cv2.resize(frame, self.display_size)
        cv2.imshow(__appname__, frame)

    def await_input(self) -> None:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            self.release()

    def release(self):
        self.pipeline.stop()
        cv2.destroyAllWindows()

def get_pcd(depth_image, intrinsic, ROI):
    mask = np.zeros_like(depth_image)
    top, left, bot, right = ROI
    mask[top:bot, left:right] = 1
    depth_image[mask == 0] = 0

    pinhole_intrinsics = o3d.camera.PinholeCameraIntrinsic(depth_image.shape[1], depth_image.shape[0], intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2])
    o3d_depth = o3d.geometry.Image(depth_image)
    pcd = o3d.geometry.PointCloud.create_from_depth_image(
        o3d_depth,
        pinhole_intrinsics,
        depth_scale=1,
        depth_trunc=0.7
    )
    return pcd

cube_size = 0.089
s = cube_size / 2
cube_corners = np.array([
    [-s,-s,-s], [s,-s,-s], [s,s,-s], [-s,s,-s],
    [-s,-s,s], [s,-s,s], [s,s,s], [-s,s,s]
])

edges = [
    (0,1),(1,2),(2,3),(3,0),
    (4,5),(5,6),(6,7),(7,4),
    (0,4),(1,5),(2,6),(3,7)
]

def visualize(color_image, pose, intrinsic):
    R_est = pose[:3, :3]
    t_est = pose[:3, 3]

    cube_proj = (R_est @ cube_corners.T + t_est.reshape(3, 1)).T
    u = (cube_proj[:, 0] * intrinsic[0, 0] / cube_proj[:, 2] + intrinsic[0, 2]).astype(int)
    v = (cube_proj[:, 1] * intrinsic[1, 1] / cube_proj[:, 2] + intrinsic[1, 2]).astype(int)

    for edge in edges:
        pt1 = (u[edge[0]], v[edge[0]])
        pt2 = (u[edge[1]], v[edge[1]])
        cv2.line(color_image, pt1, pt2, (0, 255, 0), 2)
    return color_image

def color_preprocess(color):
    wb = cv2.xphoto.createSimpleWB()
    wb.setP(0.5)
    balanced = wb.balanceWhite(color)

    hsv = cv2.cvtColor(balanced, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    v_eq = clahe.apply(v)

    hsv_eq = cv2.merge([h, s, v_eq])
    preprocessed = cv2.cvtColor(hsv_eq, cv2.COLOR_HSV2BGR)

    return preprocessed
import argparse
from app.cube import CubeInteractor
from app.models.color_classifier import KNNClassifier
from app.models.cube_detector import TFLiteDetector
from app.utils import *
from app.webcam import *
from fast_sdrsac import *
import open3d as o3d

# 입력 순서: U, R, F, D, L, B
# 면 (Face)	센터 색상 (예시)	해당 면의 '위(Up)' 방향에 있어야 할 색상
# U (Up)		흰색			파란색 (Back)
# D (Down)	노란색			초록색 (Front)
# L (Left)	주황색			흰색 (Up)
# R (Right)	빨간색			흰색 (Up)
# F (Front)	초록색			흰색 (Up)
# B (Back)	파란색			흰색 (Up)

cube_size = 0.089 # 큐브 길이
s = cube_size / 2
cube_corners = np.array([
    [-s,-s,-s], [s,-s,-s], [s,s,-s], [-s,s,-s],  # 아래면
    [-s,-s,s], [s,-s,s], [s,s,s], [-s,s,s]       # 위면
])
diameter = (3**(0.5))*2*s

def main(detector_path: str, classifier_path: str) -> None:
    classifier = KNNClassifier(classifier_path)
    detector = TFLiteDetector(detector_path)

    cube = CubeInteractor()
    webcam = WebcamInteractor()
    intrinsic = webcam.intrinsic

    mesh_cube = o3d.geometry.TriangleMesh.create_box(width=cube_size, height=cube_size, depth=cube_size)
    mesh_cube.translate(-np.array([cube_size / 2, cube_size / 2, cube_size / 2]))
    cube_pcd = mesh_cube.sample_points_uniformly(number_of_points=1000)
    cube_pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.01, max_nn=30))

    Do_init = True
    while True:
        frame, depth = webcam.get_frame()
        frame = color_preprocess(frame)
        detection = detector.detect(frame)

        if detection.score < 0.5:
            webcam.show_frame(frame)
            webcam.await_input()
            continue

        # detection.draw(frame)
        position = detection.get_position(frame)
        pcd = get_pcd(depth, intrinsic, position)

        if Do_init:
            try:
                f_sdrsac = FAST_SDRSAC(np.array(cube_pcd.points), np.array(pcd.points), np.array(cube_pcd.normals), 0.4, diameter,
                                       subset_size=5, num_iters=130)
                pose = f_sdrsac.run()
            except:
                continue
            if pose is None:
                print('pose is none')
                continue
            Do_init = False
        else:
            result = o3d.pipelines.registration.registration_icp(
                pcd, cube_pcd, max_correspondence_distance=0.015, init=np.linalg.inv(pose),
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane())
            result = o3d.pipelines.registration.registration_icp(
                pcd, cube_pcd, max_correspondence_distance=0.01, init=result.transformation,
                estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint())

            if result.fitness > 0.3 and result.inlier_rmse < 0.03:
                pose = np.linalg.inv(result.transformation)
                Do_init = False
            else:
                Do_init = True

            cube_pcd_tr = copy.copy(cube_pcd).transform(pose)
            filp_error = np.mean(np.array(cube_pcd_tr.points), axis=0)[-1] - np.mean(np.array(pcd.points), axis=0)[-1]
            if filp_error < 0:
                Do_init = False

        colors_est = color_based_on_pose(frame, pose, intrinsic, cube_size)
        top, left, bot, right = position

        colors = classifier.my_get_colors(colors_est)

        cube.register_face(colors, frame)
        if cube.is_solvable():
            print(cube.solve())
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
            cv2.putText( frame, msg, (50, 50 + msg_height),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 200), 1, cv2.LINE_AA)
            msg_height += 20

        webcam.show_frame(frame)
        webcam.await_input()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--detector_path', required=True)
    parser.add_argument('-c', '--classifier_path', required=True)

    args = parser.parse_args()
    main(args.detector_path, args.classifier_path)

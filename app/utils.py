from itertools import product
import numpy as np
from app.enums.colors import Color
import cv2


def get_virtual_cube(colors: list[Color], cube_size: int) -> np.ndarray:
    centers = 1 / 6, 3 / 6, 5 / 6

    patch_height = int(cube_size / 3  - 2)
    patch_width = int(cube_size / 3 - 2)

    virtual_cube = np.zeros((cube_size, cube_size, 3)).astype(np.uint8)

    for color, (center_y, center_x) in zip(colors, product(centers, repeat=2)):
        start_y = int(center_y * cube_size - patch_height / 2)
        end_y = start_y + patch_height

        start_x = int(center_x * cube_size - patch_width / 2)
        end_x = start_x + patch_width

        color_patch = np.tile(color.bgr, (patch_height, patch_width, 1))
        virtual_cube[start_y:end_y, start_x:end_x] = color_patch

    return virtual_cube

def reorder_cells_for_image(cell_3d_coords, rvec, t, K):
    centers_3d = [corners.mean(axis=0) for corners in cell_3d_coords]
    centers_3d = np.array(centers_3d)

    proj, _ = cv2.projectPoints(centers_3d, rvec, t, K, None)
    proj = proj.reshape(-1, 2)
    x = proj[:, 0]
    y = proj[:, 1]

    rows = []
    row_indices = np.argsort(y)

    sorted_indices = row_indices[np.argsort(y[row_indices])]
    y_sorted = y[sorted_indices]
    for i in range(0, 9, 3):
        row = sorted_indices[i:i + 3]
        row_sorted = row[np.argsort(x[row])]
        rows.extend(row_sorted)

    reordered_cells = [cell_3d_coords[i] for i in rows]
    return reordered_cells

def color_based_on_pose(image, pose, K, cube_size):

    R = pose[:3, :3]
    t = pose[:3, 3].reshape(3,1)
    rvec, _ = cv2.Rodrigues(R)


    faces = {
        "front":  (np.array([0,0,1]),  np.array([1,0,0]),  np.array([0,-1,0])),
        "back":   (np.array([0,0,-1]), np.array([-1,0,0]), np.array([0,-1,0])),
        "left":   (np.array([-1,0,0]), np.array([0,0,1]),  np.array([0,-1,0])),
        "right":  (np.array([1,0,0]),  np.array([0,0,-1]), np.array([0,-1,0])),
        "top":    (np.array([0,1,0]),  np.array([1,0,0]),  np.array([0,0,1])),
        "bottom": (np.array([0,-1,0]), np.array([1,0,0]),  np.array([0,0,-1]))
    }


    cam_pos_cube = -R.T @ t


    best_face = None
    max_dot = -np.inf
    half = cube_size / 2

    for name, (n, u, v) in faces.items():
        face_center = n * half
        view_vec = cam_pos_cube.flatten() - face_center
        view_vec /= np.linalg.norm(view_vec)
        dot = np.dot(n, view_vec)
        if dot > max_dot:
            max_dot = dot
            best_face = name

    if best_face is None:
        return None

    n, u, v = faces[best_face]

    cell_size = cube_size / 3
    cell_3d_coords = []

    face_origin = n * half

    for i in range(3):
        for j in range(3):
            du0 = -half + j*cell_size
            dv0 = half - i*cell_size
            du1 = du0 + cell_size
            dv1 = dv0 - cell_size

            corners = np.array([
                face_origin + du0*u + dv0*v,
                face_origin + du1*u + dv0*v,
                face_origin + du1*u + dv1*v,
                face_origin + du0*u + dv1*v
            ])
            cell_3d_coords.append(corners)
    cell_3d_coords = reorder_cells_for_image(cell_3d_coords, rvec, t, K)

    colors = []
    h, w = image.shape[:2]

    for corners in cell_3d_coords:
        proj, _ = cv2.projectPoints(corners, rvec, t, K, None)
        pts = proj.reshape(-1,2)
        pts[:,0] = np.clip(pts[:,0], 0, w-1)
        pts[:,1] = np.clip(pts[:,1], 0, h-1)

        mask = np.zeros((h,w), np.uint8)
        cv2.fillConvexPoly(mask, np.int32(pts), 255)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.erode(mask, kernel, iterations=1)

        if cv2.countNonZero(mask) == 0:
            colors.append((0,0,0))
            continue

        pixels = image[mask > 0]
        avg = np.median(pixels, axis=0).astype(np.uint8)
        colors.append(avg)

    vis = image.copy()
    for corners in cell_3d_coords:
        proj,_ = cv2.projectPoints(corners, rvec, t, K, None)
        pts = proj.reshape(-1,2)
        cv2.polylines(vis, [np.int32(pts)], True, (0,255,0), 2)

    # cv2.imshow("Cells", vis)
    # print("Visible face:", best_face)

    return colors


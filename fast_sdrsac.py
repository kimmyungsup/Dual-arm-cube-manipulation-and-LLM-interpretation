from search_corr import *
from scipy.spatial.distance import cdist
import open3d as o3d
import copy

def build_affinity(model_np, scene_np):
    A_x = cdist(model_np, model_np)
    A_y = cdist(scene_np, scene_np)

    A_x = A_x[..., np.newaxis][..., np.newaxis]
    A_y = A_y[np.newaxis, ...][np.newaxis, ...]

    D = np.abs(A_x - A_y) / (A_x + A_y + 1e-6)
    A_xy = np.exp(-10 * D)
    A_xy = A_xy.transpose(0, 2, 1, 3)
    A_xy = A_xy.reshape(A_x.shape[0] * A_y.shape[2], A_x.shape[0] * A_y.shape[2]).transpose(0, 1)
    return A_xy

def estimate_rigid_transform(P, Q):
    Pc = P - P.mean(axis=0)
    Qc = Q - Q.mean(axis=0)

    H = Pc.T @ Qc
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    t = Q.mean(axis=0) - R @ P.mean(axis=0)
    return R, t

class FAST_SDRSAC:
    def __init__(self, P, Q, P_n, ths, diameter, subset_size=3, num_iters=50):
        self.P = P
        self.Q = Q
        self.P_n = P_n
        self.k = subset_size
        self.num_iters = num_iters
        self.ths = ths
        self.diameter = diameter
        self.downsample_size_start_ratio = 0.075
        self.downsample_size = self.diameter * self.downsample_size_start_ratio
        self.pts_y_o3d_mean = np.mean(Q, axis=0)

    def sample_subset(self):
        idx_p = np.random.choice(len(self.P), self.k, replace=False)
        idx_q = np.random.choice(len(self.Q), self.k, replace=False)
        return self.P[idx_p], self.Q[idx_q]

    def sdrsac_subset(self, P_sub, Q_sub):
        A = build_affinity(P_sub, Q_sub)
        X_step1 = step1_linearized_matching(self.k, self.k, A)
        X, B = step2_lowrank_admm(A, X_step1, m=self.k, lr=0.5, rho=1.0, iters=50)
        rows, cols = linear_sum_assignment(-X)

        R, t = estimate_rigid_transform(P_sub[rows],Q_sub[cols])
        ini_score = 1 / (np.linalg.norm((R @ P_sub.T).T + t - Q_sub, axis=1).mean())

        if ini_score > 10:
            pts_x_o3d = o3d.geometry.PointCloud()
            pts_x_o3d.points = o3d.utility.Vector3dVector(self.P)
            pts_x_o3d.normals = o3d.utility.Vector3dVector(self.P_n)
            pts_y_o3d = o3d.geometry.PointCloud()
            pts_y_o3d.points = o3d.utility.Vector3dVector(self.Q)
            initial_T = np.identity(4)
            initial_T[:3, :3] = R
            initial_T[:3, 3] = t
            ICP_Result = o3d.pipelines.registration.registration_icp(pts_y_o3d, pts_x_o3d, self.downsample_size,
                                                                     np.linalg.inv(initial_T),
                                                                     o3d.pipelines.registration.TransformationEstimationPointToPlane())
            T_pr = np.array(ICP_Result.transformation, dtype=np.float32)
            T_pr = np.linalg.inv(T_pr)
            eval_result = o3d.pipelines.registration.evaluate_registration(pts_x_o3d, pts_y_o3d, self.downsample_size, T_pr)

            pts_x_o3d_tr = copy.copy(pts_x_o3d).transform(T_pr)
            filp_error = np.mean(np.array(pts_x_o3d_tr.points), axis=0)[-1] - self.pts_y_o3d_mean[-1]
            if filp_error < 0:
                return T_pr, 0.000
            if eval_result.fitness < 0.001:
                return T_pr, 0.001
            return T_pr, eval_result.fitness
        else:
            return None, 0.001

    def run(self):
        best_T = np.identity(4)
        best_fitness = 0.0
        for it in range(self.num_iters):
            P_sub, Q_sub = self.sample_subset()

            T, fitness = self.sdrsac_subset(P_sub, Q_sub)
            if best_fitness < fitness:
                best_fitness = fitness
                best_T = T
        return best_T


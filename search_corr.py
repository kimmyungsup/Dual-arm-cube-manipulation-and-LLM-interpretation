import numpy as np
from scipy.linalg import svd
import time
from scipy.optimize import linear_sum_assignment

def build_score_matrix_marginal(A, n):
    N = n * n
    A_reshaped = A.reshape(n, n, n, n)

    row_score = A_reshaped.sum(axis=(1, 3))
    col_score = A_reshaped.sum(axis=(0, 2))

    C = row_score + col_score
    return C

def step1_linearized_matching(n, m, A, top_k=5):

    X_relaxed = build_score_matrix_marginal(A, n)

    X_int = np.zeros_like(X_relaxed)
    X_copy = X_relaxed.copy()
    selected = 0
    while selected < m:
        i,j = np.unravel_index(np.argmax(X_copy), X_copy.shape)
        X_int[i,j] = 1
        selected += 1
        X_copy[i,:] = 0
        X_copy[:,j] = 0

    return X_int

def project_partial_perm(X_mat, m):
    n = X_mat.shape[0]

    for i in range(n):
        row = X_mat[i,:]
        if np.sum(row) > 1:
            idx = np.argmax(row)
            row[:] = 0
            row[idx] = 1
            X_mat[i,:] = row

    for j in range(n):
        col = X_mat[:,j]
        if np.sum(col) > 1:
            idx = np.argmax(col)
            col[:] = 0
            col[idx] = 1
            X_mat[:,j] = col

    flat = X_mat.flatten()
    top_idx = np.argsort(flat)[::-1][:m]
    new_X = np.zeros_like(flat)
    new_X[top_idx] = 1
    return new_X.reshape(n,n)


def step2_lowrank_admm(
    A,
    X_int,
    rank_r=3,
    m=None,
    lr=1.0,
    rho=1.0,
    iters=50
):
    n = X_int.shape[0]

    idx = np.flatnonzero(X_int.flatten())
    m_eff = len(idx)

    if m is None:
        m = m_eff

    A_I = A[np.ix_(idx, idx)]

    U, S, _ = svd(A_I, full_matrices=False)
    r = min(rank_r, len(S))

    U_r = U[:, :r]
    S_r = np.sqrt(np.diag(S[:r]))

    B = S_r @ U_r.T

    x = np.ones(m_eff)
    z = x.copy()
    u = np.zeros_like(x)

    for _ in range(iters):
        grad = 2 * (B.T @ (B @ x)) - rho * (x - z + u)
        x = x + lr * grad
        x = np.clip(x, 0.0, 1.0)

        if m < m_eff:
            z[:] = 0.0
            top_idx = np.argsort(-(x + u))[:m]
            z[top_idx] = 1.0
        else:
            z = np.clip(x + u, 0.0, 1.0)

        u = u + x - z

    X_final = np.zeros(n * n)
    X_final[idx] = z
    X_final = X_final.reshape(n, n)

    return X_final, B

if __name__ == "__main__":

    n = 5  # 4x4 points
    m = 5  # number of matches
    A = np.random.randn(n*n, n*n)
    s = time.time()
    X_step1 = step1_linearized_matching(n, m, A)
    X, B = step2_lowrank_admm(A, X_step1, m=m, lr=0.5, rho=1.0, iters=50)
    rows, cols = linear_sum_assignment(-X)
    print(time.time() - s)

    print("final X:")
    print(np.round(X, 1))

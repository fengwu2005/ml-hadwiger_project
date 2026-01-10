import math
import numpy as np


def pairwise_distances(points: np.ndarray) -> np.ndarray:
    """Compute Euclidean pairwise distances for an (n, 2) array of 2D points."""
    diff = points[:, None, :] - points[None, :, :]
    return np.linalg.norm(diff, axis=-1)


def unit_distance_edges(points: np.ndarray, tol: float = 1e-3):
    """Return list of edges (i, j) whose distance is within [1-tol, 1+tol]."""
    n = points.shape[0]
    dists = pairwise_distances(points)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if abs(dists[i, j] - 1.0) <= tol:
                edges.append((i, j))
    return edges


def epsilon_edges(points: np.ndarray, epsilon: float):
    """Return list of edges (i, j) whose distance is <= epsilon.

    这是一个更通用的 ε-邻接图构造：只要两点距离不超过 epsilon 就连边，
    对应题目里“设置 epsilon 连边”的要求。
    """
    n = points.shape[0]
    dists = pairwise_distances(points)
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if dists[i, j] <= epsilon:
                edges.append((i, j))
    return edges


def random_points_in_square(n: int, side: float = 5.0, seed: int | None = None) -> np.ndarray:
    """Sample n random points in [0, side] x [0, side]."""
    rng = np.random.default_rng(seed)
    return rng.random((n, 2)) * side

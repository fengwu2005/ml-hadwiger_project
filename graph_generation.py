from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .geometry import unit_distance_edges


@dataclass
class GeometricGraph:
    points: np.ndarray  # (n, 2)
    edges: List[Tuple[int, int]]  # list of (i, j)

    @property
    def n(self) -> int:
        return self.points.shape[0]


def parallelogram_motif(side: float = 1.0) -> np.ndarray:
    """Return 4 points 1,2,3,4 where 1-2-3 and 2-3-4 are equilateral triangles.

    按你的描述：123 构成正三角形，234 构成正三角形。一个简单的坐标系选择是：

    - 2 = (0, 0)
    - 3 = (side, 0)
    - 1 = (side / 2,  sqrt(3)/2 * side)
    - 4 = (side / 2, -sqrt(3)/2 * side)

    这样 1-2-3 是朝上的正三角形，2-3-4 是朝下的正三角形，它们共有底边 2-3，
    合起来是一个菱形 / 平行四边形，每条边长都是 side。
    """

    h = np.sqrt(3.0) / 2.0 * side
    return np.array(
        [
            [0.5 * side,  h],   # 点 1
            [0.0,         0.0], # 点 2
            [side,        0.0], # 点 3
            [0.5 * side, -h],   # 点 4
        ],
        dtype=float,
    )


def random_parallelogram_field(
    n_motifs: int,
    epsilon: float,
    side: float = 10.0,
    seed: int | None = None,
    reuse_prob: float = 0.3,
) -> GeometricGraph:
    """Randomly scatter parallelogram motifs over [0, side]^2.

    - 一共撒 n_motifs 个“两个公边正三角形拼成的菱形” motif；
    - 对每个 motif：
        * 以概率 reuse_prob 选取一个已有点作为某个顶点，
          通过平移让该顶点与已有点重合（刻意产生重复点 / 重叠边）；
        * 否则就在区域内随机选一个中心点放置该 motif；
    - 所有 motif 撒完后，用 [1-epsilon, 1+epsilon] 的距离约束连边。
    """

    rng = np.random.default_rng(seed)
    motif = parallelogram_motif(side=1.0)

    pts = np.zeros((0, 2), dtype=float)

    margin = 1.0
    low = 0.0 if side <= 2 * margin else margin
    high = side if side <= 2 * margin else side - margin

    for _ in range(max(0, n_motifs)):
        use_reuse = pts.size > 0 and 0.0 < reuse_prob and rng.random() < reuse_prob

        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        R = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ]
        )
        base = motif @ R.T

        if use_reuse:
            # 随机选一个已有点，把 motif 的某个顶点对齐到该点
            idx_old = int(rng.integers(pts.shape[0]))
            target = pts[idx_old]
            idx_vertex = int(rng.integers(base.shape[0]))
            shift = target - base[idx_vertex]
            pts_new = base + shift
        else:
            cx = float(rng.uniform(low, high))
            cy = float(rng.uniform(low, high))
            pts_new = base + np.array([cx, cy])

        pts = np.vstack([pts, pts_new])

    if pts.size == 0:
        edges: List[Tuple[int, int]] = []
    else:
        edges = unit_distance_edges(pts, tol=epsilon)

    return GeometricGraph(points=pts, edges=edges)


def grow_on_unit_circles(
    base_graph: GeometricGraph,
    n_new_points: int,
    epsilon: float,
    seed: int | None = None,
) -> GeometricGraph:
    """Start from an existing graph and add points on unit circles of existing points.

    算法思想：
    - 初始有一批点（例如由若干平行四边形 motif 构成的结构化点集）；
    - 每一步：
      1. 从已有点中随机选一个 i；
      2. 在以该点为圆心、半径为 1 的圆上随机采样一个角度 θ，生成新点：
         x_new = x_i + (cos θ, sin θ)；
    - 重复 n_new_points 次；
    - 最后对所有点重新用 [1-epsilon, 1+epsilon] 的距离约束连边。

    这样可以保证：
    - 每个新点至少与它的“父点”之间距离为 1；
    - 由于多个单位圆之间会有交集，还会出现额外的单位距离边，整体图会更“稠密”、
      更有可能需要更多颜色。
    """

    rng = np.random.default_rng(seed)

    if base_graph.points.size == 0 or n_new_points <= 0:
        return base_graph

    pts = base_graph.points.copy()

    for _ in range(n_new_points):
        n = pts.shape[0]
        idx = int(rng.integers(n))
        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        offset = np.array([np.cos(theta), np.sin(theta)], dtype=float)
        new_pt = pts[idx] + offset
        pts = np.vstack([pts, new_pt])

    edges = unit_distance_edges(pts, tol=epsilon)
    return GeometricGraph(points=pts, edges=edges)


def equilateral_triangle_motif(side: float = 1.0) -> np.ndarray:
    """Return 3 points forming an equilateral triangle of edge length `side`."""

    h = np.sqrt(3.0) / 2.0 * side
    return np.array(
        [
            [0.0, 0.0],
            [side, 0.0],
            [0.5 * side, h],
        ],
        dtype=float,
    )


def add_random_triangles(
    base_graph: GeometricGraph,
    n_triangles: int,
    epsilon: float,
    side: float = 10.0,
    seed: int | None = None,
    reuse_prob: float = 0.3,
) -> GeometricGraph:
    """On top of an existing graph, randomly sprinkle equilateral triangles as whole motifs.

    - 每次撒入一个正三角形：随机选择中心位置 (在 [0, side]^2 内) 和旋转角度，
      然后将边长为 1 的 equilateral_triangle_motif 刚性变换过去；
    - 所有 motif 加入后，重新用 [1-eps, 1+eps] 的距离约束生成边集。

    这样最后得到的图中，会同时包含：
    - 初始若干平行四边形 motif；
    - 额外撒入的许多正三角形 motif；
    - 以及基于已有点单位圆撒点得到的复杂结构。
    """

    rng = np.random.default_rng(seed)
    tri = equilateral_triangle_motif(side=1.0)

    # 先收集所有点
    if base_graph.points.size == 0:
        pts = np.zeros((0, 2), dtype=float)
    else:
        pts = base_graph.points.copy()

    for _ in range(max(0, n_triangles)):
        use_reuse = pts.size > 0 and 0.0 < reuse_prob and rng.random() < reuse_prob

        theta = float(rng.uniform(0.0, 2.0 * np.pi))
        R = np.array(
            [
                [np.cos(theta), -np.sin(theta)],
                [np.sin(theta), np.cos(theta)],
            ]
        )

        base = tri @ R.T

        if use_reuse:
            idx_old = int(rng.integers(pts.shape[0]))
            target = pts[idx_old]
            idx_vertex = int(rng.integers(base.shape[0]))
            shift = target - base[idx_vertex]
            pts_tri = base + shift
        else:
            cx = float(rng.uniform(0.0, side))
            cy = float(rng.uniform(0.0, side))
            pts_tri = base + np.array([cx, cy])

        pts = np.vstack([pts, pts_tri])

    edges = unit_distance_edges(pts, tol=epsilon)
    return GeometricGraph(points=pts, edges=edges)


def merge_duplicate_points(graph: GeometricGraph, tol: float = 1e-6) -> GeometricGraph:
    """Merge points that are numerically almost identical into single vertices.

    - 先按坐标 / tol 做四舍五入分桶，把非常接近的点视为同一个等价类；
    - 对每个等价类保留第一出现的那个点坐标，得到新的点集；
    - 用等价类编号对原来的边做映射，去掉自环和重复边。
    """

    pts = graph.points
    edges = graph.edges

    if pts.size == 0:
        return graph

    # 用四舍五入后的整数坐标作为“去重 key”
    keys = np.round(pts / tol).astype(np.int64)
    _, idx_unique, inv = np.unique(keys, axis=0, return_index=True, return_inverse=True)

    new_points = pts[idx_unique]

    edge_set: set[tuple[int, int]] = set()
    for u, v in edges:
        uu = int(inv[u])
        vv = int(inv[v])
        if uu == vv:
            continue
        if uu > vv:
            uu, vv = vv, uu
        edge_set.add((uu, vv))

    new_edges = sorted(edge_set)
    return GeometricGraph(points=new_points, edges=new_edges)


def random_gaussian_field(
    n_seed_points: int,
    n_new_points: int,
    epsilon: float,
    side: float = 10.0,
    sigma: float = 0.5,
    seed: int | None = None,
) -> GeometricGraph:
    """Strategy 2: random points from a Gaussian mixture around existing points.

    - 先在 [0, side]^2 内均匀撒 n_seed_points 个初始点；
    - 然后迭代 n_new_points 次：
        * 从当前所有点中随机选一个作为中心；
        * 在该中心附近按 N(0, sigma^2 I) 采样一个偏移，得到新点；
        * 裁剪到 [0, side]^2 后加入点集；
    - 最后用 [1-epsilon, 1+epsilon] 的距离约束连边。
    """

    rng = np.random.default_rng(seed)

    n_seed_points = max(0, n_seed_points)
    n_new_points = max(0, n_new_points)

    if n_seed_points > 0:
        pts = rng.uniform(0.0, side, size=(n_seed_points, 2))
    else:
        pts = np.zeros((0, 2), dtype=float)

    for _ in range(n_new_points):
        if pts.size == 0:
            center = rng.uniform(0.0, side, size=(2,))
        else:
            idx = int(rng.integers(pts.shape[0]))
            center = pts[idx]

        delta = rng.normal(loc=0.0, scale=sigma, size=(2,))
        new_pt = center + delta
        new_pt = np.clip(new_pt, 0.0, side)
        pts = np.vstack([pts, new_pt])

    if pts.size == 0:
        edges: List[Tuple[int, int]] = []
    else:
        edges = unit_distance_edges(pts, tol=epsilon)

    return GeometricGraph(points=pts, edges=edges)


def _random_chain_motif(
    rng: np.random.Generator,
    epsilon: float,
    min_len: int,
    max_len: int,
    min_extra_pairs: int,
    max_shape_tries: int = 50,
    max_pair_tries: int = 200,
) -> np.ndarray | None:
    """Generate a chain of length L with unit edges plus at least
    one extra non-consecutive pair at distance ~1.

        - 先生成一条长度 L (min_len-max_len) 的链，相邻点距离恰为 1；
        - 随机挑若干对非相邻点，若至少有 min_extra_pairs 对距离落在 [1-eps,1+eps] 内，
            则认为该链“合法”，返回这些点的坐标；
    - 若尝试多次仍找不到这样的链，则返回 None。
    """

    for _ in range(max_shape_tries):
        L = int(rng.integers(min_len, max_len + 1))
        pts = np.zeros((L, 2), dtype=float)

        # 先固定第一条边在 x 轴方向，长度 1
        if L >= 2:
            pts[1] = np.array([1.0, 0.0], dtype=float)

        # 后续每一步随机转向，步长固定为 1
        for i in range(2, L):
            theta = float(rng.uniform(0.0, 2.0 * np.pi))
            step = np.array([np.cos(theta), np.sin(theta)], dtype=float)
            pts[i] = pts[i - 1] + step

        # 尝试在非相邻点中找到若干对距离 ~ 1
        ok = False
        if L >= 3:
            cnt = 0
            for _ in range(max_pair_tries):
                i = int(rng.integers(0, L))
                j = int(rng.integers(0, L))
                if i == j or abs(i - j) == 1:
                    continue
                d = float(np.linalg.norm(pts[i] - pts[j]))
                if abs(d - 1.0) <= epsilon:
                    cnt += 1
                    if cnt >= max(1, min_extra_pairs):
                        ok = True
                        break

        if ok:
            return pts

    return None


def random_chain_field(
    n_chains: int,
    epsilon: float,
    side: float = 10.0,
    seed: int | None = None,
    reuse_point_prob: float = 0.3,
    reuse_edge_prob: float = 0.3,
    min_chain_len: int = 3,
    max_chain_len: int = 15,
    min_extra_pairs: int = 1,
) -> GeometricGraph:
    """Strategy 3: place multiple unit-length chains with extra constraints.

    对于每一条链：
    - 先调用 _random_chain_motif 生成一条合法链（相邻边长为 1，且存在至少一对
      非相邻点之间距离也约为 1）；
    - 然后以三种方式之一放入全局图中：
        * 公共点：将链上的某个点对齐到已有点；
        * 公共边：将链上相邻两点对齐到已有的一条单位边；
        * 随机放：在区域 [0, side]^2 内随机平移放置。
    链本身的具体边集合最终还是通过 unit_distance_edges 统一确定。
    """

    rng = np.random.default_rng(seed)

    pts = np.zeros((0, 2), dtype=float)

    margin = 1.0
    low = 0.0 if side <= 2 * margin else margin
    high = side if side <= 2 * margin else side - margin

    for _ in range(max(0, n_chains)):
        chain = _random_chain_motif(
            rng,
            epsilon,
            min_len=min_chain_len,
            max_len=max_chain_len,
            min_extra_pairs=min_extra_pairs,
        )
        if chain is None:
            continue

        use_point = pts.size > 0 and rng.random() < max(0.0, reuse_point_prob)
        use_edge = False
        edge_anchors: list[tuple[int, int]] = []

        if not use_point and pts.shape[0] >= 2 and reuse_edge_prob > 0.0:
            # 先找已有图中的单位边作为“公共边”锚点
            edge_anchors = unit_distance_edges(pts, tol=epsilon)
            if edge_anchors and rng.random() < reuse_edge_prob:
                use_edge = True

        if use_edge and edge_anchors:
            # 公共边：选已有的一条单位边 (u,v)，把链上某个相邻点对 (i,i+1)
            # 刚性变换对齐到 (u,v)
            u, v = edge_anchors[int(rng.integers(len(edge_anchors)))]
            target_u = pts[u]
            target_v = pts[v]

            # 随机选择链上的一条边
            L = chain.shape[0]
            i = int(rng.integers(0, L - 1))
            p_i = chain[i]
            p_j = chain[i + 1]

            seg = p_j - p_i
            tgt = target_v - target_u
            if np.linalg.norm(seg) == 0.0 or np.linalg.norm(tgt) == 0.0:
                placed = None
            else:
                # 计算将 seg 旋转到 tgt 的旋转矩阵
                a = np.arctan2(seg[1], seg[0])
                b = np.arctan2(tgt[1], tgt[0])
                theta = b - a
                R = np.array(
                    [
                        [np.cos(theta), -np.sin(theta)],
                        [np.sin(theta), np.cos(theta)],
                    ]
                )
                base = (chain - p_i) @ R.T + target_u
                placed = base
        elif use_point:
            # 公共点：将链上的某个点对齐到已有点
            idx_old = int(rng.integers(pts.shape[0]))
            target = pts[idx_old]
            idx_v = int(rng.integers(chain.shape[0]))
            shift = target - chain[idx_v]
            placed = chain + shift
        else:
            # 随机放：平移到区域内部
            cx = float(rng.uniform(low, high))
            cy = float(rng.uniform(low, high))
            centroid = chain.mean(axis=0)
            shift = np.array([cx, cy]) - centroid
            placed = chain + shift

        if placed is None:
            continue

        pts = np.vstack([pts, placed])

    if pts.size == 0:
        edges: List[Tuple[int, int]] = []
    else:
        edges = unit_distance_edges(pts, tol=epsilon)

    return GeometricGraph(points=pts, edges=edges)


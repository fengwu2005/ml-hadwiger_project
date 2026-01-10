from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from ..geometry import unit_distance_edges
from .abstract_graph import AbstractGraph


@dataclass
class EmbeddingResult:
    points: np.ndarray  # (n, 2)
    loss: float
    n_edge_violations: int
    n_nonedge_violations: int


def _loss_and_grad(
    pts: np.ndarray,
    graph: AbstractGraph,
    epsilon: float,
    nonedge_margin: float,
    nonedge_weight: float,
) -> Tuple[float, np.ndarray, int, int]:
    """对给定坐标计算损失和梯度（更直观的带状约束形式）。

    记 d_ij = ||x_i - x_j||。

    - 对于边 (i, j)：
        希望长度落在单位带 [1 - epsilon, 1 + epsilon] 里。
        若 |d_ij - 1| <= epsilon，则不惩罚；
        否则按 (|d_ij - 1| - epsilon)^2 罚，这只惩罚“超出带宽的部分”。

    - 对于非边 (i, j)：
        希望远离 1，一般 nonedge_margin > epsilon。
        若 |d_ij - 1| >= nonedge_margin，则不惩罚；
        否则按 (nonedge_margin - |d_ij - 1|)^2 罚，
        表示“离开危险带 [1 - nonedge_margin, 1 + nonedge_margin] 不够远”。

    总损失 = 平均边损失 + nonedge_weight * 平均非边损失。
    """

    n = graph.n
    edges = graph.edge_tuples()
    pts = pts.reshape(n, 2)

    # 邻接矩阵用于快速判断是否为边
    is_edge = np.zeros((n, n), dtype=bool)
    for i, j in edges:
        is_edge[i, j] = True
        is_edge[j, i] = True

    grad = np.zeros_like(pts)
    edge_loss = 0.0
    nonedge_loss = 0.0
    n_edge_pairs = 0
    n_nonedge_pairs = 0

    n_edge_viol = 0
    n_nonedge_viol = 0

    for i in range(n):
        for j in range(i + 1, n):
            diff = pts[i] - pts[j]
            d = float(np.linalg.norm(diff) + 1e-9)

            if is_edge[i, j]:
                # 边：只惩罚“跑出 [1-eps, 1+eps] 的那一截”
                delta = abs(d - 1.0)
                if delta > epsilon:
                    over = delta - epsilon
                    l = over * over
                    edge_loss += l
                    n_edge_pairs += 1
                    n_edge_viol += 1

                    # dL / dd = 2 * over * sign(d - 1)
                    sign = 1.0 if (d - 1.0) > 0.0 else -1.0
                    dL_dd = 2.0 * over * sign
                    coeff = dL_dd / d
                    g = coeff * diff
                    grad[i] += g
                    grad[j] -= g
                else:
                    # 在允许带宽里，不计入损失，但计数一次 pair 方便平均
                    n_edge_pairs += 1
            else:
                # 非边：只惩罚“进入 [1-nonedge_margin, 1+nonedge_margin] 危险带”的情况
                delta = abs(d - 1.0)
                if delta < nonedge_margin:
                    gap = nonedge_margin - delta
                    l = gap * gap
                    nonedge_loss += l
                    n_nonedge_pairs += 1
                    n_nonedge_viol += 1

                    # dL / dd = -2 * gap * sign(d - 1)
                    sign = 1.0 if (d - 1.0) > 0.0 else -1.0
                    dL_dd = -2.0 * gap * sign
                    coeff = dL_dd / d
                    g = coeff * diff
                    grad[i] += g
                    grad[j] -= g
                else:
                    # 足够远，不计入损失但计数一次 pair 方便平均
                    n_nonedge_pairs += 1

    if n_edge_pairs > 0:
        edge_loss /= n_edge_pairs
    if n_nonedge_pairs > 0:
        nonedge_loss /= n_nonedge_pairs

    total_loss = edge_loss + nonedge_weight * nonedge_loss
    return total_loss, grad, n_edge_viol, n_nonedge_viol


def embed_abstract_graph(
    graph: AbstractGraph,
    epsilon: float = 0.05,
    side: float = 5.0,
    n_restarts: int = 5,
    n_steps: int = 2000,
    step_size: float = 0.01,
    nonedge_margin: float = 0.2,
    nonedge_weight: float = 0.5,
    seed: int | None = None,
) -> EmbeddingResult:
    """尝试将抽象图嵌入到平面上，使得：

    - 所有边距离接近 1；
    - 非边尽量不要距离接近 1；

    采用简单的梯度下降 + 多次重启，返回最好的那次结果。
    """

    rng = np.random.default_rng(seed)
    n = graph.n

    best_result: EmbeddingResult | None = None

    for r in range(n_restarts):
        pts = rng.uniform(0.0, side, size=(n, 2))

        for _ in range(n_steps):
            loss, grad, n_ev, n_nev = _loss_and_grad(
                pts, graph, epsilon, nonedge_margin, nonedge_weight
            )
            pts = pts - step_size * grad
            pts = np.clip(pts, 0.0, side)

        loss, _, n_ev, n_nev = _loss_and_grad(
            pts, graph, epsilon, nonedge_margin, nonedge_weight
        )

        if best_result is None or loss < best_result.loss:
            best_result = EmbeddingResult(
                points=pts.copy(),
                loss=float(loss),
                n_edge_violations=n_ev,
                n_nonedge_violations=n_nev,
            )

    assert best_result is not None
    return best_result


def realized_geometric_graph(
    emb: EmbeddingResult,
    epsilon: float,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """基于嵌入结果构造几何图：用当前代码的 unit_distance_edges 连边。"""

    pts = emb.points
    edges = unit_distance_edges(pts, tol=epsilon)
    return pts, edges

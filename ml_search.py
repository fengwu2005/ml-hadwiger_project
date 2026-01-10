from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .graph_generation import GeometricGraph
from .soft_coloring import train_soft_coloring
from .exact_coloring import greedy_dsat_coloring, chromatic_number_exact


@dataclass
class MLGraphScore:
    graph: GeometricGraph
    loss_3: float
    loss_4: float
    score: float  # loss_3 - loss_4
    approx_k: int
    exact_k: int
    exact_colors: List[int]


def evaluate_graph_ml(
    graph: GeometricGraph,
    n_soft_steps: int = 800,
) -> MLGraphScore:
    """对给定几何图做机器学习风格的评分。

    - 用 soft coloring 在 3 色和 4 色下分别训练，得到 loss_3 和 loss_4；
    - 使用 score = loss_3 - loss_4 作为“难以 3 染、相对容易 4 染”的度量；
    - 再用贪心近似色数 approx_k 和精确色数 exact_k 做对比与验证。
    """

    # soft 3-coloring
    res3 = train_soft_coloring(
        n_nodes=graph.n,
        edges=graph.edges,
        n_colors=3,
        n_steps=n_soft_steps,
        lr=0.1,
        n_restarts=1,
    )

    # soft 4-coloring
    res4 = train_soft_coloring(
        n_nodes=graph.n,
        edges=graph.edges,
        n_colors=4,
        n_steps=n_soft_steps,
        lr=0.1,
        n_restarts=1,
    )

    loss_3 = res3.loss
    loss_4 = res4.loss
    score = loss_3 - loss_4

    # 贪心近似色数
    _, approx_k = greedy_dsat_coloring(graph.n, graph.edges)

    # 精确色数与染色方案
    exact_k, exact_colors = chromatic_number_exact(graph.n, graph.edges)

    return MLGraphScore(
        graph=graph,
        loss_3=loss_3,
        loss_4=loss_4,
        score=score,
        approx_k=approx_k,
        exact_k=exact_k,
        exact_colors=exact_colors,
    )


def evaluate_graph_soft_greedy(
    graph: GeometricGraph,
    n_soft_steps: int = 400,
) -> Tuple[float, float, float, int]:
    """只用 soft 3/4-coloring + 贪心近似，快速评估一张图。

    返回 (loss_3, loss_4, score, approx_k)，不做精确回溯，
    适合在多次 trial 中作为 ML 筛选指标，最后再对少数候选跑 chromatic_number_exact。
    """

    res3 = train_soft_coloring(
        n_nodes=graph.n,
        edges=graph.edges,
        n_colors=3,
        n_steps=n_soft_steps,
        lr=0.1,
        n_restarts=1,
    )
    res4 = train_soft_coloring(
        n_nodes=graph.n,
        edges=graph.edges,
        n_colors=4,
        n_steps=n_soft_steps,
        lr=0.1,
        n_restarts=1,
    )

    loss_3 = res3.loss
    loss_4 = res4.loss
    score = loss_3 - loss_4

    _, approx_k = greedy_dsat_coloring(graph.n, graph.edges)
    return loss_3, loss_4, score, approx_k


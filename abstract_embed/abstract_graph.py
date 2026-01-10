from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from ..exact_coloring import chromatic_number_exact


@dataclass
class AbstractGraph:
    n: int
    edges: List[Tuple[int, int]]  # list of (i, j), 0 <= i < j < n

    def edge_tuples(self) -> List[Tuple[int, int]]:
        return [(int(i), int(j)) for (i, j) in self.edges]


def random_erdos_renyi_graph(
    n: int,
    p_edge: float,
    seed: int | None = None,
) -> AbstractGraph:
    """简单的 G(n, p) 抽象图生成器，不考虑几何结构。

    仅作为在抽象层面上随机搜索高色数图的起点。
    """

    rng = np.random.default_rng(seed)
    edges: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < p_edge:
                edges.append((i, j))
    return AbstractGraph(n=n, edges=edges)


def compute_chromatic_number(graph: AbstractGraph) -> int:
    """调用已有的精确染色算法，计算抽象图的色数。"""

    chi, _ = chromatic_number_exact(graph.n, graph.edge_tuples())
    return chi

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np

from .graph_generation import GeometricGraph, random_unit_distance_graph
from .soft_coloring import train_soft_coloring
from .exact_coloring import chromatic_number_upper_bound


@dataclass
class CandidateGraphInfo:
    graph: GeometricGraph
    soft_loss_4color: float
    chromatic_upper_bound: int | None


def evaluate_graph(graph: GeometricGraph, n_soft_steps: int = 1500) -> CandidateGraphInfo:
    # 1) ML 风格的“软 4 染色”
    soft_res = train_soft_coloring(
        n_nodes=graph.n,
        edges=graph.edges,
        n_colors=4,
        n_steps=n_soft_steps,
        lr=0.05,
        n_restarts=3,
    )

    # 2) 若软 loss 很大，再做精确 4- 可染性检查
    if soft_res.loss > 0.02:  # 阈值可调，越大越“保守”
        chi_ub = chromatic_number_upper_bound(graph.n, graph.edges, k_max=6)
    else:
        chi_ub = 4  # 基本可以找到近似 4- 染色

    return CandidateGraphInfo(
        graph=graph,
        soft_loss_4color=soft_res.loss,
        chromatic_upper_bound=chi_ub,
    )


def random_search(
    n_trials: int = 50,
    n_points: int = 30,
    seed: int | None = None,
) -> List[CandidateGraphInfo]:
    rng = np.random.default_rng(seed)
    results: List[CandidateGraphInfo] = []

    for t in range(n_trials):
        g = random_unit_distance_graph(
            n_points=n_points,
            side=5.0,
            tol=1e-1,
            min_avg_degree=2.0,
            max_tries=50,
            seed=int(rng.integers(1e9)),
        )
        info = evaluate_graph(g)
        results.append(info)
        print(
            f"Trial {t+1}/{n_trials}: n={g.n}, |E|={len(g.edges)}, "
            f"soft_loss_4={info.soft_loss_4color:.4f}, chi_ub={info.chromatic_upper_bound}",
            flush=True,
        )

    return results

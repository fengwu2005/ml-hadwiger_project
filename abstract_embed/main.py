from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from ..exact_coloring import chromatic_number_exact
from ..geometry import unit_distance_edges
from ..main import plot_colored_graph
from .abstract_graph import AbstractGraph, random_erdos_renyi_graph, compute_chromatic_number
from .embedder import EmbeddingResult, embed_abstract_graph, realized_geometric_graph


ROOT = Path(__file__).resolve().parent


def run_experiment():
    # 抽象图参数
    # 适当增加点数和样本数，提高出现 4-色抽象图的概率
    n_vertices = 10
    p_edge = 0.5
    n_graph_samples = 40
    target_chi = 4

    # 嵌入参数
    epsilon = 0.05   # 单位距离容差
    side = 5.0
    # 增加重启次数与步数，并稍微减小步长，让收敛更稳定
    n_restarts = 8
    n_steps = 2500
    step_size = 0.005
    # 非边的危险带设得比 epsilon 稍大一些；权重略小，主要优先满足边
    nonedge_margin = 0.25
    nonedge_weight = 0.3

    rng = np.random.default_rng(0)

    abstract_candidates: List[Tuple[AbstractGraph, int]] = []

    print("Sampling abstract graphs and computing exact chromatic numbers...")
    for t in range(n_graph_samples):
        g = random_erdos_renyi_graph(
            n=n_vertices,
            p_edge=p_edge,
            seed=int(rng.integers(1_000_000_000)),
        )
        chi = compute_chromatic_number(g)
        print(f"  graph {t+1}/{n_graph_samples}: n={g.n}, |E|={len(g.edges)}, chi={chi}")
        if chi >= target_chi:
            abstract_candidates.append((g, chi))

    if not abstract_candidates:
        print("No abstract graphs with chi >= target_chi were found.")
        return

    print(f"\nFound {len(abstract_candidates)} abstract graphs with chi >= {target_chi}.")

    best_realized_chi = -1
    best_realized_pts: np.ndarray | None = None
    best_realized_edges: List[Tuple[int, int]] | None = None
    best_colors: List[int] | None = None

    for idx, (g, chi_abs) in enumerate(abstract_candidates, start=1):
        print(
            f"\nEmbedding candidate {idx}/{len(abstract_candidates)}: "
            f"abstract chi={chi_abs}, |E|={len(g.edges)}"
        )

        emb: EmbeddingResult = embed_abstract_graph(
            g,
            epsilon=epsilon,
            side=side,
            n_restarts=n_restarts,
            n_steps=n_steps,
            step_size=step_size,
            nonedge_margin=nonedge_margin,
            nonedge_weight=nonedge_weight,
            seed=int(rng.integers(1_000_000_000)),
        )

        pts, edges_real = realized_geometric_graph(emb, epsilon=epsilon)
        chi_real, colors = chromatic_number_exact(len(pts), edges_real)

        print(
            f"  embedding loss={emb.loss:.4f}, "
            f"edge_viol={emb.n_edge_violations}, nonedge_viol={emb.n_nonedge_violations}"
        )
        print(
            f"  realized geometric graph: n={len(pts)}, |E|={len(edges_real)}, "
            f"chi_real={chi_real}"
        )

        if chi_real > best_realized_chi:
            best_realized_chi = chi_real
            best_realized_pts = pts
            best_realized_edges = edges_real
            best_colors = colors

    if best_realized_pts is None or best_realized_edges is None or best_colors is None:
        print("No realized geometric graphs were produced.")
        return

    print(
        f"\nBest realized geometric graph: chi_real={best_realized_chi}, "
        f"n={best_realized_pts.shape[0]}, |E|={len(best_realized_edges)}"
    )

    # 用已有的画图函数画出最好的例子
    from ..graph_generation import GeometricGraph

    graph = GeometricGraph(points=best_realized_pts, edges=best_realized_edges)

    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    save_path = out_dir / "abstract_embed_best.png"

    title = (
        f"abstract-embed: n={graph.n}, |E|={len(graph.edges)}, "
        f"chi_real={best_realized_chi}, eps_band={epsilon}"
    )

    plot_colored_graph(graph, best_colors, title=str(title), save_path=str(save_path))
    print(f"Saved best realized geometric graph to {save_path}")


if __name__ == "__main__":
    run_experiment()

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .graph_generation import (
    GeometricGraph,
    random_parallelogram_field,
    grow_on_unit_circles,
    add_random_triangles,
    merge_duplicate_points,
    random_gaussian_field,
    random_chain_field,
)
from .exact_coloring import greedy_dsat_coloring, chromatic_number_exact


ROOT = Path(__file__).resolve().parent


def plot_colored_graph(
    graph: GeometricGraph,
    colors: list[int],
    title: str,
    save_path: str | None = None,
):
    pts = graph.points
    edges = graph.edges

    plt.figure(figsize=(4, 4))
    # 先画边
    for i, j in edges:
        x = [pts[i, 0], pts[j, 0]]
        y = [pts[i, 1], pts[j, 1]]
        plt.plot(x, y, "k-", linewidth=0.5, alpha=0.4)

    # 再按颜色画点
    colors_arr = np.array(colors)
    num_colors = colors_arr.max() + 1 if colors_arr.size > 0 else 0
    cmap = plt.get_cmap("tab10")  # 最多 10 种明显不同的颜色
    for c in range(num_colors):
        mask = colors_arr == c
        if not np.any(mask):
            continue
        plt.scatter(pts[mask, 0], pts[mask, 1], c=[cmap(c)], s=40, label=f"color {c}")

    plt.title(title)
    plt.axis("equal")
    plt.legend(loc="best", fontsize=8)
    plt.tight_layout()

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=200)
        print(f"Saved figure to {save_path}")
    else:
        plt.show()

    plt.close()


def main():
    # ========== 全局参数：epsilon 控制 [1-eps, 1+eps] 单位距离带宽 ==========
    epsilon = 0.05           # 所有 unit_distance_edges 都使用这个容差：|d-1| <= epsilon
    side = 5                 # 正方形区域 [0, side]^2

    # strategy = 1 使用“菱形 + 正三角形 + 单位圆”结构化撒点；
    # strategy = 2 使用“高斯混合随机撒点”策略；
    # strategy = 3 使用“单位长度链 + 额外约束”策略。
    strategy = 1

    n_trials = 100            # 每种撒点策略下的重复次数

    # 方法一相关参数
    n_parallelograms = 30    # 随机撒入的平行四边形 motif 个数
    n_triangles = 10         # 额外撒入的正三角形个数
    n_extra = 30             # 在单位圆上新增点的数量

    # 方法二相关参数
    n_seed_points = 1       # 初始均匀撒点个数
    n_gaussian_points = 80   # 按高斯混合继续撒入的点数
    sigma = 0.2              # 高斯噪声标准差
    
    # 方法三相关参数
    n_chains = 20           # 随机加入的链 motif 数量
    min_chain_len = 3        # 链最短长度
    max_chain_len = 9       # 链最长长度
    min_extra_pairs = 3      # 每条链要求的“额外约束”对数（非相邻点距离≈1）

    rng = np.random.default_rng(0)

    # 候选列表：每个元素是 (图, 贪心近似色数)
    candidates: list[tuple[GeometricGraph, int]] = []

    for t in range(n_trials):
        base_seed = int(rng.integers(1_000_000_000))

        if strategy == 1:
            # 方法一：菱形 + 正三角形 + 单位圆撒点
            graph = random_parallelogram_field(
                n_motifs=n_parallelograms,
                epsilon=epsilon,
                side=side,
                seed=base_seed,
            )

            graph = add_random_triangles(
                base_graph=graph,
                n_triangles=n_triangles,
                epsilon=epsilon,
                side=side,
                seed=base_seed + 1,
            )

            graph = grow_on_unit_circles(
                base_graph=graph,
                n_new_points=n_extra,
                epsilon=epsilon,
                seed=base_seed + 2,
            )
        elif strategy == 2:
            # 方法二：以当前点为中心的二维高斯混合撒点
            graph = random_gaussian_field(
                n_seed_points=n_seed_points,
                n_new_points=n_gaussian_points,
                epsilon=epsilon,
                side=side,
                sigma=sigma,
                seed=base_seed,
            )
        else:
            # 方法三：由单位长度链 + 额外约束构成的 motif 随机场
            graph = random_chain_field(
                n_chains=n_chains,
                epsilon=epsilon,
                side=side,
                seed=base_seed,
                min_chain_len=min_chain_len,
                max_chain_len=max_chain_len,
                min_extra_pairs=min_extra_pairs,
            )

        # 合并数值上重复的点，去掉重复边
        graph = merge_duplicate_points(graph)

        # Step 5: 只用组合算法评估：贪心 DSAT 得到近似色数 approx_k
        _, approx_k = greedy_dsat_coloring(graph.n, graph.edges)

        print(
            f"Trial {t+1}/{n_trials}: n={graph.n}, |E|={len(graph.edges)}, "
            f"approx_k={approx_k}",
        )

        candidates.append((graph, approx_k))

    # trial 结束后，只对少数最有希望的候选图做精确回溯，保证 exact_k 精确同时控制时间
    if not candidates:
        print("No candidates generated.")
        return

    # 先按 approx_k 再按边数从大到小排序
    candidates_sorted = sorted(
        candidates,
        key=lambda x: (x[1], len(x[0].edges)),  # 先看贪心近似色数，再看边数
        reverse=True,
    )

    # 优先对所有 approx_k == 4 的图做精确染色
    approx4_candidates = [(g, ak) for (g, ak) in candidates_sorted if ak == 4]

    best_graph = None
    best_exact_k = -1
    best_colors = None

    if approx4_candidates:
        total = len(approx4_candidates)
        print(f"\nFound {total} candidates with approx_k = 4. Running exact coloring on all of them...")
        for idx, (g, ak) in enumerate(approx4_candidates, start=1):
            print(
                f"  [{idx}/{total}] evaluating: n={g.n}, |E|={len(g.edges)}",
            )
            exact_k, colors = chromatic_number_exact(g.n, g.edges)
            print(
                f"    -> approx_k={ak}, exact_k={exact_k}",
            )

            if exact_k > best_exact_k:
                best_exact_k = exact_k
                best_graph = g
                best_colors = colors
    else:
        # 若没有 approx_k=4 的图，则退化为：只评估 top_k 个候选
        top_k = min(3, len(candidates_sorted))
        print(f"\nNo candidates with approx_k = 4. Running exact coloring on top {top_k} candidates instead...")

        for idx in range(top_k):
            g, ak = candidates_sorted[idx]
            print(
                f"  [{idx+1}/{top_k}] evaluating: n={g.n}, |E|={len(g.edges)}",
            )
            exact_k, colors = chromatic_number_exact(g.n, g.edges)
            print(
                f"    -> approx_k={ak}, exact_k={exact_k}",
            )

            if exact_k > best_exact_k:
                best_exact_k = exact_k
                best_graph = g
                best_colors = colors

    assert best_graph is not None and best_colors is not None

    graph = best_graph
    # 重新算一次贪心近似色数（也可以直接沿用上面的 ak）
    _, approx_k = greedy_dsat_coloring(graph.n, graph.edges)
    exact_k = best_exact_k
    exact_colors = best_colors

    print("\nBest candidate over all trials:")
    print(
        f"n={graph.n}, |E|={len(graph.edges)}, "
        f"approx_k={approx_k}, exact_k={exact_k}",
    )

    # ========== 保存带有精确染色的图片 ==========
    out_dir = ROOT / "outputs"
    out_dir.mkdir(exist_ok=True)
    save_path = out_dir / "parallelogram_unitcircle_colored.png"

    title = (
        f"n={graph.n}, |E|={len(graph.edges)}, "
        f"approx_k={approx_k}, exact_k={exact_k}, "
        f"eps_band={epsilon}"
    )

    plot_colored_graph(graph, exact_colors, title=title, save_path=str(save_path))

    print("Done. Colored graph saved.")


if __name__ == "__main__":
    main()

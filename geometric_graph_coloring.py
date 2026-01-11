#!/usr/bin/env python3
from __future__ import annotations

from typing import List, Optional, Set, Tuple
import numpy as np

# ======== 配置 ========
N: int = 100
EPSILON: float = 0.15
SEED: Optional[int] = None  # 设为 None 以获得非确定性结果
# =====================


def sample_points(n: int, rng: np.random.Generator) -> np.ndarray:
    # 每个坐标 ~ N(0, 1)
    return rng.normal(loc=0.0, scale=1.0, size=(n, 2))


def build_edges(points: np.ndarray, epsilon: float) -> Set[Tuple[int, int]]:
    n = points.shape[0]
    edges: Set[Tuple[int, int]] = set()
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(points[i] - points[j]))
            if (1.0 - epsilon) <= d <= (1.0 + epsilon):
                edges.add((i, j))
    return edges


def build_adjacency(n: int, edges: Set[Tuple[int, int]]) -> List[Set[int]]:
    adj: List[Set[int]] = [set() for _ in range(n)]
    for i, j in edges:
        adj[i].add(j)
        adj[j].add(i)
    return adj


def k_colorable(adj: List[Set[int]], k: int) -> Optional[List[int]]:
    """尝试用 k 种颜色给图着色，成功返回颜色数组，否则返回 None。"""
    n = len(adj)
    order = sorted(range(n), key=lambda v: len(adj[v]), reverse=True)
    color = [-1] * n

    def backtrack(idx: int) -> bool:
        if idx == n:
            return True
        v = order[idx]
        forbidden = {color[u] for u in adj[v] if color[u] != -1}
        for c in range(k):
            if c not in forbidden:
                color[v] = c
                if backtrack(idx + 1):
                    return True
                color[v] = -1
        return False

    if backtrack(0):
        remapped = [-1] * n
        for idx, v in enumerate(order):
            remapped[v] = color[v]
        return remapped
    return None


def chromatic_number(adj: List[Set[int]]) -> Tuple[int, List[int]]:
    """精确计算染色数（最小颜色数），返回 (chi, coloring)。"""
    n = len(adj)
    for k in range(1, n + 1):
        coloring = k_colorable(adj, k)
        if coloring is not None:
            return k, coloring
    return n, list(range(n))


def main() -> None:
    rng = np.random.default_rng(SEED)
    print(f"N={N}, EPSILON={EPSILON}, SEED={SEED}")

    pts = sample_points(N, rng)
    edges = build_edges(pts, EPSILON)
    adj = build_adjacency(N, edges)

    print(f"生成点数: {N}")
    for i, (x, y) in enumerate(pts):
        print(f"  {i}: {x:.6f}, {y:.6f}")

    print(f"边数: {len(edges)}")
    if edges:
        print("边列表:")
        for (i, j) in sorted(edges):
            print(f"  ({i}, {j})")

    chi, coloring = chromatic_number(adj)
    print(f"染色数: {chi}")
    
    print("顶点颜色（按索引）:")
    for i, c in enumerate(coloring):
        print(f"  {i}: {c}")
        

if __name__ == "__main__":
    main()
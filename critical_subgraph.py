from __future__ import annotations

from typing import List, Tuple

from .graph_generation import GeometricGraph
from .exact_coloring import chromatic_number_exact


def _induced_subgraph(graph: GeometricGraph, keep_vertices: List[int]) -> GeometricGraph:
    """按给定顶点集合取诱导子图，并重新压缩顶点编号。

    keep_vertices: 原图中的顶点编号列表，例如 [0, 3, 5, ...]。
    返回的新图中，顶点会被重编号为 0..k-1，保持几何坐标不变。
    """

    keep_set = set(keep_vertices)
    if not keep_set:
        return GeometricGraph(points=graph.points[:0], edges=[])

    # 旧编号 -> 新编号
    mapping = {old: new for new, old in enumerate(sorted(keep_set))}

    # 点坐标按新顺序排列
    new_points = graph.points[sorted(keep_set)]

    # 过滤并重映射边
    new_edges: List[Tuple[int, int]] = []
    for u, v in graph.edges:
        if u in keep_set and v in keep_set:
            uu = mapping[u]
            vv = mapping[v]
            if uu == vv:
                continue
            if uu > vv:
                uu, vv = vv, uu
            new_edges.append((uu, vv))

    # 去重边
    new_edges = sorted(set(new_edges))
    return GeometricGraph(points=new_points, edges=new_edges)


def find_vertex_critical_4chromatic_subgraph(
    graph: GeometricGraph,
    *,
    max_states: int | None = None,
) -> GeometricGraph:
    """在保持原图色数不变的前提下，尽量删点，得到一个顶点临界子图。

    记输入图的色数为 k = χ(G)。算法返回的子图 H 满足：
    - χ(H) = k（与原图相同的色数）；
    - 对 H 中任意顶点 v，删去 v 得到的诱导子图 H - v 的色数 < k。

    即：在“保持色数不降”的约束下做贪心删点，直到再删任何一个点都会降低色数，
    得到一个顶点临界的 k-色图。k 可以是 4，也可以是其它值。
    """

    target_chi, _ = chromatic_number_exact(
        graph.n,
        graph.edges,
        max_states=max_states,
    )

    # 对于色数 < 2 的平凡情况，直接返回原图
    if target_chi <= 1:
        return graph

    # 初始顶点集合：全部顶点；按度数从小到大排序，优先尝试删除低度顶点
    degrees = [0] * graph.n
    for u, v in graph.edges:
        degrees[u] += 1
        degrees[v] += 1

    # vertices 始终使用“原图编号”的坐标系
    vertices = list(range(graph.n))
    vertices.sort(key=lambda v: degrees[v])

    # 贪心删点：每一轮尽量删除一个还不影响色数的顶点
    # 理论上一个 k-色图至少需要 k 个顶点，这里用 target_chi 作为下界，
    # 在 |V| > target_chi 的前提下尽可能多删点。
    changed = True
    while changed and len(vertices) > target_chi:
        changed = False
        for v in list(vertices):  # 遍历一个拷贝，避免原地修改干扰
            trial_vertices = [u for u in vertices if u != v]
            trial_graph = _induced_subgraph(graph, trial_vertices)
            chi_trial, _ = chromatic_number_exact(
                trial_graph.n,
                trial_graph.edges,
                max_states=max_states,
            )
            if chi_trial == target_chi:
                # 删除 v 不改变色数，可以接受这次删点
                vertices = trial_vertices
                changed = True
                break

    # 最终再从原图上取一次诱导子图，避免编号/索引错位
    return _induced_subgraph(graph, vertices)

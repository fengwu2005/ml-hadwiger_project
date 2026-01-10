from __future__ import annotations

from typing import List, Tuple


def build_adjacency(n_nodes: int, edges: List[Tuple[int, int]]) -> List[List[int]]:
    adj: List[List[int]] = [[] for _ in range(n_nodes)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return adj


def greedy_dsat_coloring(n_nodes: int, edges: List[Tuple[int, int]]) -> Tuple[List[int], int]:
    """Fast heuristic coloring using a DSAT-like greedy strategy.

    返回：每个点的颜色编号以及使用的颜色数 (这是一个上界，近似地反映“至少需要多少颜色”)。
    """

    adj = build_adjacency(n_nodes, edges)
    colors = [-1] * n_nodes
    neighbor_colors = [set() for _ in range(n_nodes)]
    uncolored = set(range(n_nodes))

    while uncolored:
        # 选 saturation（邻居已用颜色数）最大的点；若相同则选度数更大的
        u = max(uncolored, key=lambda v: (len(neighbor_colors[v]), len(adj[v])))

        used = neighbor_colors[u]
        c = 0
        while c in used:
            c += 1
        colors[u] = c
        uncolored.remove(u)

        for v in adj[u]:
            if v in uncolored:
                neighbor_colors[v].add(c)

    num_colors = max(colors) + 1 if colors else 0
    return colors, num_colors


def chromatic_number_exact(
    n_nodes: int,
    edges: List[Tuple[int, int]],
    max_states: int | None = None,
) -> Tuple[int, List[int]]:
    """Exact chromatic number via DSATUR-style backtracking.

    使用贪心 DSAT 得到一个初始上界，然后在同样的顺序上做回溯搜索，
    得到真正的最小颜色数以及对应的一个最优染色方案。
    """

    adj = build_adjacency(n_nodes, edges)

    # 先用贪心得到一个上界和初始染色（不一定最优）
    greedy_colors, greedy_k = greedy_dsat_coloring(n_nodes, edges)

    best_k = greedy_k
    best_colors = greedy_colors[:]

    colors = [-1] * n_nodes
    neighbor_colors = [set() for _ in range(n_nodes)]
    nodes_visited = 0

    def choose_vertex() -> int | None:
        uncolored = [i for i in range(n_nodes) if colors[i] == -1]
        if not uncolored:
            return None
        # DSATUR 选择规则：饱和度最大，若相同则度数最大
        return max(uncolored, key=lambda v: (len(neighbor_colors[v]), len(adj[v])))

    def backtrack(current_k: int):
        nonlocal best_k, best_colors, nodes_visited

        # 可选的状态数量上限；默认 None 表示不裁剪，保证精确性
        if max_states is not None:
            if nodes_visited >= max_states:
                return
            nodes_visited += 1

        v = choose_vertex()
        if v is None:
            # 所有点都已染色，更新最优解
            if current_k < best_k:
                best_k = current_k
                best_colors = colors[:]
            return

        used = neighbor_colors[v]

        # 颜色尝试顺序：从 0 到 current_k，如果还能引入新颜色，则为 current_k
        for c in range(current_k + 1):
            if c in used:
                continue
            # 剪枝：如果使用新颜色会使颜色数 >= 当前最优解，则没必要继续
            new_k = max(current_k, c + 1)
            if new_k >= best_k:
                continue

            # 赋色并更新邻居的邻接颜色集合
            colors[v] = c
            changed_neighbors: List[Tuple[int, int]] = []
            for u in adj[v]:
                if colors[u] == -1 and c not in neighbor_colors[u]:
                    neighbor_colors[u].add(c)
                    changed_neighbors.append((u, c))

            backtrack(new_k)

            # 回溯撤销修改
            colors[v] = -1
            for u, col in changed_neighbors:
                neighbor_colors[u].remove(col)

    # 从 0 种颜色开始向上搜索，best_k 作为上界做剪枝
    backtrack(current_k=0)

    return best_k, best_colors

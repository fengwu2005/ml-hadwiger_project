#!/usr/bin/env python3
"""
Search for a 2D point set X of size N such that the geometric graph
constructed by connecting pairs whose distance lies in [1 - epsilon, 1 + epsilon]
contains a required edge-set substructure.

Configuration:
- Edit the GLOBALS below (N, REQUIRED_EDGES, EPSILON, MAX_ATTEMPTS, SEED)
- REQUIRED_EDGES are vertex index pairs (i, j) with 0 <= i < j < N

By default, the script tries a simple target (one edge), so it should
find a solution quickly. Adjust as needed for your case.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple
import itertools

import numpy as np

# ======== GLOBALS: Edit these to your needs ========
N: int = 7
REQUIRED_EDGES: List[Tuple[int, int]] = [
    # Examples; ensure values are within [0, N-1] and i<j
    (0, 1),
    (0, 2),
    (1, 2),
    (1, 3),
    (2, 3),

    (0, 4),
    (0, 5),
    (4, 5),
    (4, 6),
    (5, 6),

    (3, 6),
]
EPSILON: float = 0.15
MAX_ATTEMPTS: int = 10000000
SEED: Optional[int] = None  # set to None for non-deterministic runs
OUTPUT_PREFIX: Path = Path("found_points")  # outputs: .npy and .csv
# ================================================


@dataclass
class SearchResult:
    points: np.ndarray  # shape (N, 2)
    attempts: int
    edges: Set[Tuple[int, int]]


def validate_globals(n: int, required_edges: Sequence[Tuple[int, int]], epsilon: float) -> None:
    if n <= 0:
        raise ValueError("N must be positive")
    if epsilon < 0 or epsilon >= 1:
        # You can allow >=1, but keep moderate so search is meaningful
        raise ValueError("EPSILON must be in [0, 1)")
    for (i, j) in required_edges:
        if not (0 <= i < j < n):
            raise ValueError(f"Invalid edge ({i}, {j}) for N={n}")


def sample_points(n: int, rng: np.random.Generator) -> np.ndarray:
    # Each coordinate ~ N(0, 1)
    return rng.normal(loc=0.0, scale=1.0, size=(n, 2))


def build_edges(points: np.ndarray, epsilon: float) -> Set[Tuple[int, int]]:
    n = points.shape[0]
    edges: Set[Tuple[int, int]] = set()
    # Compute pairwise distances; simple double loop is fine for moderate n
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(points[i] - points[j]))
            if (1.0 - epsilon) <= d <= (1.0 + epsilon):
                edges.add((i, j))
    return edges


def has_required(edges: Set[Tuple[int, int]], required: Iterable[Tuple[int, int]]) -> bool:
    # Must contain all required edges as a subset
    re = False
    ps = list(itertools.permutations(list(range(N))))

    for permutation in ps:
        edges_new = [(permutation[x], permutation[y]) for x, y in edges]
        if all(e in edges for e in required): re =True

    return re


def search_until_found(n: int, required: Sequence[Tuple[int, int]], epsilon: float, max_attempts: int, seed: Optional[int]) -> Optional[SearchResult]:
    validate_globals(n, required, epsilon)
    rng = np.random.default_rng(seed)
    mx = 0

    for attempt in range(1, max_attempts + 1):
        pts = sample_points(n, rng)
        edges = build_edges(pts, epsilon)
        mx = len(edges) if mx < len(edges) else mx

        if has_required(edges, required):
            return SearchResult(points=pts, attempts=attempt, edges=edges)
        if attempt % 5000 == 0:
            print(f"Attempts: {attempt}, max edge count={mx}")
    return None


def save_points(prefix: Path, result: SearchResult) -> None:
    np.save(prefix.with_suffix(".npy"), result.points)
    with prefix.with_suffix(".csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["x", "y"])  # header
        writer.writerows(result.points.tolist())


def main() -> None:
    print(f"Searching with N={N}, EPSILON={EPSILON}, MAX_ATTEMPTS={MAX_ATTEMPTS}")
    print(f"Required edges: {REQUIRED_EDGES}")

    res = search_until_found(N, REQUIRED_EDGES, EPSILON, MAX_ATTEMPTS, SEED)
    if res is None:
        print("No point set found within max attempts. Try increasing MAX_ATTEMPTS or EPSILON.")
        return

    print(f"Found after {res.attempts} attempts; edge count={len(res.edges)}")
    print("Points (x, y):")
    for i, (x, y) in enumerate(res.points):
        print(f"  {i}: {x:.6f}, {y:.6f}")

    OUTPUT_PREFIX.parent.mkdir(parents=True, exist_ok=True)
    save_points(OUTPUT_PREFIX, res)
    print(f"Saved points to {OUTPUT_PREFIX.with_suffix('.npy')} and {OUTPUT_PREFIX.with_suffix('.csv')}")


if __name__ == "__main__":
    main()

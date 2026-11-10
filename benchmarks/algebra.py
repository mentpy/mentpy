# Copyright 2026 Luis Mantilla
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Run ``python -m benchmarks.algebra`` to compare complete native calls.

The closure reference retains the previous pair queue using the current, faster
PauliOp front end. Input creation is excluded; result construction is included.
"""

from collections import deque
from itertools import combinations
import platform
import time

import networkx as nx
import numpy as np
import mentpy as mp
from mentpy.utils import lc_equivalence as lc


def python_closure(generators):
    """Prior Pauli-object closure, retained here only as a benchmark reference."""
    result = mp.PauliOp(generators.matrix.copy())
    queue = deque(combinations(result, 2))
    seen = set()
    while queue:
        left, right = queue.popleft()
        pair = tuple(sorted((left.txt, right.txt)))
        if pair in seen:
            continue
        seen.add(pair)
        product = left.commutator(right)
        if isinstance(product, mp.PauliOp) and product not in result:
            result.append(product)
            queue.extend((product, old) for old in result if old != product)
    identity = mp.PauliOp("I" * result.number_of_qubits)
    if identity not in result:
        result.append(identity)
    return result


def timed(function, repeat=3):
    """Return the median after one warmup."""
    function()
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        samples.append(time.perf_counter() - start)
    return float(np.median(samples))


def main():
    """Check results before reporting warmed CPU timings."""
    print(platform.platform(), "NumPy", np.__version__)
    for n in (2, 3, 4):
        circuit = mp.templates.grid_cluster(n, n + 3)
        generators = mp.utils.calculate_gens(circuit)
        native = lambda: mp.utils.lie_algebra_completion(generators, 100000)
        reference = lambda: python_closure(generators)
        assert native() == reference()
        print(
            f"closure {n} wires, {len(native())} terms: Python {timed(reference):.6f}s, Rust {timed(native):.6f}s",
            flush=True,
        )
    for n in (16, 32, 64):
        graph = nx.gnp_random_graph(n, 0.15, seed=44)
        graph.add_edges_from(nx.path_graph(n).edges)
        other = graph.copy()
        for u, v in combinations(list(graph[0]), 2):
            if other.has_edge(u, v):
                other.remove_edge(u, v)
            else:
                other.add_edge(u, v)

        def reference():
            g = nx.to_numpy_array(graph, dtype=int, weight=None)
            h = nx.to_numpy_array(other, dtype=int, weight=None)
            return lc.search_nullspace(
                lc.nullspace_basis(lc.lc_constraint_system(g, h))
            )

        native = lambda: lc.are_lc_equivalent(graph, other)
        assert native()[0] and reference() is not None
        print(
            f"LC {n} nodes: Python {timed(reference):.6f}s, Rust {timed(native):.6f}s",
            flush=True,
        )


if __name__ == "__main__":
    main()

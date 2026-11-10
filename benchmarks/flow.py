# Copyright 2026 Luis Mantilla
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Compare complete Rust flow solves with the Python reference implementations.

Run ``python -m benchmarks.flow`` from an editable MentPy install. Includes
graph-to-array conversion and Python result construction, but excludes graph
generation. No hardware-independent speedup is assumed.
"""

import importlib
import platform
import time

import networkx as nx
import numpy as np
import mentpy as mp

flow = importlib.import_module("mentpy.mbqc.flow")
linalg2 = importlib.import_module("mentpy.calculator.linalg2")


def timed(function, repeat=3):
    """Return median wall time after one warmup."""
    function()
    times = []
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def main():
    """Time generalized flow and square/rectangular algebraic Pauli flow."""
    native = flow._rust_ext
    native_solver = linalg2._rust_ext
    if native is None:
        raise RuntimeError("Build the MentPy Rust extension before benchmarking.")
    print(platform.platform(), "NumPy", np.__version__)
    rng = np.random.default_rng(13)
    cases = []
    for n in (32, 64, 128):
        graph = nx.Graph()
        graph.add_nodes_from(range(2 * n))
        matrix = np.triu(rng.integers(0, 2, (n, n), dtype=np.uint8), 1)
        matrix |= np.eye(n, dtype=np.uint8)
        graph.add_edges_from((int(i), int(j) + n) for i, j in zip(*matrix.nonzero()))
        cases.append(
            (
                f"gflow {2*n} nodes",
                flow.find_gflow,
                (graph, list(range(n)), list(range(n, 2 * n))),
            )
        )
    for width in (2, 4, 6):
        pattern = mp.templates.grid_cluster(width, 12)
        planes = {node: "XY" for node in pattern.graph}
        for extra in (0, 1):
            cases.append(
                (
                    f"pflow {len(pattern)} nodes, {extra} free columns",
                    flow.find_pflow,
                    (
                        pattern.graph,
                        pattern.input_nodes[extra:],
                        pattern.output_nodes,
                        planes,
                    ),
                )
            )
    try:
        for label, function, args in cases:
            times = {}
            for name, extension in (("Python", None), ("Rust", native)):
                flow._rust_ext = extension
                linalg2._rust_ext = extension
                result = function(*args)
                assert result[0] is not None and result[0] is not False
                times[name] = timed(lambda: function(*args))
            print(
                f"{label}: Python {times['Python']:.6f}s, Rust {times['Rust']:.6f}s, ratio {times['Python']/times['Rust']:.2f}"
            )
    finally:
        flow._rust_ext = native
        linalg2._rust_ext = native_solver


if __name__ == "__main__":
    main()

# Copyright 2026 Luis Mantilla
# Licensed under the Apache License, Version 2.0.
"""Compare exact native Pauli and tensor gradients; time wide stabilizers.

Run ``python -m benchmarks.pauli`` from a release build. Small-case values and
gradients are checked against the tensor backend before timing. Wide graph
states use an analytically known stabilizer and never allocate output states.
"""

import platform
import time

import networkx as nx
import numpy as np
import mentpy as mp


def median(call, repeat=20):
    samples = []
    for _ in range(repeat):
        start = time.perf_counter()
        call()
        samples.append(time.perf_counter() - start)
    return np.median(samples)


def main():
    print(platform.platform(), "NumPy", np.__version__)
    print("pattern,backend,gradient_s,peak_terms")
    for name, pattern in [
        ("chain101", mp.templates.linear_cluster(101)),
        ("chain1001", mp.templates.linear_cluster(1001)),
        ("grid3x6", mp.templates.grid_cluster(3, 6)),
        ("grid5x5", mp.templates.grid_cluster(5, 5)),
    ]:
        angles = np.linspace(0.1, 0.7, len(pattern.trainable_nodes))
        n = len(pattern.quantum_output_nodes)
        obs = mp.Observable({"Y" * n: 0.7, "Z" * n: 0.3})
        ref = mp.PatternSimulator(pattern, backend="rust")
        expected, diff = ref.expectation_and_grad(angles, obs)
        for backend in ("pauli", "rust"):
            sim = mp.PatternSimulator(pattern, backend=backend)
            value, gradient = sim.expectation_and_grad(angles, obs)
            np.testing.assert_allclose(value, expected, atol=1e-11)
            np.testing.assert_allclose(gradient, diff, atol=1e-11)
            elapsed = median(lambda: sim.expectation_and_grad(angles, obs))
            print(
                f"{name},{backend},{elapsed:.8f},{getattr(sim, 'peak_terms', '')}",
                flush=True,
            )
    print("wide_graph,expectation_s,peak_terms")
    for n in (100, 500, 1024):
        pattern = mp.MBQCircuit(mp.GraphState(nx.path_graph(n)), [], list(range(n)))
        sim = mp.PatternSimulator(pattern, backend="pauli", max_terms=1)
        obs = mp.Observable("XZ" + "I" * (n - 2))
        assert sim.expectation([], obs) == 1
        elapsed = median(lambda: sim.expectation([], obs))
        print(f"{n},{elapsed:.8f},{sim.peak_terms}", flush=True)


if __name__ == "__main__":
    main()

# Copyright 2026 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Compare complete sampled trajectory and finite-shot calls.

Run ``python -m benchmarks.trajectories`` from a release build. Native and
NumPy paths use the same random draws and are checked before reporting times.
"""

import platform
import time

import numpy as np
import mentpy as mp


def timed(function, repeat=7):
    durations = []
    for _ in range(repeat):
        start = time.perf_counter()
        function()
        durations.append(time.perf_counter() - start)
    return float(np.median(durations))


def main():
    print(platform.platform(), "NumPy", np.__version__)
    print("nodes,backend,trajectory_s,shots_1024_s")
    for size in [21, 101, 1001]:
        pattern = mp.templates.linear_cluster(size)
        angles = np.linspace(0.1, 0.7, len(pattern.trainable_nodes))
        native = mp.PatternSimulator(pattern, backend="rust", force0=False)
        reference = mp.PatternSimulator(pattern, backend="numpy", force0=False)
        np.testing.assert_allclose(
            native.run(angles, seed=19), reference.run(angles, seed=19), atol=1e-11
        )
        np.testing.assert_array_equal(
            native.sample(angles, shots=1024, seed=23).bitstrings,
            reference.sample(angles, shots=1024, seed=23).bitstrings,
        )
        for backend, sim in [("rust", native), ("numpy", reference)]:
            run = timed(lambda: sim.run(angles, seed=19))
            shots = timed(lambda: sim.sample(angles, shots=1024, seed=23), repeat=3)
            print(f"{size},{backend},{run:.8f},{shots:.8f}", flush=True)


if __name__ == "__main__":
    main()

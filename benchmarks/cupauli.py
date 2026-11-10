# Copyright 2026 Luis Mantilla
# Licensed under the Apache License, Version 2.0.
"""Compare complete native CPU/GPU Pauli calls, including gradients.

Run ``python -m benchmarks.cupauli --device cuda:0`` on an NVIDIA GPU.
First-call latency includes initialization. Repeated calls include Python/Rust
marshalling, GPU allocations, propagation, and readout. This compares the same
Pauli algorithm on two devices; tensor contraction can be better on some graphs.
"""

import argparse
import platform
import time

import numpy as np
import mentpy as mp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--repeat", type=int, default=5)
    args = parser.parse_args()
    print(platform.platform(), "NumPy", np.__version__)
    print("width,depth,gradient,device,first_s,median_s,peak_terms")
    for width, depth in [(4, 8), (6, 6), (8, 5), (10, 4)]:
        pattern = mp.templates.grid_cluster(width, depth)
        angles = np.linspace(0.1, 0.7, len(pattern.trainable_nodes))
        observable = mp.Observable({"Z" * width: 0.7, "Y" * width: 0.3})
        for gradient in (False, True):
            expected = None
            for device in ("cpu", args.device):
                sim = mp.PatternSimulator(
                    pattern,
                    backend="pauli",
                    device=device,
                    max_terms=2000000,
                    max_memory=2**30,
                )

                def evaluate():
                    if gradient:
                        return sim.expectation_and_grad(angles, observable)
                    return sim.expectation(angles, observable), np.array([])

                start = time.perf_counter()
                result = evaluate()
                first = time.perf_counter() - start
                if expected is None:
                    expected = result
                else:
                    np.testing.assert_allclose(result[0], expected[0], atol=1e-11)
                    np.testing.assert_allclose(result[1], expected[1], atol=1e-11)
                durations = []
                for _ in range(args.repeat):
                    start = time.perf_counter()
                    evaluate()
                    durations.append(time.perf_counter() - start)
                print(
                    f"{width},{depth},{gradient},{device},{first:.8f},{np.median(durations):.8f},{sim.peak_terms}",
                    flush=True,
                )


if __name__ == "__main__":
    main()

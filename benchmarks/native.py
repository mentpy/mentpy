# Copyright 2026 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Benchmark complete native gradient calls, including the Python interface.

Run ``python -m benchmarks.native --jax`` from a release build. Use
``--device cuda:0`` with a CUDA build. This benchmark checks results before
reporting timings and separates JIT compilation from warmed execution.
"""

import argparse
import platform
import time

import numpy as np
import mentpy as mp


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--jax", action="store_true")
    parser.add_argument("--repeat", type=int, default=15)
    args = parser.parse_args()
    print(platform.platform(), "NumPy", np.__version__)
    backends = [("rust", args.device)]
    if args.jax:
        import jax

        jax.config.update("jax_enable_x64", True)
        print("JAX", jax.__version__, jax.devices())
        backends.append(("jax-tn", None))
    print("pattern,backend,device,setup_s,first_s,warm_median_s")
    patterns = [
        ("chain21", mp.templates.linear_cluster(21)),
        ("chain101", mp.templates.linear_cluster(101)),
        ("grid3x6", mp.templates.grid_cluster(3, 6)),
    ]
    for name, pattern in patterns:
        angles = np.linspace(0.1, 0.7, len(pattern.trainable_nodes))
        n = len(pattern.quantum_output_nodes)
        observable = mp.Observable({"Y" * n: 0.7, "Z" * n: 0.3})
        reference = mp.PatternSimulator(pattern, backend="numpy")
        expected = reference.expectation(angles, observable)
        derivative = mp.gradients.get_gradient(
            lambda a: reference.expectation(a, observable), angles, method="fd", h=1e-5
        )
        for backend, device in backends:
            start = time.perf_counter()
            sim = mp.PatternSimulator(
                pattern, backend=backend, **({"device": device} if device else {})
            )
            setup = time.perf_counter() - start

            def evaluate():
                value, grad = sim.expectation_and_grad(angles, observable)
                return float(value), np.asarray(grad)

            start = time.perf_counter()
            value, grad = evaluate()
            first = time.perf_counter() - start
            np.testing.assert_allclose(value, expected, atol=1e-10)
            np.testing.assert_allclose(grad, derivative, atol=1e-7)
            times = []
            for _ in range(args.repeat):
                start = time.perf_counter()
                evaluate()
                times.append(time.perf_counter() - start)
            print(
                f"{name},{backend},{device or 'jax default'},{setup:.8f},{first:.8f},{np.median(times):.8f}",
                flush=True,
            )


if __name__ == "__main__":
    main()

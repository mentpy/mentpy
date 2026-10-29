# Copyright 2026 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Compare warmed autodiff, parameter shift, and optional GF(2) kernels.

Run ``python benchmarks/simulators.py`` from an editable ``mentpy[dev]`` install.
Compilation is timed separately; benchmark results depend on graph and hardware.
"""

import importlib
import platform
import time

import jax
import jax.numpy as jnp
import numpy as np
import mentpy as mp


def timed(function, repeat=5):
    """Return the median wall time, waiting for asynchronous JAX results."""
    durations = []
    for _ in range(repeat):
        start = time.perf_counter()
        result = function()
        jax.block_until_ready(result)
        durations.append(time.perf_counter() - start)
    return float(np.median(durations))


def main():
    """Print timings and verify that both gradient methods agree."""
    print(platform.platform(), "JAX", jax.__version__, "NumPy", np.__version__)
    pattern = mp.templates.linear_cluster(21)
    sim = mp.PatternSimulator(pattern, backend="jax-tn")
    observable = mp.Observable({"X": 0.5, "Z": 0.7})
    angles = jnp.linspace(0.1, 0.7, len(pattern.trainable_nodes))
    cost = jax.jit(lambda a: sim.expectation(a, observable))
    autodiff = jax.jit(jax.value_and_grad(cost))
    compilation = timed(lambda: autodiff(angles), repeat=1)
    cost(angles).block_until_ready()
    psr = lambda: mp.gradients.get_value_and_gradient(
        cost, np.asarray(angles), method="psr"
    )
    np.testing.assert_allclose(autodiff(angles)[1], psr()[1], atol=1e-8)
    print("Autodiff first call (s):", compilation)
    print("Autodiff warmed (s):", timed(lambda: autodiff(angles)))
    print("Parameter shift warmed (s):", timed(psr))

    flow = importlib.import_module("mentpy.mbqc.flow")
    native = flow._rust_ext
    matrix = np.random.default_rng(7).integers(0, 2, (128, 256), dtype=np.uint8)
    try:
        for name, backend in [("Rust", native), ("NumPy", None)]:
            if name == "Rust" and native is None:
                continue
            flow._rust_ext = backend
            print(
                name,
                "GF(2) right inverse (s):",
                timed(lambda: flow._gf2_right_inverse_and_kernel(matrix)),
            )
    finally:
        flow._rust_ext = native


if __name__ == "__main__":
    main()

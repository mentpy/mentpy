# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Simulator for measuring patterns of MBQC circuits."""

from typing import List
import numpy as np

from mentpy.mbqc.mbqcircuit import MBQCircuit
from .numpy_simulator import NumpySimulator

try:
    from mentpy.simulators.jax_tn_simulator import JaxTNSimulator

    _HAS_JAX = True
except ImportError as exc:
    if exc.name not in {"jax", "jax.numpy", "opt_einsum"}:
        raise
    _HAS_JAX = False

__all__ = ["PatternSimulator"]


class PatternSimulator:
    """Simulator for measuring patterns of MBQC circuits.

    Parameters
    ----------
    mbqcircuit: mp.MBQCircuit
        The MBQC circuit used for the simulation.
    backend: str
        Use "numpy" for branch simulation or "jax-tn" for differentiable
        zero-outcome tensor contraction. "jax-tn" requires the JAX extra.

    See Also
    --------
    :class:`mp.NumpySimulator`, :class:`mp.JaxTNSimulator`

    Group
    -----
    simulators
    """

    def __init__(
        self,
        mbqcircuit: MBQCircuit,
        input_state: np.ndarray = None,
        backend="numpy",
        *args,
        **kwargs,
    ) -> None:
        supported_backends = {"numpy": NumpySimulator}
        if _HAS_JAX:
            supported_backends["jax-tn"] = JaxTNSimulator

        backend = backend.lower()
        if backend in {"numpy-dm", "numpy-sv", "trajectory", "pennylane"}:
            backend = "numpy"
        if backend == "jax-tn" and not _HAS_JAX:
            raise ImportError("Install 'mentpy[jax]' to use the jax-tn backend.")
        if backend not in supported_backends:
            raise ValueError(
                f"Backend {backend} not supported. Supported backends are {supported_backends.keys()}"
            )

        self.simulator = supported_backends[backend](
            mbqcircuit, input_state, *args, **kwargs
        )

    def __getattr__(self, name):
        return getattr(self.simulator, name)

    def __call__(self, angles: List[float] = (), **kwargs):
        return self.run(angles, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} ({self.simulator!r})"

    def measure(self, angle: float, **kwargs):
        return self.simulator.measure(angle, **kwargs)

    def run(self, angles: List[float] = (), **kwargs) -> np.ndarray:
        return self.simulator.run(angles, **kwargs)

    def expectation(self, angles, observable, shots=None, seed=None, **kwargs):
        return self.simulator.expectation(
            angles, observable, shots=shots, seed=seed, **kwargs
        )

    def reset(self, input_state: np.ndarray = None):
        return self.simulator.reset(input_state)

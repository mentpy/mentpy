# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""VQE helpers for MBQC ansatzes."""

from dataclasses import dataclass

import numpy as np

from mentpy.operators import Observable
from mentpy.optimizers import AdamOpt, SGDOpt
from mentpy.optimizers.base import BaseOpt
from mentpy.simulators import PatternSimulator

__all__ = ["VQE", "VQEResult", "train_vqe"]


@dataclass
class VQEResult:
    """Result returned by :class:`VQE`.

    Group
    -----
    qml
    """

    angles: np.ndarray
    energy: float
    history: dict


class VQE:
    """Train MBQC measurement angles against an observable.

    Parameters
    ----------
    circuit : MBQCircuit
        MBQC ansatz to train.
    observable : Observable or compatible constructor input
        Hamiltonian or cost observable acting on the quantum output wires.
    backend : str
        Simulator backend. ``"jax-tn"`` enables autodiff when the JAX extra is
        installed.
    optimizer : str
        ``"adam"``, ``"sgd"``, or an optimizer instance implementing
        ``step`` and ``reset``.
    step_size : float
        Optimizer learning rate.
    gradient_method : str
        ``"auto"`` uses backend autodiff for exact JAX tensor-network
        simulations and parameter-shift for finite-shot objectives. Explicit
        values include ``"jax"``, ``"parameter-shift"``, and ``"fd"``.
    shots : int or None
        If ``None``, evaluate exact observable expectations. If an integer, use
        finite synthetic Pauli-measurement shots per observable term.

    Group
    -----
    qml
    """

    def __init__(
        self,
        circuit,
        observable,
        backend="jax-tn",
        optimizer="adam",
        step_size=0.05,
        gradient_method="auto",
        gradient_kwargs=None,
        shots=None,
        seed=None,
        optimizer_kwargs=None,
        simulator=None,
        **simulator_kwargs,
    ):
        self.circuit = circuit
        self.observable = (
            observable if isinstance(observable, Observable) else Observable(observable)
        )
        self.backend = backend
        self.step_size = step_size
        self.gradient_method = gradient_method
        self.gradient_kwargs = {} if gradient_kwargs is None else dict(gradient_kwargs)
        self.shots = shots
        self._rng = np.random.default_rng(seed)
        self.simulator = simulator or PatternSimulator(
            circuit, backend=backend, **simulator_kwargs
        )
        self.optimizer = self._build_optimizer(optimizer, step_size, optimizer_kwargs)
        self.reset()

    def reset(self):
        """Reset optimizer state."""
        self._iteration = 0
        if hasattr(self.optimizer, "reset"):
            self.optimizer.reset()

    def energy(self, angles, shots=None, seed=None):
        """Return the observable expectation for ``angles``."""
        shots = self.shots if shots is None else shots
        if seed is None and shots is not None:
            seed = int(self._rng.integers(0, np.iinfo(np.uint32).max))

        if hasattr(self.simulator, "expectation"):
            return self.simulator.expectation(
                angles,
                self.observable,
                shots=shots,
                seed=seed,
            )
        state = self.simulator.run(angles, output_form="sv")
        if shots is not None:
            return self.observable.sample_expectation(state, shots=shots, seed=seed)
        return self.observable.expectation(state)

    def value_and_grad(self, angles):
        """Return energy and gradient for ``angles``."""
        method = self._resolved_gradient_method()
        if (
            method in {"jax", "autodiff"}
            and self.shots is None
            and hasattr(self.simulator, "expectation_and_grad")
        ):
            value, gradient = self.simulator.expectation_and_grad(
                angles, self.observable
            )
            return float(value), np.asarray(gradient, dtype=float)

        from mentpy.gradients import get_gradient

        value = self.energy(angles)
        gradient = get_gradient(
            lambda x: self.energy(x),
            angles,
            method=method,
            **self.gradient_kwargs,
        )
        return float(value), np.asarray(gradient, dtype=float)

    def step(self, angles):
        """Take one optimizer step and return ``(new_angles, energy, gradient)``."""
        angles = np.asarray(angles, dtype=float)
        value, gradient = self.value_and_grad(angles)
        angles = self.optimizer.step(
            lambda x: self.energy(x),
            angles,
            self._iteration,
            gradient=gradient,
        )
        self._iteration += 1
        return angles, value, gradient

    def fit(self, initial_angles, steps=100, callback=None, verbose=False):
        """Optimize angles and return a :class:`VQEResult`."""
        angles = np.asarray(initial_angles, dtype=float)
        history = {"energy": [], "grad_norm": [], "angles": []}

        for step in range(steps):
            angles, energy, gradient = self.step(angles)
            grad_norm = float(np.linalg.norm(gradient))
            history["energy"].append(energy)
            history["grad_norm"].append(grad_norm)
            history["angles"].append(angles.copy())

            if callback is not None:
                callback(angles, energy, gradient, step)
            if verbose:
                print(f"Iteration {step + 1}/{steps} - energy: {energy}")

        final_energy = float(self.energy(angles))
        return VQEResult(angles=angles, energy=final_energy, history=history)

    def _build_optimizer(self, optimizer, step_size, optimizer_kwargs):
        optimizer_kwargs = {} if optimizer_kwargs is None else dict(optimizer_kwargs)
        if isinstance(optimizer, BaseOpt):
            return optimizer
        if not isinstance(optimizer, str):
            if hasattr(optimizer, "step"):
                return optimizer
            raise TypeError("optimizer must be a string or optimizer instance.")

        name = optimizer.lower()
        if name == "adam":
            return AdamOpt(step_size=step_size, **optimizer_kwargs)
        if name == "sgd":
            return SGDOpt(step_size=step_size, **optimizer_kwargs)
        raise ValueError("optimizer must be 'adam', 'sgd', or an optimizer instance.")

    def _resolved_gradient_method(self):
        method = self.gradient_method.lower()
        if method != "auto":
            return method
        if self.shots is not None:
            return "parameter-shift"
        if hasattr(self.simulator, "expectation_and_grad"):
            return "jax"
        return "finite-difference"


def train_vqe(circuit, observable, initial_angles, steps=100, **kwargs):
    """Convenience wrapper around :class:`VQE`.

    Group
    -----
    qml
    """
    return VQE(circuit, observable, **kwargs).fit(initial_angles, steps=steps)

# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""VQE helpers for MBQC ansatzes."""

from dataclasses import dataclass

import numpy as np

from mentpy.operators import Observable
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
        Currently ``"adam"`` or ``"sgd"``.
    step_size : float
        Optimizer learning rate.

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
        simulator=None,
        **simulator_kwargs,
    ):
        self.circuit = circuit
        self.observable = (
            observable if isinstance(observable, Observable) else Observable(observable)
        )
        self.backend = backend
        self.optimizer = optimizer.lower()
        self.step_size = step_size
        self.simulator = simulator or PatternSimulator(
            circuit, backend=backend, **simulator_kwargs
        )
        self.reset()

    def reset(self):
        """Reset optimizer state."""
        self._iteration = 0
        self._m = None
        self._v = None

    def energy(self, angles):
        """Return the observable expectation for ``angles``."""
        if hasattr(self.simulator, "expectation"):
            return self.simulator.expectation(angles, self.observable)
        state = self.simulator.run(angles, output_form="sv")
        return self.observable.expectation(state)

    def value_and_grad(self, angles):
        """Return energy and gradient for ``angles``."""
        if hasattr(self.simulator, "expectation_and_grad"):
            value, gradient = self.simulator.expectation_and_grad(
                angles, self.observable
            )
            return float(value), np.asarray(gradient, dtype=float)

        from mentpy.gradients import get_gradient

        value = self.energy(angles)
        gradient = get_gradient(lambda x: self.energy(x), angles, method="fd")
        return float(value), np.asarray(gradient, dtype=float)

    def step(self, angles):
        """Take one optimizer step and return ``(new_angles, energy, gradient)``."""
        angles = np.asarray(angles, dtype=float)
        value, gradient = self.value_and_grad(angles)

        if self.optimizer == "adam":
            update = self._adam_update(gradient)
        elif self.optimizer == "sgd":
            update = self.step_size * gradient
        else:
            raise ValueError("optimizer must be 'adam' or 'sgd'.")

        self._iteration += 1
        return angles - update, value, gradient

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

    def _adam_update(self, gradient):
        if self._m is None:
            self._m = np.zeros_like(gradient)
        if self._v is None:
            self._v = np.zeros_like(gradient)

        b1 = 0.9
        b2 = 0.999
        eps = 1e-8
        self._m = b1 * self._m + (1 - b1) * gradient
        self._v = b2 * self._v + (1 - b2) * gradient**2
        m_hat = self._m / (1 - b1 ** (self._iteration + 1))
        v_hat = self._v / (1 - b2 ** (self._iteration + 1))
        return self.step_size * m_hat / (np.sqrt(v_hat) + eps)


def train_vqe(circuit, observable, initial_angles, steps=100, **kwargs):
    """Convenience wrapper around :class:`VQE`.

    Group
    -----
    qml
    """
    return VQE(circuit, observable, **kwargs).fit(initial_angles, steps=steps)

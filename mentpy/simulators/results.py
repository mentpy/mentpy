# Copyright 2026 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Finite-shot output measurements in quantum-output-node order."""

from dataclasses import dataclass
from numbers import Integral
import numpy as np

__all__ = ["Samples"]


def validate_shots(shots):
    if (
        isinstance(shots, (bool, np.bool_))
        or not isinstance(shots, Integral)
        or shots <= 0
    ):
        raise ValueError("shots must be a positive integer.")
    return int(shots)


@dataclass(frozen=True)
class Samples:
    """Output bits; zero/one denotes the +1/-1 eigenstate of each basis axis.

    Columns follow ``output_nodes`` (leftmost bit is the first output), with
    no Qiskit-style reversal. ``expectation`` is the product of measured signs.
    Identity axes contribute +1. Arrays are copied and made read-only.

    Group
    -----
    simulators
    """

    bitstrings: np.ndarray
    output_nodes: tuple
    basis: str

    def __post_init__(self):
        bits = np.array(self.bitstrings, dtype=np.int8, copy=True)
        if bits.ndim != 2 or bits.shape[1] != len(self.output_nodes):
            raise ValueError("bitstrings must have one column per output node.")
        if not np.all((bits == 0) | (bits == 1)) or bits.shape[0] == 0:
            raise ValueError("bitstrings must contain positive shots of binary data.")
        if len(self.basis) != bits.shape[1] or not set(self.basis) <= set("IXYZ"):
            raise ValueError("basis must contain one I, X, Y or Z per output.")
        bits.setflags(write=False)
        object.__setattr__(self, "bitstrings", bits)
        object.__setattr__(self, "output_nodes", tuple(self.output_nodes))

    @property
    def shots(self):
        return len(self.bitstrings)

    @property
    def counts(self):
        strings, counts = np.unique(self.bitstrings, axis=0, return_counts=True)
        return {
            "".join(map(str, bits)): int(count) for bits, count in zip(strings, counts)
        }

    @property
    def probabilities(self):
        """Empirical frequencies, keyed by observed bitstrings."""
        return {bits: count / self.shots for bits, count in self.counts.items()}

    @property
    def eigenvalues(self):
        axes = [i for i, char in enumerate(self.basis) if char != "I"]
        return 1 - 2 * (self.bitstrings[:, axes].sum(axis=1) % 2)

    @property
    def expectation(self):
        return float(np.mean(self.eigenvalues))

    @property
    def standard_error(self):
        if self.shots < 2:
            return float("nan")
        return float(np.std(self.eigenvalues, ddof=1) / np.sqrt(self.shots))


def measurement_probabilities(state, basis):
    """Rotate a vector/density matrix to a tensor-product Pauli eigenbasis."""
    state = np.asarray(state, dtype=complex)
    n = len(basis)
    if state.shape not in {(2**n,), (2**n, 2**n)}:
        raise ValueError("State dimensions do not match the measurement basis.")
    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    rotations = {"I": np.eye(2), "Z": np.eye(2), "X": h, "Y": h @ np.diag([1, -1j])}
    # Apply local basis changes without constructing a dense 2**n square matrix.
    tensor = state.reshape((2,) * (n * state.ndim))
    for axis, char in enumerate(basis):
        rotation = rotations[char]
        tensor = np.moveaxis(np.tensordot(rotation, tensor, axes=(1, axis)), 0, axis)
        if state.ndim == 2:
            tensor = np.moveaxis(
                np.tensordot(rotation.conj(), tensor, axes=(1, n + axis)), 0, n + axis
            )
    state = tensor.reshape(state.shape)
    probs = np.abs(state) ** 2 if state.ndim == 1 else np.real(np.diag(state))
    if not np.all(np.isfinite(probs)) or np.min(probs) < -1e-9 or probs.sum() <= 0:
        raise ValueError("Simulation did not produce valid probabilities.")
    probs = np.maximum(probs, 0)
    return probs / probs.sum()

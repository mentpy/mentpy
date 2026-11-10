# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Base class for simulators."""

import abc
import numpy as np
from typing import Union, List, Tuple, Optional

from mentpy.mbqc.mbqcircuit import MBQCircuit
from mentpy.operators import Observable
from .results import Samples, measurement_probabilities, validate_shots

__all__ = ["BaseSimulator"]


def _window_size(circuit, schedule, measurements, size):
    """Validate a sliding window and choose the smallest complete default."""
    n = len(circuit)
    if len(schedule) != n or set(schedule) != set(circuit.graph):
        raise ValueError("Schedule must contain every graph node once.")
    n_inputs = len(circuit.input_nodes)
    if set(schedule[:n_inputs]) != set(circuit.input_nodes):
        raise ValueError("NumPy schedules must start with all input nodes.")

    if list(schedule[: len(measurements)]) != measurements:
        raise ValueError(
            "NumPy schedules must place measurements before quantum outputs."
        )
    positions = {node: i for i, node in enumerate(schedule)}
    minimum = max(
        n_inputs,
        n - len(measurements),
        max(
            (abs(positions[u] - positions[v]) + 1 for u, v in circuit.graph.edges),
            default=0,
        ),
    )
    if not max(minimum, min(n, 1)) <= size <= n:
        raise ValueError(
            f"Window size {size} is invalid for this schedule; use {minimum} to {n} nodes."
        )
    return size


class BaseSimulator(abc.ABC):
    """Base class for simulators.

    Note
    ----
    This class should not be used directly. Instead, use one of the subclasses.

    Args
    ----
    mbqcircuit: mp.MBQCircuit
        The MBQC circuit used for the simulation.
    input_state: np.ndarray
        The input state of the simulator.

    See Also
    --------
    :class:`mp.PatternSimulator`, :class:`mp.NumpySimulator`

    Group
    -----
    simulators
    """

    _statevector_output = False

    def __init__(
        self,
        mbqcircuit: MBQCircuit,
        input_state: np.ndarray = None,
    ) -> None:
        self._mbqcirc = mbqcircuit
        self._input_state = input_state
        self._outcomes = {}

    @property
    def mbqcircuit(self) -> MBQCircuit:
        """The MBQC circuit used for the simulation."""
        return self._mbqcirc

    @property
    def input_state(self) -> np.ndarray:
        """The input state of the simulator."""
        return self._input_state

    @input_state.setter
    def input_state(self, input_state: np.ndarray):
        """Sets the input state of the simulator."""
        self._input_state = input_state

    @property
    def outcomes(self) -> dict:
        """The outcomes of the simulation."""
        return self._outcomes

    @outcomes.setter
    def outcomes(self, outcomes: dict):
        """Sets the outcomes of the simulation."""
        self._outcomes = outcomes

    def __call__(self, angles: List[float], **kwargs):
        return self.run(angles, **kwargs)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__} for {self.mbqcircuit}"

    @abc.abstractmethod
    def measure(self, angle: float, **kwargs):
        """Measures the state of the system.

        Parameters
        ----------
        angle: float
            The angle of measurement.
        """
        pass

    @abc.abstractmethod
    def run(self, angles: List[float], **kwargs) -> Tuple[List[int], np.ndarray]:
        """Measures the state of the system.

        Parameters
        ----------
        angles: List[float]
            The parameters of the MBQC circuit (if any).
        """
        pass

    def probabilities(self, angles=(), basis=None, **kwargs):
        """Exact output probabilities, ordered lexicographically by output bits.

        ``basis`` is a Pauli string in quantum-output-node order (default Z).
        """
        n = len(self.mbqcircuit.quantum_output_nodes)
        basis = "Z" * n if basis is None else basis.upper()
        if len(basis) != n or not set(basis) <= set("IXYZ"):
            raise ValueError("basis must contain one I, X, Y or Z per output.")
        kwargs.pop("output_form", None)
        if self._statevector_output:
            kwargs["output_form"] = "sv"
        return measurement_probabilities(self.run(angles, **kwargs), basis)

    def sample(self, angles=(), *, shots=1024, basis=None, seed=None, **kwargs):
        """Draw finite output shots. Intermediate-state simulation remains exact.

        For deterministic flow patterns, corrected output samples have the same
        law whether intermediate measurements are sampled or forced to zero.
        Non-flow patterns with forced outcomes describe a postselected branch.
        """
        shots = validate_shots(shots)
        nodes = tuple(self.mbqcircuit.quantum_output_nodes)
        basis = "Z" * len(nodes) if basis is None else basis.upper()
        rng = np.random.default_rng(
            seed if seed is not None else getattr(self, "rng", None)
        )
        if not getattr(self, "force0", True):
            self.reset(kwargs.pop("input_state", None))
            indices = np.empty(shots, dtype=np.intp)
            for i in range(shots):
                probs = self.probabilities(angles, basis=basis, seed=rng, **kwargs)
                indices[i] = rng.choice(len(probs), p=probs)
            shifts = np.arange(len(nodes) - 1, -1, -1)
            return Samples((indices[:, None] >> shifts) & 1, nodes, basis)
        probs = np.asarray(
            self.probabilities(angles, basis=basis, **kwargs), dtype=float
        )
        probs = probs / probs.sum()
        indices = rng.choice(len(probs), size=shots, p=probs)
        shifts = np.arange(len(nodes) - 1, -1, -1)
        bits = (indices[:, None] >> shifts) & 1
        return Samples(bits, nodes, basis)

    def expectation(self, angles, observable, shots=None, seed=None, **kwargs):
        """Evaluate an observable exactly or with finite synthetic shots."""
        if not isinstance(observable, Observable):
            observable = Observable(observable)

        if shots is not None and not getattr(self, "force0", True):
            shots = validate_shots(shots)
            observable._validate_qubits(len(self.mbqcircuit.quantum_output_nodes))
            rng = np.random.default_rng(
                seed if seed is not None else getattr(self, "rng", None)
            )
            value = observable.constant
            for basis, coefficient in observable.terms.items():
                samples = self.sample(
                    angles, basis=basis, shots=shots, seed=rng, **kwargs
                )
                value += coefficient * samples.expectation
            return np.real(value)

        kwargs.pop("output_form", None)
        if self._statevector_output:
            kwargs["output_form"] = "sv"
        state = self.run(angles, seed=seed, **kwargs)
        if shots is None:
            return observable.expectation(state)
        seed = seed if seed is not None else getattr(self, "rng", None)
        return observable.sample_expectation(state, shots=shots, seed=seed)

    @abc.abstractmethod
    def reset(self, input_state=None):
        """Resets the simulator to the initial state."""
        pass

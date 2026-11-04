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
    :class:`mp.PatternSimulator`, :class:`mp.PennylaneSimulator`, :class:`mp.CirqSimulator`

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
        self.reset()
        try:
            if not self._statevector_output:
                state = self.run(angles, **kwargs)
            else:
                state = self.run(angles, output_form="sv", **kwargs)
            return measurement_probabilities(state, basis)
        finally:
            self.reset()

    def sample(self, angles=(), *, shots=1024, basis=None, seed=None, **kwargs):
        """Draw finite output shots. Intermediate-state simulation remains exact.

        For deterministic flow patterns, corrected output samples have the same
        law whether intermediate measurements are sampled or forced to zero.
        Non-flow patterns with forced outcomes describe a postselected branch.
        """
        shots = validate_shots(shots)
        nodes = tuple(self.mbqcircuit.quantum_output_nodes)
        basis = "Z" * len(nodes) if basis is None else basis.upper()
        probs = self.probabilities(angles, basis=basis, **kwargs)
        indices = np.random.default_rng(seed).choice(len(probs), size=shots, p=probs)
        shifts = np.arange(len(nodes) - 1, -1, -1)
        bits = (indices[:, None] >> shifts) & 1
        return Samples(bits, nodes, basis)

    def expectation(self, angles, observable, shots=None, seed=None, **kwargs):
        """Evaluate an observable exactly or with finite synthetic shots."""
        if not isinstance(observable, Observable):
            observable = Observable(observable)

        self.reset()
        try:
            if self._statevector_output:
                state = self.run(angles, output_form="sv", **kwargs)
            else:
                state = self.run(angles, **kwargs)
            if shots is None:
                return observable.expectation(state)
            return observable.sample_expectation(state, shots=shots, seed=seed)
        finally:
            self.reset()

    @abc.abstractmethod
    def reset(self, input_state=None):
        """Resets the simulator to the initial state."""
        pass

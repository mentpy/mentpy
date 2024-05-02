# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Base class for simulators.""" ""
import abc
import numpy as np
from typing import List, Tuple

from mentpy.mbqc.mbqcircuit import MBQCircuit

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

    def __init__(
        self,
        mbqcircuit: MBQCircuit,
        input_state: np.ndarray = None,
        *args,
        **kwargs,
    ) -> None:
        self._check_flow(mbqcircuit)
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

    @abc.abstractmethod
    def reset(self, input_state=None):
        """Resets the simulator to the initial state."""
        pass

    def _check_flow(self, mbqcircuit: MBQCircuit) -> None:
        """Checks if the MBQC circuit has a flow."""
        if not mbqcircuit.flow:
            raise ValueError("Cannot simulate a circuit without a flow.")

    def num_qubits_layer_pairs(self) -> int:
        """Returns the maximum number of qubits in a layer pair."""
        num_qubits_layer = np.array([len(layer) for layer in self.mbqcircuit.layers])
        num_qubits_layer_pairs = num_qubits_layer[:-1] + num_qubits_layer[1:]
        return num_qubits_layer_pairs

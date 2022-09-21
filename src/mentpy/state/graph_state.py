"""The graph_state module"""

from functools import cached_property
from typing import Optional, List, Tuple, Union

import numpy as np
import scipy as scp
import networkx as nx
import mentpy as mtp
import cirq


class GraphState:
    r"""The GraphState class that deals with operations and manipulations of graph states
    Args
    ----
    graph: nx.Graph
    simulation_type: "qubit" or "density_matrix".

    Examples
    --------
    Create a 1D cluster state :math:`|G>` of five qubits

    .. ipython:: python

        g = nx.Graph()
        g.add_edges_from([(0,1), (1,2), (2,3), (3, 4)])
        state = mtp.GraphState(g, input_nodes=[0], output_nodes=[4])

    :group: states
    """

    def __init__(
        self,
        graph: nx.Graph,
        input_state: Optional[np.ndarray] = None,
        input_nodes: np.ndarray = np.array([]),
        output_nodes: np.ndarray = np.array([]),
    ) -> None:
        """Initializes a graph state"""
        self._graph = graph
        if input_state is not None:
            self._input_state = input_state
        else:
            in_states = len(input_nodes) * [cirq.KET_PLUS.state_vector()]
            self._input_state = cirq.kron(*in_states)

        self._input_nodes = input_nodes
        self._output_nodes = output_nodes

    def __repr__(self) -> str:
        """Return the representation of the current graph state"""
        return (
            f"Graph state with {self.graph.number_of_nodes()} nodes and "
            f"{self.graph.number_of_edges()} edges."
        )

    @property
    def graph(self) -> nx.Graph:
        r"""Return the graph of the resource state."""
        return self._graph

    @property
    def input_state(self):
        r"""Return the input state :math:`|\psi\rangle` of the MBQC circuit"""
        return self._input_state

    @property
    def input_nodes(self) -> np.ndarray:
        r"""Return the input nodes of the MBQC circuit."""
        return self._input_nodes

    @property
    def output_nodes(self) -> np.ndarray:
        r"""Return the output nodes of the MBQC circuit."""
        return self._output_nodes

    @cached_property
    def outputc(self):
        r"""Returns :math:`O^c`, the complement of output nodes."""
        return [v for v in self.graph.nodes() if v not in self.output_nodes]

    @cached_property
    def inputc(self):
        r"""Returns :math:`I^c`, the complement of input nodes."""
        return [v for v in self.graph.nodes() if v not in self.input_nodes]

    def create_plus_states(self, n):
        r"""Returns the quantum state :math:`|+\rangle^n`."""


def lc_reduce(state: GraphState):
    """Reduce graph state

    :group: states
    """
    raise NotImplementedError


def merge(
    state1: GraphState, state2: GraphState, indices_tuple: List[Tuple]
) -> GraphState:
    """Merge two graph states into a larger graph state

    :group: states
    """
    raise NotImplementedError


def entanglement_entropy(
    state: GraphState, subRegionA: List, subRegionB: Optional[List] = None
):
    """Calculates the entanglement entropy between subRegionA and subRegionB
    of state. If subRegionB is None, then :python:`subRegionB = set(state.graph.nodes()) - set(subRegionA)`
    by default.

    :group: states
    """

    G = state.graph.copy()

    # minimum_cut requires the capacity kwarg.
    nx.set_edge_attributes(G, 1, name="capacity")
    if subRegionB is None:
        subRegionB = set(state.graph.nodes()) - set(subRegionA)

    # Allow subregions. These are merged into a supernode to calculate
    # the minimum cut between them.
    if isinstance(subRegionA, List):
        for v in subRegionA:
            G = nx.contracted_nodes(G, subRegionA[0], v)
        subRegionA = subRegionA[0]
    if isinstance(subRegionB, List):
        for v in subRegionA:
            G = nx.contracted_nodes(G, subRegionB[0], v)
        subRegionB = subRegionB[0]

    return nx.minimum_cut(G, subRegionA, subRegionB)[0]

# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Graph state class and related functions."""

import numpy as np
from mentpy.operators import PauliOp
import networkx as nx
from typing import Optional, List

__all__ = ["GraphState", "entanglement_entropy"]


class GraphState(nx.Graph):
    """A graph state class that inherits from networkx.Graph.

    Examples
    --------
    Create a 1D cluster state :math:`|G>` of five qubits

    .. ipython:: python

        g = mp.GraphState()
        g.add_edges_from([(0,1), (1,2), (2,3), (3, 4)])
        print(g)

    See Also
    --------
    :class:`mentpy.mbqc.MBQCircuit`

    Group
    -----
    mbqc
    """

    def __init__(self, *args, **kwargs):
        """Initialize a graph state. See networkx.Graph for more information."""
        super().__init__(*args, **kwargs)

    def __repr__(self):
        return f"GraphState with {self.number_of_nodes()} nodes and {self.number_of_edges()} edges."

    def __len__(self):
        return self.number_of_nodes()

    def __eq__(self, other):
        if not isinstance(other, nx.Graph):
            return NotImplemented
        return nx.is_isomorphic(self, other)

    def index_mapping(self):
        """Return a mapping of the nodes to their indices."""
        return {v: i for i, v in enumerate(self.nodes())}

    def stabilizers(self):
        """
        Generate the stabilizers of a graph state.

        Examples
        --------
        Calculate the stabilizers of a 1D cluster state :math:`|G>` of five qubits

        .. ipython:: python
            :okwarning:

            g = mp.GraphState()
            g.add_edges_from([(0,1), (1,2), (2,3), (3, 4)])
            print(g.stabilizers())
        """
        return _get_stabilizers(self)


def entanglement_entropy(
    state: GraphState, subRegionA: List, subRegionB: Optional[List] = None
) -> float:
    """Calculate bipartite entanglement entropy for a graph state.

    For graph states, the entropy across a bipartition is the GF(2) rank of
    the adjacency submatrix connecting the two regions.

    Group
    -----
    mbqc
    """
    nodes = list(state.nodes())
    subRegionA = _as_node_list(subRegionA)
    region_a = set(subRegionA)
    subRegionB = (
        [node for node in nodes if node not in region_a]
        if subRegionB is None
        else _as_node_list(subRegionB)
    )

    overlap = set(subRegionA) & set(subRegionB)
    if overlap:
        raise ValueError(f"Subregions must be disjoint; overlap is {overlap}.")

    missing = (set(subRegionA) | set(subRegionB)) - set(nodes)
    if missing:
        raise ValueError(f"Subregions contain nodes not in the graph: {missing}.")

    cut_matrix = np.zeros((len(subRegionA), len(subRegionB)), dtype=np.uint8)
    columns = {node: i for i, node in enumerate(subRegionB)}
    for i, node in enumerate(subRegionA):
        cut_matrix[i, [columns[v] for v in state[node] if v in columns]] = 1
    return float(_gf2_rank(cut_matrix))


def _as_node_list(nodes):
    if isinstance(nodes, (str, bytes)):
        return [nodes]
    try:
        return list(nodes)
    except TypeError:
        return [nodes]


def _gf2_rank(matrix):
    matrix = np.array(matrix, dtype=np.uint8, copy=True) % 2
    rows, cols = matrix.shape
    rank = 0
    for col in range(cols):
        pivot_rows = np.flatnonzero(matrix[rank:, col])
        if pivot_rows.size == 0:
            continue
        pivot = rank + pivot_rows[0]
        if pivot != rank:
            matrix[[rank, pivot]] = matrix[[pivot, rank]]
        for row in range(rows):
            if row != rank and matrix[row, col]:
                matrix[row] ^= matrix[rank]
        rank += 1
        if rank == rows:
            break
    return rank


def _get_stabilizers(graph: GraphState) -> PauliOp:
    """Generate the stabilizers of a graph state."""
    z_mat = nx.to_numpy_array(graph, dtype=int, weight=None)
    x_mat = np.eye(graph.number_of_nodes(), dtype=int)

    return PauliOp(np.hstack((x_mat, z_mat)))

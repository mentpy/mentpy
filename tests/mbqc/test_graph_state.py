# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Tests for the graph state module."""

import pytest
import mentpy as mp
from mentpy.mbqc.states.graphstate import GraphState, entanglement_entropy


def test_create_graphstate():
    """Test the creation of a graph state."""
    my_graph = GraphState()
    my_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
    assert my_graph.number_of_nodes() == 5


def test_graphstate_helpers_preserve_graph_behavior():
    graph = GraphState()
    graph.add_nodes_from(["a", "b", "c"])
    graph.add_edge("a", "b")

    isomorphic_graph = GraphState()
    isomorphic_graph.add_nodes_from([0, 1, 2])
    isomorphic_graph.add_edge(0, 1)

    different_graph = GraphState()
    different_graph.add_edges_from([(0, 1), (1, 2)])

    assert len(graph) == 3
    assert repr(graph) == "GraphState with 3 nodes and 1 edges."
    assert graph.index_mapping() == {"a": 0, "b": 1, "c": 2}
    assert graph == isomorphic_graph
    assert graph != different_graph


def test_pauli_stabilizers():
    """Test the generation of the stabilizers of a graph state."""
    my_graph = GraphState()
    my_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
    stabs = my_graph.stabilizers()
    assert len(stabs) == 5


def test_entanglement_entropy_uses_gf2_cut_rank():
    """A two-edge cut with dependent GF(2) rows has entropy one, not two."""
    graph = GraphState()
    graph.add_edges_from([(0, 1), (2, 1)])

    assert entanglement_entropy(graph, [0, 2], [1]) == 1.0


def test_entanglement_entropy_defaults_to_complement_for_scalar_node():
    graph = GraphState()
    graph.add_edges_from([("left", "middle"), ("middle", "right")])

    assert entanglement_entropy(graph, "left") == 1.0


def test_entanglement_entropy_validates_bipartition():
    graph = GraphState()
    graph.add_edge(0, 1)

    with pytest.raises(ValueError, match="disjoint"):
        entanglement_entropy(graph, [0], [0, 1])


def test_entanglement_entropy_validates_missing_nodes():
    graph = GraphState()
    graph.add_edge(0, 1)

    with pytest.raises(ValueError, match="not in the graph"):
        entanglement_entropy(graph, [0], [2])


def test_stale_state_placeholders_are_not_exported():
    assert not hasattr(mp, "ClusterState")
    assert not hasattr(mp, "AKLTState")

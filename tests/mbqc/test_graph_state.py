# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Tests for the graph state module."""

import pytest
from mentpy.mbqc.states.graphstate import GraphState


def test_create_graphstate():
    """Test the creation of a graph state."""
    my_graph = GraphState()
    my_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
    assert my_graph.number_of_nodes() == 5


def test_pauli_stabilizers():
    """Test the generation of the stabilizers of a graph state."""
    my_graph = GraphState()
    my_graph.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4)])
    stabs = my_graph.stabilizers()
    assert len(stabs) == 5

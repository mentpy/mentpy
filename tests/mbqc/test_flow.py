# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Tests for the flow module."""

import pytest
from mentpy.mbqc import find_cflow, find_gflow, find_pflow
from mentpy import MBQCircuit, GraphState
from mentpy import Ment

# Test case functions

@pytest.fixture
def line_state():
    graph_state = GraphState()
    graph_state.add_edges_from([(1, 2), (2, 3), (3, 4), (4, 5)])
    circuit = MBQCircuit(graph_state, input_nodes=[1], output_nodes=[5])
    return circuit

@pytest.fixture
def gflow_no_cflow_graph():
    """
    Creates a graph with generalized flow but no classical flow.
    Reference: 10.1088/1367-2630/9/8/250 Fig 2
    """
    graph_state = GraphState()
    graph_state.add_edges_from([(1, 4), (2, 5), (3, 4), (3, 6)])
    circuit = MBQCircuit(graph_state)
    return circuit

@pytest.fixture
def cflow_graph():
    """
    Creates a graph with classical flow.
    Reference: 10.1088/1367-2630/9/8/250 Fig 3
    """
    graph_state = GraphState()
    graph_state.add_edges_from([(1, 4), (2, 5), (3, 6)])
    circuit = MBQCircuit(graph_state)
    return circuit

@pytest.fixture
def open_pflow_no_gflow_graph():
    """
    Creates a graph with open pattern flow but no generalized flow.
    Reference: 10.1088/1367-2630/9/8/250 Fig 7
    """
    graph_state = GraphState()
    graph_state.add_edges_from([
        (1, 3),
        (2, 5),
        (3, 4),
        (3, 6),
        (4, 7),
        (4, 5),
        (5, 8),
        (6, 9),
        (6, 7),
        (7, 8),
        (8, 10),
        (9, 11),
        (10, 12)
    ])
    circuit = MBQCircuit(graph_state, input_nodes=[1, 2], output_nodes=[11, 12], default_measurement=Ment("X"))
    return circuit

@pytest.fixture
def open_no_pflow_no_gflow_graph():
    """
    Creates a graph without pattern flow or generalized flow.
    Reference: 10.1088/1367-2630/9/8/250 p14 Fig 8
    """
    graph_state = GraphState()
    graph_state.add_edges_from([
        (1, 3),
        (2, 5),
        (3, 4),
        (3, 6),
        (4, 5),
        (4, 7),
        (5, 8),
        (6, 7),
        (6, 9),
        (7, 8),
        (7, 10),
        (8, 11),
        (9, 12),
        (9, 10),
        (10, 11),
        (11, 13)
    ])
    circuit = MBQCircuit(graph_state, input_nodes=[1, 2], output_nodes=[12, 13])
    return circuit

# Test functions

def test_cflow(cflow_graph, gflow_no_cflow_graph):
    """Tests the classical flow detection."""
    assert find_cflow(cflow_graph.graph, cflow_graph.input_nodes, cflow_graph.output_nodes)[0] == True
    assert find_cflow(gflow_no_cflow_graph.graph, gflow_no_cflow_graph.input_nodes, gflow_no_cflow_graph.output_nodes)[0] == False

def test_gflow(gflow_no_cflow_graph, open_pflow_no_gflow_graph, open_no_pflow_no_gflow_graph):
    """Tests the generalized flow detection."""
    assert find_gflow(gflow_no_cflow_graph.graph, gflow_no_cflow_graph.input_nodes, gflow_no_cflow_graph.output_nodes)[0] == True
    assert find_gflow(open_pflow_no_gflow_graph.graph, open_pflow_no_gflow_graph.input_nodes, open_no_pflow_no_gflow_graph.output_nodes)[0] == False
    assert find_gflow(open_no_pflow_no_gflow_graph.graph, open_no_pflow_no_gflow_graph.input_nodes, open_no_pflow_no_gflow_graph.output_nodes)[0] == False

def test_pflow(line_state, open_pflow_no_gflow_graph, open_no_pflow_no_gflow_graph):
    """Tests the pattern flow detection."""
    assert find_pflow(line_state, line_state.graph, line_state.input_nodes, line_state.output_nodes)[0] == True
    assert find_pflow(open_pflow_no_gflow_graph, open_pflow_no_gflow_graph.graph, open_pflow_no_gflow_graph.input_nodes, open_pflow_no_gflow_graph.output_nodes)[0] == True
    assert find_pflow(open_pflow_no_gflow_graph, open_no_pflow_no_gflow_graph.graph, open_no_pflow_no_gflow_graph.input_nodes, open_no_pflow_no_gflow_graph.output_nodes)[0] == False
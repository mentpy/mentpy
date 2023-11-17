# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Tests for the flow module."""

import pytest
from mentpy.mbqc import flow, find_cflow, find_pflow, find_gflow
from mentpy import MBQCircuit, GraphState

# Test case functions

def gflow_no_cflow_graph():
    """
    Creates a graph with generalized flow but no classical flow.
    Reference: 10.1088/1367-2630/9/8/250 p11
    """
    graph_state = GraphState()
    graph_state.add_edges_from([(1, 4), (2, 5), (3, 4), (3, 6)])
    circuit = MBQCircuit(graph_state)
    return circuit.graph()

def cflow_graph():
    """
    Creates a graph with classical flow.
    Reference: 10.1088/1367-2630/9/8/250 p12
    """
    graph_state = GraphState()
    graph_state.add_edges_from([(1, 4), (2, 5), (3, 6)])
    circuit = MBQCircuit(graph_state)
    return circuit.graph()

def open_pflow_no_gflow_graph():
    """
    Creates a graph with open pattern flow but no generalized flow.
    Reference: 10.1088/1367-2630/9/8/250 p14 Fig 7
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
    circuit = MBQCircuit(graph_state)
    return circuit.graph()

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
    circuit = MBQCircuit(graph_state)
    return circuit.graph()

# Test functions

def test_cflow():
    """Tests the classical flow detection."""
    classical_flow_graph = cflow_graph()
    gen_flow_no_class_flow_graph = gflow_no_cflow_graph()
    
    assert find_cflow(classical_flow_graph)[0] == True
    assert find_cflow(gen_flow_no_class_flow_graph)[0] == False

def test_gflow():
    """Tests the generalized flow detection."""
    gen_flow_graph = gflow_no_cflow_graph()
    open_pf_no_gen_flow_graph = open_pflow_no_gflow_graph()
    open_no_pf_no_gen_flow_graph = open_no_pflow_no_gflow_graph()

    assert find_gflow(gen_flow_graph)[0] == True
    assert find_gflow(open_pf_no_gen_flow_graph)[0] == False
    assert find_gflow(open_no_pf_no_gen_flow_graph)[0] == False

def test_pflow():
    """Tests the pattern flow detection."""
    open_pf_no_gen_flow_graph = open_pflow_no_gflow_graph()
    open_no_pf_no_gen_flow_graph = open_no_pflow_no_gflow_graph()

    assert find_pflow(open_pf_no_gen_flow_graph)[0] == True
    assert find_pflow(open_no_pf_no_gen_flow_graph)[0] == True

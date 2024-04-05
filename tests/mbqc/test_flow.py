# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Tests for the flow module."""

import pytest

import mentpy as mp
from mentpy.mbqc import flow


def test_cflow():
    """Test the cflow function."""
    gs = mp.templates.linear_cluster(5).graph
    cond, _, _, _ = flow.find_cflow(gs, set([0]), set([4]))
    assert cond


def test_gflow():
    """Test the gflow function."""
    gs = mp.templates.linear_cluster(5).graph
    cond, _, _, _ = flow.find_gflow(gs, set([0]), set([4]))
    assert cond


def test_pflow():
    """Test the pflow function."""
    gs = mp.templates.linear_cluster(5).graph
    cond, p, d = flow.find_pflow(gs, set([0]), set([4]), {v: "XY" for v in gs.nodes})
    assert cond

    gs = mp.GraphState()
    gs.add_edges_from([(0, 1), (1, 2), (1, 3), (3, 4)])
    circ = mp.MBQCircuit(
        gs,
        input_nodes=[0],
        output_nodes=[2],
        measurements={
            0: mp.Ment("XY"),
            1: mp.Ment("X"),
            3: mp.Ment("XY"),
            4: mp.Ment("X"),
        },
    )
    circ.flow.initialize_flow()

    assert circ.flow.depth == 1

    gs = mp.GraphState()
    gs.add_edges_from([(0, 1), (1, 2), (2, 3), (1, 4), (2, 4), (4, 5), (5, 2), (5, 6)])
    circ = mp.MBQCircuit(
        gs,
        input_nodes=[0],
        output_nodes=[3, 6],
        measurements={
            0: mp.Ment("XY"),
            1: mp.Ment("XY"),
            2: mp.Ment("Y"),
            4: mp.Ment("YZ"),
            5: mp.Ment("XY"),
        },
    )

    circ.flow.initialize_flow()
    assert circ.flow.depth == 6

    gs = mp.GraphState()

    gs.add_edges_from(
        [
            (0, 3),
            (1, 3),
            (1, 4),
            (2, 4),
            (0, 5),
            (2, 5),
        ]
    )

    circ = mp.MBQCircuit(
        gs,
        input_nodes=[0, 1, 2],
        output_nodes=[0, 1, 2],
        measurements={3: mp.Ment("YZ"), 4: mp.Ment("YZ"), 5: mp.Ment("YZ")},
    )

    circ.flow.initialize_flow()
    assert circ.flow.depth == 1

# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Tests for the MBQC templates."""

import pytest
import mentpy as mp


@pytest.mark.parametrize("n_wires", [2, 3, 4])
@pytest.mark.parametrize("n_layers", [2, 3, 4])
@pytest.mark.parametrize("periodic", [True, False])
def test_grid_cluster(n_wires, n_layers, periodic):
    """Test the grid_cluster template."""
    gc = mp.templates.grid_cluster(n_wires, n_layers, periodic=periodic)
    assert gc.flow is not None
    assert gc.graph.number_of_nodes() == n_wires * n_layers


def test_from_pauli():
    """Test the from_pauli template."""
    gs = mp.templates.from_pauli(mp.PauliOp("XYY"))
    assert gs.flow is not None
    assert gs.graph.number_of_nodes() == 3 * 3 + 2

    gs = mp.templates.from_pauli(mp.PauliOp("IIIX"))
    assert gs.flow is not None
    assert gs.graph.number_of_nodes() == 4 * 3 + 2

    gs = mp.templates.from_pauli(mp.PauliOp("XYXZZ"))
    assert gs.flow is not None
    assert gs.graph.number_of_nodes() == 5 * 3 + 2


def test_hstack_can_skip_eager_flow_initialization():
    left = mp.templates.linear_cluster(2)
    right = mp.templates.linear_cluster(2)

    stacked = mp.hstack([left, right], initialize_flow=False)

    assert stacked.flow is not None
    assert not stacked.flow.flow_initialized
    assert stacked.measurement_order == list(stacked.graph.nodes)
    assert not stacked.flow.flow_initialized


def test_single_site_clusters_have_one_input_and_output():
    for pattern in (mp.templates.linear_cluster(1), mp.templates.grid_cluster(1, 1)):
        assert len(pattern) == 1
        assert pattern.input_nodes == pattern.output_nodes == [0]
        assert not pattern.trainable_nodes


def test_stack_preserves_named_nodes_and_idle_wires():
    named = mp.MBQCircuit(
        mp.GraphState([("left", "right")]),
        ["left"],
        ["right"],
        {"left": mp.Ment(0.3, "XY")},
        relabel_indices=False,
    )
    idle = mp.templates.linear_cluster(1)
    composed = mp.hstack([idle, named])
    assert len(composed) == len(named)
    assert composed[composed.input_nodes[0]].angle == 0.3
    disjoint = mp.vstack([named, named])
    assert len(disjoint) == 4
    assert [disjoint[i].angle for i in disjoint.input_nodes] == [0.3, 0.3]
    merged = mp.merge(named, named, along=[("right", "left")])
    assert len(merged) == 3
    assert [merged[i].angle for i in merged.outputc] == [0.3, 0.3]


class TestMuTA:
    """Test the MuTA template."""

    @pytest.mark.parametrize("n_wires", [1, 2, 3, 4])
    @pytest.mark.parametrize("n_layers", [1, 2, 3, 4])
    def test_trainable_nodes(self, n_wires, n_layers):
        muta = mp.templates.muta(n_wires, n_layers)
        if n_wires != 1 and n_layers != 1:
            assert len(muta.trainable_nodes) == (4 * n_layers * (n_wires**2))

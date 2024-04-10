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


class TestMuTA:
    """Test the MuTA template."""

    @pytest.mark.parametrize("n_wires", [1, 2, 3, 4])
    @pytest.mark.parametrize("n_layers", [1, 2, 3, 4])
    def test_trainable_nodes(self, n_wires, n_layers):
        muta = mp.templates.muta(n_wires, n_layers)
        if n_wires != 1 and n_layers != 1:
            assert len(muta.trainable_nodes) == (4 * n_layers * (n_wires**2))

# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Tests for the MBQC templates."""
import pytest
import mentpy as mp


class TestSpturb:
    """Test the spturb template."""

    @pytest.mark.parametrize("n_layer", [1, 2])
    @pytest.mark.parametrize("n_qubits", [4, 5])
    @pytest.mark.parametrize("periodic", [True, False])
    def test_trainable_nodes(self, n_layer, n_qubits, periodic):
        spt = mp.templates.spturb(n_qubits, n_layer, periodic=periodic)
        periodic_params = n_qubits if periodic else n_qubits - 2
        assert (
            len(spt.trainable_nodes)
            == n_qubits * n_layer + 2 * n_layer * periodic_params
        )

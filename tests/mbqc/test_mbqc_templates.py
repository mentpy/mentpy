# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Tests for the MBQC templates."""
import pytest
import mentpy as mp


class TestMuTA:
    """Test the MuTA template."""

    @pytest.mark.parametrize("n_wires", [1, 2, 3, 4])
    @pytest.mark.parametrize("n_layers", [1, 2, 3, 4])
    def test_trainable_nodes(self, n_wires, n_layers):
        muta = mp.templates.muta(n_wires, n_layers)
        if n_wires != 1 and n_layers != 1:
            assert len(muta.trainable_nodes) == (4 * n_layers * (n_wires**2))

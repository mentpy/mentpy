# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Tests for the gates module."""

import mentpy as mp
import numpy as np
import pytest


def test_isingxx_gate():
    """Test the IsingXX gate."""
    assert np.allclose(mp.gates.ising_xx(0), np.eye(4))


def test_swap_qubits_matches_swap_operator():
    state = np.array([1, 2j, 3, 4j, 5, 6j, 7, 8j])
    np.testing.assert_array_equal(
        mp.gates.swap_qubits(state, 0, 2), mp.gates.swap_ij(0, 2, 3) @ state
    )
    np.testing.assert_array_equal(mp.gates.swap_qubits(state, 1, 1), state)
    np.testing.assert_array_equal(mp.gates.swap_ij(1, 1, 3), np.eye(8))


def test_cnot_preserves_one_based_convention():
    np.testing.assert_array_equal(mp.gates.cnot_ij(1, 2, 2), mp.gates.CNOT)
    with pytest.raises(ValueError, match="distinct qubit"):
        mp.gates.cnot_ij(0, 1, 2)

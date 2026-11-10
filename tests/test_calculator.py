# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Tests for the calculator module."""

import pytest

import numpy as np
import mentpy as mp


def test_pure2density():
    """Test the pure to density matrix function."""
    psi = np.array([1, 0])
    rho_expected = np.array([[1, 0], [0, 0]])
    rho_result = mp.calculator.pure2density(psi)
    assert np.allclose(rho_result, rho_expected)

    psi = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
    rho_expected = np.array([[0.5, 0.5], [0.5, 0.5]])
    rho_result = mp.calculator.pure2density(psi)
    assert np.allclose(rho_result, rho_expected)


def test_partial_trace():
    """Test the partial trace function."""
    # Simple test with mixed state
    rho = np.array([[0.5, 0], [0, 0.5]])
    indices = [0]
    expected = 1
    result = mp.calculator.partial_trace(rho, indices)
    assert np.allclose(result, expected)

    plus = np.array([1, 1]) / np.sqrt(2)
    sigma = np.outer(plus, plus)

    product_state = np.kron(rho, sigma)
    result1 = mp.calculator.partial_trace(product_state, [0])
    result2 = mp.calculator.partial_trace(product_state, [1])
    assert np.allclose(result1, sigma)
    assert np.allclose(result2, rho)

    # Test with pure state
    psi = np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])
    indices = [0]
    expected = 1
    result = mp.calculator.partial_trace(psi, indices)
    assert np.allclose(result, expected)

    product_state = np.kron(psi, plus)
    result1 = mp.calculator.partial_trace(product_state, [0])
    result2 = mp.calculator.partial_trace(product_state, [1])
    assert np.allclose(result1, sigma)
    assert np.allclose(result2, mp.calculator.pure2density(psi))


def test_partial_trace_entangled_and_multiple_axes():
    bell = np.array([1, 0, 0, -1j]) / np.sqrt(2)
    state = np.kron(bell, [0, 1])
    rho = mp.calculator.pure2density(state)
    for representation in (state, rho):
        np.testing.assert_allclose(
            mp.calculator.partial_trace(representation, [2, 0]), np.eye(2) / 2
        )
        np.testing.assert_allclose(mp.calculator.partial_trace(representation, []), rho)
        with pytest.raises(ValueError, match="unique"):
            mp.calculator.partial_trace(representation, [0, 0])


@pytest.mark.parametrize("column", [False, True])
def test_gf2_solver_handles_free_variables_and_rhs_shapes(column):
    matrix = np.array([[1, 1, 0], [0, 1, 1], [0, 0, 0]])
    rhs = np.array([0, 1, 0])
    solution = mp.calculator.solve(
        matrix, rhs[:, None] if column else rhs, check_solution=True
    )
    np.testing.assert_array_equal(solution[:, 0], [1, 1, 0])
    with pytest.raises(ValueError, match="Solution not found"):
        mp.calculator.solve(matrix, np.array([0, 1, 1]), check_solution=True)

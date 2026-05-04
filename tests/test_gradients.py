"""Tests for numerical and autodiff gradient helpers."""

import numpy as np
import pytest

import mentpy as mp


def test_parameter_shift_gradient_matches_trig_function():
    x = np.array([0.2, -0.4])

    def cost(params):
        return np.sin(params[0]) + np.cos(params[1])

    gradient = mp.gradients.get_gradient(cost, x, method="parameter-shift")

    assert np.allclose(gradient, [np.cos(x[0]), -np.sin(x[1])])


def test_get_value_and_gradient_finite_difference():
    x = np.array([1.5, -2.0])

    def cost(params):
        return np.sum(params**2)

    value, gradient = mp.gradients.get_value_and_gradient(cost, x, method="fd")

    assert np.allclose(value, 6.25)
    assert np.allclose(gradient, 2 * x, atol=1e-5)


def test_get_value_and_gradient_jax_autodiff():
    jnp = pytest.importorskip("jax.numpy")

    x = jnp.array([1.5, -2.0])

    def cost(params):
        return jnp.sum(params**2)

    value, gradient = mp.gradients.get_value_and_gradient(cost, x, method="jax")

    assert np.allclose(value, 6.25)
    assert np.allclose(gradient, 2 * np.asarray(x))

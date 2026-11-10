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


def test_value_and_gradient_keeps_numerical_options_separate():
    value, gradient = mp.gradients.get_value_and_gradient(
        lambda x: np.sum(x**2), np.array([2.0]), method="fd", h=1e-4
    )
    np.testing.assert_allclose(value, 4)
    np.testing.assert_allclose(gradient, [4], atol=1e-6)


@pytest.mark.parametrize("kind", ["central", "forward", "backward"])
def test_finite_difference_gradient_and_mixed_hessian(kind):
    x = np.array([0.3, -0.7])

    def cost(values):
        a, b = values
        return 2 * a**2 + 3 * a * b + 4 * b**2 + 5 * a - 7 * b + 8

    expected_gradient = np.array([4 * x[0] + 3 * x[1] + 5, 3 * x[0] + 8 * x[1] - 7])
    gradient = mp.gradients.get_gradient(cost, x, method="fd", type=kind, h=1e-3)
    hessian = mp.gradients.get_hessian(cost, x, method="fd", type=kind, h=1e-3)
    np.testing.assert_allclose(gradient, expected_gradient, atol=0.005)
    np.testing.assert_allclose(hessian, [[4, 3], [3, 8]], atol=1e-6)


@pytest.mark.parametrize("shift", [np.pi / 2, np.pi / 3])
def test_parameter_shift_mixed_hessian(shift):
    x = np.array([0.3, -0.7])
    cost = lambda values: np.sin(values[0]) * np.cos(values[1])
    a, b = x
    expected_gradient = [np.cos(a) * np.cos(b), -np.sin(a) * np.sin(b)]
    diagonal = -np.sin(a) * np.cos(b)
    off_diagonal = -np.cos(a) * np.sin(b)
    expected_hessian = [[diagonal, off_diagonal], [off_diagonal, diagonal]]
    np.testing.assert_allclose(
        mp.gradients.get_gradient(cost, x, method="psr", shift=shift), expected_gradient
    )
    np.testing.assert_allclose(
        mp.gradients.get_hessian(cost, x, method="psr", shift=shift), expected_hessian
    )


def test_autodiff_hessian_and_automatic_selection():
    jnp = pytest.importorskip("jax.numpy")
    x = jnp.array([0.3, -0.7])
    cost = (
        lambda values: values[0] ** 2 + 3 * values[0] * values[1] + 2 * values[1] ** 2
    )
    np.testing.assert_allclose(
        mp.gradients.get_hessian(cost, x, method="autodiff"), [[2, 3], [3, 4]]
    )
    value, gradient = mp.gradients.get_value_and_gradient(cost, x, method=None)
    np.testing.assert_allclose(value, cost(x))
    np.testing.assert_allclose(gradient, [2 * x[0] + 3 * x[1], 3 * x[0] + 4 * x[1]])
    np.testing.assert_allclose(
        mp.gradients.get_gradient(
            lambda a: np.sum(a**2), np.array([2.0]), method="auto"
        ),
        [4],
    )


@pytest.mark.parametrize(
    "function", [mp.gradients.get_gradient, mp.gradients.get_hessian]
)
def test_gradient_method_errors(function):
    with pytest.raises(ValueError, match="denominator"):
        function(np.sin, np.array([0.3]), method="psr", shift=0)
    with pytest.raises(UserWarning, match="Expected"):
        function(np.sin, np.array([0.3]), method="unknown")
    with pytest.raises(UserWarning, match="Expected"):
        function(np.sin, np.array([0.3]), method="fd", type="unknown")


def test_autodiff_missing_dependency(monkeypatch):
    import importlib

    gradient_module = importlib.import_module("mentpy.gradients.grad")
    monkeypatch.setattr(gradient_module, "_HAS_JAX", False)
    for function in [
        mp.gradients.get_gradient,
        mp.gradients.get_hessian,
        mp.gradients.get_value_and_gradient,
    ]:
        with pytest.raises(ImportError, match="standard dependencies"):
            function(np.sin, np.array([0.3]), method="jax")

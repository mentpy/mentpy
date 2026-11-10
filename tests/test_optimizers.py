"""Tests for optimizer gradient injection APIs."""

import numpy as np
import pytest

import mentpy as mp


def test_adam_step_accepts_precomputed_gradient():
    opt = mp.optimizers.AdamOpt(step_size=0.1)
    x = np.array([1.0, -2.0])

    updated = opt.step(
        lambda _: (_ for _ in ()).throw(AssertionError), x, 0, gradient=x
    )

    assert np.linalg.norm(updated) < np.linalg.norm(x)


def test_sgd_step_accepts_value_and_grad_callable():
    opt = mp.optimizers.SGDOpt(step_size=0.1)
    x = np.array([1.0, -2.0])

    def value_and_grad(params):
        return np.sum(params**2), 2 * params

    updated = opt.step(
        lambda params: np.sum(params**2), x, 0, value_and_grad=value_and_grad
    )

    assert np.allclose(updated, np.array([0.8, -1.6]))


def test_rcd_optimizes_integer_parameters_without_mutating_input():
    x0 = np.array([2])
    opt = mp.optimizers.RCDOpt(step_size=0.25)
    cost = lambda x: x[0] ** 2
    np.testing.assert_allclose(opt.optimize(cost, x0, num_iters=1), [1])
    result, norms = opt.optimize_and_gradient_norm(cost, x0, num_iters=1)
    np.testing.assert_allclose(result, [1])
    np.testing.assert_allclose(norms, [4])
    np.testing.assert_array_equal(x0, [2])


@pytest.mark.parametrize(
    "make",
    [
        mp.optimizers.AdamOpt,
        lambda: mp.optimizers.SGDOpt(momentum=0.5),
        lambda: mp.optimizers.RCDOpt(adaptive=True, seed=4),
    ],
)
def test_optimization_methods_share_updates_and_continuation(make):
    def cost(x):
        return np.dot(x, x)

    x = np.array([1.0, 2.0])
    optimizer = make()
    whole = optimizer.optimize(cost, x, num_iters=8)
    optimizer.reset()
    part = optimizer.optimize(cost, x, num_iters=3)
    rest, norms = optimizer.optimize_and_gradient_norm(cost, part, num_iters=5)
    np.testing.assert_allclose(rest, whole)
    assert norms.shape == (5,)

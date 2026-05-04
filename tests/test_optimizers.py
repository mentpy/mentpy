"""Tests for optimizer gradient injection APIs."""

import numpy as np

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

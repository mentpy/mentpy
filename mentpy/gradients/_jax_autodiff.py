# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Module to calculate gradients using JAX autodiff."""

import jax
import jax.numpy as jnp

__all__ = ["jax_gradient", "jax_value_and_gradient", "jax_hessian"]


def jax_gradient(cost, x, *args, **kwargs):
    """Calculate the gradient of a cost function using JAX autodiff.

    Args:
        cost (callable): Cost function to differentiate.
        x (array): Input at which to evaluate the gradient.

    Returns:
        jnp.ndarray: Gradient of the cost function.
    """
    return jax.grad(lambda x_: cost(x_, *args, **kwargs))(
        jnp.asarray(x, dtype=jnp.float64)
    )


def jax_value_and_gradient(cost, x, *args, **kwargs):
    """Calculate the value and gradient of a cost function using JAX autodiff.

    Args:
        cost (callable): Cost function to differentiate.
        x (array): Input at which to evaluate.

    Returns:
        tuple: (cost_value, gradient)
    """
    return jax.value_and_grad(lambda x_: cost(x_, *args, **kwargs))(
        jnp.asarray(x, dtype=jnp.float64)
    )


def jax_hessian(cost, x, *args, **kwargs):
    """Calculate the Hessian of a cost function using JAX autodiff.

    Args:
        cost (callable): Cost function to differentiate.
        x (array): Input at which to evaluate the Hessian.

    Returns:
        jnp.ndarray: Hessian matrix of the cost function.
    """
    return jax.hessian(lambda x_: cost(x_, *args, **kwargs))(
        jnp.asarray(x, dtype=jnp.float64)
    )

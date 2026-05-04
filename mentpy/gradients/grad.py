# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Module that contains functions to calculate gradients of cost functions."""

import numpy as np
from ._finite_difference import fd_gradient, fd_hessian
from ._parameter_shift import psr_gradient, psr_hessian

try:
    from ._jax_autodiff import jax_gradient, jax_hessian, jax_value_and_gradient

    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False

__all__ = ["get_gradient", "get_value_and_gradient", "get_hessian"]


def _canonical_method(method, x=None):
    if method is None:
        method = "auto"
    method = method.lower()
    if method in {"auto", "best"}:
        if _HAS_JAX and x is not None and "jax" in type(x).__module__:
            return "jax"
        return "finite-differences"
    return method


def get_gradient(cost, x, method="parameter-shift", *args, **kwargs):
    """Calculate the gradient of a cost function.

    Args:
        cost (callable): Cost function to calculate the gradient of.
        x (array): Input to the cost function.
        method (str, optional): Method to use to calculate the gradient. Defaults to 'parameter-shift'.

    Returns:
        array: Gradient of the cost function.
    """

    method = _canonical_method(method, x)

    match method:
        case "parameter-shift" | "psr" | "parametershift":
            return psr_gradient(cost, x, *args, **kwargs)
        case "finite-difference" | "finite-differences" | "fd" | "finitedifferences":
            return fd_gradient(cost, x, *args, **kwargs)
        case "jax" | "autodiff":
            if not _HAS_JAX:
                raise ImportError(
                    "JAX is required for autodiff gradients. "
                    "Install it with: pip install 'mentpy[jax]'"
                )
            return jax_gradient(cost, x, *args, **kwargs)
        case _:
            raise UserWarning(
                f"Expected method to be 'parameter-shift', 'finite-difference', or 'jax' but {method} was given"
            )


def get_value_and_gradient(cost, x, method="auto", *args, **kwargs):
    """Return ``(value, gradient)`` for a scalar cost.

    ``method="jax"`` uses :func:`jax.value_and_grad` and evaluates the cost once.
    Numerical methods evaluate the value once and then reuse
    :func:`get_gradient`.
    """
    method = _canonical_method(method, x)
    if method in {"jax", "autodiff"}:
        if not _HAS_JAX:
            raise ImportError(
                "JAX is required for autodiff gradients. "
                "Install it with: pip install 'mentpy[jax]'"
            )
        return jax_value_and_gradient(cost, x, *args, **kwargs)

    value = cost(x)
    gradient = get_gradient(cost, x, method, *args, **kwargs)
    return value, gradient


def get_hessian(cost, x, method="parameter-shift", *args, **kwargs):
    """Calculate the Hessian of a cost function.

    Args:
        cost (callable): Cost function to calculate the Hessian of.
        x (array): Input to the cost function.
        method (str, optional): Method to use to calculate the Hessian. Defaults to 'parameter-shift'.

    Returns:
        array: Hessian of the cost function.
    """

    method = _canonical_method(method, x)

    match method:
        case "parameter-shift" | "psr" | "parametershift":
            return psr_hessian(cost, x, *args, **kwargs)
        case "finite-difference" | "finite-differences" | "fd" | "finitedifferences":
            return fd_hessian(cost, x, *args, **kwargs)
        case "jax" | "autodiff":
            if not _HAS_JAX:
                raise ImportError(
                    "JAX is required for autodiff Hessians. "
                    "Install it with: pip install 'mentpy[jax]'"
                )
            return jax_hessian(cost, x, *args, **kwargs)
        case _:
            raise UserWarning(
                f"Expected method to be 'parameter-shift', 'finite-difference', or 'jax' but {method} was given"
            )

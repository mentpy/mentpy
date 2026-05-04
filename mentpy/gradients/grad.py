# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Module that contains functions to calculate gradients of cost functions."""

import numpy as np
from ._finite_difference import fd_gradient, fd_hessian
from ._parameter_shift import psr_gradient, psr_hessian

try:
    from ._jax_autodiff import jax_gradient, jax_hessian

    _HAS_JAX = True
except ImportError:
    _HAS_JAX = False

__all__ = ["get_gradient", "get_hessian"]


def get_gradient(cost, x, method="parameter-shift", *args, **kwargs):
    """Calculate the gradient of a cost function.

    Args:
        cost (callable): Cost function to calculate the gradient of.
        x (array): Input to the cost function.
        method (str, optional): Method to use to calculate the gradient. Defaults to 'parameter-shift'.

    Returns:
        array: Gradient of the cost function.
    """

    match method:
        case "parameter-shift" | "psr" | "parametershift":
            return psr_gradient(cost, x, *args, **kwargs)
        case "finite-differences" | "fd" | "finitedifferences":
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


def get_hessian(cost, x, method="parameter-shift", *args, **kwargs):
    """Calculate the Hessian of a cost function.

    Args:
        cost (callable): Cost function to calculate the Hessian of.
        x (array): Input to the cost function.
        method (str, optional): Method to use to calculate the Hessian. Defaults to 'parameter-shift'.

    Returns:
        array: Hessian of the cost function.
    """

    match method:
        case "parameter-shift" | "psr" | "parametershift":
            return psr_hessian(cost, x, *args, **kwargs)
        case "finite-differences" | "fd" | "finitedifferences":
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

# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Module to calculate gradients using the parameter shift rule."""

import numpy as np


def psr_gradient(cost, x, shift=np.pi / 2):
    """Calculate the gradient of a cost function using the parameter shift rule.

    Args:
        cost (callable): Cost function to calculate the gradient of.
        x (array): Input to the cost function.
        shift (float, optional): Shift to use in the parameter shift rule.
            Defaults to :math:`\\pi / 2`.

    Returns:
        array: Gradient of the cost function.
    """
    x = np.asarray(x, dtype=float)
    denominator = 2 * np.sin(shift)
    if np.isclose(denominator, 0.0):
        raise ValueError("Parameter-shift denominator is zero for this shift.")

    grad = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        x_forward = x.copy()
        x_forward[i] += shift
        x_backward = x.copy()
        x_backward[i] -= shift
        grad[i] = (cost(x_forward) - cost(x_backward)) / denominator
    return grad


def psr_hessian(cost, x, shift=np.pi / 2):
    """Calculate the Hessian of a cost function using the parameter shift rule.

    Args:
        cost (callable): Cost function to calculate the Hessian of.
        x (array): Input to the cost function.
        shift (float, optional): Shift to use in the parameter shift rule.
            Defaults to :math:`\\pi / 2`.

    Returns:
        array: Hessian of the cost function.
    """
    x = np.asarray(x, dtype=float)
    denominator = 4 * np.sin(shift) ** 2
    if np.isclose(denominator, 0.0):
        raise ValueError("Parameter-shift denominator is zero for this shift.")

    hess = np.zeros((len(x), len(x)), dtype=float)
    for i in range(len(x)):
        for j in range(len(x)):
            x_ff = x.copy()
            x_ff[i] += shift
            x_ff[j] += shift
            x_fb = x.copy()
            x_fb[i] += shift
            x_fb[j] -= shift
            x_bf = x.copy()
            x_bf[i] -= shift
            x_bf[j] += shift
            x_bb = x.copy()
            x_bb[i] -= shift
            x_bb[j] -= shift
            hess[i, j] = (
                cost(x_ff) - cost(x_fb) - cost(x_bf) + cost(x_bb)
            ) / denominator
    return hess

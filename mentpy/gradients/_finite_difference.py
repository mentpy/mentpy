# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Module to calculate gradients using the finite difference method."""

import numpy as np


def fd_gradient(f, x, h=1e-5, type="central"):
    if type not in ["central", "forward", "backward"]:
        raise UserWarning(
            f"Expected type to be 'central', 'forward', or 'backward' but {type} was given"
        )

    x = np.asarray(x, dtype=float)
    grad = np.zeros_like(x, dtype=float)
    f0 = None
    if type in {"forward", "backward"}:
        f0 = f(x)

    for i in range(len(x)):
        x_forward = x.copy()
        x_forward[i] += h
        x_backward = x.copy()
        x_backward[i] -= h
        if type == "central":
            grad[i] = (f(x_forward) - f(x_backward)) / (2 * h)
        elif type == "forward":
            grad[i] = (f(x_forward) - f0) / h
        elif type == "backward":
            grad[i] = (f0 - f(x_backward)) / h
    return grad


def fd_hessian(f, x, h=1e-5, type="central"):
    if type not in ["central", "forward", "backward"]:
        raise UserWarning(
            f"Expected type to be 'central', 'forward', or 'backward' but {type} was given"
        )

    x = np.asarray(x, dtype=float)
    hess = np.zeros((len(x), len(x)), dtype=float)
    f0 = None
    if type in {"forward", "backward"}:
        f0 = f(x)

    for i in range(len(x)):
        for j in range(len(x)):
            x_i_forward = x.copy()
            x_i_forward[i] += h
            x_j_forward = x.copy()
            x_j_forward[j] += h
            x_i_backward = x.copy()
            x_i_backward[i] -= h
            x_j_backward = x.copy()
            x_j_backward[j] -= h
            x_ff = x.copy()
            x_ff[i] += h
            x_ff[j] += h
            x_fb = x.copy()
            x_fb[i] += h
            x_fb[j] -= h
            x_bf = x.copy()
            x_bf[i] -= h
            x_bf[j] += h
            x_bb = x.copy()
            x_bb[i] -= h
            x_bb[j] -= h
            if type == "central":
                hess[i, j] = (f(x_ff) - f(x_fb) - f(x_bf) + f(x_bb)) / (4 * h**2)
            elif type == "forward":
                hess[i, j] = (f(x_ff) - f(x_i_forward) - f(x_j_forward) + f0) / h**2
            elif type == "backward":
                hess[i, j] = (f0 - f(x_i_backward) - f(x_j_backward) + f(x_bb)) / h**2
    return hess

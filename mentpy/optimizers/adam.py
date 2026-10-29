# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""This module contains the Adam optimizer."""

import numpy as np

from mentpy.optimizers.base import BaseOpt


class AdamOpt(BaseOpt):
    """Class for the Adam optimizer.

    Parameters
    ----------
    step_size : float, optional
        The step size of the optimizer, by default 0.1
    b1 : float, optional
        The first moment decay rate, by default 0.9
    b2 : float, optional
        The second moment decay rate, by default 0.999
    eps : float, optional
        A small number to avoid division by zero, by default 10**-8

    Examples
    --------
    Create an Adam optimizer

    .. ipython:: python

        opt = mp.optimizers.AdamOpt()
        print(opt)


    See Also
    --------
    :class:`mp.optimizers.SGDOpt`

    Group
    -----
    optimizers
    """

    def __init__(
        self,
        step_size=0.1,
        b1=0.9,
        b2=0.999,
        eps=10**-8,
        gradient_method="parameter-shift",
        gradient_kwargs=None,
    ) -> None:
        """Initialize the Adam optimizer."""
        super().__init__(
            gradient_method=gradient_method, gradient_kwargs=gradient_kwargs
        )
        self.step_size = step_size
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.m = None
        self.v = None

    def step(self, f, x, i, **kwargs):
        """Take a step of the optimizer."""
        x = np.asarray(x, dtype=float)
        g = self._gradient(f, x, **kwargs)
        if self.m is None:
            self.m = np.zeros_like(x, dtype=float)
        if self.v is None:
            self.v = np.zeros_like(x, dtype=float)
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g**2
        m_hat = self.m / (1 - self.b1 ** (i + 1))
        v_hat = self.v / (1 - self.b2 ** (i + 1))
        x = x - self.step_size * m_hat / (np.sqrt(v_hat) + self.eps)
        return x

    def optimize(self, f, x0, num_iters=100, callback=None, verbose=False, **kwargs):
        """Optimize a function f using the Adam optimizer."""
        x = np.asarray(x0, dtype=float)
        for i in range(num_iters):
            x = self.step(f, x, i, **kwargs)
            if callback is not None:
                callback(x, i)
            if verbose:
                print(f"Iteration {i+1}/{num_iters} - x: {x}")
        return x

    def update_step_size(self, x, i, factor=0.99):
        """Update the step size of the optimizer."""
        self.step_size = self.step_size * factor

    def optimize_and_gradient_norm(
        self, f, x0, num_iters=100, callback=None, verbose=False, **kwargs
    ):
        """Optimize a function f using the Adam optimizer."""
        x = np.asarray(x0, dtype=float)
        norm = np.zeros(num_iters)

        for i in range(num_iters):
            g = self._gradient(f, x, **kwargs)
            x = self.step(f, x, i, gradient=g)
            norm[i] = np.linalg.norm(g)
            if callback is not None:
                callback(x, i)
            if verbose:
                print(f"Iteration {i+1} of {num_iters}: {x} with value {f(x)}")

        return x, norm

    def reset(self):
        """Reset the optimizer."""
        self.m = None
        self.v = None

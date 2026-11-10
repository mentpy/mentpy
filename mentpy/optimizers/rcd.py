# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""This module contains the random coordinate descent optimizer."""

from mentpy.optimizers.base import BaseOpt

import numpy as np


class RCDOpt(BaseOpt):
    """Class for the random coordinate descent optimizer.

    Args
    ----
    step_size : float, optional
        The initial step size of the optimizer, by default 0.1
    adaptive : bool, optional
        Whether to use an adaptive step size, by default False

    Examples
    --------
    Create a random coordinate descent optimizer

    .. ipython:: python

        opt = mp.optimizers.RCDOpt()
        print(opt)

    Group
    -----
    optimizers
    """

    def __init__(self, step_size=0.1, adaptive=False, seed=None) -> None:
        """Initialize the random coordinate descent optimizer."""
        super().__init__()
        self.step_size = step_size
        self.adaptive = adaptive
        self._seed = seed
        self.reset()

    def _step(self, f, x, h=1e-5, **kwargs):
        if not np.isfinite(h) or h <= 0:
            raise ValueError("Finite-difference step must be positive and finite.")
        if not x.size:
            return x.copy(), 0.0
        coordinate = self._rng.integers(x.size)
        updated = x.copy()
        if kwargs:
            gradient = self._gradient(f, x, **kwargs).flat[coordinate]
        else:
            updated.flat[coordinate] += h
            forward = f(updated)
            updated.flat[coordinate] = x.flat[coordinate] - h
            gradient = (forward - f(updated)) / (2 * h)
        self._steps += 1
        step_size = (
            self.step_size / np.sqrt(self._steps) if self.adaptive else self.step_size
        )
        updated.flat[coordinate] = x.flat[coordinate] - step_size * gradient
        return updated, abs(gradient)

    def update_step_size(self, x, i, factor=0.99):
        """Update the step size of the optimizer."""
        self.step_size = self.step_size * factor

    def reset(self):
        """Clear the schedule and restart the seeded random stream."""
        self._steps = 0
        self._rng = np.random.default_rng(self._seed)

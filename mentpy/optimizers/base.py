# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Base classes and utilities for optimizers."""

import abc

import numpy as np

from mentpy.gradients import get_gradient


class BaseOpt(abc.ABC):
    """Base class for optimizers.

    Note
    ----
    This class should not be used directly. Instead, use one of the subclasses.

    See Also
    --------
    :class:`mp.optimizers.SGDOpt`, :class:`mp.optimizers.AdamOpt`

    Group
    -----
    optimizers
    """

    def __init__(self, gradient_method="finite-difference", gradient_kwargs=None):
        self.gradient_method = gradient_method
        self.gradient_kwargs = {} if gradient_kwargs is None else dict(gradient_kwargs)

    def _gradient(
        self,
        f,
        x,
        *,
        gradient=None,
        grad=None,
        value_and_grad=None,
        method=None,
        gradient_method=None,
        gradient_kwargs=None,
        **kwargs,
    ):
        """Resolve a gradient from explicit data, a callable, or a method name."""
        if gradient is not None:
            return np.asarray(gradient, dtype=float)

        if grad is not None:
            if not callable(grad):
                return np.asarray(grad, dtype=float)
            return np.asarray(grad(x), dtype=float)

        if value_and_grad is not None:
            _value, resolved_gradient = value_and_grad(x)
            return np.asarray(resolved_gradient, dtype=float)

        resolved_method = (
            method
            if method is not None
            else (
                gradient_method if gradient_method is not None else self.gradient_method
            )
        )
        resolved_kwargs = dict(self.gradient_kwargs)
        if gradient_kwargs is not None:
            resolved_kwargs.update(gradient_kwargs)
        resolved_kwargs.update(kwargs)
        return np.asarray(
            get_gradient(f, x, method=resolved_method, **resolved_kwargs), dtype=float
        )

    def step(self, f, x, i=None, **kwargs):
        """Update the parameters once using the optimizer's current state."""
        return self._step(f, np.asarray(x, dtype=float), **kwargs)[0]

    @abc.abstractmethod
    def _step(self, f, x, **kwargs):
        """Return the updated parameters and the norm of the gradient used."""

    def optimize(self, f, x0, num_iters=100, callback=None, verbose=False, **kwargs):
        """Run consecutive steps, continuing state until reset() is called."""
        return self._optimize(f, x0, num_iters, callback, verbose, kwargs)[0]

    def optimize_and_gradient_norm(
        self, f, x0, num_iters=100, callback=None, verbose=False, **kwargs
    ):
        """Run consecutive steps and return one gradient norm per update."""
        return self._optimize(f, x0, num_iters, callback, verbose, kwargs)

    def _optimize(self, f, x0, num_iters, callback, verbose, kwargs):
        x = np.array(x0, dtype=float, copy=True)
        norms = np.empty(num_iters)
        for i in range(num_iters):
            x, norms[i] = self._step(f, x, **kwargs)
            if callback is not None:
                callback(x, i)
            if verbose:
                print(f"Iteration {i + 1}/{num_iters}, x = {x}")
        return x, norms

    @abc.abstractmethod
    def reset(self, *args, **kwargs):
        """Clear the optimizer state for a new run."""

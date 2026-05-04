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

    def __init__(self, gradient_method="parameter-shift", gradient_kwargs=None):
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

    @abc.abstractmethod
    def step(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def optimize(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def optimize_and_gradient_norm(self, *args, **kwargs):
        pass

    @abc.abstractmethod
    def reset(self, *args, **kwargs):
        pass

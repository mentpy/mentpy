# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.

from .grad import *

try:
    from ._jax_autodiff import *
except ImportError as exc:
    if exc.name not in {"jax", "jaxlib"}:
        raise

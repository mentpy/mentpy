# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""
The Measurement-Based Quantum computing simulator.
"""
from . import calculator

from .mbqc import *
from .operators import *
from .simulators import *

from . import gradients
from . import optimizers
from . import utils

from packaging.version import parse as parse_version

try:
    import importlib.metadata

    __version__ = importlib.metadata.version("mentpy")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

__version_info__ = parse_version(__version__)

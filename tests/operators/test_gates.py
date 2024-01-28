# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Tests for the gates module."""
import mentpy as mp
import numpy as np
import pytest


def test_isingxx_gate():
    """Test the IsingXX gate."""
    assert np.allclose(mp.gates.ising_xx(0), np.eye(4))

# Copyright 2026 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Run the molecular optimization examples through the public primitives."""

from pathlib import Path
import runpy

import matplotlib.pyplot as plt
import numpy as np
import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"


def test_molecular_energy_loop():
    pytest.importorskip("jax")
    result = runpy.run_path(str(EXAMPLES / "vqe.py"))
    try:
        energies = np.asarray(result["energies"])
        assert len(energies) == result["steps"] + 1
        assert np.all(np.isfinite(energies))
        assert energies[-1] < energies[0] - 0.001
        assert abs(energies[-1] - result["exact_energy"]) < 1e-5
        np.testing.assert_allclose(energies[-1], result["energy"](result["angles"]))
    finally:
        plt.close(result["fig"])


def test_molecular_geometry_sweep():
    pytest.importorskip("jax")
    pyscf = pytest.importorskip("pyscf")
    pytest.importorskip("openfermionpyscf")
    previous_threads = pyscf.lib.num_threads()
    try:
        pyscf.lib.num_threads(1)
        result = runpy.run_path(str(EXAMPLES / "pes.py"))
    finally:
        pyscf.lib.num_threads(previous_threads)
    try:
        energies = np.asarray(result["energies"])
        exact = np.asarray(result["exact_energies"])
        hf = np.asarray(result["hf_energies"])
        assert energies.shape == result["bond_lengths"].shape
        assert np.all(np.isfinite(energies))
        assert np.all(energies >= exact - 1e-9)
        assert np.all(energies < hf)
        assert np.max(energies - exact) < 1e-5
        recorded = np.loadtxt(
            EXAMPLES / "data" / "h2-pes.csv", delimiter=",", comments="#"
        )
        np.testing.assert_allclose(recorded[:, 0], result["bond_lengths"])
        np.testing.assert_allclose(recorded[:, 1], energies, atol=1e-8)
    finally:
        plt.close(result["fig"])

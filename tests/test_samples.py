# Copyright 2026 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Output ordering, Pauli bases, reproducibility, and shot validation."""

import numpy as np
import pytest
import mentpy as mp
from mentpy.simulators.results import measurement_probabilities


@pytest.mark.parametrize(
    "backend", ["trajectory", "numpy-sv", "numpy-dm", "jax-tn", "pennylane"]
)
def test_output_sampling(backend):
    pattern = mp.templates.linear_cluster(5)
    sim = mp.PatternSimulator(pattern, backend=backend)
    angles = np.zeros(len(pattern.trainable_nodes))
    np.testing.assert_allclose(sim.probabilities(angles), [0.5, 0.5], atol=1e-9)
    samples = sim.sample(angles, shots=1000, seed=29)
    repeat = sim.sample(angles, shots=1000, seed=29)
    np.testing.assert_array_equal(samples.bitstrings, repeat.bitstrings)
    assert samples.shots == sum(samples.counts.values()) == 1000
    assert abs(samples.probabilities["0"] - 0.5) < 0.06
    assert samples.output_nodes == tuple(pattern.quantum_output_nodes)
    x_samples = sim.sample(angles, shots=32, basis="X", seed=4)
    assert x_samples.counts == {"0": 32}
    assert x_samples.expectation == 1
    assert x_samples.standard_error == 0


def test_pauli_y_basis_and_density_matrix():
    state = np.array([1, 1j]) / np.sqrt(2)
    for value in [state, np.outer(state, state.conj())]:
        np.testing.assert_allclose(
            measurement_probabilities(value, "Y"), [1, 0], atol=1e-10
        )
    samples = mp.Samples(np.array([[0, 1], [1, 0]]), (3, 2), "ZY")
    assert samples.counts == {"01": 1, "10": 1}
    assert samples.expectation == -1
    with pytest.raises(ValueError):
        samples.bitstrings[0, 0] = 1


@pytest.mark.parametrize("shots", [True, False, 0, -1, 3.2, "2"])
def test_reject_invalid_shots(shots):
    sim = mp.PatternSimulator(mp.templates.linear_cluster(3), backend="trajectory")
    with pytest.raises(ValueError, match="positive integer"):
        sim.sample([0, 0], shots=shots)
    with pytest.raises(ValueError, match="positive integer"):
        mp.Observable("Z").sample_expectation(np.array([1, 0]), shots)


def test_frontier_handles_thousands_of_graph_nodes():
    pattern = mp.templates.linear_cluster(2001)
    sim = mp.PatternSimulator(pattern, backend="trajectory", force0=False, seed=7)
    np.testing.assert_allclose(
        sim.probabilities(np.zeros(2000), basis="X"), [1, 0], atol=1e-8
    )
    assert sim.max_frontier <= 3

# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Tests for the simulator module."""

import pytest
import numpy as np
import mentpy as mp

all_backends = ["numpy-dm", "numpy-sv", "pennylane"]


@pytest.mark.parametrize("backend", all_backends)
class TestBackends(object):
    def test_teleportation(cls, backend):
        """Test teleportation in one wire."""
        for i in range(1, 5):
            gs = mp.templates.linear_cluster(2 * i + 1)
            sts = mp.utils.generate_haar_random_states(1, 5)
            ps = mp.simulators.PatternSimulator(gs, backend=backend)
            for st in sts:
                ps.reset(input_state=st)
                assert (
                    len(ps.mbqcircuit.trainable_nodes) == 2 * i
                ), f"{ps.mbqcircuit.trainable_nodes} != {2*i}, {backend} failed"
                output_dm = ps([0] * (2 * i))
                expected_dm = np.outer(st, st.conj().T)
                assert np.allclose(
                    output_dm, expected_dm, atol=1e-3
                ), f"{backend} failed, {2*i+1} qubits"


param_templates = [
    lambda i: mp.templates.grid_cluster(2, 3 + i),
    lambda i: mp.templates.linear_cluster(3 + i),
    lambda i: mp.templates.muta(2, 1),
]


@pytest.mark.parametrize("templ", param_templates)
def test_random_measurements_equal(templ):
    """Test random measurements."""
    for i in range(1, 3):
        gs = templ(i)
        if np.random.rand() < 0.5:
            indx = np.random.choice(gs.outputc, 2, replace=False)
            gs[indx[0]] = mp.Ment("X")
            gs[indx[1]] = mp.Ment("X")

        # random measurements
        angles = 2 * np.pi * np.random.rand(len(gs.trainable_nodes))
        results = []
        for backend in all_backends:
            ps = mp.simulators.PatternSimulator(
                gs, backend=backend, window_size=min(5, len(gs.trainable_nodes))
            )
            ps.reset()
            results.append(ps.run(angles))

        # check that the results are equal
        for j in range(len(results) - 1):
            assert np.allclose(
                results[j], results[j + 1]
            ), f"results not equal for {all_backends[j]} and {all_backends[j+1]}"


def test_pattern_simulator_expectation_exact_and_shots():
    gs = mp.templates.linear_cluster(3)
    angles = np.zeros(len(gs.trainable_nodes))
    observable = mp.Observable({"X": 1.0})
    ps = mp.PatternSimulator(gs, backend="numpy-sv")

    exact = ps.expectation(angles, observable)
    sampled = ps.expectation(angles, observable, shots=100, seed=123)

    assert np.allclose(exact, 1.0)
    assert np.allclose(sampled, 1.0)


@pytest.mark.parametrize("backend", ["numpy-sv", "numpy-dm"])
def test_numpy_minimal_pattern_and_repeated_runs(backend):
    pattern = mp.templates.linear_cluster(2)
    sim = mp.PatternSimulator(pattern, np.array([1, 0]), backend=backend)
    expected = np.ones((2, 2)) / 2
    np.testing.assert_allclose(sim.run([0.0]), expected)
    np.testing.assert_allclose(sim.run([0.0]), expected)


def test_numpy_sv_does_not_project_removed_qubit_twice():
    pattern = mp.templates.linear_cluster(3)
    sim = mp.PatternSimulator(pattern, np.array([1, 0]), backend="numpy-sv")
    state = sim.run([np.pi, 0], output_form="sv")
    np.testing.assert_allclose(np.abs(state) ** 2, [1, 0], atol=1e-12)


@pytest.mark.parametrize("backend", ["numpy-sv", "numpy-dm"])
def test_numpy_window_covers_all_graph_edges(backend):
    pattern = mp.templates.muta(2, 1)
    angles = np.linspace(0.2, 1.0, len(pattern.trainable_nodes))
    expected = mp.PatternSimulator(pattern, backend="trajectory").run(angles)
    sim = mp.PatternSimulator(pattern, backend=backend)
    np.testing.assert_allclose(sim.run(angles), expected, atol=1e-12)
    with pytest.raises(ValueError, match="invalid for this schedule"):
        mp.PatternSimulator(pattern, backend=backend, window_size=3)


def test_numpy_sv_reduces_measured_output():
    pattern = mp.templates.linear_cluster(3)
    pattern[2] = mp.Ment("X")
    sim = mp.PatternSimulator(pattern, backend="numpy-sv")
    np.testing.assert_allclose(sim.run([0, 0]), [[1]])
    assert sim.outcomes == {0: 0, 1: 0, 2: 0}


def test_pennylane_supports_named_nodes():
    graph = mp.GraphState([("left", "middle"), ("middle", "right")])
    pattern = mp.MBQCircuit(graph, ["left"], ["right"], relabel_indices=False)
    sim = mp.PatternSimulator(pattern, np.array([1, 0]), backend="pennylane")
    np.testing.assert_allclose(sim.run([0, 0]), [[1, 0], [0, 0]], atol=1e-12)

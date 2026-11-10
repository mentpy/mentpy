# Copyright 2026 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Seeded native trajectories, adaptive control, and independent shot laws."""

import networkx as nx
import numpy as np
import pytest

import mentpy as mp
from mentpy import _rust


@pytest.mark.parametrize("plane", ["XY", "XZ", "YZ", "X", "Y", "Z"])
def test_trajectories_match_numpy(plane):
    rng = np.random.default_rng(518)
    pattern = mp.templates.grid_cluster(2, 4)
    for node in pattern.outputc:
        pattern[node] = mp.Ment(plane)
    angles = rng.normal(size=len(pattern.trainable_nodes))
    state = rng.normal(size=4) + 1j * rng.normal(size=4)
    state /= np.linalg.norm(state)
    native = mp.PatternSimulator(pattern, state, backend="rust", force0=False)
    reference = mp.PatternSimulator(pattern, state, backend="numpy", force0=False)
    for seed in range(8):
        np.testing.assert_allclose(
            native.run(angles, seed=seed), reference.run(angles, seed=seed), atol=1e-12
        )
        assert native.outcomes == reference.outcomes
    assert native.max_frontier <= len(pattern)


def test_adaptive_plane_and_parameter_selection():
    pattern = mp.templates.linear_cluster(5)
    pattern[1] = mp.ControlMent(
        ~pattern[0].outcome,
        true_plane="YZ",
        true_angle=None,
        false_plane="X",
        false_angle=0,
    )
    pattern[3] = mp.ControlMent(
        pattern[0].outcome ^ pattern[2].outcome,
        true_plane="XZ",
        true_angle=None,
        false_plane="XY",
        false_angle=0.7,
    )
    angles = np.linspace(-0.7, 0.3, len(pattern.trainable_nodes))
    native = mp.PatternSimulator(pattern, [1, 0], backend="rust", force0=False)
    reference = mp.PatternSimulator(pattern, [1, 0], backend="numpy", force0=False)
    for seed in range(20):
        np.testing.assert_allclose(
            native.run(angles, seed=seed), reference.run(angles, seed=seed), atol=1e-12
        )
        assert native.outcomes == reference.outcomes


def test_no_flow_named_and_reordered_nodes():
    graph = mp.GraphState([("in", "a"), ("a", "out"), ("in", "out"), ("other", "b")])
    pattern = mp.MBQCircuit(
        graph,
        ["other", "in"],
        ["out", "b"],
        initialize_flow=False,
        relabel_indices=False,
    )
    rng = np.random.default_rng(11)
    state = rng.normal(size=4) + 1j * rng.normal(size=4)
    state /= np.linalg.norm(state)
    angles = [0.7] * len(pattern.trainable_nodes)
    native = mp.PatternSimulator(pattern, state, backend="rust", force0=False)
    reference = mp.PatternSimulator(pattern, state, backend="numpy", force0=False)
    for seed in range(10):
        np.testing.assert_allclose(
            native.run(angles, seed=seed), reference.run(angles, seed=seed), atol=1e-12
        )
        assert native.outcomes == reference.outcomes


def test_xyz_trajectory():
    pattern = mp.templates.linear_cluster(3)
    with pytest.warns(UserWarning):
        pattern[0] = mp.Ment("XYZ")
    angles = [(0.3, 0.2), -0.4]
    native = mp.PatternSimulator(pattern, backend="rust", force0=False)
    reference = mp.PatternSimulator(pattern, backend="numpy", force0=False)
    for seed in range(5):
        with pytest.warns(UserWarning):
            expected = reference.run(angles, seed=seed)
        np.testing.assert_allclose(native.run(angles, seed=seed), expected, atol=1e-12)
        assert native.outcomes == reference.outcomes


def test_shots_resample_adaptive_branches_and_match_seeds():
    pattern = mp.templates.linear_cluster(3)
    pattern[0] = mp.Ment("X")
    pattern[1] = mp.ControlMent(~pattern[0].outcome, true_angle=np.pi)
    native = mp.PatternSimulator(pattern, [1, 0], backend="rust", force0=False, seed=16)
    reference = mp.PatternSimulator(
        pattern, [1, 0], backend="numpy", force0=False, seed=16
    )
    samples = native.sample([], shots=2000, seed=7)
    expected = reference.sample([], shots=2000, seed=7)
    np.testing.assert_array_equal(samples.bitstrings, expected.bitstrings)
    assert abs(samples.expectation) < 0.05
    assert native.expectation([], "Z", shots=2000, seed=19) == reference.expectation(
        [], "Z", shots=2000, seed=19
    )
    assert abs(native.expectation([], "Z", seed=19)) == 1
    with pytest.raises(ValueError, match="postselected"):
        native.expectation_and_grad([], "Z")


def test_joint_samples_keep_correlations():
    graph = mp.GraphState([(0, 1)])
    pattern = mp.MBQCircuit(graph, [], [0, 1])
    sim = mp.PatternSimulator(pattern, backend="rust", force0=False)
    samples = sim.sample([], shots=300, basis="XZ", seed=33)
    assert samples.expectation == 1
    assert set(samples.counts) == {"00", "11"}


@pytest.mark.parametrize("basis", ["IY", "XY", "YZ", "ZZ"])
def test_batched_shots_preserve_rng_order_and_pauli_readout(basis):
    pattern = mp.templates.grid_cluster(2, 4)
    angles = np.linspace(-0.4, 0.7, len(pattern.trainable_nodes))
    native = mp.PatternSimulator(pattern, backend="rust", force0=False)
    reference = mp.PatternSimulator(pattern, backend="numpy", force0=False)
    first = native.sample(angles, shots=513, basis=basis, seed=71)
    second = reference.sample(angles, shots=513, basis=basis, seed=71)
    np.testing.assert_array_equal(first.bitstrings, second.bitstrings)
    assert native.outcomes == reference.outcomes
    observable = mp.Observable({basis: 0.7, "XX": -0.2}, constant=0.3)
    assert native.expectation(
        angles, observable, shots=257, seed=48
    ) == reference.expectation(angles, observable, shots=257, seed=48)


def test_implicit_inputs_stay_local_and_reset_is_current():
    graph = mp.GraphState(nx.empty_graph(100))
    pattern = mp.MBQCircuit(graph, list(graph), [], initialize_flow=False)
    sim = mp.PatternSimulator(pattern, backend="rust", force0=False)
    np.testing.assert_allclose(sim.run(np.zeros(100), seed=1), [[1]])
    assert sim.max_frontier == 1 and sim.input_state is None
    pattern = mp.templates.linear_cluster(3)
    sim = mp.PatternSimulator(pattern, backend="rust", force0=False)
    sim.reset([1, 0])
    np.testing.assert_allclose(sim.run([0, 0], seed=1), [[1, 0], [0, 0]], atol=1e-12)
    sim.input_state[:] = [0, 1]
    np.testing.assert_allclose(sim.run([0, 0], seed=1), [[0, 0], [0, 1]], atol=1e-12)


def test_native_step_boundaries_and_limits():
    state = _rust.Frontier(2, [(0, 1)], [0], [1], [0], [[]], max_elements=4)
    with pytest.raises(ValueError, match="all measurements"):
        state.finish()
    with pytest.raises(ValueError, match="uniform"):
        state.step(0, 0, 1.0)
    assert state.step(0, 0, 0.1) == 0
    with pytest.raises(ValueError, match="already"):
        state.step(0, 0, 0.1)
    state.reset()
    with pytest.raises(ValueError, match="one angle"):
        state.run([])
    with pytest.raises(ValueError, match="together"):
        state.set_input([1, 0])
    with pytest.raises(ValueError, match="normalized"):
        state.set_input([0, 0], [0, 0])
    small = _rust.Frontier(3, [(0, 1), (0, 2)], [0], [1, 2], [0], [[]], max_elements=4)
    with pytest.raises(ValueError, match="max_elements"):
        small.run([0, 0], [0.2])
    with pytest.raises(ValueError, match="future"):
        _rust.Frontier(2, [(0, 1)], [0], [1], [0], [[(0, 0)]])
    with pytest.raises(ValueError, match="sampling"):
        state.sample([0, 0], [0.1], "Z")
    with pytest.raises(ValueError, match="sampling"):
        state.sample([0, 0], [0.1, np.nan], "Z")

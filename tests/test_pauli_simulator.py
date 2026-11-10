# Copyright 2026 Luis Mantilla
# Licensed under the Apache License, Version 2.0.
"""Pauli expectations and gradients against independent dense simulation."""

import itertools

import networkx as nx
import numpy as np
import pytest

import mentpy as mp
from mentpy import _rust


@pytest.mark.parametrize("plane", ["X", "Y", "XY"])
@pytest.mark.parametrize("explicit", [False, True])
def test_pauli_words_and_gradients(plane, explicit):
    rng = np.random.default_rng(82)
    pattern = mp.templates.grid_cluster(2, 4)
    for node in pattern.outputc:
        pattern[node] = mp.Ment(plane)
    state = None
    if explicit:
        state = rng.normal(size=4) + 1j * rng.normal(size=4)
        state /= np.linalg.norm(state)
    angles = rng.normal(size=len(pattern.trainable_nodes))
    sim = mp.PatternSimulator(pattern, state, backend="pauli")
    ref = mp.PatternSimulator(pattern, state, backend="numpy")
    for chars in itertools.product("IXYZ", repeat=2):
        obs = mp.Observable("".join(chars)) + 0.13
        expected = ref.expectation(angles, obs)
        np.testing.assert_allclose(sim.expectation(angles, obs), expected, atol=1e-12)
        value, gradient = sim.expectation_and_grad(angles, obs)
        np.testing.assert_allclose(value, expected, atol=1e-12)
        if len(angles):
            diff = mp.gradients.get_gradient(
                lambda a: ref.expectation(a, obs), angles, method="fd", h=1e-5
            )
            np.testing.assert_allclose(gradient, diff, atol=1e-7)
        else:
            assert gradient.shape == (0,)


@pytest.mark.parametrize(
    "angle", [0, np.pi / 2, np.pi, -np.pi / 2, 0.0000001, 2**40 * np.pi]
)
def test_clifford_zero_coefficients_keep_derivatives(angle):
    pattern = mp.templates.linear_cluster(2)
    sim = mp.PatternSimulator(pattern, backend="pauli")
    value, gradient = sim.expectation_and_grad([angle], "Y")
    np.testing.assert_allclose(value, np.sin(angle), atol=1e-15)
    np.testing.assert_allclose(gradient, [np.cos(angle)], atol=1e-15)
    value, gradient = sim.expectation_and_grad([angle], "Z")
    np.testing.assert_allclose(value, np.cos(angle), atol=1e-15)
    np.testing.assert_allclose(gradient, [-np.sin(angle)], atol=1e-15)


def test_mixed_observable_and_named_reordered_wires():
    graph = mp.GraphState(
        [("in_a", "a"), ("a", "out_a"), ("in_b", "b"), ("b", "out_b"), ("a", "b")]
    )
    pattern = mp.MBQCircuit(
        graph, ["in_b", "in_a"], ["out_a", "out_b"], relabel_indices=False
    )
    state = np.array([1, 1j, 2, -1j]) / np.sqrt(7)
    angles = [0.4, -0.6, 0.7, 1.2]
    observable = mp.Observable({"XY": 0.8, "YZ": -0.9, "ZZ": 0.2}, constant=-0.3)
    sim = mp.PatternSimulator(pattern, state, backend="pauli")
    ref = mp.PatternSimulator(pattern, state, backend="rust")
    np.testing.assert_allclose(
        sim.expectation_and_grad(angles, observable)[0],
        ref.expectation(angles, observable),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        sim.expectation_and_grad(angles, observable)[1],
        ref.expectation_and_grad(angles, observable)[1],
        atol=1e-12,
    )
    sim.reset([1, 0, 0, 0])
    ref.reset([1, 0, 0, 0])
    np.testing.assert_allclose(
        sim.expectation(angles, observable),
        ref.expectation(angles, observable),
        atol=1e-12,
    )


def test_isometry_plus_ancillas_and_graph_states():
    graph = mp.GraphState([(0, 1), (1, 2), (3, 4), (1, 3)])
    pattern = mp.MBQCircuit(graph, [0], [4, 2])
    sim = mp.PatternSimulator(pattern, [1, 0], backend="pauli")
    ref = mp.PatternSimulator(pattern, [1, 0], backend="numpy")
    for word in ("XY", "YZ", "ZZ", "II"):
        np.testing.assert_allclose(
            sim.expectation([0.3, 0.5, 0.8], word),
            ref.expectation([0.3, 0.5, 0.8], word),
            atol=1e-12,
        )
    pattern = mp.MBQCircuit(mp.GraphState([(0, 1)]), [], [1, 0])
    sim = mp.PatternSimulator(pattern, backend="pauli")
    assert sim.expectation([], "XZ") == 1
    assert sim.expectation([], "ZZ") == 0


@pytest.mark.parametrize("width", [9, 33, 129, 513, 1024])
def test_wide_expectations_never_construct_output_states(width, monkeypatch):
    pattern = mp.MBQCircuit(mp.GraphState(nx.path_graph(width)), [], list(range(width)))
    sim = mp.PatternSimulator(pattern, backend="pauli", max_terms=1)

    def forbidden(*args, **kwargs):
        raise AssertionError("Dense simulation must not be called.")

    monkeypatch.setattr(mp.RustSimulator, "__init__", forbidden)
    assert sim.expectation([], "XZ" + "I" * (width - 2)) == 1
    assert sim.peak_terms == 1 and sim.input_state is None


def test_adaptive_trajectories_and_sampling_law():
    pattern = mp.templates.linear_cluster(3)
    pattern[0] = mp.Ment("X")
    pattern[1] = mp.ControlMent(~pattern[0].outcome, true_angle=np.pi)
    sim = mp.PatternSimulator(pattern, [1, 0], backend="pauli", force0=False)
    ref = mp.PatternSimulator(pattern, [1, 0], backend="numpy", force0=False)
    for seed in range(12):
        np.testing.assert_allclose(
            sim.expectation([], "Z", seed=seed),
            ref.expectation([], "Z", seed=seed),
            atol=1e-12,
        )
        assert sim.outcomes == ref.outcomes
    assert abs(sim.expectation([], "Z", shots=3000, seed=71)) < 0.06
    assert sim.expectation([], "Z", shots=50, seed=71) == sim.expectation(
        [], "Z", shots=50, seed=71
    )
    with pytest.raises(ValueError, match="force0"):
        sim.expectation_and_grad([], "Z")


def test_inactive_parameters_and_dense_outputs():
    pattern = mp.templates.linear_cluster(4)
    pattern[1] = mp.ControlMent(
        pattern[0].outcome, true_angle=None, false_angle=0.4, false_plane="XY"
    )
    angles = [0.7, 0.6, 0.5]
    sim = mp.PatternSimulator(pattern, backend="pauli")
    ref = mp.PatternSimulator(pattern, backend="rust")
    value, grad = sim.expectation_and_grad(angles, "Z")
    np.testing.assert_allclose(value, ref.expectation(angles, "Z"), atol=1e-12)
    np.testing.assert_allclose(
        grad, ref.expectation_and_grad(angles, "Z")[1], atol=1e-12
    )
    assert grad[1] == 0
    np.testing.assert_allclose(sim.run(angles), ref.run(angles))
    np.testing.assert_allclose(
        sim.probabilities(angles, basis="Y"), ref.probabilities(angles, basis="Y")
    )
    np.testing.assert_array_equal(
        sim.sample(angles, shots=31, seed=51).bitstrings,
        ref.sample(angles, shots=31, seed=51).bitstrings,
    )
    assert abs(sim.expectation(angles, "Z", shots=5000, seed=18) - value) < 0.05
    with pytest.raises(ValueError, match="exact"):
        sim.expectation_and_grad(angles, "Z", shots=10)


def test_limits_and_native_validation():
    pattern = mp.templates.linear_cluster(2)
    sim = mp.PatternSimulator(pattern, backend="pauli", max_terms=1)
    with pytest.raises(MemoryError, match="max_terms"):
        sim.expectation([0.3], "Z")
    assert sim.expectation([0], "Z") == 1
    sim = mp.PatternSimulator(pattern, backend="pauli", max_tape=1)
    with pytest.raises(MemoryError, match="max_tape"):
        sim.expectation_and_grad([0.3], "Y")
    with pytest.raises(ValueError, match="finite"):
        sim.expectation([np.nan], "Y")
    with pytest.raises(ValueError, match="positive"):
        mp.PatternSimulator(pattern, backend="pauli", max_terms=True)
    pattern[0] = mp.Ment("XZ")
    with pytest.raises(ValueError):
        mp.PatternSimulator(pattern, backend="pauli")
    with pytest.raises(ValueError, match="permutation"):
        _rust.PauliCircuit(2, 1, [0, 0], [], 0)
    with pytest.raises(ValueError, match="operation"):
        _rust.PauliCircuit(2, 1, [0, 1], [(1, 0, 0)], 0)
    core = _rust.PauliCircuit(1, 1, [0], [], 0)
    with pytest.raises(ValueError, match="normalized"):
        core.expectation([], [("X", 1)], input_re=[0, 0], input_im=[0, 0])
    with pytest.raises(ValueError, match="coefficients"):
        core.expectation([], [("XY", 1)])


def test_long_chain_gradient_tape_is_iterative():
    pattern = mp.templates.linear_cluster(1001)
    angles = np.full(1000, 0.3)
    sim = mp.PatternSimulator(pattern, backend="pauli")
    ref = mp.PatternSimulator(pattern, backend="rust")
    value, gradient = sim.expectation_and_grad(angles, "Z")
    expected, diff = ref.expectation_and_grad(angles, "Z")
    np.testing.assert_allclose(value, expected, atol=1e-11)
    np.testing.assert_allclose(gradient, diff, atol=1e-11)

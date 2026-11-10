# Copyright 2026 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Native contractions and gradients against independent reference methods."""

import itertools
import os

import networkx as nx
import numpy as np
import pytest

import mentpy as mp
from mentpy import _rust

DEVICE = os.environ.get("MENTPY_TEST_DEVICE", "cpu")


@pytest.mark.parametrize("plane", ["XY", "XZ", "YZ", "X", "Y", "Z"])
def test_planes_and_gradients(plane):
    rng = np.random.default_rng(31)
    pattern = mp.templates.grid_cluster(2, 3)
    for node in pattern.outputc:
        pattern[node] = mp.Ment(plane)
    angles = rng.uniform(-2, 2, len(pattern.trainable_nodes))
    state = rng.normal(size=4) + 1j * rng.normal(size=4)
    state /= np.linalg.norm(state)
    native = mp.PatternSimulator(pattern, state, backend="rust", device=DEVICE)
    reference = mp.PatternSimulator(pattern, state, backend="numpy")
    np.testing.assert_allclose(native.run(angles), reference.run(angles), atol=1e-12)
    for letters in itertools.product("IXYZ", repeat=2):
        observable = mp.Observable("".join(letters)) + 0.17
        value, gradient = native.expectation_and_grad(angles, observable)
        np.testing.assert_allclose(
            value, reference.expectation(angles, observable), atol=1e-12
        )
        if len(angles):
            expected = mp.gradients.get_gradient(
                lambda a: reference.expectation(a, observable),
                angles,
                method="fd",
                h=1e-5,
            )
            np.testing.assert_allclose(gradient, expected, atol=1e-7)
        else:
            assert gradient.shape == (0,)


def test_random_graphs_and_reversed_outputs():
    rng = np.random.default_rng(44)
    for seed in range(12):
        graph = mp.GraphState(nx.gnp_random_graph(7, 0.35, seed=seed))
        pattern = mp.MBQCircuit(graph, [1, 0], [6, 4], initialize_flow=False)
        state = rng.normal(size=4) + 1j * rng.normal(size=4)
        state /= np.linalg.norm(state)
        angles = rng.normal(size=len(pattern.trainable_nodes))
        native = mp.PatternSimulator(pattern, state, backend="rust", device=DEVICE)
        reference = mp.PatternSimulator(pattern, state, backend="numpy")
        np.testing.assert_allclose(
            native.run(angles), reference.run(angles), atol=1e-12
        )
        observable = mp.Observable({"YX": 0.7, "ZZ": -0.4, "IY": 0.8})
        value, derivative = native.expectation_and_grad(angles, observable)
        np.testing.assert_allclose(
            value, reference.expectation(angles, observable), atol=1e-12
        )
        expected = mp.gradients.get_gradient(
            lambda a: reference.expectation(a, observable), angles, method="fd", h=1e-5
        )
        np.testing.assert_allclose(derivative, expected, atol=1e-7)


def test_larger_contractions_and_adjoint():
    graph = mp.GraphState(nx.gnp_random_graph(14, 0.45, seed=49))
    pattern = mp.MBQCircuit(graph, [0, 1], list(range(7, 14)), initialize_flow=False)
    angles = np.linspace(0.2, 1.3, 7)
    native = mp.PatternSimulator(pattern, backend="rust", device=DEVICE)
    reference = mp.PatternSimulator(pattern, backend="numpy")
    observable = mp.Observable({"XYZIYXZ": 0.7, "ZZZZZZZ": -0.4})
    value, derivative = native.expectation_and_grad(angles, observable)
    np.testing.assert_allclose(native.run(angles), reference.run(angles), atol=1e-12)
    np.testing.assert_allclose(
        value, reference.expectation(angles, observable), atol=1e-12
    )
    expected = mp.gradients.get_gradient(
        lambda a: reference.expectation(a, observable), angles, method="fd", h=1e-5
    )
    np.testing.assert_allclose(derivative, expected, atol=1e-7)


@pytest.mark.parametrize("condition", [True, False])
def test_controlled_angles_keep_parameter_order(condition):
    pattern = mp.templates.linear_cluster(5)
    pattern[1] = mp.ControlMent(
        condition, true_angle=None, true_plane="XZ", false_angle=0, false_plane="X"
    )
    angles = np.full(len(pattern.trainable_nodes), 0.3)
    native = mp.PatternSimulator(pattern, backend="rust", device=DEVICE)
    reference = mp.PatternSimulator(pattern, backend="numpy")
    value, derivative = native.expectation_and_grad(angles, "Z")
    np.testing.assert_allclose(value, reference.expectation(angles, "Z"), atol=1e-12)
    expected = mp.gradients.get_gradient(
        lambda a: reference.expectation(a, "Z"), angles, method="fd"
    )
    np.testing.assert_allclose(derivative, expected, atol=1e-6)
    if not condition:
        assert derivative[pattern.trainable_nodes.index(1)] == 0


def test_xyz_pair_gradient():
    pattern = mp.templates.linear_cluster(3)
    with pytest.warns(UserWarning):
        pattern[0] = mp.Ment("XYZ")
        pattern[1] = mp.Ment("XYZ")
    angles = np.array([[0.2, 0.4], [-0.1, 0.3]])
    native = mp.PatternSimulator(pattern, backend="rust", device=DEVICE)
    reference = mp.PatternSimulator(pattern, backend="numpy")
    value, gradient = native.expectation_and_grad(angles, "Y")

    def cost(a):
        with pytest.warns(UserWarning):
            return reference.expectation([tuple(x) for x in a.reshape(2, 2)], "Y")

    np.testing.assert_allclose(value, cost(angles), atol=1e-12)
    expected = mp.gradients.get_gradient(cost, angles.ravel(), method="fd", h=1e-5)
    np.testing.assert_allclose(gradient, expected.reshape(2, 2), atol=1e-7)


def test_clifford_angle_retains_derivative():
    pattern = mp.templates.linear_cluster(2)
    sim = mp.PatternSimulator(pattern, backend="rust", device=DEVICE)
    value, gradient = sim.expectation_and_grad([0.0], "Y")
    np.testing.assert_allclose(value, 0, atol=1e-14)
    np.testing.assert_allclose(gradient, [1], atol=1e-14)


def test_long_chain_and_high_degree_remain_small():
    pattern = mp.templates.linear_cluster(1201)
    sim = mp.PatternSimulator(pattern, backend="rust", device=DEVICE)
    value, gradient = sim.expectation_and_grad(np.zeros(1200), "Y")
    assert sim.peak_elements <= 16
    np.testing.assert_allclose(value, 0, atol=1e-12)
    assert np.all(np.isfinite(gradient))
    assert np.linalg.norm(gradient) > 1
    graph = mp.GraphState(nx.star_graph(150))
    pattern = mp.MBQCircuit(graph, [], [0], initialize_flow=False)
    sim = mp.PatternSimulator(pattern, backend="rust", device=DEVICE)
    assert sim.peak_elements <= 16
    np.testing.assert_allclose(np.trace(sim.run(np.zeros(150))), 1, atol=1e-12)


def test_implicit_inputs_do_not_allocate_dense_vector():
    graph = mp.GraphState(nx.empty_graph(100))
    pattern = mp.MBQCircuit(graph, list(graph), [], initialize_flow=False)
    sim = mp.PatternSimulator(pattern, backend="rust", device=DEVICE)
    assert sim.input_state is None
    assert sim.peak_elements == 2
    np.testing.assert_allclose(sim.run(np.zeros(100)), [[1]])


def test_empty_graph_and_measured_output():
    empty = mp.MBQCircuit(mp.GraphState(), [], [], initialize_flow=False)
    sim = mp.PatternSimulator(empty, backend="rust", device=DEVICE)
    np.testing.assert_allclose(sim.run([]), [[1]])
    value, gradient = sim.expectation_and_grad([], mp.Observable(constant=2))
    assert value == 2 and gradient.size == 0
    pattern = mp.templates.linear_cluster(3)
    pattern[2] = mp.Ment("X")
    sim = mp.PatternSimulator(pattern, backend="rust", device=DEVICE)
    np.testing.assert_allclose(sim.run([0, 0]), [[1]])
    assert sim.outcomes == {0: 0, 1: 0, 2: 0}


def test_state_reset_sampling_and_validation():
    pattern = mp.templates.linear_cluster(3)
    sim = mp.PatternSimulator(pattern, backend="rust", device=DEVICE)
    angles = [0.3, -0.2]
    sim.reset([1, 0])
    obs = mp.Observable({"X": 0.3, "Y": -0.7, "Z": 0.5})
    expected = mp.PatternSimulator(pattern, [1, 0]).expectation(angles, obs)
    np.testing.assert_allclose(sim.expectation_and_grad(angles, obs)[0], expected)
    np.testing.assert_allclose(
        sim.expectation(angles, obs, shots=10000, seed=31), expected, atol=0.03
    )
    np.testing.assert_array_equal(
        sim.sample(angles, shots=50, seed=17).bitstrings,
        sim.sample(angles, shots=50, seed=17).bitstrings,
    )
    with pytest.raises(ValueError, match="Number of angles"):
        sim.run([])
    with pytest.raises(ValueError, match="finite"):
        sim.run([np.nan, 0])
    with pytest.raises(ValueError, match="normalized"):
        sim.reset([0, 0])
    with pytest.raises(ValueError, match="Shot-sampled"):
        sim.expectation_and_grad(angles, obs, shots=1)
    with pytest.raises(ValueError, match="max_elements"):
        mp.PatternSimulator(
            mp.templates.grid_cluster(3, 3),
            backend="rust",
            device=DEVICE,
            max_elements=4,
        )
    with pytest.raises(ValueError, match="device"):
        mp.PatternSimulator(pattern, backend="rust", device="invalid")


def test_impossible_branch_raises_without_nan():
    graph = mp.GraphState(nx.empty_graph(2))
    pattern = mp.MBQCircuit(graph, [0], [1], initialize_flow=False)
    sim = mp.PatternSimulator(
        pattern, [2**-0.5, -(2**-0.5)], backend="rust", device=DEVICE
    )
    for evaluate in [lambda: sim.run([0]), lambda: sim.expectation_and_grad([0], "Z")]:
        with pytest.raises(ValueError, match="zero probability"):
            evaluate()
    np.testing.assert_allclose(sim.run([np.pi]), np.ones((2, 2)) / 2, atol=1e-12)


@pytest.mark.parametrize(
    "inputs, outputs, measured, edges",
    [
        ([0, 0], [1], [0], []),
        ([], [1, 1], [0], []),
        ([], [1], [1], []),
        ([], [1], [0], [(0, 1), (1, 0)]),
        ([], [2], [0], []),
    ],
)
def test_native_boundary_rejects_invalid_graphs(inputs, outputs, measured, edges):
    with pytest.raises(ValueError):
        _rust.TensorNetwork(2, edges, inputs, outputs, measured)


def test_expectation_seed_controls_intermediate_outcomes():
    pattern = mp.templates.linear_cluster(3)
    pattern[0] = mp.Ment("X")
    pattern[1] = mp.ControlMent(~pattern[0].outcome, true_angle=np.pi)
    sim = mp.PatternSimulator(pattern, [1, 0], force0=False, seed=7)
    expected = mp.Observable("Z").expectation(sim.run([], seed=19))
    assert [sim.expectation([], "Z", seed=19) for _ in range(8)] == [expected] * 8

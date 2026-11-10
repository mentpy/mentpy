# Copyright 2026 Luis Mantilla
# Licensed under the Apache License, Version 2.0.
"""Hardware tests for the runtime-loaded cuPauliProp interface."""

import itertools
import os

import networkx as nx
import numpy as np
import pytest

import mentpy as mp

DEVICE = os.environ.get("MENTPY_TEST_CUPAULI")
pytestmark = pytest.mark.skipif(
    DEVICE is None, reason="Requires cuPauliProp and an NVIDIA GPU"
)


@pytest.mark.parametrize("plane", ["XY", "X", "Y"])
@pytest.mark.parametrize("explicit", [False, True])
def test_gpu_values_and_gradients(plane, explicit):
    pattern = mp.templates.grid_cluster(2, 4)
    for node in pattern.outputc:
        pattern[node] = mp.Ment(plane)
    angles = np.linspace(-0.7, 0.9, len(pattern.trainable_nodes))
    state = np.array([1, 1j, 2, -1j]) / np.sqrt(7) if explicit else None
    cpu = mp.PatternSimulator(pattern, state, backend="pauli")
    gpu = mp.PatternSimulator(pattern, state, backend="pauli", device=DEVICE)
    for letters in itertools.product("IXYZ", repeat=2):
        observable = mp.Observable("".join(letters)) + 0.31
        expected, grad = cpu.expectation_and_grad(angles, observable)
        value, derivative = gpu.expectation_and_grad(angles, observable)
        np.testing.assert_allclose(value, expected, atol=1e-12)
        np.testing.assert_allclose(derivative, grad, atol=1e-12)
        np.testing.assert_allclose(
            gpu.expectation(angles, observable), expected, atol=1e-12
        )


@pytest.mark.parametrize("angle", [0, np.pi / 2, np.pi, -np.pi / 2, 2**40 * np.pi])
def test_zero_coefficients_and_large_angles(angle):
    gpu = mp.PatternSimulator(
        mp.templates.linear_cluster(2), backend="pauli", device=DEVICE
    )
    value, gradient = gpu.expectation_and_grad([angle], "Y")
    np.testing.assert_allclose(value, np.sin(angle), atol=1e-12)
    np.testing.assert_allclose(gradient, [np.cos(angle)], atol=1e-12)
    value, gradient = gpu.expectation_and_grad([angle], "Z")
    np.testing.assert_allclose(value, np.cos(angle), atol=1e-12)
    np.testing.assert_allclose(gradient, [-np.sin(angle)], atol=1e-12)


@pytest.mark.parametrize("explicit", [False, True])
def test_joint_observable_and_wire_order(explicit):
    pattern = mp.MBQCircuit(
        mp.GraphState([(0, 1), (1, 2), (3, 4), (1, 3)]), [0], [4, 2]
    )
    obs = mp.Observable({"XY": 0.7, "YZ": -0.6, "ZZ": 0.2, "II": -0.3}, constant=0.8)
    state = [1 / np.sqrt(2), 1j / np.sqrt(2)] if explicit else None
    cpu = mp.PatternSimulator(pattern, state, backend="pauli")
    gpu = mp.PatternSimulator(pattern, state, backend="pauli", device=DEVICE)
    expected, derivative = cpu.expectation_and_grad([0.2, -0.3, 0.7], obs)
    actual, gradient = gpu.expectation_and_grad([0.2, -0.3, 0.7], obs)
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    np.testing.assert_allclose(gradient, derivative, atol=1e-12)
    assert gpu.expectation([0.2, -0.3, 0.7], mp.Observable()) == 0


@pytest.mark.parametrize("width", [65, 129])
def test_wide_stabilizer(width):
    pattern = mp.MBQCircuit(mp.GraphState(nx.path_graph(width)), [], list(range(width)))
    gpu = mp.PatternSimulator(pattern, backend="pauli", device=DEVICE)
    assert gpu.expectation([], "XZ" + "I" * (width - 2)) == 1


def test_adaptive_trajectory_and_shots():
    pattern = mp.templates.linear_cluster(3)
    pattern[1] = mp.ControlMent(~pattern[0].outcome, true_angle=np.pi)
    gpu = mp.PatternSimulator(pattern, backend="pauli", device=DEVICE, force0=False)
    cpu = mp.PatternSimulator(pattern, backend="pauli", force0=False)
    for seed in range(10):
        np.testing.assert_allclose(
            gpu.expectation([0.3], "Y", seed=seed),
            cpu.expectation([0.3], "Y", seed=seed),
            atol=1e-12,
        )
        assert gpu.outcomes == cpu.outcomes
    np.testing.assert_allclose(
        gpu.expectation([0.3], "Y", shots=50, seed=17),
        cpu.expectation([0.3], "Y", shots=50, seed=17),
    )


def test_gpu_limits_and_recovery():
    pattern = mp.templates.linear_cluster(2)
    gpu = mp.PatternSimulator(pattern, backend="pauli", device=DEVICE, max_terms=1)
    with pytest.raises(MemoryError, match="max_terms"):
        gpu.expectation([0.3], "Z")
    assert gpu.expectation([0], "X") == 0
    low_memory = mp.PatternSimulator(
        pattern, backend="pauli", device=DEVICE, max_memory=1
    )
    with pytest.raises(MemoryError, match="max_memory"):
        low_memory.expectation([0.3], "Z")
    assert gpu.expectation([0], "X") == 0


def test_independent_gpu_calls_from_threads():
    from concurrent.futures import ThreadPoolExecutor

    pattern = mp.templates.grid_cluster(3, 4)
    cpu = mp.PatternSimulator(pattern, backend="pauli")
    angles = [np.full(len(pattern.trainable_nodes), a) for a in (0.2, -0.4, 0.6, 0.9)]
    expected = [cpu.expectation_and_grad(a, "XYZ") for a in angles]

    def evaluate(values):
        sim = mp.PatternSimulator(pattern, backend="pauli", device=DEVICE)
        return sim.expectation_and_grad(values, "XYZ")

    with ThreadPoolExecutor(max_workers=2) as pool:
        actual = list(pool.map(evaluate, angles))
    for (value, grad), (reference, derivative) in zip(actual, expected):
        np.testing.assert_allclose(value, reference, atol=1e-12)
        np.testing.assert_allclose(grad, derivative, atol=1e-12)

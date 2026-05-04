"""Tests for VQE helpers."""

import numpy as np
import pytest

import mentpy as mp

jax = pytest.importorskip("jax")


def test_vqe_fit_decreases_observable_energy():
    circuit = mp.templates.linear_cluster(3)
    observable = mp.Observable({"X": 1.0})
    initial_angles = np.array([1.0, 1.0])
    trainer = mp.qml.VQE(circuit, observable, step_size=0.1)

    initial_energy = trainer.energy(initial_angles)
    result = trainer.fit(initial_angles, steps=15)

    assert result.energy < initial_energy
    assert result.angles.shape == initial_angles.shape
    assert len(result.history["energy"]) == 15


def test_train_vqe_convenience_wrapper():
    circuit = mp.templates.linear_cluster(3)
    observable = mp.Observable({"X": 1.0})
    initial_angles = np.array([1.0, 1.0])

    result = mp.qml.train_vqe(
        circuit,
        observable,
        initial_angles,
        steps=3,
        optimizer="sgd",
        step_size=0.1,
    )

    assert np.isfinite(result.energy)
    assert result.angles.shape == initial_angles.shape


def test_vqe_supports_explicit_optimizer_instance():
    circuit = mp.templates.linear_cluster(3)
    observable = mp.Observable({"X": 1.0})
    initial_angles = np.array([1.0, 1.0])
    optimizer = mp.optimizers.SGDOpt(step_size=0.1)

    trainer = mp.qml.VQE(
        circuit,
        observable,
        optimizer=optimizer,
        gradient_method="auto",
    )
    angles, energy, gradient = trainer.step(initial_angles)

    assert np.isfinite(energy)
    assert angles.shape == initial_angles.shape
    assert gradient.shape == initial_angles.shape


def test_vqe_finite_shots_uses_sampled_objective_with_parameter_shift():
    circuit = mp.templates.linear_cluster(3)
    observable = mp.Observable({"X": 1.0})
    initial_angles = np.array([0.2, -0.1])

    trainer = mp.qml.VQE(
        circuit,
        observable,
        shots=200,
        seed=123,
        gradient_method="auto",
        optimizer="sgd",
        step_size=0.02,
    )
    value, gradient = trainer.value_and_grad(initial_angles)

    assert np.isfinite(value)
    assert gradient.shape == initial_angles.shape
    assert np.all(np.isfinite(gradient))

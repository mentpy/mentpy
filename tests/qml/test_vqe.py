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

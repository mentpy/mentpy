# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Tests for the JAX tensor network simulator."""

import pytest
import numpy as np

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

import mentpy as mp


class TestJaxTNCorrectness:
    """Test JAX TN simulator correctness against numpy-sv."""

    @pytest.mark.parametrize("n", [3, 5, 7, 9])
    def test_teleportation(self, n):
        """Test teleportation on linear cluster of size n."""
        gs = mp.templates.linear_cluster(n)
        n_trainable = len(gs.trainable_nodes)
        angles = np.zeros(n_trainable)

        ps_sv = mp.PatternSimulator(gs, backend="numpy-sv")
        dm_sv = ps_sv.run(angles)

        ps_jax = mp.PatternSimulator(gs, backend="jax-tn")
        dm_jax = ps_jax.run(angles)

        assert np.allclose(
            np.array(dm_jax), dm_sv, atol=1e-6
        ), f"Teleportation failed for linear_cluster({n})"

    @pytest.mark.parametrize("n", [3, 5, 7])
    def test_random_angles_linear(self, n):
        """Test random angles on linear cluster."""
        gs = mp.templates.linear_cluster(n)
        n_trainable = len(gs.trainable_nodes)
        rng = np.random.default_rng(42)
        angles = 2 * np.pi * rng.random(n_trainable)

        ps_sv = mp.PatternSimulator(gs, backend="numpy-sv")
        dm_sv = ps_sv.run(angles)

        ps_jax = mp.PatternSimulator(gs, backend="jax-tn")
        dm_jax = ps_jax.run(angles)

        assert np.allclose(
            np.array(dm_jax), dm_sv, atol=1e-6
        ), f"Random angles failed for linear_cluster({n})"

    def test_random_angles_grid(self):
        """Test random angles on grid cluster."""
        gs = mp.templates.grid_cluster(2, 3)
        n_trainable = len(gs.trainable_nodes)
        rng = np.random.default_rng(123)
        angles = 2 * np.pi * rng.random(n_trainable)

        ps_sv = mp.PatternSimulator(
            gs, backend="numpy-sv", window_size=min(5, n_trainable)
        )
        dm_sv = ps_sv.run(angles)

        ps_jax = mp.PatternSimulator(gs, backend="jax-tn")
        dm_jax = ps_jax.run(angles)

        assert np.allclose(
            np.array(dm_jax), dm_sv, atol=1e-6
        ), "Random angles failed for grid_cluster(2,3)"

    def test_random_angles_muta(self):
        """Test random angles on muta ansatz."""
        gs = mp.templates.muta(2, 1)
        n_trainable = len(gs.trainable_nodes)
        rng = np.random.default_rng(456)
        angles = 2 * np.pi * rng.random(n_trainable)

        ps_sv = mp.PatternSimulator(
            gs, backend="numpy-sv", window_size=min(5, n_trainable)
        )
        dm_sv = ps_sv.run(angles)

        ps_jax = mp.PatternSimulator(gs, backend="jax-tn")
        dm_jax = ps_jax.run(angles)

        assert np.allclose(
            np.array(dm_jax), dm_sv, atol=1e-6
        ), "Random angles failed for muta(2,1)"

    def test_statevector_output(self):
        """Test state vector output mode."""
        gs = mp.templates.linear_cluster(5)
        n_trainable = len(gs.trainable_nodes)
        angles = np.zeros(n_trainable)

        ps_sv = mp.PatternSimulator(gs, backend="numpy-sv")
        sv_np = ps_sv.run(angles, output_form="sv")

        ps_jax = mp.PatternSimulator(gs, backend="jax-tn")
        sv_jax = ps_jax.run(angles, output_form="sv")

        # State vectors may differ by global phase
        dm_np = np.outer(sv_np, np.conj(sv_np))
        dm_jax = np.array(jnp.outer(sv_jax, jnp.conj(sv_jax)))
        assert np.allclose(dm_jax, dm_np, atol=1e-6)

    def test_fixed_angle_nodes(self):
        """Test circuits with fixed-angle measurement nodes."""
        gs = mp.templates.linear_cluster(5)
        # Fix some nodes to X measurement
        non_output = gs.outputc
        if len(non_output) >= 2:
            gs[non_output[0]] = mp.Ment("X")

        n_trainable = len(gs.trainable_nodes)
        rng = np.random.default_rng(789)
        angles = 2 * np.pi * rng.random(n_trainable)

        ps_sv = mp.PatternSimulator(gs, backend="numpy-sv")
        dm_sv = ps_sv.run(angles)

        ps_jax = mp.PatternSimulator(gs, backend="jax-tn")
        dm_jax = ps_jax.run(angles)

        assert np.allclose(np.array(dm_jax), dm_sv, atol=1e-6)

    def test_fixed_y_measurement_nodes(self):
        """Test fixed Y-plane measurements are mapped to XY angle pi/2."""
        gs = mp.templates.linear_cluster(5)
        non_output = gs.outputc
        gs[non_output[0]] = mp.Ment("Y")

        n_trainable = len(gs.trainable_nodes)
        rng = np.random.default_rng(321)
        angles = 2 * np.pi * rng.random(n_trainable)

        ps_sv = mp.PatternSimulator(gs, backend="numpy-sv")
        dm_sv = ps_sv.run(angles)

        ps_jax = mp.PatternSimulator(gs, backend="jax-tn")
        dm_jax = ps_jax.run(angles)

        assert np.allclose(np.array(dm_jax), dm_sv, atol=1e-6)

    def test_custom_input_state(self):
        """Test arbitrary input states match numpy-sv teleportation."""
        gs = mp.templates.linear_cluster(7)
        rng = np.random.default_rng(654)
        input_state = rng.normal(size=2) + 1j * rng.normal(size=2)
        input_state = input_state / np.linalg.norm(input_state)
        angles = np.zeros(len(gs.trainable_nodes))

        ps_sv = mp.PatternSimulator(gs, input_state=input_state, backend="numpy-sv")
        dm_sv = ps_sv.run(angles)

        ps_jax = mp.PatternSimulator(gs, input_state=input_state, backend="jax-tn")
        dm_jax = ps_jax.run(angles)

        assert np.allclose(np.array(dm_jax), dm_sv, atol=1e-6)

    @pytest.mark.parametrize("condition", [True, False])
    def test_controlled_measurement_branches(self, condition):
        """Test deterministic controlled measurements in force0 mode."""
        gs = mp.templates.linear_cluster(5)
        gs[1] = mp.ControlMent(
            condition,
            true_angle=None,
            true_plane="XY",
            false_angle=0,
            false_plane="X",
        )
        rng = np.random.default_rng(987)
        angles = 2 * np.pi * rng.random(len(gs.trainable_nodes))

        ps_sv = mp.PatternSimulator(gs, backend="numpy-sv")
        dm_sv = ps_sv.run(angles)

        ps_jax = mp.PatternSimulator(gs, backend="jax-tn")
        dm_jax = ps_jax.run(angles)

        assert np.allclose(np.array(dm_jax), dm_sv, atol=1e-6)


class TestJaxTNScalability:
    """Test that JAX TN simulator can handle large circuits."""

    def test_large_linear_cluster(self):
        """Test linear_cluster(100) runs successfully."""
        gs = mp.templates.linear_cluster(101)
        n_trainable = len(gs.trainable_nodes)
        angles = jnp.zeros(n_trainable)

        ps = mp.PatternSimulator(gs, backend="jax-tn")
        dm = ps.run(angles)

        # Check it's a valid density matrix
        dm_np = np.array(dm)
        assert dm_np.shape == (2, 2)
        assert np.allclose(np.trace(dm_np), 1.0, atol=1e-6)
        # Should be a pure state (trace of rho^2 = 1)
        assert np.real(np.trace(dm_np @ dm_np)) > 0.99


class TestJaxFeatures:
    """Test JAX-specific features (JIT, autodiff)."""

    def test_jax_grad(self):
        """Test that jax.grad works through the simulation."""
        gs = mp.templates.linear_cluster(5)
        n_trainable = len(gs.trainable_nodes)

        sim = mp.simulators.JaxTNSimulator(gs)

        def cost(angles):
            dm = sim._forward(angles, "dm")
            return jnp.real(jnp.trace(dm))

        angles = jnp.zeros(n_trainable)
        grad = jax.grad(cost)(angles)

        assert grad.shape == (n_trainable,)
        assert jnp.all(jnp.isfinite(grad))

    def test_run_and_grad(self):
        """Test the run_and_grad convenience method."""
        gs = mp.templates.linear_cluster(5)
        n_trainable = len(gs.trainable_nodes)

        sim = mp.simulators.JaxTNSimulator(gs)
        angles = jnp.ones(n_trainable) * 0.1

        def cost_fn(dm):
            return 1.0 - jnp.real(jnp.trace(dm))

        cost_val, grad = sim.run_and_grad(angles, cost_fn)
        assert jnp.isfinite(cost_val)
        assert grad.shape == (n_trainable,)
        assert jnp.all(jnp.isfinite(grad))

    def test_expectation_and_grad_observable(self):
        """Test observable convenience methods for VQE-style costs."""
        gs = mp.templates.linear_cluster(5)
        n_trainable = len(gs.trainable_nodes)

        sim = mp.simulators.JaxTNSimulator(gs)
        observable = mp.Observable({"Z": 1.0})
        angles = jnp.ones(n_trainable) * 0.2

        value, grad = sim.expectation_and_grad(angles, observable)

        assert jnp.isfinite(value)
        assert grad.shape == (n_trainable,)
        assert jnp.all(jnp.isfinite(grad))

    def test_shot_sampled_expectation(self):
        """Test finite-shot observable estimates through the TN backend."""
        gs = mp.templates.linear_cluster(5)
        sim = mp.simulators.JaxTNSimulator(gs)
        observable = mp.Observable({"X": 1.0})
        angles = jnp.zeros(len(gs.trainable_nodes))

        exact = sim.expectation(angles, observable)
        sampled = sim.expectation(angles, observable, shots=100, seed=7)

        assert np.allclose(exact, 1.0)
        assert np.allclose(sampled, 1.0)

    def test_gradient_method_jax(self):
        """Test get_gradient with method='jax'."""
        gs = mp.templates.linear_cluster(5)
        n_trainable = len(gs.trainable_nodes)

        sim = mp.simulators.JaxTNSimulator(gs)

        def cost(angles):
            dm = sim._forward(angles, "dm")
            return jnp.real(jnp.trace(dm))

        angles = jnp.zeros(n_trainable)
        grad = mp.gradients.get_gradient(cost, angles, method="jax")

        assert grad.shape == (n_trainable,)
        assert jnp.all(jnp.isfinite(grad))

    def test_gradient_vs_finite_difference(self):
        """Compare JAX autodiff gradient vs central finite differences."""
        gs = mp.templates.linear_cluster(5)
        n_trainable = len(gs.trainable_nodes)

        sim_jax = mp.simulators.JaxTNSimulator(gs)

        # Target: identity channel (teleportation with zero angles)
        target_dm = np.array([[0.5, 0.5], [0.5, 0.5]])  # |+><+|

        def cost_jax(angles):
            dm = sim_jax._forward(angles, "dm")
            return 1.0 - jnp.real(jnp.trace(dm @ jnp.array(target_dm)))

        angles = np.array([0.1] * n_trainable)
        grad_jax = np.array(jax.grad(cost_jax)(jnp.array(angles)))

        # Finite-difference gradient using numpy-sv
        ps_sv = mp.PatternSimulator(gs, backend="numpy-sv")

        def cost_fd(angles_np):
            ps_sv.reset()
            dm = ps_sv.run(angles_np)
            return 1.0 - np.real(np.trace(dm @ target_dm))

        grad_fd = mp.gradients.get_gradient(cost_fd, angles, method="fd", h=1e-6)

        assert np.allclose(
            grad_jax, grad_fd, atol=1e-5
        ), f"Gradient mismatch: jax={grad_jax}, finite-difference={grad_fd}"

    def test_reset_and_rerun(self):
        """Test that reset allows re-running the simulation."""
        gs = mp.templates.linear_cluster(5)
        n_trainable = len(gs.trainable_nodes)

        ps = mp.PatternSimulator(gs, backend="jax-tn")
        angles = jnp.zeros(n_trainable)

        dm1 = ps.run(angles)
        dm2 = ps.run(angles)

        assert np.allclose(np.array(dm1), np.array(dm2), atol=1e-10)

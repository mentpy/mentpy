"""Tests for coefficient-bearing observables."""

import numpy as np
import pytest

import mentpy as mp


def test_observable_statevector_expectation():
    obs = mp.Observable({"Z": 1.0, "X": 0.5}, constant=0.25)
    state = np.array([1.0, 0.0], dtype=np.complex128)

    assert np.allclose(obs.expectation(state), 1.25)


def test_observable_density_matrix_expectation():
    obs = mp.Observable({"XX": 1.0, "ZI": -0.5})
    plus_plus = np.ones(4, dtype=np.complex128) / 2
    density_matrix = np.outer(plus_plus, plus_plus.conj())

    assert np.allclose(obs(density_matrix), 1.0)


def test_observable_complex_expectation_can_keep_imaginary_part():
    obs = mp.Observable({"Z": 1j})
    state = np.array([1.0, 0.0], dtype=np.complex128)

    assert np.allclose(obs.expectation(state, real=False), 1j)
    assert np.allclose(obs.expectation(state), 0.0)


def test_observable_from_pauliop():
    obs = mp.Observable.from_pauliop(mp.PauliOp("ZI;IZ"), coeffs=[1.0, -1.0])

    assert obs.n_qubits == 2
    assert obs.terms == {"ZI": 1.0, "IZ": -1.0}


def test_observable_parses_iterable_pairs_and_combines_terms():
    obs = mp.Observable([("z i", 1.0), (2.5, "ZI"), ("XX", 0.0)])

    assert obs.n_qubits == 2
    assert obs.terms == {"ZI": 3.5}
    assert repr(obs) == "3.5 * ZI"


def test_observable_identity_arithmetic_copy_and_matrix():
    obs = mp.Observable({"ZI": 1.0})
    identity = mp.Observable.identity(2, coeff=0.5)

    combined = 2 * (obs + identity) - obs + 0.25
    shifted = 2.0 - obs
    copied = combined.copy()
    copied.add_term("ZI", -1.0)

    assert combined.constant == 1.25
    assert combined.terms == {"ZI": 1.0}
    assert shifted.constant == 2.0
    assert shifted.terms == {"ZI": -1.0}
    assert copied.terms == {}
    assert combined.terms == {"ZI": 1.0}
    assert np.allclose(combined.matrix(), 1.25 * np.eye(4) + obs.matrix("ZI"))


def test_observable_rejects_mismatched_terms():
    with pytest.raises(ValueError, match="Expected Pauli string"):
        mp.Observable({"Z": 1.0, "XX": 1.0})


def test_observable_rejects_invalid_terms_and_states():
    with pytest.raises(ValueError, match="length-2 pairs"):
        mp.Observable([("Z", 1.0, 2.0)])

    with pytest.raises(TypeError, match="Pauli terms must be strings"):
        mp.Observable({("Z",): 1.0})

    with pytest.raises(ValueError, match="Invalid Pauli"):
        mp.Observable({"ZA": 1.0})

    with pytest.raises(ValueError, match="empty"):
        mp.Observable({"": 1.0})

    with pytest.raises(ValueError, match="power of two"):
        mp.Observable({"Z": 1.0}).expectation(np.ones(3))

    with pytest.raises(ValueError, match="square density matrix"):
        mp.Observable({"Z": 1.0}).expectation(np.ones((2, 3)))

    with pytest.raises(ValueError, match="state has 1"):
        mp.Observable({"ZZ": 1.0}).expectation(np.array([1.0, 0.0]))


def test_observable_jax_autodiff_with_tn_simulator():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")

    gs = mp.templates.linear_cluster(5)
    sim = mp.simulators.JaxTNSimulator(gs)
    obs = mp.Observable({"Z": 1.0})

    def cost(angles):
        state = sim._forward(angles, "sv")
        return obs.expectation(state)

    angles = jnp.zeros(len(gs.trainable_nodes))
    grad = jax.grad(cost)(angles)

    assert grad.shape == angles.shape
    assert jnp.all(jnp.isfinite(grad))

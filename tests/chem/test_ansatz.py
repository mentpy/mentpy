"""Tests for chemistry ansatz helpers."""

from types import SimpleNamespace

import numpy as np
import pytest

import mentpy as mp

openfermion = pytest.importorskip("openfermion")


def test_uccsd_pauli_terms_from_openfermion():
    terms, coefficients, parameter_size = mp.chem.uccsd_pauli_terms(4, 2)

    assert parameter_size == 2
    assert len(terms) == len(coefficients)
    assert len(terms) > 0
    assert all(len(term) == 4 for term in terms)


def test_uccsd_builds_mbqcircuit():
    ansatz = mp.chem.uccsd(4, 2)

    assert len(ansatz.pauli_terms) == len(ansatz.coefficients)
    assert len(ansatz.circuit.input_nodes) == 4
    assert len(ansatz.circuit.output_nodes) == 4
    assert len(ansatz.circuit.trainable_nodes) == len(ansatz.pauli_terms)
    assert ansatz.initial_angles().shape == (len(ansatz.pauli_terms),)
    assert np.allclose(ansatz.initial_angles(), 0.0)


def test_uccsd_infers_active_system_from_hamiltonian_and_molecule():
    hamiltonian = mp.Observable.identity(4)
    molecule = SimpleNamespace(n_electrons=4)
    hamiltonian.chemistry = mp.chem.MolecularHamiltonianMetadata(
        molecule=molecule,
        mapping="jordan_wigner",
        occupied_indices=(0,),
        active_indices=(1, 2),
        n_electrons=2,
    )

    ansatz = mp.chem.uccsd(hamiltonian)

    assert ansatz.n_qubits == 4
    assert ansatz.n_electrons == 2
    assert ansatz.molecule is molecule
    assert ansatz.occupied_indices == (0,)
    assert ansatz.active_indices == (1, 2)


def test_uccsd_requires_electron_count_without_molecule():
    with pytest.raises(ValueError, match="n_electrons"):
        mp.chem.uccsd(mp.Observable.identity(4))

"""Tests for chemistry helper surfaces."""

import pytest

import mentpy as mp
from mentpy.chem import hamiltonians


def test_lih_geometry():
    geometry = mp.chem.lih_geometry(1.6)

    assert geometry == [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, 1.6))]


def test_lih_geometry_axis():
    geometry = mp.chem.lih_geometry(1.2, axis="x")

    assert geometry == [("Li", (0.0, 0.0, 0.0)), ("H", (1.2, 0.0, 0.0))]


def test_lih_geometry_rejects_bad_axis():
    with pytest.raises(ValueError, match="axis"):
        mp.chem.lih_geometry(1.0, axis="bad")


def test_chem_namespace_has_no_packaged_pes_workflow():
    assert not hasattr(mp.chem, "lih" + "_pes")


def test_mp2_energy_reads_populated_molecule():
    class Molecule:
        mp2_energy = -7.8

    assert mp.chem.mp2_energy(Molecule()) == -7.8


def test_mp2_energy_requires_value():
    class Molecule:
        mp2_energy = None

    with pytest.raises(ValueError, match="MP2"):
        mp.chem.mp2_energy(Molecule())


def test_molecular_hamiltonian_omits_none_description(monkeypatch):
    captured = {}

    class FakeMolecularHamiltonian:
        n_qubits = 2

    class FakeMolecule:
        n_qubits = 2
        n_electrons = 4

        def __init__(self, **kwargs):
            if kwargs.get("description") is None and "description" in kwargs:
                raise TypeError("description must be a string.")
            captured["molecule_kwargs"] = kwargs

        def get_molecular_hamiltonian(self, **kwargs):
            captured["active_space_kwargs"] = kwargs
            return FakeMolecularHamiltonian()

    class FakeOpenFermion:
        MolecularData = FakeMolecule

        @staticmethod
        def jordan_wigner(_operator):
            return object()

    def fake_run_pyscf(molecule, **kwargs):
        captured["pyscf_kwargs"] = kwargs
        return molecule

    monkeypatch.setattr(
        hamiltonians,
        "_require_chem",
        lambda: (FakeOpenFermion, fake_run_pyscf),
    )
    monkeypatch.setattr(
        hamiltonians,
        "from_openfermion",
        lambda _operator, n_qubits: mp.Observable.identity(n_qubits),
    )

    observable, _molecule = mp.chem.molecular_hamiltonian(
        mp.chem.lih_geometry(1.6),
        occupied_indices=[0],
        active_indices=[1],
    )

    assert "description" not in captured["molecule_kwargs"]
    assert captured["active_space_kwargs"] == {
        "occupied_indices": [0],
        "active_indices": [1],
    }
    assert observable.n_qubits == 2
    assert observable.chemistry.occupied_indices == (0,)
    assert observable.chemistry.active_indices == (1,)
    assert observable.chemistry.n_electrons == 2


def test_active_electrons_subtracts_frozen_spatial_orbitals():
    class Molecule:
        n_electrons = 4

    assert mp.chem.active_electrons(Molecule(), occupied_indices=[0]) == 2


def test_hartree_fock_state_uses_first_spin_orbitals():
    state = mp.chem.hartree_fock_state(4, 2)

    assert state.shape == (16,)
    assert state[0b1100] == 1.0
    assert state.sum() == 1.0


def test_hartree_fock_state_validates_electron_count():
    with pytest.raises(ValueError, match="n_electrons"):
        mp.chem.hartree_fock_state(2, 3)

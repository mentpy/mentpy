# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Molecular Hamiltonian helpers."""

from dataclasses import dataclass

import numpy as np

from mentpy.utils import from_openfermion

__all__ = [
    "active_electrons",
    "hartree_fock_state",
    "lih_geometry",
    "MolecularHamiltonianMetadata",
    "molecular_hamiltonian",
    "mp2_energy",
]


@dataclass(frozen=True)
class MolecularHamiltonianMetadata:
    """Active-space metadata attached to chemistry observables.

    Group
    -----
    chemistry
    """

    molecule: object
    mapping: str
    occupied_indices: tuple
    active_indices: object
    n_electrons: int


def lih_geometry(bond_length, axis="z"):
    """Return a linear LiH geometry for a bond length in Angstrom.

    Group
    -----
    chemistry
    """
    axes = {
        "x": np.array([1.0, 0.0, 0.0]),
        "y": np.array([0.0, 1.0, 0.0]),
        "z": np.array([0.0, 0.0, 1.0]),
    }
    if axis not in axes:
        raise ValueError("axis must be 'x', 'y', or 'z'.")
    h_position = tuple((bond_length * axes[axis]).tolist())
    return [("Li", (0.0, 0.0, 0.0)), ("H", h_position)]


def molecular_hamiltonian(
    geometry,
    basis="sto-3g",
    multiplicity=1,
    charge=0,
    mapping="jordan_wigner",
    occupied_indices=None,
    active_indices=None,
    run_scf=True,
    run_mp2=False,
    run_cisd=False,
    run_ccsd=False,
    run_fci=False,
    description=None,
    **kwargs,
):
    """Build a qubit Hamiltonian using OpenFermion-PySCF.

    Returns
    -------
    observable : mentpy.Observable
        Qubit Hamiltonian converted to MentPy's observable format. The
        observable carries a ``chemistry`` metadata attribute with the active
        space used to build it.
    molecule : openfermion.MolecularData
        The populated OpenFermion molecule object, including requested PySCF
        energies such as MP2 when ``run_mp2=True``.

    Group
    -----
    chemistry
    """
    openfermion, run_pyscf = _require_chem()

    molecule_kwargs = {
        "geometry": geometry,
        "basis": basis,
        "multiplicity": multiplicity,
        "charge": charge,
    }
    if description is not None:
        molecule_kwargs["description"] = description
    molecule = openfermion.MolecularData(**molecule_kwargs)
    molecule = run_pyscf(
        molecule,
        run_scf=run_scf,
        run_mp2=run_mp2,
        run_cisd=run_cisd,
        run_ccsd=run_ccsd,
        run_fci=run_fci,
        **kwargs,
    )

    fermion_hamiltonian = molecule.get_molecular_hamiltonian(
        occupied_indices=occupied_indices,
        active_indices=active_indices,
    )
    mapper = _mapping_function(openfermion, mapping)
    qubit_operator = mapper(fermion_hamiltonian)
    n_qubits = getattr(fermion_hamiltonian, "n_qubits", molecule.n_qubits)
    observable = from_openfermion(qubit_operator, n_qubits=n_qubits)
    observable.chemistry = MolecularHamiltonianMetadata(
        molecule=molecule,
        mapping=mapping,
        occupied_indices=tuple([] if occupied_indices is None else occupied_indices),
        active_indices=None if active_indices is None else tuple(active_indices),
        n_electrons=active_electrons(molecule, occupied_indices),
    )
    return observable, molecule


def mp2_energy(molecule):
    """Return the MP2 energy stored on an OpenFermion molecule.

    Group
    -----
    chemistry
    """
    energy = getattr(molecule, "mp2_energy", None)
    if energy is None:
        raise ValueError("MP2 energy is not available; call with run_mp2=True first.")
    return energy


def active_electrons(molecule, occupied_indices=None):
    """Return the active electron count after freezing occupied orbitals.

    ``occupied_indices`` are spatial orbitals, so each frozen occupied orbital
    removes two electrons for closed-shell molecules.

    Group
    -----
    chemistry
    """
    occupied_indices = [] if occupied_indices is None else list(occupied_indices)
    return molecule.n_electrons - 2 * len(occupied_indices)


def hartree_fock_state(n_qubits, n_electrons):
    """Return a computational-basis Hartree-Fock statevector.

    The first ``n_electrons`` spin orbitals are occupied, matching the standard
    OpenFermion Jordan-Wigner ordering used by the chemistry helpers.

    Group
    -----
    chemistry
    """
    if n_electrons < 0 or n_electrons > n_qubits:
        raise ValueError("n_electrons must be between 0 and n_qubits.")
    bitstring = "1" * n_electrons + "0" * (n_qubits - n_electrons)
    index = int(bitstring, 2) if bitstring else 0
    state = np.zeros(2**n_qubits, dtype=np.complex128)
    state[index] = 1.0
    return state


def _mapping_function(openfermion, mapping):
    mapping = mapping.replace("-", "_").lower()
    if mapping in ("jw", "jordan_wigner"):
        return openfermion.jordan_wigner
    if mapping in ("bk", "bravyi_kitaev"):
        return openfermion.bravyi_kitaev
    raise ValueError("mapping must be 'jordan_wigner' or 'bravyi_kitaev'.")


def _require_chem():
    try:
        import openfermion
        from openfermionpyscf import run_pyscf
    except ImportError as exc:
        raise ImportError(
            "Chemistry helpers require the 'chem' extra: " "pip install 'mentpy[chem]'."
        ) from exc
    return openfermion, run_pyscf

# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Chemistry ansatz helpers."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np
import networkx as nx

from mentpy.mbqc import MBQCircuit
from mentpy.mbqc.templates import from_pauli
from mentpy.operators import PauliOp

__all__ = [
    "UCCSDAnsatz",
    "uccsd",
    "uccsd_pauli_terms",
    "uccsd_generator",
    "uccsd_singlet_generator",
    "uccsd_singlet_paramsize",
]


@dataclass
class UCCSDAnsatz:
    """Trotterized MBQC UCCSD ansatz metadata.

    Group
    -----
    chemistry
    """

    circuit: object
    pauli_terms: list
    coefficients: np.ndarray
    n_qubits: int
    n_electrons: int
    parameter_size: int
    mapping: str = "jordan_wigner"
    molecule: object = None
    occupied_indices: tuple = ()
    active_indices: object = None

    def initial_angles(self, scale=0.0):
        """Return initial MBQC rotation angles for the Pauli-term circuit."""
        return scale * np.real_if_close(1j * self.coefficients).astype(float)


def uccsd(
    system=None,
    n_electrons=None,
    amplitudes=None,
    mapping="jordan_wigner",
    spin_adapted=True,
    coefficient_tol=1e-12,
    *,
    hamiltonian=None,
    n_qubits=None,
    molecule=None,
    occupied_indices=None,
    active_indices=None,
):
    """Build a first-order MBQC UCCSD Pauli-rotation ansatz.

    The returned :class:`UCCSDAnsatz` contains a stacked MBQC circuit with one
    trainable MBQC rotation per mapped Pauli term. This is a practical
    first-order product formula surface for VQE training.

    ``system`` may be either the number of active qubits or a molecular
    Hamiltonian/observable with an ``n_qubits`` attribute. Hamiltonians returned
    by :func:`mentpy.chem.molecular_hamiltonian` carry enough active-space
    metadata for the electron count to be inferred.

    Group
    -----
    chemistry
    """
    n_qubits, n_electrons, molecule, occupied_indices, active_indices = (
        _resolve_uccsd_system(
            system=system,
            hamiltonian=hamiltonian,
            n_qubits=n_qubits,
            n_electrons=n_electrons,
            molecule=molecule,
            occupied_indices=occupied_indices,
            active_indices=active_indices,
        )
    )
    terms, coefficients, parameter_size = uccsd_pauli_terms(
        n_qubits,
        n_electrons,
        amplitudes=amplitudes,
        mapping=mapping,
        spin_adapted=spin_adapted,
        coefficient_tol=coefficient_tol,
    )
    if not terms:
        raise ValueError("UCCSD generator produced no Pauli terms.")

    circuits = [from_pauli(PauliOp(term)) for term in terms]
    circuit = _stack_rotation_circuits(circuits)
    return UCCSDAnsatz(
        circuit=circuit,
        pauli_terms=terms,
        coefficients=np.asarray(coefficients),
        n_qubits=n_qubits,
        n_electrons=n_electrons,
        parameter_size=parameter_size,
        mapping=mapping,
        molecule=molecule,
        occupied_indices=tuple([] if occupied_indices is None else occupied_indices),
        active_indices=None if active_indices is None else tuple(active_indices),
    )


def _resolve_uccsd_system(
    system,
    hamiltonian,
    n_qubits,
    n_electrons,
    molecule,
    occupied_indices,
    active_indices,
):
    if hamiltonian is not None:
        if system is not None:
            raise ValueError("Pass either system or hamiltonian, not both.")
        system = hamiltonian

    chemistry = getattr(system, "chemistry", None)
    if molecule is None and chemistry is not None:
        molecule = getattr(chemistry, "molecule", None)
    if occupied_indices is None and chemistry is not None:
        occupied_indices = getattr(chemistry, "occupied_indices", None)
    if active_indices is None and chemistry is not None:
        active_indices = getattr(chemistry, "active_indices", None)

    if n_qubits is not None:
        if system is not None:
            raise ValueError("Pass either n_qubits or a Hamiltonian system, not both.")
        n_qubits = _validate_n_qubits(n_qubits)
    elif isinstance(system, Integral):
        n_qubits = _validate_n_qubits(system)
    elif system is not None:
        n_qubits = getattr(system, "n_qubits", None)
        if n_qubits is None:
            raise ValueError("Hamiltonian system must expose a finite n_qubits value.")
        n_qubits = _validate_n_qubits(n_qubits)
    else:
        raise ValueError("uccsd requires n_qubits or a Hamiltonian system.")

    if n_electrons is None:
        if (
            chemistry is not None
            and getattr(chemistry, "n_electrons", None) is not None
        ):
            n_electrons = chemistry.n_electrons
        elif molecule is None or not hasattr(molecule, "n_electrons"):
            raise ValueError(
                "n_electrons is required unless a molecule with n_electrons is supplied."
            )
        else:
            occupied_indices = (
                [] if occupied_indices is None else list(occupied_indices)
            )
            n_electrons = molecule.n_electrons - 2 * len(occupied_indices)

    if occupied_indices is None:
        occupied_indices = ()
    if active_indices is not None:
        active_indices = tuple(active_indices)

    n_electrons = int(n_electrons)
    if n_electrons < 0 or n_electrons > n_qubits:
        raise ValueError("n_electrons must be between 0 and n_qubits.")
    return n_qubits, n_electrons, molecule, tuple(occupied_indices), active_indices


def _validate_n_qubits(n_qubits):
    if not isinstance(n_qubits, Integral):
        raise TypeError("n_qubits must be an integer.")
    n_qubits = int(n_qubits)
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive.")
    return n_qubits


def uccsd_pauli_terms(
    n_qubits,
    n_electrons,
    amplitudes=None,
    mapping="jordan_wigner",
    spin_adapted=True,
    coefficient_tol=1e-12,
):
    """Return Pauli terms generated by OpenFermion's UCCSD generator.

    Group
    -----
    chemistry
    """
    openfermion = _require_openfermion()
    circuits = _require_openfermion_circuits()

    if spin_adapted:
        parameter_size = circuits.uccsd_singlet_paramsize(n_qubits, n_electrons)
        if amplitudes is None:
            amplitudes = np.ones(parameter_size)
        fermion_generator = circuits.uccsd_singlet_generator(
            amplitudes, n_qubits, n_electrons
        )
    else:
        if amplitudes is None:
            raise ValueError("amplitudes are required for non-spin-adapted UCCSD.")
        parameter_size = len(amplitudes)
        fermion_generator = circuits.uccsd_generator(amplitudes, n_qubits)

    qubit_generator = _mapping_function(openfermion, mapping)(fermion_generator)
    qubit_generator.compress(abs_tol=coefficient_tol)

    terms = []
    coefficients = []
    for term, coefficient in qubit_generator.terms.items():
        if term == ():
            continue
        if abs(coefficient) <= coefficient_tol:
            continue
        terms.append(_term_to_pauli_string(term, n_qubits))
        coefficients.append(coefficient)

    return terms, coefficients, parameter_size


def uccsd_generator(*args, **kwargs):
    """Return OpenFermion's general UCCSD generator.

    Group
    -----
    chemistry
    """
    circuits = _require_openfermion_circuits()
    return circuits.uccsd_generator(*args, **kwargs)


def uccsd_singlet_generator(*args, **kwargs):
    """Return OpenFermion's spin-adapted singlet UCCSD generator.

    Group
    -----
    chemistry
    """
    circuits = _require_openfermion_circuits()
    return circuits.uccsd_singlet_generator(*args, **kwargs)


def uccsd_singlet_paramsize(*args, **kwargs):
    """Return OpenFermion's singlet UCCSD parameter count.

    Group
    -----
    chemistry
    """
    circuits = _require_openfermion_circuits()
    return circuits.uccsd_singlet_paramsize(*args, **kwargs)


def _require_openfermion_circuits():
    try:
        import openfermion.circuits as circuits
    except ImportError as exc:
        raise ImportError(
            "Chemistry ansatz helpers require the 'chem' extra: "
            "pip install 'mentpy[chem]'."
        ) from exc
    return circuits


def _require_openfermion():
    try:
        import openfermion
    except ImportError as exc:
        raise ImportError(
            "Chemistry ansatz helpers require OpenFermion. Install it with "
            "pip install 'mentpy[chem]' or 'mentpy[interop]'."
        ) from exc
    return openfermion


def _mapping_function(openfermion, mapping):
    mapping = mapping.replace("-", "_").lower()
    if mapping in ("jw", "jordan_wigner"):
        return openfermion.jordan_wigner
    if mapping in ("bk", "bravyi_kitaev"):
        return openfermion.bravyi_kitaev
    raise ValueError("mapping must be 'jordan_wigner' or 'bravyi_kitaev'.")


def _term_to_pauli_string(term, n_qubits):
    pauli = ["I"] * n_qubits
    for wire, op in term:
        pauli[wire] = op
    return "".join(pauli)


def _stack_rotation_circuits(circuits):
    """Stack equal-width MBQC circuits while initializing flow only once."""
    if len(circuits) == 1:
        return circuits[0]

    graph = circuits[0].graph.copy()
    input_nodes = list(circuits[0].input_nodes)
    output_nodes = list(circuits[0].output_nodes)
    measurements = dict(circuits[0].measurements)

    for circuit in circuits[1:]:
        if len(output_nodes) != len(circuit.input_nodes):
            raise ValueError("All UCCSD rotation circuits must have the same width.")

        offset = max(graph.nodes) + 1 if len(graph.nodes) else 0
        mapping = {node: node + offset for node in circuit.graph.nodes}
        next_graph = nx.relabel_nodes(circuit.graph, mapping, copy=True)
        graph = nx.compose(graph, next_graph)
        measurements.update(
            {mapping[node]: ment for node, ment in circuit.measurements.items()}
        )

        next_inputs = [mapping[node] for node in circuit.input_nodes]
        next_outputs = [mapping[node] for node in circuit.output_nodes]
        for old_output, next_input in zip(output_nodes, next_inputs):
            graph.add_edge(old_output, next_input)
            graph = nx.contracted_edge(
                graph, (next_input, old_output), self_loops=False
            )
            measurements.pop(old_output, None)

        output_nodes = next_outputs

    return MBQCircuit(
        graph,
        input_nodes=input_nodes,
        output_nodes=output_nodes,
        measurements=measurements,
    )

# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""A module for calculating the Lie algebra of a given MBQC circuit"""

import operator

import numpy as np
from mentpy import MBQCircuit, PauliOp, _rust
from mentpy.mbqc.flow import _ordered_graph_nodes

__all__ = [
    "calculate_complete_gens",
    "calculate_ordered_gens",
    "calculate_gens",
    "lie_algebra_completion",
    "calculate_lie_algebra",
    "dim_su",
    "dim_so",
    "dim_sp",
]


def calculate_complete_gens(circuit: MBQCircuit):
    """Calculates the complete generators for the Lie algebra of a given circuit.

    Group
    -----
    utils
    """
    if not circuit.trainable_nodes:
        return PauliOp(np.empty((0, 2 * len(circuit)), dtype=np.uint8))
    circuit.flow.initialize_flow()
    if circuit.flow.func is None:
        raise ValueError("Cannot calculate the Lie algebra of a circuit without a flow")
    rows = [circuit.flow.generator_op(node).matrix for node in circuit.trainable_nodes]
    return PauliOp(np.vstack(rows))


def calculate_gens(circuit: MBQCircuit):
    """Calculates the generators of the Lie algebra of a given circuit removing repeated operators

    Group
    -----
    utils
    """
    return remove_repeated_ops(calculate_ordered_gens(circuit))


def calculate_ordered_gens(circuit: MBQCircuit):
    """Calculates the generators of a given circuit in order. The i-th generator
    corresponds to the i-th trainable node in the circuit.

    Args:
        circuit (MBQCircuit): The state for which to calculate the possible rotations

    Returns:
        PauliOp: The possible rotations achievable by the circuit

    Group
    -----
    utils
    """
    generators = calculate_complete_gens(circuit)
    indices = {
        node: index for index, node in enumerate(_ordered_graph_nodes(circuit.graph))
    }
    return generators.get_subset([indices[node] for node in circuit.output_nodes])


def lie_algebra_completion(generators: PauliOp, max_iter: int = 1000):
    """Complete Pauli generators under nonzero commutators, modulo phase.

    The Rust solver preserves first occurrence and discovery order, removes
    duplicates, and includes the identity, following MentPy's convention.
    Subtract one from the returned size when counting traceless generators.
    ``max_iter`` bounds the number of pairs examined; exhaustion raises instead
    of returning a partial basis. The basis can grow exponentially with qubits.

    Parameters
    ----------
    generators : PauliOp
        Initial Pauli generators. An empty tableau returns just the identity.
    max_iter : int
        Positive maximum number of commutator checks.

    Returns
    -------
    PauliOp
        Completed basis including the identity.

    Group
    -----
    utils
    """
    if isinstance(max_iter, (bool, np.bool_)):
        raise ValueError("max_iter must be a positive integer.")
    try:
        max_iter = operator.index(max_iter)
    except TypeError as exc:
        raise ValueError("max_iter must be a positive integer.") from exc
    if max_iter < 1:
        raise ValueError("max_iter must be a positive integer.")
    qubits = generators.number_of_qubits
    data, rows = _rust.lie_closure(
        np.asarray(generators.matrix, dtype=np.uint8).tobytes(),
        len(generators),
        qubits,
        max_iter,
    )
    tableau = np.frombuffer(data, dtype=np.uint8).reshape(rows, 2 * qubits)
    return PauliOp(tableau)


def calculate_lie_algebra(circuit: MBQCircuit, max_iter: int = 10000):
    """Calculates the Lie algebra of a given MBQCircuit

    Examples
    --------
    Calculate the Lie algebra of a 3x3 grid cluster state

    .. ipython:: python

        circ = mp.templates.grid_cluster(3,3)
        lie_alg = mp.utils.calculate_lie_algebra(circ)
        print(lie_alg)

    Group
    -----
    utils
    """
    generators = calculate_gens(circuit)
    return lie_algebra_completion(generators, max_iter=max_iter)


def remove_repeated_ops(ops: PauliOp):
    """Removes repeated Pauli operators from a set"""
    _, indices = np.unique(np.asarray(ops.matrix), axis=0, return_index=True)
    return PauliOp(ops.matrix[np.sort(indices)])


def dim_su(n):
    """Calculates the dimension of :math:`\\mathfrak{su}(n)`

    Group
    -----
    utils
    """
    return int(n**2 - 1)


def dim_so(n):
    """Calculates the dimension of :math:`\\mathfrak{so}(n)`

    Group
    -----
    utils
    """
    return int(n * (n - 1) // 2)


def dim_sp(n):
    """Calculates the dimension of :math:`\\mathfrak{sp}(n)`

    Group
    -----
    utils
    """
    assert n % 2 == 0, "n must be even"
    n = n // 2
    return int(n * (2 * n + 1))

# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""
This is the common_ansatz module.
It has several common ansatzes that can be used for MBQC algorithms
"""

from typing import List
from mentpy.operators import Ment, PauliOp
from mentpy.mbqc.states.graphstate import GraphState
from mentpy.mbqc.mbqcircuit import MBQCircuit, hstack

__all__ = [
    "linear_cluster",
    "many_wires",
    "grid_cluster",
    "muta",
    "from_pauli",
]


def linear_cluster(n, **kwargs) -> MBQCircuit:
    r"""Returns a linear cluster state of n qubits.

    Args
    ----
    n: int
        The number of qubits in the cluster state.

    Returns
    -------
    The linear cluster state of n qubits.

    Examples
    --------
    Create a 1D cluster state :math:`|G>` of five qubits

    .. ipython:: python

        g = mp.templates.linear_cluster(5)
        @savefig linear_cluster.png width=1000px
        mp.draw(g)

    Group
    -----
    templates
    """
    g = GraphState()
    g.add_edges_from([(i, i + 1) for i in range(n - 1)])
    gs = MBQCircuit(g, input_nodes=[0], output_nodes=[n - 1], **kwargs)
    return gs


def many_wires(n_wires: List, **kwargs) -> MBQCircuit:
    r"""Returns a graph state with many wires.

    Args
    ----
    n_wires: List
        A list of the number of qubits in each wire.

    Returns
    -------
    The graph state with many wires.

    Examples
    --------
    Create a graph state with three wires of 2, 3, and 4 qubits respectively

    .. ipython:: python

        g = mp.templates.many_wires([2, 3, 4])
        @savefig many_wires.png width=1000px
        mp.draw(g)

    Group
    -----
    templates
    """
    # require n_wires to be a list of integers greater than 1
    if not all([isinstance(n, int) and n > 1 for n in n_wires]):
        raise ValueError("n_wires must be a list of integers greater than 1")

    g = GraphState()
    for i, n in enumerate(n_wires):
        g.add_edges_from(
            [(j + sum(n_wires[:i]), j + sum(n_wires[:i]) + 1) for j in range(n - 1)]
        )

    # input nodes are the first qubit in each wire and output nodes are the last qubit in each wire
    gs = MBQCircuit(
        g,
        input_nodes=[sum(n_wires[:i]) for i in range(len(n_wires))],
        output_nodes=[sum(n_wires[: i + 1]) - 1 for i in range(len(n_wires))],
        **kwargs,
    )
    return gs


def grid_cluster(n, m, periodic=False, **kwargs) -> MBQCircuit:
    r"""Returns a grid cluster state of n x m qubits.

    Args
    ----
    n: int
        The number of rows in the cluster state.
    m: int
        The number of columns in the cluster state.
    periodic: bool
        If True, the returned state will be a cylinder.

    Returns
    -------
    The grid cluster state of n x m qubits.

    Examples
    --------
    Create a 2D cluster state :math:`|G>` of five qubits

    .. ipython:: python

        g = mp.templates.grid_cluster(2, 3)
        @savefig grid_cluster.png width=1000px
        mp.draw(g)

    Group
    -----
    templates
    """
    g = GraphState()
    # add edges between rows
    n_wires = [m] * n
    for i, m in enumerate(n_wires):
        g.add_edges_from(
            [(j + sum(n_wires[:i]), j + sum(n_wires[:i]) + 1) for j in range(m - 1)]
        )

    # add edges between columns
    for i in range(n - 1):
        g.add_edges_from([(i * m + j, (i + 1) * m + j) for j in range(m)])

    if periodic and n > 1:
        # add edges between first and last row
        for j in range(m):
            g.add_edge(j, (n - 1) * m + j)

    gs = MBQCircuit(
        g,
        input_nodes=[sum(n_wires[:i]) for i in range(len(n_wires))],
        output_nodes=[sum(n_wires[: i + 1]) - 1 for i in range(len(n_wires))],
        **kwargs,
    )
    return gs


def muta(n_wires, n_layers, **kwargs) -> MBQCircuit:
    """This is the Multiple Triangle Ansatz (MuTA) template.

    Args
    ----
    n_wires: int
        The number of wires in the graph state.
    n_layers: int
        The number of layers in the graph state.

    Keyword Args
    ------------
    one_column: bool
        Whether to use only one column of triangles.

    Returns
    -------
    The graph state with the MuTA template.

    Examples
    --------
    Create a MuTA ansatz with 3 wires and 2 layers

    .. ipython:: python

        g = mp.templates.muta(3, 2)
        @savefig muta.png width=1000px
        mp.draw(g, figsize=(16,5))

    Group
    -----
    templates
    """
    options = {
        "restrict_trainable": True,
        "one_column": False,
    }
    options.update(kwargs)

    SIZE_TRIANGLE = 5

    big_graph = None
    for wire in range(n_wires):
        if options["one_column"] and wire != 0:
            break
        g = many_wires([SIZE_TRIANGLE] * n_wires)
        if options["restrict_trainable"]:
            g.trainable_nodes = list(
                set(g.trainable_nodes) - set([i - 1 for i in g.output_nodes])
            )
            # TODO! Make something with this...

        for connect in range(n_wires):
            if connect != wire:
                g.add_edge(SIZE_TRIANGLE * wire + 1, SIZE_TRIANGLE * connect)
                g.add_edge(SIZE_TRIANGLE * wire + 1, SIZE_TRIANGLE * connect + 2)

        if big_graph is None:
            big_graph = g
        else:
            big_graph = hstack((big_graph, g))

    bigger_graph = None
    for layer in range(n_layers):
        if bigger_graph is None:
            bigger_graph = big_graph
        else:
            bigger_graph = hstack((bigger_graph, big_graph))

    # TODO: I think this is ending in a Hadamard rotated state (odd)
    # It might need a padding of 1 extra qubit in each wire.
    return bigger_graph


def from_pauli(pauli_op: PauliOp) -> MBQCircuit:
    """Returns a graph state that can implement :math:`U=e^{-i \\theta P}`

    Args
    ----
    pauli_op: PauliOp
        The Pauli operator to implement.

    Returns
    -------
    The graph state that can implement the Pauli operator.

    Examples
    --------
    Create a graph state that can implement a rotation around :math:`XYY`.

    .. ipython:: python

            g = mp.templates.from_pauli(mp.PauliOp("XYY"))
            p_op = g.flow.correction_op(10)
            @savefig from_pauli.png width=1000px
            mp.draw(g, pauliop=p_op)

    Group
    -----
    templates
    """

    if len(pauli_op) != 1:
        raise ValueError(
            f"Can only create the template for a single Pauli operator, but {len(pauli_op)} were given."
        )

    n_qubits = pauli_op.number_of_qubits
    exp_ansatz = many_wires([3] * n_qubits).graph
    exp_ansatz.add_nodes_from([3 * n_qubits, 3 * n_qubits + 1])
    exp_ansatz.add_edge(3 * n_qubits, 3 * n_qubits + 1)

    measurements = {3 * n_qubits + 1: Ment("XY")}

    for q in range(n_qubits):
        has_x, has_z = False, False
        if pauli_op.matrix[0, q] == 1:
            exp_ansatz.add_edge(3 * q + 1, 3 * n_qubits)

        if pauli_op.matrix[0, q + n_qubits] == 1:
            exp_ansatz.add_edge(3 * q, 3 * n_qubits)

    n_y = pauli_op.txt.count("Y")
    if n_y % 2 == 1:
        measurements[3 * n_qubits] = Ment("Y")

    input_nodes = [3 * q for q in range(n_qubits)]
    output_nodes = [3 * q + 2 for q in range(n_qubits)]

    return MBQCircuit(
        exp_ansatz,
        measurements=measurements,
        default_measurement=Ment("X"),
        input_nodes=input_nodes,
        output_nodes=output_nodes,
    )

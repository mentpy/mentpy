# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""This is the Flow module. It deals with the flow of a given graph state"""

from typing import List
import importlib
import warnings

import math
import numpy as np
import networkx as nx

from mentpy.calculator import linalg2
from mentpy.operators.pauliop import PauliOp

try:
    _rust_ext = importlib.import_module("mentpy._rust")
except ImportError:
    _rust_ext = None

__all__ = ["Flow", "find_cflow", "find_gflow", "find_pflow", "odd_neighborhood"]


class Flow:
    """This class deals with the flow of a given graph state

    Parameters
    ----------
    graph : mp.GraphState
        The graph state to find the flow of.
    input_nodes : list
        The input nodes of the graph state.
    output_nodes : list
        The output nodes of the graph state.
    planes : dict
        The measurement planes of the graph state. The keys are the nodes and the values are the planes. If None, the algorithm will assume that the measurement planes are all XY.

    Group
    -----
    flow
    """

    def __init__(self, graph, input_nodes, output_nodes, planes=None):
        """
        Initializes the flow of a given graph state.
        Assumes that graph has nodes labeled with numbers from 0 to n-1, where n is the number of nodes in the graph.
        """
        self.graph = graph
        self.input_nodes = input_nodes
        self.output_nodes = output_nodes
        self.planes = planes
        self.flow_initialized = False
        self.name = "Flow"

    def __repr__(self):
        return f"{self.name}(n={self.graph.number_of_nodes()})"

    def __call__(self, node):
        self.initialize_flow()
        return self.func(node) if self.func else None

    def initialize_flow(self):
        """Lazily initializes the flow properties when needed."""
        if not self.flow_initialized:
            self._find_flow()
            self.flow_initialized = True

    def _initialize_layers(self, layers):
        """Initializes layers and measurement order based on the provided layers dict."""
        if layers:
            self.layers = [[] for _ in range(max(layers.values()) + 1)]
            for node, layer in layers.items():
                self.layers[-1 - layer].append(node)
            order = [item for sublist in self.layers for item in sublist]
            for i in self.input_nodes[::-1]:
                order.remove(i)
                order.insert(0, i)
            self.measurement_order = order
        else:
            self.layers = None
            self.measurement_order = None

    def _find_flow(self):
        """Attempts to find various types of flow, prioritizing causal flow, then generalized, then pflow."""
        flow_function = partial_order = depth = layers = None
        name = "No Flow "
        planes = self.planes or {}
        output_nodes = set(self.output_nodes)
        if all(
            callable(plane) or plane in {"XY", "X", "Y"}
            for node, plane in planes.items()
            if node not in output_nodes
        ):
            flow_function, partial_order, depth, layers = find_cflow(
                self.graph, self.input_nodes, self.output_nodes
            )
            name = "cFlow"
            if flow_function is None:
                flow_function, partial_order, depth, layers = find_gflow(
                    self.graph, self.input_nodes, self.output_nodes
                )
                name = "gFlow"

        if flow_function is None and self.planes is not None:
            _, flow_function, layers = find_pflow(
                self.graph, self.input_nodes, self.output_nodes, self.planes
            )
            partial_order = lambda u, v: layers[u] > layers[v]
            depth = None if len(layers) == 0 else max(layers.values())
            name = "pFlow"

        if flow_function is None:
            warnings.warn(
                "No flow found. The flow function will return None for all nodes.",
                stacklevel=2,
            )
            name = "No Flow "
            partial_order = None

        self.func = flow_function
        self.partial_order = partial_order
        self.depth = depth
        self.name = name
        self._initialize_layers(layers)

    def _corrections(self, node):
        target = self(node)
        if target is None:
            raise ValueError("Correction operators require a flow.")
        if self.name == "cFlow":
            return {target}, set(self.graph.neighbors(target))
        nodes = _ordered_graph_nodes(self.graph)
        x = {nodes[i] for i in np.flatnonzero(target)}
        return x, odd_neighborhood(self.graph, x)

    def correction_op(self, node):
        """Return the correction operator for a given node."""
        nodes = _ordered_graph_nodes(self.graph)
        indices = {vertex: i for i, vertex in enumerate(nodes)}
        n = len(nodes)
        x, z = self._corrections(node)
        pauli = np.zeros((1, 2 * n), dtype=int)
        pauli[0, [indices[v] for v in x]] = 1
        pauli[0, [n + indices[v] for v in z]] = 1
        return PauliOp(pauli)

    def generator_op(self, node):
        """Returns the generator operator for a given node."""
        nodes = _ordered_graph_nodes(self.graph)
        op = self.correction_op(node)
        cond = False
        while not cond:
            x, z = np.split(op.matrix[0], 2)
            z_places = {nodes[i] for i in np.flatnonzero((z != 0) & (x == 0))}
            nodes_allowed = set([node, *self.output_nodes])

            z_mult = None

            if z_places.issubset(nodes_allowed):
                cond = True

            else:
                z_mult = z_places - nodes_allowed

            if z_mult:
                for z in z_mult:
                    op = op * self.correction_op(z)

        return op


# Implementation of Causal Flow. Time complexity: O(min(m, kn))


def find_cflow(graph, input_nodes, output_nodes) -> object:
    """Finds the causal flow a graph.

    Parameters
    ----------
    graph : mp.GraphState
        The graph state to find the flow of.
    input_nodes : list
        The input nodes of the graph state.
    output_nodes : list
        The output nodes of the graph state.

    Returns
    -------
    flow : function
        The flow function.
    partial_order : function
        The partial order function.
    depth : int
        The depth of the flow.
    layers : dict
        The layers of the flow.


    References
    ----------
    Implementation of algorithm in https://arxiv.org/pdf/0709.2670v1.pdf.

    Group
    -----
    flow
    """

    if _rust_ext is not None:
        nodes = _ordered_graph_nodes(graph)
        indices = {node: i for i, node in enumerate(nodes)}
        result = _rust_ext.cflow(
            [[indices[u] for u in graph.neighbors(v)] for v in nodes],
            [indices[v] for v in input_nodes],
            [indices[v] for v in output_nodes],
        )
        if result is None:
            return None, None, None, None
        successors, native_layers = result
        layers = dict(zip(nodes, native_layers))
        corrections = {
            node: nodes[target]
            for node, target in zip(nodes, successors)
            if target < len(nodes)
        }
        return (
            corrections.__getitem__,
            lambda u, v: layers[u] > layers[v],
            max(native_layers, default=0),
            layers,
        )

    l = {}
    g = {}
    past = {}
    C_set = set()

    for v in graph.nodes():
        l[v] = 0
        past[v] = 0

    inputs, outputs = set(input_nodes), set(output_nodes)
    for v in outputs - inputs:
        past[v] = sum(u not in outputs for u in graph.neighbors(v))
        if past[v] == 1:
            C_set.add(v)

    flow, ln = causal_flow_aux(graph, inputs, outputs, C_set, past, 1, g, l)

    if len(flow) != len(graph.nodes()) - len(output_nodes):
        return None, None, None, None

    return (
        lambda x: flow[x],
        lambda u, v: ln[u] > ln[v],
        max(ln.values(), default=0),
        ln,
    )


def causal_flow_aux(graph, inputs, outputs, C, past, k, g, l) -> object:
    """Aux function for causal_flow"""
    while C:
        C_prime = set()
        for v in C:
            intersection = {u for u in graph.neighbors(v) if u not in outputs}
            if len(intersection) == 1:
                u = intersection.pop()
                g[u] = v
                l[u] = k
                outputs.add(u)
                if u not in inputs:
                    past[u] = sum(w not in outputs for w in graph.neighbors(u))
                    if past[u] == 1:
                        C_prime.add(u)
                for w in graph.neighbors(u):
                    if past[w] > 0:
                        past[w] -= 1
                        if past[w] == 1:
                            C_prime.add(w)
        C = C_prime
        k += 1
    return g, l


def check_if_cflow(
    graph, input_nodes: List, output_nodes: List, flow, partial_order
) -> bool:
    """Checks if flow satisfies conditions on state."""
    conds = True
    for i in [v for v in graph.nodes() if v not in output_nodes]:
        nfi = list(graph.neighbors(flow(i)))
        c1 = i in nfi
        c2 = partial_order(i, flow(i))
        c3 = math.prod([partial_order(i, k) for k in set(nfi) - {i}])
        conds = conds * c1 * c2 * c3
        if not c1:
            print(f"Condition 1 failed for node {i}. {i} not in {nfi}")
        if not c2:
            print(f"Condition 2 failed for node {i}. {i} ≮ {flow(i)}")
        if not c3:
            print(f"Condition 3 failed for node {i}.")
            for k in set(nfi) - {i}:
                if not partial_order(i, k):
                    print(f"{i} ≮ {k}")
    return conds


# Implementation of gFlow. Time complexity: O(n^4)


def find_gflow(graph, input_nodes, output_nodes) -> object:
    """Finds the generalized flow of a graph.

    Parameters
    ----------
    graph : mp.GraphState
        The graph state to find the flow of.
    input_nodes : list
        The input nodes of the graph state.
    output_nodes : list
        The output nodes of the graph state.

    Returns
    -------
    flow : function
        The flow function.
    partial_order : function
        The partial order function.
    depth : int
        The depth of the flow.
    layers : dict
        The layers of the flow.

    References
    ----------
    Implementation of algorithm in https://arxiv.org/pdf/0709.2670v1.pdf.

    Group
    -----
    flow
    """
    nodes = _ordered_graph_nodes(graph)
    gamma = nx.to_numpy_array(graph, nodelist=nodes, dtype=np.uint8, weight=None)

    if _rust_ext is not None:
        indices = {node: i for i, node in enumerate(nodes)}
        result = _rust_ext.gflow(
            gamma.tobytes(),
            len(nodes),
            [indices[node] for node in input_nodes],
            [indices[node] for node in output_nodes],
        )
        if result is None:
            return None, None, None, None
        data, native_layers = result
        corrections = np.frombuffer(data, dtype=np.uint8).reshape(
            len(nodes), len(nodes)
        )
        layers = dict(zip(nodes, native_layers))
        measured = {node: indices[node] for node in nodes if layers[node] > 0}
        return (
            lambda node: corrections[measured[node]].astype(int).reshape(-1, 1),
            lambda u, v: layers[u] > layers[v],
            max(native_layers, default=0),
            layers,
        )

    l = {}
    g = {}

    for v in output_nodes:
        l[v] = 0

    result, gn, ln = gflowaux(
        graph,
        gamma,
        set(input_nodes),
        set(output_nodes),
        1,
        g,
        l,
    )

    if result == False:
        return None, None, None, None

    return lambda x: gn[x], lambda u, v: ln[u] > ln[v], max(ln.values(), default=0), ln


def gflowaux(graph, gamma, inputs, outputs, k, g, l) -> object:
    """Aux function for gFlow"""

    nodes = _ordered_graph_nodes(graph)
    while len(outputs) < len(nodes):
        rows = [i for i, node in enumerate(nodes) if node not in outputs]
        cols = [
            i for i, node in enumerate(nodes) if node in outputs and node not in inputs
        ]
        submatrix = gamma[np.ix_(rows, cols)]
        layer = set()
        b = np.zeros(len(rows), dtype=int)
        for i, row in enumerate(rows):
            b[i] = 1
            try:
                solution = linalg2.solve(submatrix, b, check_solution=True)
            except ValueError:
                continue
            finally:
                b[i] = 0
            node = nodes[row]
            l[node] = k
            layer.add(node)
            correction = np.zeros((len(nodes), 1), dtype=int)
            correction[cols] = solution
            g[node] = correction
        if not layer:
            return False, g, l
        outputs = outputs | layer
        k += 1
    return True, g, l


# Implementation of PauliFlow. Time complexity: O(n^3)


def find_pflow(graph, I, O, planes):
    """
    Find a p-flow in a given graph.

    Parameters
    ----------
    graph : mp.GraphState
        The graph state to find the flow of.
    I : list
        The input nodes of the graph state.
    O : list
        The output nodes of the graph state.
    planes : dict
        The measurement planes of the graph state. The keys are the nodes and the values are the planes.

    Returns
    -------
    condition : bool
        True if a p-flow was found, False otherwise.
    flow : function
        The flow function.
    layers : dict
        The layers of the flow.

    References
    ----------
    Implementation of the algebraic flow-demand/order-demand matrix algorithm in
    https://arxiv.org/abs/2410.23439. This improves the previous O(n^5)
    recursive Pauli-flow finder from https://arxiv.org/pdf/2109.05654v1.pdf to
    O(n^3).


    Group
    -----
    flow
    """
    try:
        C, R, row_vertices, column_vertices = _find_pflow_matrices(graph, I, O, planes)
    except ValueError:
        return False, None, {}

    layers = _pflow_layers_from_relation(R, row_vertices, O)
    node_to_column = {node: i for i, node in enumerate(row_vertices)}
    node_to_index = {node: i for i, node in enumerate(_ordered_graph_nodes(graph))}
    n_nodes = graph.number_of_nodes()

    def flow_fn(node):
        column = node_to_column[node]
        correction = np.zeros((n_nodes, 1), dtype=int)
        for row, vertex in enumerate(column_vertices):
            if C[row, column]:
                correction[node_to_index[vertex]] = 1
        return correction

    return True, flow_fn, layers


def _find_pflow_matrices(graph, I, O, planes):
    """Return algebraic Pauli-flow matrices ``C`` and ``R = N @ C``.

    Rows of ``C`` correspond to non-input vertices and columns of ``C``/``R``
    correspond to non-output vertices. ``R[w, v] = 1`` means vertex ``v`` must
    precede vertex ``w`` in the induced relation.
    """
    I, O = set(I), set(O)

    if len(I) > len(O):
        raise ValueError(
            "Pauli flow cannot exist when there are more inputs than outputs."
        )

    M, N, row_vertices, column_vertices = _pflow_demand_matrices(graph, I, O, planes)
    n_rows = M.shape[0]
    n_cols = M.shape[1]

    if _rust_ext is not None:
        result = _rust_ext.pflow_matrices(M.tobytes(), N.tobytes(), n_rows, n_cols)
        if result is None:
            raise ValueError("No right inverse yields an acyclic Pauli-flow relation.")
        c_data, r_data = result
        return (
            np.frombuffer(c_data, dtype=np.uint8).reshape(n_cols, n_rows).copy(),
            np.frombuffer(r_data, dtype=np.uint8).reshape(n_rows, n_rows).copy(),
            row_vertices,
            column_vertices,
        )

    if n_rows == 0:
        return (
            np.zeros((n_cols, 0), dtype=np.uint8),
            np.zeros((0, 0), dtype=np.uint8),
            row_vertices,
            column_vertices,
        )

    if n_rows == n_cols:
        C = _gf2_inverse(M)
        if C is None:
            raise ValueError("The flow-demand matrix is singular.")

        R = _gf2_matmul(N, C)
        if not _is_dag_adjacency(R):
            raise ValueError("The induced Pauli-flow relation is cyclic.")
        return C, R, row_vertices, column_vertices

    C0, kernel = _gf2_right_inverse_and_kernel(M)
    if C0 is None:
        raise ValueError("The flow-demand matrix is not right-invertible.")

    basis_change = np.hstack((C0, kernel)).astype(np.uint8, copy=False)
    changed_N = _gf2_matmul(N, basis_change)
    NL = changed_N[:, :n_rows]
    NR = changed_N[:, n_rows:]
    P = _solve_dag_right_inverse(NL, NR)
    if P is None:
        raise ValueError("No right inverse yields an acyclic Pauli-flow relation.")

    changed_C = np.vstack((np.eye(n_rows, dtype=np.uint8), P))
    C = _gf2_matmul(basis_change, changed_C)
    R = _gf2_matmul(N, C)
    if not _is_dag_adjacency(R):
        raise ValueError("The induced Pauli-flow relation is cyclic.")
    return C, R, row_vertices, column_vertices


def _pflow_demand_matrices(graph, I, O, planes):
    """Construct the flow-demand matrix ``M`` and order-demand matrix ``N``."""
    nodes = _ordered_graph_nodes(graph)
    node_to_index = {node: i for i, node in enumerate(nodes)}
    row_vertices = [node for node in nodes if node not in O]
    column_vertices = [node for node in nodes if node not in I]

    adjacency = nx.to_numpy_array(graph, nodelist=nodes, dtype=np.uint8, weight=None)
    row_indices = [node_to_index[node] for node in row_vertices]
    column_indices = [node_to_index[node] for node in column_vertices]
    reduced_adjacency = adjacency[np.ix_(row_indices, column_indices)]

    M = reduced_adjacency.copy()
    N = reduced_adjacency.copy()
    column_index = {node: i for i, node in enumerate(column_vertices)}

    for row, vertex in enumerate(row_vertices):
        plane = _normalise_plane(planes, vertex)

        if plane in {"Z", "YZ", "XZ"}:
            M[row, :] = 0
        if plane in {"Y", "Z", "YZ", "XZ"} and vertex in column_index:
            M[row, column_index[vertex]] = 1

        if plane in {"X", "Y", "Z", "XY"}:
            N[row, :] = 0
        if plane in {"XY", "XZ"} and vertex in column_index:
            N[row, column_index[vertex]] = 1

    return M, N, row_vertices, column_vertices


def _ordered_graph_nodes(graph):
    nodes = list(graph.nodes())
    try:
        if set(nodes) == set(range(len(nodes))):
            return sorted(nodes)
    except TypeError:
        pass
    return nodes


def _normalise_plane(planes, vertex):
    try:
        plane = planes[vertex]
    except KeyError as exc:
        raise ValueError(
            f"Missing measurement plane for non-output vertex {vertex}."
        ) from exc

    plane = plane() if callable(plane) else plane
    if hasattr(plane, "plane"):
        plane = plane.plane
    plane = str(plane).upper()
    if plane not in {"X", "Y", "Z", "XY", "XZ", "YZ"}:
        raise ValueError(f"Plane {plane} is not supported by Pauli flow.")
    return plane


def _gf2_matmul(A, B):
    return ((A.astype(np.uint8) @ B.astype(np.uint8)) & 1).astype(np.uint8)


def _gf2_rref_augmented(A, rhs=None, max_cols=None):
    A = np.array(A, dtype=np.uint8, copy=True) & 1
    if rhs is not None:
        rhs = np.array(rhs, dtype=np.uint8, copy=True) & 1
        A = np.hstack((A, rhs))

    n_rows = A.shape[0]
    max_cols = A.shape[1] if max_cols is None else max_cols
    pivot_cols = []
    pivot_row = 0

    for col in range(max_cols):
        pivot_offsets = np.flatnonzero(A[pivot_row:, col])
        if pivot_offsets.size == 0:
            continue

        pivot = pivot_row + int(pivot_offsets[0])
        if pivot != pivot_row:
            A[[pivot_row, pivot]] = A[[pivot, pivot_row]]

        rows = np.flatnonzero(A[:, col])
        rows = rows[rows != pivot_row]
        if rows.size:
            A[rows] ^= A[pivot_row]

        pivot_cols.append(col)
        pivot_row += 1
        if pivot_row == n_rows:
            break

    return A, pivot_cols


def _gf2_inverse(A):
    A = np.asarray(A, dtype=np.uint8) & 1
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("Only square matrices can be inverted.")

    if _rust_ext is not None:
        A_contiguous = np.ascontiguousarray(A, dtype=np.uint8)
        inverse = _rust_ext.gf2_inverse_bytes(
            A_contiguous.tobytes(), A.shape[0], A.shape[1]
        )
        if inverse is None:
            return None
        data, rows, cols = inverse
        return np.frombuffer(data, dtype=np.uint8).reshape((rows, cols)).copy()

    n = A.shape[0]
    augmented, pivot_cols = _gf2_rref_augmented(A, np.eye(n, dtype=np.uint8), n)
    if len(pivot_cols) != n:
        return None
    return augmented[:, n:]


def _gf2_right_inverse_and_kernel(A):
    """Return one right inverse and a column basis for ``ker(A)`` over GF(2)."""
    A = np.asarray(A, dtype=np.uint8) & 1
    n_rows, n_cols = A.shape

    if _rust_ext is not None:
        A_contiguous = np.ascontiguousarray(A, dtype=np.uint8)
        result = _rust_ext.gf2_right_inverse_and_kernel_bytes(
            A_contiguous.tobytes(), n_rows, n_cols
        )
        if result is None:
            return None, None
        C0_data, C0_rows, C0_cols, kernel_data, kernel_rows, kernel_cols = result
        C0 = np.frombuffer(C0_data, dtype=np.uint8).reshape((C0_rows, C0_cols))
        kernel = np.frombuffer(kernel_data, dtype=np.uint8).reshape(
            (kernel_rows, kernel_cols)
        )
        return C0.copy(), kernel.copy()

    augmented, pivot_cols = _gf2_rref_augmented(
        A, np.eye(n_rows, dtype=np.uint8), n_cols
    )
    rank = len(pivot_cols)
    if rank != n_rows:
        return None, None

    row_transform = augmented[:n_rows, n_cols:]
    C0 = np.zeros((n_cols, n_rows), dtype=np.uint8)
    for row, pivot_col in enumerate(pivot_cols):
        C0[pivot_col, :] = row_transform[row, :]

    pivot_col_set = set(pivot_cols)
    free_cols = [col for col in range(n_cols) if col not in pivot_col_set]
    kernel = np.zeros((n_cols, len(free_cols)), dtype=np.uint8)
    for basis_col, free_col in enumerate(free_cols):
        kernel[free_col, basis_col] = 1
        for row, pivot_col in enumerate(pivot_cols):
            kernel[pivot_col, basis_col] = augmented[row, free_col]

    return C0, kernel


def _gf2_row_echelon_inplace(A, max_cols):
    pivot_row = 0
    n_rows = A.shape[0]
    for col in range(max_cols):
        pivot_offsets = np.flatnonzero(A[pivot_row:, col])
        if pivot_offsets.size == 0:
            continue
        pivot = pivot_row + int(pivot_offsets[0])
        if pivot != pivot_row:
            A[[pivot_row, pivot]] = A[[pivot, pivot_row]]

        rows = np.flatnonzero(A[pivot_row + 1 :, col]) + pivot_row + 1
        if rows.size:
            A[rows] ^= A[pivot_row]

        pivot_row += 1
        if pivot_row == n_rows:
            break


def _leading_one(row):
    entries = np.flatnonzero(row)
    return int(entries[0]) if entries.size else None


def _sort_echelon_rows(A, max_cols):
    leading = []
    for row in range(A.shape[0]):
        lead = _leading_one(A[row, :max_cols])
        leading.append(max_cols if lead is None else lead)
    order = sorted(range(A.shape[0]), key=lambda row: (leading[row], row))
    A[:] = A[order]


def _solve_from_row_echelon(echelon, rhs):
    echelon = np.array(echelon, dtype=np.uint8, copy=False) & 1
    rhs = np.array(rhs, dtype=np.uint8, copy=False).reshape(-1) & 1
    n_cols = echelon.shape[1]
    solution = np.zeros(n_cols, dtype=np.uint8)

    for row in range(echelon.shape[0] - 1, -1, -1):
        pivot = _leading_one(echelon[row])
        if pivot is None:
            if rhs[row]:
                raise ValueError("Inconsistent GF(2) linear system.")
            continue
        tail = np.dot(echelon[row, pivot + 1 :], solution[pivot + 1 :]) & 1
        solution[pivot] = rhs[row] ^ tail

    return solution


def _solve_dag_right_inverse(NL, NR):
    """Find ``P`` such that ``NL + NR @ P`` is a DAG, if one exists."""
    NL = np.array(NL, dtype=np.uint8, copy=True) & 1
    NR = np.array(NR, dtype=np.uint8, copy=True) & 1
    n_vertices = NL.shape[0]
    free_dim = NR.shape[1]
    if _rust_ext is not None:
        result = _rust_ext.solve_dag(NL.tobytes(), NR.tobytes(), n_vertices, free_dim)
        return (
            None
            if result is None
            else np.frombuffer(result, dtype=np.uint8)
            .reshape(free_dim, n_vertices)
            .copy()
        )
    if free_dim == 0:
        if _is_dag_adjacency(NL):
            return np.zeros((0, n_vertices), dtype=np.uint8)
        return None

    ILS = np.hstack((NR, NL, np.eye(n_vertices, dtype=np.uint8)))
    LS = ILS.copy()
    _gf2_row_echelon_inplace(LS, free_dim)

    solved = np.zeros(n_vertices, dtype=bool)
    P = np.zeros((free_dim, n_vertices), dtype=np.uint8)

    while not np.all(solved):
        zero_rows = np.where(~LS[:, :free_dim].any(axis=1))[0]
        first_zero_row = int(zero_rows[0]) if zero_rows.size else n_vertices
        constants = LS[first_zero_row:, free_dim : free_dim + n_vertices]

        to_solve = [
            vertex
            for vertex in range(n_vertices)
            if not solved[vertex]
            and (constants.shape[0] == 0 or not np.any(constants[:, vertex]))
        ]
        if len(to_solve) == 0:
            return None

        for vertex in to_solve:
            P[:, vertex] = _solve_from_row_echelon(
                LS[:, :free_dim], LS[:, free_dim + vertex]
            )

        for vertex in to_solve:
            solved[vertex] = True
            tracker_col = free_dim + n_vertices + vertex
            dependent_rows = np.flatnonzero(LS[:, tracker_col])
            if dependent_rows.size == 0:
                continue

            last_row = int(dependent_rows[-1])
            for row in dependent_rows[:-1]:
                LS[int(row)] ^= LS[last_row]

            LS[last_row] ^= ILS[vertex]
            for row in range(n_vertices):
                if row == last_row:
                    continue
                pivot = _leading_one(LS[row, :free_dim])
                if pivot is None:
                    break
                if LS[last_row, pivot]:
                    LS[last_row] ^= LS[row]

            _sort_echelon_rows(LS, free_dim)

    R = (NL ^ _gf2_matmul(NR, P)).astype(np.uint8, copy=False)
    return P if _is_dag_adjacency(R) else None


def _is_dag_adjacency(adjacency):
    adjacency = np.array(adjacency, dtype=np.uint8, copy=False) & 1
    if adjacency.shape[0] != adjacency.shape[1]:
        return False
    if np.any(np.diag(adjacency)):
        return False

    graph = nx.DiGraph()
    graph.add_nodes_from(range(adjacency.shape[0]))
    rows, cols = np.nonzero(adjacency)
    graph.add_edges_from(zip(cols.tolist(), rows.tolist()))
    return nx.is_directed_acyclic_graph(graph)


def _pflow_layers_from_relation(relation, row_vertices, output_nodes):
    """Convert the algebraic induced relation into MentPy layer numbers."""
    layers = {node: 0 for node in output_nodes}
    relation = np.array(relation, dtype=np.uint8, copy=False) & 1
    if _rust_ext is not None:
        layers.update(
            zip(
                row_vertices,
                _rust_ext.flow_layers(relation.tobytes(), len(row_vertices)),
            )
        )
        return layers
    successors = {
        vertex: {row_vertices[row] for row in np.flatnonzero(relation[:, col])}
        for col, vertex in enumerate(row_vertices)
    }

    for vertex in reversed(list(nx.topological_sort(_relation_digraph(relation)))):
        if len(successors[row_vertices[vertex]]) == 0:
            layers[row_vertices[vertex]] = 1
        else:
            layers[row_vertices[vertex]] = 1 + max(
                layers[successor] for successor in successors[row_vertices[vertex]]
            )
    return layers


def _relation_digraph(relation):
    graph = nx.DiGraph()
    graph.add_nodes_from(range(relation.shape[0]))
    rows, cols = np.nonzero(relation)
    graph.add_edges_from(zip(cols.tolist(), rows.tolist()))
    return graph


def odd_neighborhood(graph, A):
    """Returns the set of nodes in the graph that have an odd number of neighbors in A.

    Group
    -----
    flow
    """
    return {w for w in graph.nodes() if len(set(graph.neighbors(w)) & A) % 2 == 1}

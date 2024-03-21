# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""This is the Flow module. It deals with the flow of a given graph state"""
from typing import List
import warnings

import math
import numpy as np
import networkx as nx

from mentpy.calculator import linalg2

__all__ = ["Flow"]


class Flow:
    """This class deals with the flow of a given graph state

    Group
    -----
    mbqc
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
            self.layers = [
                [n for n, l in layers.items() if l == j]
                for j in range(max(layers.values()) + 1)
            ][::-1]
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
            condition, flow_function, layers = find_pflow(
                self.graph, self.input_nodes, self.output_nodes, self.planes
            )
            partial_order = lambda u, v: layers[u] > layers[v]
            depth = None if len(layers) == 0 else max(layers.values())
            name = "pFlow"

        if flow_function is None:
            warnings.warn(
                "No flow found. The flow function will return None for all nodes."
            )
            name = "No Flow "

        self.func = flow_function
        self.partial_order = partial_order
        self.depth = depth
        self.name = name
        self._initialize_layers(layers)

    def adapt_angles(self, angles, outcomes):
        raise NotImplementedError

    def adapt_angle(self, angle, node, previous_outcomes):
        raise NotImplementedError


# Implementation of Causal Flow. Time complexity: O(min(m, kn))


def find_cflow(graph, input_nodes, output_nodes) -> object:
    """Finds the causal flow a graph. Retrieved from https://arxiv.org/pdf/0709.2670v1.pdf."""

    l = {}
    g = {}
    past = {}
    C_set = set()

    for v in graph.nodes():
        l[v] = 0
        past[v] = 0

    for v in set(output_nodes) - set(input_nodes):
        past[v] = len(
            set(graph.neighbors(v)) & (set(graph.nodes() - set(output_nodes)))
        )
        if past[v] == 1:
            C_set = C_set.union({v})

    flow, ln = causal_flow_aux(
        graph, set(input_nodes), set(output_nodes), C_set, past, 1, g, l
    )

    if len(flow) != len(graph.nodes()) - len(output_nodes):
        return None, None, None, None

    return lambda x: flow[x], lambda u, v: ln[u] > ln[v], max(flow.values()), ln


def causal_flow_aux(graph, inputs, outputs, C, past, k, g, l) -> object:
    """Aux function for causal_flow"""
    V = set(graph.nodes())
    C_prime = set()

    for _, v in enumerate(C):
        intersection = set(graph.neighbors(v)) & (V - outputs)
        if len(intersection) == 1:
            u = intersection.pop()
            g[u] = v
            l[u] = k
            outputs.add(u)
            if u not in inputs:
                past[u] = len(set(graph.neighbors(u)) & (V - outputs))
                if past[u] == 1:
                    C_prime.add(u)
            for w in set(graph.neighbors(u)):
                if past[w] > 0:
                    past[w] -= 1
                    if past[w] == 1:
                        C_prime.add(w)

    if len(C_prime) == 0:
        return g, l

    else:
        return causal_flow_aux(
            graph,
            inputs,
            outputs,
            C_prime,
            past,
            k + 1,
            g,
            l,
        )


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
    """Finds the generalized flow of a ``MBQCGraph`` if it exists.
    Retrieved from https://arxiv.org/pdf/0709.2670v1.pdf.
    """
    gamma = nx.adjacency_matrix(graph).toarray()

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

    return lambda x: gn[x], lambda u, v: ln[u] > ln[v], max(ln.values()), ln


def gflowaux(graph, gamma, inputs, outputs, k, g, l) -> object:
    """Aux function for gFlow"""

    V = set(graph.nodes())
    C = set()
    vmol = list(V - outputs)
    for u in vmol:
        submatrix = np.zeros((len(vmol), len(outputs - inputs)), dtype=int)
        submatrix = gamma[np.ix_(vmol, list(outputs - inputs))]

        b = np.zeros((len(vmol), 1), dtype=int)
        b[vmol.index(u)] = 1

        try:
            solution = linalg2.solve(submatrix, b, check_solution=True).reshape(-1, 1)
            l[u] = k
            C.add(u)
            sol_extended = np.zeros((len(V), 1), dtype=int)
            sol_extended[list(outputs - inputs)] = solution
            g[u] = sol_extended
        except Exception as e:
            pass

    if len(C) == 0:
        if set(outputs) == V:
            return True, g, l
        else:
            return False, g, l

    else:
        return gflowaux(graph, gamma, inputs, outputs | C, k + 1, g, l)


# Implementation of PauliFlow. Time complexity: O(n^5)


def find_pflow(graph, I, O, λ):
    """
    Find a p-flow in a given graph. Implementation of pauli flow algorithm in https://arxiv.org/pdf/2109.05654v1.pdf
    """
    V = set(graph.nodes())
    Γ = nx.adjacency_matrix(graph).toarray()
    I, O = set(I), set(O)

    LX, LY, LZ = set(), set(), set()
    d = {}

    for v in V:
        if v in O:
            d[v] = 0
        elif λ[v] == "X":
            LX.add(v)
        elif λ[v] == "Y":
            LY.add(v)
        elif λ[v] == "Z":
            LZ.add(v)

    p = {}
    return pflowaux(V, Γ, I, O, λ, LX, LY, LZ, set(), O, d, 0, graph, p)


def solve_constraints(u, V, Γ, I, O, λ, LX, LY, LZ, A, B, d, k, graph, plane):
    solution = None
    KAu = get_KAu(Γ, A, u, V, I, B, λ)
    PAu = get_PAu(Γ, A, u, V, I, B, λ)
    YAu = get_YAu(Γ, A, u, V, I, B, λ)

    solution_executed = False

    if len(KAu) != 0 and len(PAu) != 0:
        MAu1 = get_MAu1(Γ, KAu, PAu)
        SLambda1 = get_SLambda1(plane, u, V, I, O, λ, graph, Γ, A, KAu, PAu)

        try:
            solution1 = linalg2.solve(MAu1.T, SLambda1, check_solution=True).reshape(
                -1, 1
            )
            solution_executed = True
        except Exception as e:
            pass

    if len(KAu) != 0 and len(YAu) != 0:
        MAu2 = get_MAu2(Γ, KAu, YAu)
        SLambda2 = get_SLambda2(plane, u, V, I, O, λ, graph, Γ, A, KAu, YAu)
        try:
            solution2 = linalg2.solve(MAu2.T, SLambda2, check_solution=True).reshape(
                -1, 1
            )
            solution_executed = True
        except Exception as e:
            pass

    if solution_executed:
        solution = np.zeros((len(V), 1), dtype=int)

        if "solution1" in locals():
            solution[KAu] = solution1

        if "solution2" in locals():
            for idx, val in np.ndenumerate(solution2):
                i = YAu[idx[0]]
                solution[i] = max(solution[i], val)

    return solution


def pflowaux(V, Γ, I, O, λ, LX, LY, LZ, A, B, d, k, graph, p):
    C = set()
    for u in set(V) - B:
        solution = None

        if λ[u] in {"XY", "X", "Y"}:
            solution = solve_constraints(
                u, V, Γ, I, O, λ, LX, LY, LZ, A, B, d, k, graph, "XY"
            )

        if λ[u] in {"XZ", "X", "Z"} and solution is None:
            solution = solve_constraints(
                u, V, Γ, I, O, λ, LX, LY, LZ, A, B, d, k, graph, "XZ"
            )

        if λ[u] in {"YZ", "Y", "Z"} and solution is None:
            solution = solve_constraints(
                u, V, Γ, I, O, λ, LX, LY, LZ, A, B, d, k, graph, "YZ"
            )

        if solution is not None:
            C.add(u)
            p[u] = solution
            d[u] = k

    if C == set() and k > 0:
        if set(B) == set(V):
            return True, lambda x: p[x], d
        else:
            return False, None, {}

    Bprime = B | C
    return pflowaux(V, Γ, I, O, λ, LX, LY, LZ, Bprime, Bprime, d, k + 1, graph, p)


def get_MAu1(Gamma, KAu, PAu):
    return Gamma[np.ix_(KAu, PAu)]


def get_MAu2(Gamma, KAu, YAu):
    return (Gamma + np.eye(Gamma.shape[0]))[np.ix_(KAu, YAu)]


def get_SLambda1(plane, u, V, I, O, planes, graph, Gamma, A, KAu, PAu):
    sl = set()
    if plane == "XY":
        sl = {u}
    elif plane == "XZ":
        neighbors = set(graph.neighbors(u))
        PAu = get_PAu(Gamma, A, u, V, I, O, planes)
        sl = (neighbors & set(PAu)) | {u}
    elif plane == "YZ":
        neighbors = set(graph.neighbors(u))
        PAu = get_PAu(Gamma, A, u, V, I, O, planes)
        sl = neighbors & set(PAu)

    vec_sl = np.zeros((len(V), 1), dtype=int)
    vec_sl[list(sl)] = 1
    vec_sl = vec_sl[PAu]
    return vec_sl


def get_SLambda2(plane, u, V, I, O, planes, graph, Gamma, A, KAu, YAu):
    sl = set()
    if plane == "XY":
        sl = set()
    elif plane in {"XZ", "YZ"}:
        neighbors = set(graph.neighbors(u))
        YAu = get_YAu(Gamma, A, u, V, I, O, planes)
        sl = neighbors & set(YAu)

    vec_sl = np.zeros((len(V), 1), dtype=int)

    vec_sl[list(sl)] = 1
    vec_sl = vec_sl[YAu]
    return vec_sl


def get_KAu(Gamma, A, u, V, I, O, planes):
    p = A | LambdaPu("X", u, V, O, planes) | LambdaPu("Y", u, V, O, planes)
    return list(p & (V - I))


def get_PAu(Gamma, A, u, V, I, O, planes):
    p = A | LambdaPu("Y", u, V, O, planes) | LambdaPu("Z", u, V, O, planes)
    return list(V - p)


def get_YAu(Gamma, A, u, V, I, O, planes):
    p = LambdaPu("Y", u, V, O, planes)
    return list(p - A)


def LambdaPu(plane, u, V, O, planes):
    return {v for v in V - O if v != u and planes[v] == plane}


if __name__ == "__main__":
    import mentpy as mp

    gs = mp.GraphState()
    gs.add_edges_from(
        [
            (0, 5),
            (1, 5),
            (1, 6),
            (2, 6),
            (2, 7),
            (3, 7),
            (3, 8),
            (4, 8),
            (0, 9),
            (2, 9),
            (1, 10),
            (3, 10),
            (2, 11),
            (4, 11),
            (0, 12),
            (3, 12),
            (1, 13),
            (4, 13),
            (0, 14),
            (4, 14),
        ]
    )

    position = {
        0: (0, 0),
        1: (1, 0),
        2: (2, 0),
        3: (3, 0),
        4: (4, 0),
        5: (0.5, 0.5),
        6: (1.5, 0.5),
        7: (2.5, 0.5),
        8: (3.5, 0.5),
        9: (1, 1.5),
        10: (2, 1.5),
        11: (3, 1.5),
        12: (1.5, 2.5),
        13: (2.5, 2.5),
        14: (2, 4),
    }

    cond, p, d = find_pflow(
        gs, set([0, 1, 2, 3, 4]), set([0, 1, 2, 3, 4]), {v: "YZ" for v in gs.nodes}
    )

    print(cond, p, d)

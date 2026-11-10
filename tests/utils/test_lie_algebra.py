# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Tests for the lie algebra module."""

import pytest
import mentpy as mp


@pytest.mark.parametrize("n_wires", [2, 3, 4])
def test_lie_algebra_grid(n_wires):
    gs = mp.templates.grid_cluster(n_wires, n_wires + 3)
    lieAlg = mp.utils.calculate_lie_algebra(gs, max_iter=100000)
    assert (
        len(lieAlg) == mp.utils.dim_su(2**n_wires) + 1
    ), f"Grid cluster of size ({n_wires}, {n_wires + 3}) failed"


@pytest.mark.parametrize("n_wires", [4])
def test_lie_algebra_cylinder(n_wires):
    gs = mp.templates.grid_cluster(n_wires, n_wires + 3, periodic=True)
    extralegs = mp.templates.many_wires([2] * n_wires)
    gs = mp.hstack((extralegs, gs, extralegs))
    lieAlg = mp.utils.calculate_lie_algebra(gs, max_iter=100000)
    assert (
        len(lieAlg) == mp.utils.dim_so(2**n_wires) + 1
    ), f"Cylinder cluster of size ({n_wires}, {n_wires + 2}) failed"


def _string_closure(words):
    # Independent small-system reference using letters rather than GF(2) arrays.
    from collections import deque
    from itertools import combinations

    basis = list(dict.fromkeys(words))
    queue = deque(combinations(basis, 2))
    while queue:
        left, right = queue.popleft()
        if sum(x != y and x != "I" and y != "I" for x, y in zip(left, right)) % 2:
            product = "".join(
                (
                    y
                    if x == "I"
                    else (
                        x
                        if y == "I"
                        else "I" if x == y else next(iter(set("XYZ") - {x, y}))
                    )
                )
                for x, y in zip(left, right)
            )
            if product not in basis:
                queue.extend((product, old) for old in basis)
                basis.append(product)
    identity = "I" * len(words[0])
    if identity not in basis:
        basis.append(identity)
    return basis


def test_closure_matches_letter_reference():
    import numpy as np

    rng = np.random.default_rng(123)
    for qubits in (1, 2, 3, 4):
        for _ in range(5):
            words = ["".join(row) for row in rng.choice(list("IXYZ"), (5, qubits))]
            result = mp.utils.lie_algebra_completion(mp.PauliOp(words), 100000)
            assert result.txt.splitlines() == _string_closure(words)


@pytest.mark.parametrize("qubits", [63, 64, 65, 129, 1025])
def test_closure_crosses_packed_word_boundaries(qubits):
    import numpy as np

    # The only anticommuting position is beyond the previous packed block.
    x = "X" + "I" * (qubits - 2) + "X"
    z = "I" * (qubits - 1) + "Z"
    result = mp.utils.lie_algebra_completion(mp.PauliOp([x, z]), np.int64(3))
    assert result.txt.splitlines() == [x, z, x[:-1] + "Y", "I" * qubits]


def test_closure_limits_duplicates_and_empty_input():
    import numpy as np
    from mentpy.utils.lie_algebra import remove_repeated_ops

    ops = mp.PauliOp("X;Z;X")
    original = ops.matrix.copy()
    assert mp.utils.lie_algebra_completion(ops, 3).txt == "X\nZ\nY\nI"
    np.testing.assert_array_equal(ops.matrix, original)
    for limit in (1, 2):
        with pytest.raises(ValueError, match="Max iterations"):
            mp.utils.lie_algebra_completion(ops, limit)
    for limit in (0, -1, 1.5, True, np.bool_(False)):
        with pytest.raises(ValueError, match="positive integer"):
            mp.utils.lie_algebra_completion(ops, limit)
    for qubits in (0, 2):
        empty = mp.PauliOp(np.empty((0, 2 * qubits), dtype=int))
        assert len(remove_repeated_ops(empty)) == 0
        result = mp.utils.lie_algebra_completion(empty)
        assert len(result) == 1
        assert result.txt == "I" * qubits


def test_generators_accept_named_reordered_nodes():
    import networkx as nx
    import numpy as np

    original = mp.templates.grid_cluster(2, 5)
    expected = mp.utils.calculate_ordered_gens(original)
    labels = {node: f"v{100-node}" for node in original.graph}
    graph = nx.Graph()
    graph.add_nodes_from(labels[node] for node in reversed(list(original.graph)))
    graph.add_edges_from((labels[u], labels[v]) for u, v in original.graph.edges)
    renamed = mp.MBQCircuit(
        graph,
        [labels[n] for n in original.input_nodes],
        [labels[n] for n in reversed(original.output_nodes)],
        relabel_indices=False,
    )
    actual = mp.utils.calculate_ordered_gens(renamed)
    for index, node in enumerate(original.trainable_nodes):
        target = renamed.trainable_nodes.index(labels[node])
        np.testing.assert_array_equal(
            actual[target].matrix, expected[index].get_subset([1, 0]).matrix
        )


def test_generators_without_trainable_measurements():
    import networkx as nx

    fixed = mp.MBQCircuit(nx.path_graph(3), [0], [2], default_measurement=mp.Ment(0))
    assert len(mp.utils.calculate_complete_gens(fixed)) == 0
    assert len(mp.utils.calculate_gens(fixed)) == 0
    assert mp.utils.calculate_lie_algebra(fixed).txt == "I"
    with pytest.warns(UserWarning, match="No flow found"):
        no_flow = mp.MBQCircuit(nx.empty_graph(2), [0], [1])
    with pytest.raises(ValueError, match="without a flow"):
        mp.utils.calculate_gens(no_flow)


def test_native_closure_rejects_invalid_tableaux():
    from mentpy import _rust

    for data, rows, qubits in ((b"", 1, 1), (b"\x02\x00", 1, 1), (b"", 1, 2**63)):
        with pytest.raises(ValueError):
            _rust.lie_closure(data, rows, qubits, 100)

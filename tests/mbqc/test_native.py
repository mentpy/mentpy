# Copyright 2026 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Packed Rust GF(2) kernels against the independent NumPy implementation."""

import importlib
import numpy as np
import pytest
import networkx as nx

flow = importlib.import_module("mentpy.mbqc.flow")
linalg2 = importlib.import_module("mentpy.calculator.linalg2")


@pytest.mark.parametrize(
    "rows,cols",
    [
        (0, 0),
        (0, 65),
        (1, 1),
        (2, 1),
        (3, 7),
        (8, 8),
        (63, 65),
        (64, 64),
        (65, 129),
        (129, 129),
    ],
)
def test_native_matches_numpy(rows, cols, monkeypatch):
    native = pytest.importorskip("mentpy._rust")
    rng = np.random.default_rng(rows + cols)
    for _ in range(3):
        matrix = rng.integers(0, 2, (rows, cols), dtype=np.uint8)
        with monkeypatch.context() as patch:
            patch.setattr(flow, "_rust_ext", None)
            expected = flow._gf2_right_inverse_and_kernel(matrix)
            inverse = flow._gf2_inverse(matrix) if rows == cols else None
        actual = flow._gf2_right_inverse_and_kernel(matrix)
        for left, right in zip(actual, expected):
            if right is None:
                assert left is None
            else:
                np.testing.assert_array_equal(left, right)
        if actual[0] is not None:
            np.testing.assert_array_equal(
                (matrix @ actual[0]) & 1, np.eye(rows, dtype=np.uint8)
            )
            assert not np.any((matrix @ actual[1]) & 1)
        if rows == cols:
            actual_inverse = flow._gf2_inverse(matrix)
            if inverse is None:
                assert actual_inverse is None
            else:
                np.testing.assert_array_equal(actual_inverse, inverse)
        # Also cover the list APIs and modulo-two input normalization.
        if rows and cols:
            result = native.gf2_right_inverse_and_kernel(
                (matrix.astype(int) + 2).tolist()
            )
            assert (result is None) == (expected[0] is None)
            if result is not None:
                np.testing.assert_array_equal(
                    np.asarray(result[0]).reshape(cols, rows), expected[0]
                )
                np.testing.assert_array_equal(
                    np.asarray(result[1]).reshape(cols, cols - rows), expected[1]
                )


def test_native_rejects_invalid_dimensions():
    native = pytest.importorskip("mentpy._rust")
    with pytest.raises(ValueError, match="rectangular"):
        native.gf2_inverse([[1], [1, 0]])
    with pytest.raises(ValueError, match="square"):
        native.gf2_inverse([[1, 0]])
    with pytest.raises(ValueError, match="length"):
        native.gf2_inverse_bytes(b"", 3, 3)
    with pytest.raises(ValueError, match="overflow"):
        native.gf2_right_inverse_and_kernel_bytes(b"", 2**63, 2)
    with pytest.raises(ValueError, match="overflow"):
        native.gf2_right_inverse_and_kernel_bytes(b"", 0, 2**63)


def test_native_generalized_flow_matches_reference(monkeypatch):
    """Compare complete solves and verify their correction certificates."""
    pytest.importorskip("mentpy._rust")
    rng = np.random.default_rng(1201)
    found = missing = 0
    for n in range(1, 10):
        for _ in range(12):
            graph = nx.gnp_random_graph(n, 0.4, seed=int(rng.integers(2**31)))
            inputs = [v for v in graph if rng.random() < 0.25]
            outputs = [v for v in graph if rng.random() < 0.5]
            with monkeypatch.context() as patch:
                patch.setattr(flow, "_rust_ext", None)
                patch.setattr(linalg2, "_rust_ext", None)
                expected, _, depth, layers = flow.find_gflow(graph, inputs, outputs)
            actual, order, actual_depth, actual_layers = flow.find_gflow(
                graph, inputs, outputs
            )
            assert (actual is None) == (expected is None)
            if actual is None:
                missing += 1
                continue
            found += 1
            assert actual_depth == depth
            assert actual_layers == layers
            for node in set(graph) - set(outputs):
                np.testing.assert_array_equal(actual(node), expected(node))
                correction = set(np.flatnonzero(actual(node)))
                assert not correction.intersection(inputs)
                odd = flow.odd_neighborhood(graph, correction)
                assert node in odd
                assert all(
                    order(node, target) for target in correction | (odd - {node})
                )
    assert found > 20 and missing > 20


def test_native_pauli_flow_matches_reference(monkeypatch):
    """Compare all six measurement labels with independent Python algebra."""
    pytest.importorskip("mentpy._rust")
    rng = np.random.default_rng(1213)
    found = missing = 0
    for n in range(1, 10):
        for _ in range(16):
            graph = nx.gnp_random_graph(n, 0.45, seed=int(rng.integers(2**31)))
            inputs = {v for v in graph if rng.random() < 0.2}
            outputs = {v for v in graph if rng.random() < 0.45}
            planes = {v: rng.choice(["X", "Y", "Z", "XY", "XZ", "YZ"]) for v in graph}
            with monkeypatch.context() as patch:
                patch.setattr(flow, "_rust_ext", None)
                expected, _, _ = flow.find_pflow(graph, inputs, outputs, planes)
            actual, _, _ = flow.find_pflow(graph, inputs, outputs, planes)
            assert actual == expected
            if not actual:
                missing += 1
                continue
            found += 1
            c, relation, _, _ = flow._find_pflow_matrices(
                graph, inputs, outputs, planes
            )
            m, order, _, _ = flow._pflow_demand_matrices(graph, inputs, outputs, planes)
            np.testing.assert_array_equal((m @ c) & 1, np.eye(len(m), dtype=np.uint8))
            np.testing.assert_array_equal((order @ c) & 1, relation)
            assert flow._is_dag_adjacency(relation)
    assert found > 20 and missing > 20


@pytest.mark.parametrize("n,free", [(0, 3), (63, 1), (64, 2), (65, 4), (129, 1)])
def test_native_dag_solver_across_word_boundaries(n, free, monkeypatch):
    native = pytest.importorskip("mentpy._rust")
    rng = np.random.default_rng(n)
    # Construct a solvable instance by adding a known NR @ P to a DAG.
    nr = rng.integers(0, 2, (n, free), dtype=np.uint8)
    p = rng.integers(0, 2, (free, n), dtype=np.uint8)
    dag = np.triu(rng.integers(0, 2, (n, n), dtype=np.uint8), k=1)
    nl = dag ^ ((nr @ p) & 1)
    with monkeypatch.context() as patch:
        patch.setattr(flow, "_rust_ext", None)
        expected = flow._solve_dag_right_inverse(nl, nr)
    actual = flow._solve_dag_right_inverse(nl, nr)
    assert actual is not None and expected is not None
    assert flow._is_dag_adjacency(nl ^ ((nr @ actual) & 1))
    with pytest.raises(ValueError, match="length"):
        native.solve_dag(b"", b"", 3, 1)


def test_native_flow_input_validation():
    native = pytest.importorskip("mentpy._rust")
    with pytest.raises(ValueError, match="length"):
        native.pflow_matrices(b"", b"", 2, 3)
    with pytest.raises(ValueError, match="vertices"):
        native.gflow(bytes(4), 2, [2], [])
    with pytest.raises(ValueError, match="overflow"):
        native.pflow_matrices(b"", b"", 0, 2**63)
    assert native.pflow_matrices(b"", b"", 0, 2048) == (b"", b"")


def test_native_causal_flow_matches_reference(monkeypatch):
    native = pytest.importorskip("mentpy._rust")
    rng = np.random.default_rng(1217)
    for n in range(1, 12):
        for _ in range(12):
            graph = nx.gnp_random_graph(n, 0.3, seed=int(rng.integers(2**31)))
            graph = nx.relabel_nodes(graph, {i: f"node-{n-i}" for i in graph})
            inputs = {v for v in graph if rng.random() < 0.3}
            outputs = {v for v in graph if rng.random() < 0.5}
            with monkeypatch.context() as patch:
                patch.setattr(flow, "_rust_ext", None)
                expected, _, _, _ = flow.find_cflow(graph, inputs, outputs)
            actual, order, depth, layers = flow.find_cflow(graph, inputs, outputs)
            assert (actual is None) == (expected is None)
            if actual is not None:
                assert flow.check_if_cflow(graph, inputs, outputs, actual, order)
                assert depth == max(layers.values())
                assert all(actual(node) not in inputs for node in set(graph) - outputs)
    with pytest.raises(ValueError, match="vertices"):
        native.cflow([[3]], [], [])


@pytest.mark.parametrize("rows,cols", [(0, 3), (3, 0), (4, 7), (8, 2), (65, 129)])
def test_native_linear_solve_matches_reference(rows, cols, monkeypatch):
    native = pytest.importorskip("mentpy._rust")
    rng = np.random.default_rng(rows + cols)
    a = rng.integers(0, 2, (rows, cols), dtype=np.uint8)
    b = (a @ rng.integers(0, 2, cols, dtype=np.uint8)) & 1
    with monkeypatch.context() as patch:
        patch.setattr(linalg2, "_rust_ext", None)
        expected = linalg2.solve(a, b, check_solution=True)
    actual = linalg2.solve(a, b, check_solution=True)
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal((a @ np.asarray(actual).ravel()) & 1, b)
    assert linalg2.solve(a, b).shape == (cols,)
    with pytest.raises(ValueError, match="Right-hand side"):
        native.gf2_solve(bytes(4), b"", 2, 2)
    with pytest.raises(ValueError, match="Solution not found"):
        linalg2.solve(
            np.zeros((1, 1), dtype=np.uint8), np.ones(1, dtype=np.uint8), True
        )

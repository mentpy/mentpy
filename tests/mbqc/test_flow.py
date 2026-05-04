# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Tests for the flow module."""

import itertools

import pytest
import numpy as np

import mentpy as mp
from mentpy.mbqc import flow


def _assert_valid_algebraic_pflow(graph, inputs, outputs, planes):
    """Assert the algebraic Pauli-flow certificate conditions."""
    C, R, rows, columns = flow._find_pflow_matrices(graph, inputs, outputs, planes)
    M, N, matrix_rows, matrix_columns = flow._pflow_demand_matrices(
        graph, set(inputs), set(outputs), planes
    )

    assert rows == matrix_rows
    assert columns == matrix_columns
    assert flow._gf2_matmul(M, C).tolist() == np.eye(len(rows), dtype=int).tolist()
    assert flow._gf2_matmul(N, C).tolist() == R.tolist()
    assert flow._is_dag_adjacency(R)


def _bruteforce_dag_right_inverse(NL, NR):
    """Exhaustively solve ``NL + NR @ P`` for small GF(2) systems."""
    n_vertices = NL.shape[0]
    free_dim = NR.shape[1]

    for bits in itertools.product([0, 1], repeat=free_dim * n_vertices):
        P = np.array(bits, dtype=np.uint8).reshape(free_dim, n_vertices)
        relation = NL ^ flow._gf2_matmul(NR, P)
        if flow._is_dag_adjacency(relation):
            return P

    return None


def test_cflow():
    """Test the cflow function."""
    gs = mp.templates.linear_cluster(5).graph
    cond, _, _, _ = flow.find_cflow(gs, set([0]), set([4]))
    assert cond


def test_gflow():
    """Test the gflow function."""
    gs = mp.templates.linear_cluster(5).graph
    cond, _, _, _ = flow.find_gflow(gs, set([0]), set([4]))
    assert cond


def test_pflow():
    """Test the pflow function."""
    gs = mp.templates.linear_cluster(5).graph
    cond, _, _ = flow.find_pflow(
        gs, set([0]), set([4]), {v: "XY" for v in gs.nodes}
    )
    assert cond

    gs = mp.GraphState()
    gs.add_edges_from([(0, 1), (1, 2), (1, 3), (3, 4)])
    circ = mp.MBQCircuit(
        gs,
        input_nodes=[0],
        output_nodes=[2],
        measurements={
            0: mp.Ment("XY"),
            1: mp.Ment("X"),
            3: mp.Ment("XY"),
            4: mp.Ment("X"),
        },
    )
    circ.flow.initialize_flow()

    assert circ.flow.depth == 2

    gs = mp.GraphState()
    gs.add_edges_from(
        [(0, 1), (1, 2), (2, 3), (1, 4), (2, 4), (4, 5), (5, 2), (5, 6)]
    )
    circ = mp.MBQCircuit(
        gs,
        input_nodes=[0],
        output_nodes=[3, 6],
        measurements={
            0: mp.Ment("XY"),
            1: mp.Ment("XY"),
            2: mp.Ment("Y"),
            4: mp.Ment("YZ"),
            5: mp.Ment("XY"),
        },
    )

    circ.flow.initialize_flow()
    assert circ.flow.depth == 6

    gs = mp.GraphState()

    gs.add_edges_from(
        [
            (0, 3),
            (1, 3),
            (1, 4),
            (2, 4),
            (0, 5),
            (2, 5),
        ]
    )

    circ = mp.MBQCircuit(
        gs,
        input_nodes=[0, 1, 2],
        output_nodes=[0, 1, 2],
        measurements={3: mp.Ment("YZ"), 4: mp.Ment("YZ"), 5: mp.Ment("YZ")},
    )

    circ.flow.initialize_flow()
    assert circ.flow.depth == 1


def test_pflow_algebraic_square_case():
    """Test the O(n^3) square flow-demand algorithm."""
    gs = mp.GraphState()
    gs.add_edges_from([(0, 2), (0, 3), (1, 3)])
    planes = {0: "X", 1: "X"}

    cond, _, d = flow.find_pflow(gs, {0, 1}, {2, 3}, planes)

    assert cond
    assert d == {2: 0, 3: 0, 0: 1, 1: 1}
    _assert_valid_algebraic_pflow(gs, {0, 1}, {2, 3}, planes)


def test_pflow_algebraic_rectangular_case():
    """Test the O(n^3) general algorithm when there are more outputs."""
    gs = mp.GraphState()
    gs.add_edges_from(
        [(0, 1), (1, 2), (2, 3), (1, 4), (2, 4), (4, 5), (5, 2), (5, 6)]
    )
    planes = {0: "XY", 1: "XY", 2: "Y", 4: "YZ", 5: "XY"}

    cond, _, d = flow.find_pflow(gs, {0}, {3, 6}, planes)

    assert cond
    assert max(d.values()) == 3
    _assert_valid_algebraic_pflow(gs, {0}, {3, 6}, planes)


def test_pflow_dag_right_inverse_solver_matches_bruteforce():
    """Check the acyclic right-inverse step against exhaustive search."""
    rng = np.random.default_rng(20261030)

    for n_vertices in range(1, 5):
        for free_dim in range(0, 4):
            for _ in range(8):
                NL = rng.integers(
                    0, 2, size=(n_vertices, n_vertices), dtype=np.uint8
                )
                NR = rng.integers(
                    0, 2, size=(n_vertices, free_dim), dtype=np.uint8
                )

                expected = _bruteforce_dag_right_inverse(NL, NR)
                actual = flow._solve_dag_right_inverse(NL, NR)

                assert (actual is None) == (expected is None)
                if actual is not None:
                    relation = NL ^ flow._gf2_matmul(NR, actual)
                    assert flow._is_dag_adjacency(relation)


def test_pflow_rust_accelerated_gf2_helpers_match_numpy_fallback():
    """Validate optional Rust helpers when the extension is available."""
    if flow._rust_ext is None:
        pytest.skip("Rust extension is optional and was not built.")

    matrix = np.array(
        [
            [1, 0, 0, 1],
            [0, 1, 0, 1],
            [0, 0, 1, 1],
        ],
        dtype=np.uint8,
    )

    rust_right_inverse, rust_kernel = flow._gf2_right_inverse_and_kernel(matrix)

    previous = flow._rust_ext
    try:
        flow._rust_ext = None
        numpy_right_inverse, numpy_kernel = flow._gf2_right_inverse_and_kernel(
            matrix
        )
    finally:
        flow._rust_ext = previous

    assert rust_right_inverse.tolist() == numpy_right_inverse.tolist()
    assert rust_kernel.tolist() == numpy_kernel.tolist()
    assert flow._gf2_matmul(matrix, rust_right_inverse).tolist() == np.eye(
        matrix.shape[0], dtype=np.uint8
    ).tolist()

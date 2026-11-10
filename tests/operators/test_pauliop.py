# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Tests for the PauliOp class."""

import numpy as np
import pytest
from mentpy import PauliOp


def test_pauli_op_init_from_matrix():
    op = PauliOp(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [1, 0, 1, 0]]))
    assert op.txt == "XI\nIX\nIZ\nYI"


def test_pauli_op_init_from_string():
    op = PauliOp("XIZ;ZII;IIZ;YYY")
    print(op.matrix)
    assert np.array_equal(
        op.matrix,
        np.array(
            [
                [1, 0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1],
            ]
        ),
    )


def test_pauli_op_init_from_list():
    op = PauliOp(["XIZ", "ZII", "IIZ", "YYY"])
    assert np.array_equal(
        op.matrix,
        np.array(
            [
                [1, 0, 0, 0, 0, 1],
                [0, 0, 0, 1, 0, 0],
                [0, 0, 0, 0, 0, 1],
                [1, 1, 1, 1, 1, 1],
            ]
        ),
    )


def test_pauli_op_init_invalid_input():
    with pytest.raises(ValueError):
        PauliOp(42)


def test_pauli_op_getitem():
    op = PauliOp("XIZ;ZII;IIZ;IZI")
    assert op[0].txt == "XIZ"
    assert op[1].txt == "ZII"
    assert op[2].txt == "IIZ"
    assert op[3].txt == "IZI"


def test_pauli_op_get_subset():
    op = PauliOp("XIZ;ZII;IIZ;IZI")
    subset_op = op.get_subset([0, 2])
    assert subset_op.txt == "XZ\nZI\nIZ\nII"


def test_pauli_op_append():
    op1 = PauliOp("XIZ;IZI")
    op2 = PauliOp("XZZ")
    op1.append(op2)
    assert op1.txt == "XIZ\nIZI\nXZZ"


def test_pauli_op_contains():
    op1 = PauliOp("XIZ;IZI")
    op2 = PauliOp("XIZ")
    assert op2 in op1


def test_pauli_op_shapes_and_alphabet():
    assert PauliOp("X") != PauliOp("X;X")
    assert PauliOp("X") != "X"
    assert PauliOp("X") not in PauliOp("XX")
    assert PauliOp("X;Y")[:0].matrix.shape == (0, 2)
    with pytest.raises(ValueError, match="only I, X, Y and Z"):
        PauliOp("XA")


def test_pauli_rows_round_trip_and_boundaries():
    rng = np.random.default_rng(31)
    matrix = rng.integers(0, 2, (100, 130), dtype=np.uint8)
    op = PauliOp(matrix)
    np.testing.assert_array_equal(PauliOp(op.txt.replace("\n", ";")).matrix, matrix)
    assert len(list(op)) == 100
    assert op[-1] == PauliOp(matrix[-1:])
    assert op[np.int64(2)] == PauliOp(matrix[2:3])
    np.testing.assert_array_equal(op[4:1:-1].matrix, matrix[4:1:-1])
    for index in (-101, 100):
        with pytest.raises(IndexError):
            op[index]
    with pytest.raises(TypeError):
        op[0.5]
    zero = PauliOp(np.empty((2, 0), dtype=int))
    assert zero.txt == "\n"
    assert len(list(zero)) == 2

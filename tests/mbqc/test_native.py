# Copyright 2026 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Packed Rust GF(2) kernels against the independent NumPy implementation."""

import importlib
import numpy as np
import pytest

flow = importlib.import_module("mentpy.mbqc.flow")


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

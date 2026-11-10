# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""A module to store common linear algebra functions."""

import numpy as np
import galois

__all__ = ["solve"]


def solve(A, b, check_solution=False):
    """
    Solve a linear system of equations Ax = b over GF(2).

    Free variables are set to zero. With ``check_solution=True``, return a
    column vector and raise ValueError if the system is inconsistent.
    """
    GF = galois.GF(2)
    A = GF(A.astype(np.uint8))
    b = GF(b.astype(np.uint8)).reshape(-1)

    rows, cols = A.shape
    augmented_matrix = np.hstack((A, b.reshape(-1, 1)))

    row = 0
    pivot_columns = []
    for col in range(cols):
        # Find pivot in current column
        for pivot_row in range(row, rows):
            if augmented_matrix[pivot_row, col]:
                break
        else:
            continue

        # Swap rows if needed
        if pivot_row != row:
            augmented_matrix[[row, pivot_row]] = augmented_matrix[[pivot_row, row]]

        # Zero out below pivot
        for i in range(row + 1, rows):
            if augmented_matrix[i, col]:
                augmented_matrix[i] ^= augmented_matrix[row]

        pivot_columns.append(col)
        row += 1

        if row >= rows:
            break

    # Back-substitution
    x = GF.Zeros(cols)
    for row, col in reversed(list(enumerate(pivot_columns))):
        x[col] = augmented_matrix[row, -1] - np.sum(
            augmented_matrix[row, col + 1 : cols] * x[col + 1 : cols]
        )

    if check_solution:
        if not np.array_equal(A @ x, b):
            raise ValueError("Solution not found")
        return x.reshape(-1, 1)

    return x

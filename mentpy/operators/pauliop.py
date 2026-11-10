# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Module for representing Pauli operators."""

import operator

import numpy as np
from typing import Union, List
import galois

__all__ = ["PauliOp"]

GF = galois.GF(2)


class PauliOp:
    """Class for representing Pauli operators as matrices and strings.

    Parameters
    ----------
    op: Union[np.ndarray, str, List[str]]
        The Pauli operator to be represented. Can be a matrix, a string, or a list of strings.

    Examples
    --------
    Create a Pauli operator from a matrix

    .. ipython:: python

        op = mp.PauliOp(np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]]))
        print(op)

    Create a Pauli operator from a string

    .. ipython:: python

        op = mp.PauliOp('XIZ;ZII;IIZ;IZI')
        print(op)

    Create a Pauli operator from a list of strings

    .. ipython:: python

        op = mp.PauliOp(['XIZ', 'ZII', 'IIZ', 'IZI'])
        print(op)

    Group
    -----
    operators
    """

    def __init__(self, op: Union[np.ndarray, str, List[str]]):
        """Initialize a PauliOp object."""

        if isinstance(op, np.ndarray):
            if op.ndim != 2 or op.shape[1] % 2 != 0:
                raise ValueError(
                    "Tableau representation must be a matrix with an even number of columns"
                )
            self.matrix = GF(op)
            self.txt = self._mat_to_txt(op)

        elif isinstance(op, str):
            op = op.replace(" ", "")
            if op.endswith(";"):
                op = op[:-1]

            op_list = op.split(";")

            if not all([len(op) == len(op_list[0]) for op in op_list]):
                raise ValueError("All Pauli operators must be the same length")

            self.txt = op.replace(";", "\n")
            self.matrix = self._txt_to_mat(op_list)

        elif isinstance(op, list):
            if not all([len(pauliop) == len(op[0]) for pauliop in op]):
                raise ValueError("All Pauli operators must be the same length")

            self.txt = "\n".join(op)
            self.matrix = self._txt_to_mat(op)

        else:
            raise ValueError(
                "PauliOp must be initialized with a string or a numpy array"
            )

    def __repr__(self):
        return self.txt

    def __len__(self):
        return self.matrix.shape[0]

    def __getitem__(self, key):
        if isinstance(key, slice):
            return PauliOp(self.matrix[key])
        index = operator.index(key)
        if not -len(self) <= index < len(self):
            raise IndexError("Pauli operator index out of range")
        if len(self) == 1:
            return self
        return PauliOp(self.matrix[index][None, :])

    def __mul__(self, other: "PauliOp"):
        """Return the product of two Pauli operators modulo global phase."""
        if self.number_of_qubits != other.number_of_qubits:
            raise ValueError("Pauli operators must act on the same number of qubits.")
        new_matrix = GF(self.matrix) + GF(other.matrix)
        return PauliOp(new_matrix)

    def __contains__(self, item: "PauliOp"):
        if (
            not isinstance(item, PauliOp)
            or self.number_of_qubits != item.number_of_qubits
        ):
            return False
        for row in item.matrix:
            if not np.any((self.matrix == row).all(axis=1)):
                return False
        return True

    def __hash__(self):
        return hash(str(self.matrix))

    def __eq__(self, other: "PauliOp"):
        if not isinstance(other, PauliOp):
            return NotImplemented
        return np.array_equal(self.matrix, other.matrix)

    def _mat_to_txt(self, op):
        values = np.asarray(op, dtype=np.uint8)
        n_qubits = values.shape[1] // 2
        letters = np.frombuffer(b"IXZY", dtype=np.uint8)
        text = letters[values[:, :n_qubits] + 2 * values[:, n_qubits:]]
        return "\n".join(row.tobytes().decode("ascii") for row in text)

    def _txt_to_mat(self, op_list):
        if not op_list or any(not set(op) <= set("IXYZ") for op in op_list):
            raise ValueError("Pauli strings must contain only I, X, Y and Z.")
        letters = np.frombuffer("".join(op_list).encode("ascii"), dtype=np.uint8)
        letters = letters.reshape(len(op_list), len(op_list[0]))
        x = (letters == ord("X")) | (letters == ord("Y"))
        z = (letters == ord("Z")) | (letters == ord("Y"))
        return GF(np.concatenate((x, z), axis=1).astype(np.uint8))

    @property
    def number_of_qubits(self):
        """Returns the number of qubits in the Pauli operator."""
        return int(self.matrix.shape[1] // 2)

    def commutator(self, other) -> "PauliOp":
        """Returns the commutator of two Pauli operators."""
        if not self.symplectic_prod(other):
            return 0
        new_matrix = GF(self.matrix) + GF(other.matrix)
        return PauliOp(new_matrix)

    def symplectic_prod(self, other):
        """Returns the symplectic product of two Pauli operators."""
        x1 = self.matrix[:, : self.matrix.shape[1] // 2]
        x2 = other.matrix[:, : other.matrix.shape[1] // 2]
        z1 = self.matrix[:, self.matrix.shape[1] // 2 :]
        z2 = other.matrix[:, other.matrix.shape[1] // 2 :]
        return x1 @ z2.T + z1 @ x2.T

    def append(self, other):
        """Appends a Pauli operator to the end of another Pauli operator.

        Examples
        --------
        .. ipython:: python

            op1 = mp.PauliOp('XIZ;IZI')
            op2 = mp.PauliOp('XZZ')
            op1.append(op2)
            print(op1)
        """
        new_mat = np.vstack((self.matrix, other.matrix))
        self.__init__(new_mat)

    def get_subset(self, indices):
        """Returns a subset of the Pauli operator.

        Parameters
        ----------
        indices: List[int]
            The indices of the Pauli operators to be returned.

        Examples
        --------
        .. ipython:: python

            op = mp.PauliOp('XIZ;ZII;IIZ;IZI')
            print(op.get_subset([0, 2]))
        """
        inds = list(indices)
        if any(not 0 <= i < self.number_of_qubits for i in inds):
            raise ValueError("Index out of range")

        inds += [i + self.matrix.shape[1] // 2 for i in inds]
        return PauliOp(self.matrix[:, inds])

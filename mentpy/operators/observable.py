# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Coefficient-bearing observables for variational algorithms."""

from collections.abc import Iterable, Mapping
from numbers import Number

import numpy as np

try:
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - exercised when jax extra is absent
    jnp = None

from .pauliop import PauliOp

__all__ = ["Observable", "Hamiltonian"]


_PAULI_CHARS = set("IXYZ")


class Observable:
    """A weighted sum of Pauli strings.

    Parameters
    ----------
    terms : mapping or iterable
        Pauli terms. Mappings are interpreted as ``{"ZI": -1.0}``; iterables
        may contain either ``("ZI", -1.0)`` or ``(-1.0, "ZI")`` pairs.
    constant : number
        Scalar identity offset.

    Examples
    --------
    >>> h = Observable({"ZI": -1.0, "IZ": -1.0, "XX": 0.5})
    >>> energy = h.expectation(state)

    Group
    -----
    operators
    """

    def __init__(self, terms=None, constant=0.0):
        self.constant = constant
        self.terms = {}
        self.n_qubits = None

        if terms is None:
            return

        if isinstance(terms, str):
            self.add_term(terms, 1.0)
        elif isinstance(terms, Mapping):
            for pauli, coeff in terms.items():
                self.add_term(pauli, coeff)
        elif isinstance(terms, Iterable):
            for term in terms:
                if len(term) != 2:
                    raise ValueError("Observable terms must be length-2 pairs.")
                first, second = term
                if isinstance(first, str):
                    pauli, coeff = first, second
                else:
                    coeff, pauli = first, second
                self.add_term(pauli, coeff)
        else:
            raise TypeError(
                "Observable terms must be a Pauli string, mapping, or iterable."
            )

    def __repr__(self):
        pieces = [f"{coeff} * {pauli}" for pauli, coeff in self.terms.items()]
        if self.constant != 0 or not pieces:
            pieces.insert(0, f"{self.constant} * I")
        return " + ".join(pieces)

    @classmethod
    def from_pauliop(cls, pauli_op: PauliOp, coeffs=None, constant=0.0):
        """Build an observable from a tableau-style :class:`PauliOp`."""
        pauli_strings = [line for line in pauli_op.txt.split("\n") if line]
        if coeffs is None:
            coeffs = [1.0] * len(pauli_strings)
        if len(coeffs) != len(pauli_strings):
            raise ValueError("Number of coefficients must match Pauli strings.")
        return cls(zip(pauli_strings, coeffs), constant=constant)

    @classmethod
    def identity(cls, n_qubits, coeff=1.0):
        """Return ``coeff * I`` on ``n_qubits`` qubits."""
        obs = cls()
        obs.n_qubits = n_qubits
        obs.constant = coeff
        return obs

    def copy(self):
        """Return a shallow copy of this observable."""
        new = Observable()
        new.constant = self.constant
        new.terms = dict(self.terms)
        new.n_qubits = self.n_qubits
        return new

    def add_term(self, pauli, coeff=1.0):
        """Add ``coeff * pauli`` to this observable."""
        pauli = self._normalize_pauli(pauli)
        if not isinstance(coeff, Number):
            coeff = complex(coeff)

        if self.n_qubits is None:
            self.n_qubits = len(pauli)
        elif len(pauli) != self.n_qubits:
            raise ValueError(
                f"Expected Pauli string with {self.n_qubits} qubits, got {len(pauli)}."
            )

        self.terms[pauli] = self.terms.get(pauli, 0.0) + coeff
        if self.terms[pauli] == 0:
            del self.terms[pauli]
        return self

    def expectation(self, state, real=True):
        """Return the expectation value on a statevector or density matrix.

        ``state`` may be a NumPy array or a JAX array. JAX inputs keep the
        computation differentiable.
        """
        xp = self._namespace(state)
        state = xp.asarray(state)
        dtype = xp.result_type(state.dtype, xp.asarray(1j).dtype)
        state = xp.asarray(state, dtype=dtype)
        n_qubits = self._infer_state_qubits(state)
        self._validate_qubits(n_qubits)

        total = xp.asarray(self.constant, dtype=state.dtype)
        for pauli, coeff in self.terms.items():
            matrix = self.matrix(pauli, xp=xp, dtype=state.dtype)
            coeff = xp.asarray(coeff, dtype=state.dtype)
            if state.ndim == 1:
                total = total + coeff * xp.vdot(state, matrix @ state)
            elif state.ndim == 2:
                total = total + coeff * xp.trace(state @ matrix)
            else:
                raise ValueError("State must be a statevector or density matrix.")

        return xp.real(total) if real else total

    def matrix(self, pauli=None, xp=np, dtype=np.complex128):
        """Return a dense matrix for this observable or one Pauli string."""
        if pauli is not None:
            return self._pauli_matrix(pauli, xp=xp, dtype=dtype)

        self._validate_qubits(self.n_qubits)
        dim = 2**self.n_qubits
        out = xp.asarray(self.constant, dtype=dtype) * xp.eye(dim, dtype=dtype)
        for term, coeff in self.terms.items():
            out = out + xp.asarray(coeff, dtype=dtype) * self._pauli_matrix(
                term, xp=xp, dtype=dtype
            )
        return out

    def __call__(self, state, real=True):
        return self.expectation(state, real=real)

    def __add__(self, other):
        if isinstance(other, Number):
            new = self.copy()
            new.constant = new.constant + other
            return new
        if not isinstance(other, Observable):
            return NotImplemented
        new = self.copy()
        new.constant = new.constant + other.constant
        for pauli, coeff in other.terms.items():
            new.add_term(pauli, coeff)
        if new.n_qubits is None:
            new.n_qubits = other.n_qubits
        return new

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, scalar):
        if not isinstance(scalar, Number):
            return NotImplemented
        new = self.copy()
        new.constant = new.constant * scalar
        new.terms = {pauli: coeff * scalar for pauli, coeff in new.terms.items()}
        return new

    def __rmul__(self, scalar):
        return self.__mul__(scalar)

    def __sub__(self, other):
        return self + (-1 * other)

    def __rsub__(self, other):
        return (-1 * self) + other

    @staticmethod
    def _normalize_pauli(pauli):
        if not isinstance(pauli, str):
            raise TypeError("Pauli terms must be strings.")
        pauli = pauli.replace(" ", "").upper()
        if pauli == "":
            raise ValueError("Pauli string cannot be empty.")
        if not set(pauli) <= _PAULI_CHARS:
            raise ValueError(f"Invalid Pauli string {pauli!r}.")
        return pauli

    @staticmethod
    def _namespace(value):
        if jnp is not None and "jax" in type(value).__module__:
            return jnp
        return np

    @staticmethod
    def _infer_state_qubits(state):
        if state.ndim == 1:
            dim = state.shape[0]
        elif state.ndim == 2 and state.shape[0] == state.shape[1]:
            dim = state.shape[0]
        else:
            raise ValueError("State must be a statevector or square density matrix.")

        n_qubits = int(np.log2(dim))
        if 2**n_qubits != dim:
            raise ValueError("State dimension must be a power of two.")
        return n_qubits

    def _validate_qubits(self, n_qubits):
        if self.n_qubits is None:
            self.n_qubits = n_qubits
        if self.n_qubits != n_qubits:
            raise ValueError(
                f"Observable acts on {self.n_qubits} qubits, state has {n_qubits}."
            )

    @staticmethod
    def _pauli_matrix(pauli, xp=np, dtype=np.complex128):
        pauli = Observable._normalize_pauli(pauli)
        matrices = {
            "I": xp.eye(2, dtype=dtype),
            "X": xp.asarray([[0, 1], [1, 0]], dtype=dtype),
            "Y": xp.asarray([[0, -1j], [1j, 0]], dtype=dtype),
            "Z": xp.asarray([[1, 0], [0, -1]], dtype=dtype),
        }
        out = matrices[pauli[0]]
        for char in pauli[1:]:
            out = xp.kron(out, matrices[char])
        return out


Hamiltonian = Observable

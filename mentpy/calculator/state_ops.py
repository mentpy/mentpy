# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
import numpy as np
from numbers import Integral

__all__ = [
    "partial_trace",
    "pure2density",
    "partial_trace_pure_state",
    "partial_trace_mixed_state",
]


def pure2density(psi):
    """
    Convert a pure state to a density matrix.

    Note
    ----
    Given a pure state :math:`|\\psi\\rangle`, this function returns its density matrix
    representation :math:`|\\psi\\rangle\\langle\\psi|`.

    Args
    ----
    psi: np.ndarray
        The pure state represented as a state vector.

    Returns
    -------
    np.ndarray
        The density matrix representation of the given pure state.

    Group
    -----
    quantum_operations
    """
    return np.outer(psi, np.conj(psi))


def _trace_axes(dimension, indices):
    if dimension < 1 or dimension & (dimension - 1):
        raise ValueError("Quantum state dimension must be a power of two.")
    n = dimension.bit_length() - 1
    traced = list(indices)
    if any(not isinstance(i, Integral) or not 0 <= i < n for i in traced):
        raise ValueError("Traced qubit indices must be within the state.")
    if len(set(traced)) != len(traced):
        raise ValueError("Traced qubit indices must be unique.")
    keep = [i for i in range(n) if i not in traced]
    return n, keep, traced


def partial_trace_pure_state(psi, indices):
    """
    Partial trace over specified indices for pure states.

    Note
    ----
    This function assumes the input is a pure quantum state.

    Args
    ----
    psi: np.ndarray
        The pure state represented as a state vector.
    indices: list[int]
        The indices over which the partial trace should be computed.

    Returns
    -------
    np.ndarray
        Reduced density matrix, including when the result is pure.

    Group
    -----
    quantum_operations
    """
    psi = np.asarray(psi)
    if psi.ndim != 1:
        raise ValueError("Expected a state vector.")
    n, keep, traced = _trace_axes(psi.size, indices)
    amplitudes = psi.reshape((2,) * n).transpose(keep + traced)
    amplitudes = amplitudes.reshape(2 ** len(keep), 2 ** len(traced))
    return amplitudes @ amplitudes.conj().T


def partial_trace_mixed_state(rho, indices):
    """
    Partial trace over specified indices for mixed states.

    Note
    ----
    This function assumes the input is a mixed quantum state represented by a density matrix.

    Args
    ----
    rho: np.ndarray
        The mixed state represented as a density matrix.
    indices: list[int]
        The indices over which the partial trace should be computed.

    Returns
    -------
    np.ndarray
        Resulting state after performing the partial trace.

    Group
    -----
    quantum_operations
    """
    rho = np.asarray(rho)
    if rho.ndim != 2 or rho.shape[0] != rho.shape[1]:
        raise ValueError("Expected a square density matrix.")
    n, keep, traced = _trace_axes(rho.shape[0], indices)
    order = keep + traced
    tensor = rho.reshape((2,) * (2 * n)).transpose(order + [i + n for i in order])
    kept_size, traced_size = 2 ** len(keep), 2 ** len(traced)
    tensor = tensor.reshape(kept_size, traced_size, kept_size, traced_size)
    return np.trace(tensor, axis1=1, axis2=3)


def partial_trace(data, indices):
    """
    Perform a partial trace over specified indices for a quantum state.

    Note
    ----
    The function determines the type of the input data (pure or mixed state)
    and delegates the actual computation to one of the specific functions.

    Args
    ----
    data: np.ndarray
        The quantum state, either a state vector (pure state) or a density matrix (mixed state).
    indices: list[int]
        The indices over which the partial trace should be computed.

    Returns
    -------
    np.ndarray
        Reduced density matrix, including for state-vector input.

    See Also
    --------
    :func:`partial_trace_pure_state`, :func:`partial_trace_mixed_state`

    Group
    -----
    quantum_operations
    """
    data = np.asarray(data)
    if data.ndim == 1:
        return partial_trace_pure_state(data, indices)
    elif data.ndim == 2:
        return partial_trace_mixed_state(data, indices)
    else:
        raise ValueError("Invalid input shape for quantum state.")

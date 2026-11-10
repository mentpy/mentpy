# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""A module for generating random quantum data."""

import numpy as np
import scipy

__all__ = [
    "generate_haar_random_states",
    "generate_random_dataset",
    "generate_random_dataset_noisy",
    "train_test_split",
]


def _generate_haar_random_state(n_qubits: int) -> np.ndarray:
    r"""Makes one Haar random state over n_qubits"""

    real, imaginary = np.random.standard_normal((2, 2**n_qubits))
    state = real + 1j * imaginary
    return state / np.linalg.norm(state)


def generate_haar_random_states(n_qubits: int, n_samples: int = 1) -> np.ndarray:
    r"""Return a Haar random state, or a list when requesting multiple states.

    Group
    -----
    utils
    """

    if n_samples == 1:
        return _generate_haar_random_state(n_qubits)
    else:
        return [_generate_haar_random_state(n_qubits) for _ in range(n_samples)]


def generate_random_dataset(
    unitary: np.ndarray, n_samples: int, test_size: float = 0.3
) -> tuple:
    """
    Return random training and test data (input, target) for a given unitary gate ``unitary``.

    Group
    -----
    utils
    """
    n_qubits = int(np.log2(unitary.shape[0]))
    random_inputs = [_generate_haar_random_state(n_qubits) for _ in range(n_samples)]
    random_targets = [(unitary @ st.T).T for st in random_inputs]
    return train_test_split(random_inputs, random_targets, test_size=test_size)


def train_test_split(inputs, targets, test_size: float = 0.3, randomize=False) -> tuple:
    """Split the data into training and test sets.

    Group
    -----
    utils
    """
    n_samples = len(inputs)
    if len(targets) != n_samples:
        raise ValueError("Inputs and targets must have the same length.")
    if not 0 <= test_size <= 1:
        raise ValueError("test_size must be between 0 and 1.")
    n_test_samples = int(n_samples * test_size)
    n_train_samples = n_samples - n_test_samples

    if randomize:
        shuffled_indices = np.random.permutation(n_samples)
        inputs = (
            inputs[shuffled_indices]
            if isinstance(inputs, np.ndarray)
            else [inputs[i] for i in shuffled_indices]
        )
        targets = (
            targets[shuffled_indices]
            if isinstance(targets, np.ndarray)
            else [targets[i] for i in shuffled_indices]
        )

    return (inputs[:n_train_samples], targets[:n_train_samples]), (
        inputs[n_train_samples:],
        targets[n_train_samples:],
    )


def brownian_circuit(dim, n, dt):
    """Returns a random unitary matrix close to the identity matrix"""
    u = np.eye(dim)
    for j in range(n):
        re = np.random.normal(size=(dim, dim))
        im = 1j * np.random.normal(size=(dim, dim))
        c = re + im
        h = (c + np.conj(c.T)) / 4
        u = u @ scipy.linalg.expm(1j * h * dt)
    return u


def randomUnitary_closetoid(dim, t, n):
    """Returns a random unitary matrix close to the identity matrix"""
    return brownian_circuit(dim, n, np.sqrt(1 / (n * dim)) * 2 * np.pi * t)


def generate_random_dataset_noisy(
    unitary: np.ndarray,
    n_samples: int,
    noise_level: float = 0.0,
    noise_type="brownian",
    test_size: float = 0.3,
) -> tuple:
    r"""Return random training data (input, target) for a given unitary gate ``unitary`` with
    brownian noise parametrized by ``noise_level``.

    Args:
        unitary (np.ndarray): unitary gate
        n_samples (int): number of samples
        noise_level (float): noise level
        noise_type (str): type of noise. Either 'brownian' or 'bitflip'
        test_size (float): percentage of test data

    Group
    -----
    utils
    """
    if noise_level > 1 or noise_level < 0:
        raise ValueError("noise_level must be between 0 and 1")
    if noise_type not in {"brownian", "bitflip"}:
        raise ValueError(f"Unsupported noise type: {noise_type}.")

    n_qubits = int(np.log2(unitary.shape[0]))
    random_inputs = [_generate_haar_random_state(n_qubits) for _ in range(n_samples)]
    targets = [unitary @ state for state in random_inputs]
    (train_inputs, _), (test_inputs, test_targets) = train_test_split(
        random_inputs, targets, test_size=test_size
    )

    random_targets_train = []

    for state, target in zip(train_inputs, targets):
        if noise_type == "brownian":
            noise_u = randomUnitary_closetoid(2**n_qubits, noise_level, 20)
            random_targets_train += [noise_u @ unitary @ noise_u.conj().T @ state]

        elif noise_type == "bitflip":
            noise_u = 1
            for i in range(n_qubits):
                if np.random.rand() < noise_level:
                    noise_u = np.kron(noise_u, np.array([[0, 1], [1, 0]]))
                else:
                    noise_u = np.kron(noise_u, np.eye(2))
            random_targets_train.append(noise_u @ target)

    return (train_inputs, random_targets_train), (test_inputs, test_targets)

# Copyright 2026 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Optimize an H2 energy with an MBQC ansatz and plot its convergence.

Run ``python examples/vqe.py`` with ``mentpy[jax]`` installed.
The Hamiltonian was generated with OpenFermion/PySCF for H2 at 0.74 Angstrom,
STO-3G, Jordan-Wigner mapping, including nuclear repulsion.
"""

import matplotlib.pyplot as plt
import numpy as np
import mentpy as mp

# Ansatz
ansatz = mp.templates.from_pauli(mp.PauliOp("YXXX"))
initial_state = np.zeros(16)
initial_state[0b1100] = 1
angles = np.array([0.1])

# Hamiltonian
hamiltonian = mp.Observable(
    {
        "ZIII": 0.171412826447769,
        "IZII": 0.171412826447769,
        "IIZI": -0.223431536908134,
        "IIIZ": -0.223431536908134,
        "ZZII": 0.168688981703612,
        "ZIZI": 0.120625234833904,
        "ZIIZ": 0.165927850337703,
        "IZZI": 0.165927850337703,
        "IZIZ": 0.120625234833904,
        "IIZZ": 0.174412876122615,
        "XXYY": -0.045302615503799,
        "XYYX": 0.045302615503799,
        "YXXY": 0.045302615503799,
        "YYXX": -0.045302615503799,
    },
    constant=-0.097066268167632,
)

# Simulator
simulator = mp.PatternSimulator(ansatz, input_state=initial_state, backend="jax-tn")


def energy(parameters):
    """Return the energy of the output state in Hartree."""
    return simulator.expectation(parameters, hamiltonian)


# Optimize
steps = 100
optimizer = mp.optimizers.AdamOpt(step_size=0.05)
energies = [float(energy(angles))]
for iteration in range(steps):
    value, gradient = simulator.expectation_and_grad(angles, hamiltonian)
    angles = optimizer.step(energy, angles, iteration, gradient=gradient)
    energies.append(float(energy(angles)))

# Plot
exact_energy = float(np.linalg.eigvalsh(hamiltonian.matrix())[0])
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(np.arange(steps + 1), energies, label="MBQC ansatz")
ax.axhline(exact_energy, color="black", linestyle="--", label="Exact diagonalization")
ax.set_xlabel("Iteration")
ax.set_ylabel("Energy (Hartree)")
ax.legend()
fig.tight_layout()

if __name__ == "__main__":
    print(f"Energy after {steps} iterations: {energies[-1]:.8f} Hartree")
    print(f"Exact energy: {exact_energy:.8f} Hartree")
    plt.show()

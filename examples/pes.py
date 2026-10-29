# Copyright 2026 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Optimize H2 at several bond lengths and plot a potential-energy surface.

Run ``OMP_NUM_THREADS=1 python examples/pes.py`` with ``mentpy[chem,jax]``
installed. Geometry is in Angstrom; energies include nuclear repulsion.
"""

import matplotlib.pyplot as plt
import numpy as np
import mentpy as mp

# Ansatz
ansatz = mp.templates.from_pauli(mp.PauliOp("YXXX"))
initial_state = np.zeros(16)
initial_state[0b1100] = 1
bond_lengths = np.array([0.5, 0.74, 1.0, 1.5, 2.0])
steps = 100
energies = []
hf_energies = []
exact_energies = []

# Geometry sweep
for distance in bond_lengths:
    geometry = [("H", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, float(distance)))]
    hamiltonian, molecule = mp.chem.molecular_hamiltonian(geometry, basis="sto-3g")
    simulator = mp.PatternSimulator(ansatz, input_state=initial_state, backend="jax-tn")
    angles = np.array([0.1])
    optimizer = mp.optimizers.AdamOpt(step_size=0.05)

    def energy(parameters):
        """Return the energy at the current bond length."""
        return simulator.expectation(parameters, hamiltonian)

    for iteration in range(steps):
        value, gradient = simulator.expectation_and_grad(angles, hamiltonian)
        angles = optimizer.step(energy, angles, iteration, gradient=gradient)

    energies.append(float(energy(angles)))
    hf_energies.append(float(hamiltonian.expectation(initial_state)))
    exact_energies.append(float(np.linalg.eigvalsh(hamiltonian.matrix())[0]))

# Plot
fig, ax = plt.subplots(figsize=(6.5, 4))
ax.plot(bond_lengths, energies, "o-", label=f"MBQC ansatz ({steps} iterations)")
ax.plot(bond_lengths, exact_energies, "k--", label="Exact diagonalization")
ax.plot(bond_lengths, hf_energies, "s:", label="Hartree-Fock")
ax.set_xlabel("H-H distance (Angstrom)")
ax.set_ylabel("Energy (Hartree)")
ax.legend()
fig.tight_layout()

if __name__ == "__main__":
    for distance, value in zip(bond_lengths, energies):
        print(f"{distance:.2f} Angstrom: {value:.8f} Hartree")
    plt.show()

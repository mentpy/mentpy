Introduction to Measurement-based VQE
=====================================

.. meta::
    :description: Measurement-based VQE
    :keywords: quantum, quantum machine learning, measurement-based VQE, variational quantum eigensolver, MB-VQE

.. admonition:: Note
   :class: note

   The molecular examples use the ``jax`` and ``chem`` extras:
   ``pip install "mentpy[jax,chem]"``.

VQE-style observable costs
--------------------------

Variational MBQC experiments usually optimize measurement angles against a
Hamiltonian expectation value. MentPy represents these costs with
``Observable`` objects:

.. ipython:: python

   circuit = mp.templates.linear_cluster(5)
   hamiltonian = mp.Observable({"Z": 1.0})
   simulator = mp.PatternSimulator(circuit, backend="jax-tn")

   angles = np.zeros(len(circuit.trainable_nodes))
   energy, gradient = simulator.expectation_and_grad(angles, hamiltonian)
   print(energy)

The same objective can be optimized with the VQE helper:

.. ipython:: python

   trainer = mp.qml.VQE(circuit, hamiltonian, step_size=0.05)
   result = trainer.fit(np.ones(len(circuit.trainable_nodes)), steps=5)
   print(result.energy)

LiH active-space PES
--------------------

Installing ``mentpy[chem]`` enables the chemistry helpers under ``mp.chem``.
They provide OpenFermion/PySCF molecular Hamiltonians in MentPy's observable
format:

.. code-block:: python

   geometry = mp.chem.lih_geometry(1.6)
   hamiltonian, molecule = mp.chem.molecular_hamiltonian(
       geometry,
       basis="sto-3g",
       occupied_indices=[0],
       active_indices=[1, 2],
       run_mp2=True,
   )
   mp2_reference = mp.chem.mp2_energy(molecule)

The active-space arguments above freeze the Li 1s spatial orbital and keep two
spatial orbitals active. This gives a four-qubit, two-electron problem, which is
small enough to run as a quick MBQC-UCCSD example while still using PySCF and
OpenFermion for the molecular integrals.

The chemistry ansatz helper builds a first-order MBQC UCCSD Pauli-rotation
ansatz from OpenFermion's generator. Hamiltonians returned by
``molecular_hamiltonian`` carry the active-space metadata needed by the ansatz:

.. code-block:: python

   uccsd = mp.chem.uccsd(hamiltonian)
   n_active_electrons = uccsd.n_electrons
   hf_state = mp.chem.hartree_fock_state(
       hamiltonian.n_qubits,
       n_active_electrons,
   )
   trainer = mp.qml.VQE(uccsd.circuit, hamiltonian, input_state=hf_state)
   result = trainer.fit(uccsd.initial_angles(), steps=100)

The full MBQC-UCCSD ansatz is a stack of one MBQC Pauli-rotation template per
mapped UCCSD Pauli term. For the four-qubit, two-electron active-space LiH
example, one such layer can be drawn directly from the first generated Pauli
term:

.. ipython:: python

   active_lih_uccsd = mp.chem.uccsd(n_qubits=4, n_electrons=2)
   pauli_term = active_lih_uccsd.pauli_terms[0]
   layer = mp.templates.from_pauli(mp.PauliOp(pauli_term))
   @savefig lih_mbqc_uccsd_layer.png width=1400px
   mp.draw(
       layer,
       layout="pauli",
       figsize=(21, 11.2),
       title=f"One MBQC-UCCSD Pauli-rotation layer: {pauli_term}",
   )

For a bond-length sweep, keep the PES explicit: rebuild the LiH Hamiltonian for
each ``r``, construct the matching MBQC-UCCSD ansatz, then warm-start the next
VQE point from the previous optimum:

.. code-block:: python

   import matplotlib.pyplot as plt
   import numpy as np
   import mentpy as mp

   bond_lengths = np.linspace(1.2, 2.4, 7)
   occupied_indices = [0]
   active_indices = [1, 2]
   previous_angles = None
   mb_vqe = []
   mp2 = []

   for r in bond_lengths:
       hamiltonian, molecule = mp.chem.molecular_hamiltonian(
           mp.chem.lih_geometry(r),
           basis="sto-3g",
           occupied_indices=occupied_indices,
           active_indices=active_indices,
           run_mp2=True,
       )
       uccsd = mp.chem.uccsd(hamiltonian)
       hf_state = mp.chem.hartree_fock_state(
           hamiltonian.n_qubits,
           uccsd.n_electrons,
       )
       initial_angles = uccsd.initial_angles()
       if previous_angles is not None and previous_angles.shape == initial_angles.shape:
           initial_angles = previous_angles

       trainer = mp.qml.VQE(
           uccsd.circuit,
           hamiltonian,
           backend="jax-tn",
           step_size=0.03,
           input_state=hf_state,
       )
       result = trainer.fit(initial_angles, steps=80)
       previous_angles = result.angles
       mb_vqe.append(result.energy)
       mp2.append(mp.chem.mp2_energy(molecule))

   mb_vqe = np.array(mb_vqe)
   mp2 = np.array(mp2)

   plt.plot(bond_lengths, mb_vqe, "o-", label="MBQC-UCCSD VQE")
   plt.plot(bond_lengths, mp2, "s--", label="MP2")
   plt.xlabel("Li-H bond length (Angstrom)")
   plt.ylabel("Energy (Hartree)")
   plt.legend()

A seven-point run of this loop gives the following MBQC-UCCSD active-space PES.
The exact active-space curve is the dense diagonalization of the same four-qubit
Hamiltonian, while the Hartree-Fock and MP2 curves are PySCF references for the
underlying molecular calculation:

.. ipython:: python

   bond_lengths = np.array([1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4])
   mb_vqe = np.array([
       -7.83576636,
       -7.86072891,
       -7.86212710,
       -7.85041250,
       -7.83153309,
       -7.80905864,
       -7.78531319,
   ])
   mp2 = np.array([
       -7.84679703,
       -7.87221179,
       -7.87476887,
       -7.86485973,
       -7.84839367,
       -7.82889009,
       -7.80854161,
   ])
   hf = np.array([
       -7.83561583,
       -7.86053866,
       -7.86186477,
       -7.85001870,
       -7.83090558,
       -7.80799437,
       -7.78338163,
   ])
   exact_active = np.array([
       -7.83576681,
       -7.86072951,
       -7.86212883,
       -7.85041279,
       -7.83153363,
       -7.80905984,
       -7.78531362,
   ])
   fig, ax = plt.subplots(figsize=(7.6, 4.8), dpi=220)
   ax.plot(bond_lengths, mb_vqe, "o-", linewidth=2.2, markersize=6, label="MBQC-UCCSD VQE")
   ax.plot(bond_lengths, exact_active, color="black", linewidth=1.8, alpha=0.75, label="Exact active space")
   ax.plot(bond_lengths, mp2, "s--", linewidth=2.0, markersize=5.5, label="PySCF MP2")
   ax.plot(bond_lengths, hf, "^:", linewidth=1.9, markersize=5.5, label="PySCF HF")
   ax.set_xlabel("Li-H bond length (Angstrom)");
   ax.set_ylabel("Energy (Hartree)");
   ax.set_title("LiH PES: active-space MBQC-UCCSD and PySCF references");
   ax.grid(True, alpha=0.28, linewidth=0.8)
   ax.legend(frameon=False);
   fig.tight_layout()
   @savefig lih_mbqc_uccsd_pes.png width=1100px
   fig.canvas.draw()

PennyLane QPE for H2 as an MBQC Circuit
=======================================

.. meta::
   :description: Import a PennyLane quantum phase estimation circuit for H2 into MentPy and simulate the MBQC pattern
   :keywords: MBQC, PennyLane, QPE, H2, STO-3G, quantum chemistry, circuit import

.. admonition:: Note
   :class: note

   This tutorial uses PennyLane's quantum-chemistry tools and MentPy's
   tensor-network simulator. For a local run, install the chemistry and JAX
   extras with ``pip install "mentpy[chem,jax]"``.

The workflow below starts with an H2 Hamiltonian generated in PennyLane in the
STO-3G basis, builds a quantum phase estimation circuit from PennyLane
operations, imports that circuit into MentPy, and simulates the resulting
graph-state measurement pattern.

The full QPE import is intentionally shown as a local script rather than an
executed docs cell. MentPy lowers every imported PennyLane operation into MBQC
Pauli-rotation templates, so even a small chemistry QPE circuit expands into a
large graph state.

Build the H2 Hamiltonian in PennyLane
-------------------------------------

PennyLane can generate the four-qubit Jordan-Wigner Hamiltonian for H2 at a
bond length of ``0.742`` Angstrom:

.. code-block:: python

   import time

   import numpy as np
   import pennylane as qml
   import mentpy as mp

   symbols = ["H", "H"]
   coordinates = np.array(
       [
           [0.0, 0.0, -0.371],
           [0.0, 0.0, 0.371],
       ]
   )

   molecule = qml.qchem.Molecule(
       symbols,
       coordinates,
       basis_name="sto-3g",
       unit="angstrom",
   )
   pl_hamiltonian, n_system_qubits = qml.qchem.molecular_hamiltonian(molecule)

   h2_hamiltonian = mp.utils.from_pennylane(
       pl_hamiltonian,
       wire_order=range(n_system_qubits),
   )

The MentPy ``Observable`` is useful for dense reference calculations and for
iterating through the Pauli terms that define the controlled time evolution.

Build QPE from supported PennyLane operations
---------------------------------------------

The PennyLane importer currently accepts ``PauliRot``, ``RX``, ``RY``, ``RZ``,
``Hadamard``, ``CZ``, and ``CNOT`` operations. A controlled Pauli rotation can
therefore be written without using a backend-specific controlled-unitary gate:

.. math::

   |0\rangle\langle 0| \otimes I
   + |1\rangle\langle 1| \otimes e^{-i\theta P/2}
   =
   R_P(\theta/2) R_{ZP}(-\theta/2).

The script below uses that identity for each Pauli term in a first-order
Trotter step, then applies an inverse QFT written with ``Hadamard``, ``RZ``,
and ``PauliRot("ZZ")`` operations.

.. code-block:: python

   def active_pauli_word(pauli, wires):
       active = [(op, wire) for op, wire in zip(pauli, wires) if op != "I"]
       if not active:
           return "", []

       ops, active_wires = zip(*active)
       return "".join(ops), list(active_wires)


   def controlled_pauli_rotation(angle, pauli, control, wires):
       """Controlled exp(-i angle * P / 2) using supported PennyLane ops."""
       word, active_wires = active_pauli_word(pauli, wires)
       if not word:
           return [qml.RZ(-angle / 2, wires=control)]

       return [
           qml.PauliRot(angle / 2, word, wires=active_wires),
           qml.PauliRot(
               -angle / 2,
               "Z" + word,
               wires=[control] + active_wires,
           ),
       ]


   def controlled_hamiltonian_evolution(
       control,
       system_wires,
       hamiltonian,
       time,
       trotter_steps,
   ):
       ops = []
       dt = time / trotter_steps

       for _ in range(trotter_steps):
           if hamiltonian.constant:
               ops.append(qml.RZ(-dt * hamiltonian.constant, wires=control))

           for pauli, coeff in hamiltonian.terms.items():
               ops.extend(
                   controlled_pauli_rotation(
                       2 * dt * float(coeff),
                       pauli,
                       control,
                       system_wires,
                   )
               )

       return ops


   def controlled_phase(theta, control, target):
       return [
           qml.RZ(theta / 2, wires=control),
           qml.RZ(theta / 2, wires=target),
           qml.PauliRot(-theta / 2, "ZZ", wires=[control, target]),
       ]


   def inverse_qft_no_swaps(wires):
       ops = []
       for target in reversed(range(len(wires))):
           for control in reversed(range(target + 1, len(wires))):
               ops.extend(
                   controlled_phase(
                       -np.pi / 2 ** (control - target),
                       wires[control],
                       wires[target],
                   )
               )
           ops.append(qml.Hadamard(wires=wires[target]))
       return ops


   def h2_qpe_operations(
       hamiltonian,
       n_phase=3,
       energy_offset=-1.5,
       energy_width=1.0,
       trotter_steps=4,
   ):
       phase_wires = list(range(n_phase))
       system_wires = list(range(n_phase, n_phase + hamiltonian.n_qubits))
       wire_order = phase_wires + system_wires

       shifted_hamiltonian = mp.Observable(
           dict(hamiltonian.terms),
           constant=hamiltonian.constant - energy_offset,
       )
       base_time = -2 * np.pi / energy_width

       ops = [qml.Hadamard(wires=wire) for wire in phase_wires]
       for power, control in enumerate(phase_wires):
           ops.extend(
               controlled_hamiltonian_evolution(
                   control,
                   system_wires,
                   shifted_hamiltonian,
                   base_time * 2**power,
                   trotter_steps,
               )
           )

       ops.extend(inverse_qft_no_swaps(phase_wires))
       return ops, wire_order, phase_wires, system_wires


   qpe_ops, wire_order, phase_wires, system_wires = h2_qpe_operations(
       h2_hamiltonian,
       n_phase=2,
       energy_offset=-1.5,
       energy_width=1.0,
       trotter_steps=1,
   )

Import the PennyLane QPE circuit into MentPy
--------------------------------------------

The import call turns the PennyLane operation list into a concrete
``MBQCircuit``. Its ``graph`` is the graph state and its ``measurements``
dictionary contains the measurement bases and angles that simulate the QPE
circuit.

.. code-block:: python

   start = time.perf_counter()
   qpe_mbqc = mp.utils.from_pennylane(qpe_ops, wire_order=wire_order)
   mentpy_import_elapsed = time.perf_counter() - start

   print(qpe_mbqc)
   print(qpe_mbqc.graph.number_of_nodes())
   print(qpe_mbqc.graph.number_of_edges())
   print(qpe_mbqc.input_nodes)
   print(qpe_mbqc.output_nodes)

At this point ``qpe_mbqc`` is a graph-state MBQC implementation of the
PennyLane QPE circuit. Increasing ``n_phase`` or ``trotter_steps`` gives a
sharper phase estimate but also increases the imported graph.

Simulate the MBQC QPE pattern
-----------------------------

For the system-register input, use the Hartree-Fock state ``|1100>``. This is
not an exact eigenstate of the H2 Hamiltonian, so QPE returns the phase
distribution induced by the Hartree-Fock state's overlap with the exact
eigenvectors.

.. code-block:: python

   hf_state = mp.chem.hartree_fock_state(
       n_qubits=n_system_qubits,
       n_electrons=2,
   )
   phase_zero = np.zeros(2 ** len(phase_wires), dtype=complex)
   phase_zero[0] = 1.0
   input_state = np.kron(phase_zero, hf_state)

First run the PennyLane gate-model circuit so there is a direct reference for
the imported MBQC pattern:

.. code-block:: python

   dev = qml.device("default.qubit", wires=wire_order)

   @qml.qnode(dev)
   def pennylane_qpe():
       qml.StatePrep(input_state, wires=wire_order, normalize=True)
       for op in qpe_ops:
           qml.apply(op)
       return qml.state()

   start = time.perf_counter()
   pennylane_state = pennylane_qpe()
   pennylane_elapsed = time.perf_counter() - start

Then simulate the imported graph-state pattern with MentPy:

.. code-block:: python

   start = time.perf_counter()
   simulator = mp.PatternSimulator(
       qpe_mbqc,
       input_state=input_state,
       backend="jax-tn",
   )
   mentpy_setup_elapsed = time.perf_counter() - start

   start = time.perf_counter()
   mentpy_state = simulator.run([], output_form="sv")
   mentpy_elapsed = time.perf_counter() - start

The imported QPE circuit has fixed measurement angles, so the call to
``run`` receives an empty parameter list.

Decode the phase register
-------------------------

The inverse QFT above omits final swaps and the controlled powers use the same
wire convention, so the phase-register basis index can be decoded directly.
With two phase qubits and ``energy_width=1.0``, each phase bin corresponds to
``0.25`` Hartree. This is intentionally coarse so the full H2 import remains a
small local benchmark.

.. code-block:: python

   def phase_energy(state, energy_offset=-1.5, energy_width=1.0):
       n_phase = len(phase_wires)
       n_system = len(system_wires)
       output_tensor = np.asarray(state).reshape(2**n_phase, 2**n_system)
       phase_probabilities = np.sum(np.abs(output_tensor) ** 2, axis=1)
       phase_index = int(np.argmax(phase_probabilities))
       estimated_energy = (
           energy_offset + energy_width * phase_index / 2**n_phase
       )
       return phase_probabilities, phase_index, estimated_energy


   pl_probs, pl_index, pl_energy = phase_energy(pennylane_state)
   mp_probs, mp_index, mp_energy = phase_energy(mentpy_state)
   bit_width = len(phase_wires)

   print(np.round(pl_probs, 6))
   print(np.round(mp_probs, 6))
   print(f"PennyLane: {pl_index:0{bit_width}b}, {pl_energy:.6f} Ha")
   print(f"MentPy:    {mp_index:0{bit_width}b}, {mp_energy:.6f} Ha")
   print(f"max probability error: {np.max(np.abs(pl_probs - mp_probs)):.2e}")
   print(f"PennyLane run: {pennylane_elapsed:.4f} s")
   print(f"MentPy import: {mentpy_import_elapsed:.4f} s")
   print(f"MentPy setup:  {mentpy_setup_elapsed:.4f} s")
   print(f"MentPy run:    {mentpy_elapsed:.4f} s")

For ``n_phase=2`` and ``trotter_steps=1``, both backends should report the same
phase-register distribution, approximately
``[0.142157, 0.203967, 0.310280, 0.343597]``. They both decode to bitstring
``11`` and the same coarse energy estimate, ``-0.75`` Hartree. A local run on
this repository produced a PennyLane median statevector time of about
``4.40`` ms, while MentPy spent about ``1.71`` s importing the graph, ``0.18``
s constructing the tensor-network simulator, and about ``147`` ms per warmed
tensor-network run. More phase qubits improve the energy resolution; more
Trotter steps reduce product-formula error, but both increase the imported
graph size.

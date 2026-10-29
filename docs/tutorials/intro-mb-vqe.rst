Molecular energies with an MBQC ansatz
====================================

A variational energy calculation needs an ansatz, a Hamiltonian, and an
optimization loop. This example uses H₂ in the STO-3G basis. We first optimize
at a fixed geometry, then repeat the calculation over several bond lengths to
plot a potential-energy surface.

Install ``mentpy[jax]`` for the first example. The geometry sweep also uses
``mentpy[chem,jax]`` for the OpenFermion/PySCF molecular integrals.

Create the ansatz
-----------------

The four quantum outputs represent four spin orbitals in Jordan-Wigner order.
We start from the Hartree-Fock state ``|1100>``. A Pauli-rotation template for
``YXXX`` mixes this determinant with ``|0011>`` using one trainable angle.
This small ansatz is sufficient for this H₂ example; other molecules generally
need more rotations or a different ansatz.

.. literalinclude:: ../../examples/vqe.py
   :language: python
   :start-after: # Ansatz
   :end-before: # Hamiltonian

Define the measured operator
----------------------------

The cost is the expectation of the electronic Hamiltonian plus nuclear
repulsion. Here are the Pauli coefficients at an H-H distance of 0.74 Angstrom,
generated with OpenFermion/PySCF. Including the coefficients lets this first
example run without a chemistry installation.

.. literalinclude:: ../../examples/vqe.py
   :language: python
   :start-after: # Hamiltonian
   :end-before: # Simulator

The Pauli-string positions follow the circuit's quantum-output order. This
Hamiltonian is an ``Observable``; the simulator measures its expectation on the
output state.

.. literalinclude:: ../../examples/vqe.py
   :language: python
   :start-after: # Simulator
   :end-before: # Optimize

Optimize the angles
-------------------

Each iteration evaluates the energy and its gradient, then updates the angles
with Adam. The loop records the initial energy and the energy after every
update, so iteration zero and the final value are both visible in the plot.
``expectation_and_grad`` uses JAX autodiff and caches its compiled objective.

.. literalinclude:: ../../examples/vqe.py
   :language: python
   :start-after: # Optimize
   :end-before: # Plot

The complete script is :download:`vqe.py <../../examples/vqe.py>`. Run it with
``python examples/vqe.py`` to print the final energy and display the convergence
plot. The docs build executes the same script below.

.. ipython:: python

   import runpy
   from pathlib import Path
   example_dir = Path(mp.__file__).resolve().parents[1] / "examples"
   run = runpy.run_path(str(example_dir / "vqe.py"))
   energies = np.asarray(run["energies"])
   print(f"Energy after {run['steps']} iterations: {energies[-1]:.8f} Hartree")
   print(f"Exact energy: {run['exact_energy']:.8f} Hartree")
   plt.close(run["fig"])
   fig, ax = plt.subplots(figsize=(6.5, 4))
   ax.plot(np.arange(len(energies)), energies, label="MBQC ansatz");
   ax.axhline(run["exact_energy"], color="black", linestyle="--", label="Exact diagonalization");
   ax.set_xlabel("Iteration");
   ax.set_ylabel("Energy (Hartree)");
   ax.legend();
   @savefig h2-energy.png width=100%
   fig.tight_layout()

For four qubits, diagonalizing ``hamiltonian.matrix()`` gives a useful reference.
It uses the same Hamiltonian as the variational calculation. Dense
diagonalization is only a reference for small problems, since its memory grows
exponentially with the number of orbitals.

Energy versus geometry
----------------------

For a potential-energy surface, build the Hamiltonian at each geometry and run
the same optimization loop. The example below uses the same initial angle and
100 Adam updates at every bond length. It records the resulting energy rather
than assuming every point has converged.

.. literalinclude:: ../../examples/pes.py
   :language: python
   :start-after: # Ansatz
   :end-before: # Plot

Run :download:`pes.py <../../examples/pes.py>` with
``OMP_NUM_THREADS=1 python examples/pes.py``. The script computes the molecular
integrals, optimizes the angles, and displays the energy-versus-distance plot.
Changing the geometry, Hamiltonian, ansatz, or number of updates only requires
editing those parts of the example.

The following plot uses the :download:`recorded output <../../examples/data/h2-pes.csv>`
of that script. Keeping this small table lets the docs build show the result
without running PySCF. Both the Hartree-Fock and exact-reference curves use the
same Hamiltonian and include nuclear repulsion.

.. ipython:: python

   pes = np.loadtxt(example_dir / "data" / "h2-pes.csv", delimiter=",", comments="#")
   fig, ax = plt.subplots(figsize=(6.5, 4))
   ax.plot(pes[:, 0], pes[:, 1], "o-", label="MBQC ansatz (100 iterations)");
   ax.plot(pes[:, 0], pes[:, 3], "k--", label="Exact diagonalization");
   ax.plot(pes[:, 0], pes[:, 2], "s:", label="Hartree-Fock");
   ax.set_xlabel("H-H distance (Angstrom)");
   ax.set_ylabel("Energy (Hartree)");
   ax.legend();
   @savefig h2-pes.png width=100%
   fig.tight_layout()

Using finite shots
------------------

To estimate the cost from output measurement shots, define the cost with a shot
count and use parameter shift for its gradient. The ansatz, Hamiltonian,
optimizer, and loop structure stay the same:

.. code-block:: python

   rng = np.random.default_rng(7)

   def energy(parameters):
       seed = int(rng.integers(0, 2**32))
       return simulator.expectation(parameters, hamiltonian, shots=4096, seed=seed)

   angles = np.array([0.1])
   optimizer = mp.optimizers.AdamOpt(step_size=0.05)
   energies = [float(energy(angles))]
   for iteration in range(steps):
       gradient = mp.gradients.get_gradient(energy, angles, method="parameter-shift")
       angles = optimizer.step(energy, angles, iteration, gradient=gradient)
       energies.append(float(energy(angles)))

Each Pauli term receives 4096 shots. Finite-shot estimates fluctuate, so use an
independent evaluation of the final angles when comparing energies or
uncertainties across geometries. JAX autodiff applies to the exact expectation;
it does not differentiate randomly sampled bitstrings.

Simulating MBQC Circuits
========================

.. meta::
   :description: Learn how to simulate MBQC circuits in MentPy
   :keywords: mbqc, measurement-based quantum computation, quantum computing

In the previous tutorial, you have learned how to create an MBQC circuit using the 
:obj:`MBQCircuit` class. Now, we will use this circuit to run an actual experiment. 
We will use the :obj:`PatternSimulator` class to create a simulator. Let's try it out


.. ipython:: python

    grid_cluster = mp.templates.grid_cluster(3, 5)
    simulator = mp.PatternSimulator(grid_cluster, backend='numpy-dm')
    print(simulator)

You can specify the backend you want to use with the keyword argument ``backend``.
This can be useful if you want to run a circuit on real hardware.

.. admonition:: Note
   :class: warning
   
   Currently, ``mentpy`` does not support running circuits on real hardware. This feature will be added in the future.

For larger forced-outcome patterns, the ``jax-tn`` backend contracts the MBQC
graph as a tensor network with JAX arrays. This backend supports JAX autodiff and
can run on GPU when JAX is installed with CUDA support. It is most useful when
the measured resource graph has low treewidth or when the output boundary is
small enough to materialize.

.. ipython:: python

    tn_simulator = mp.PatternSimulator(grid_cluster, backend='jax-tn')
    output_state = tn_simulator(np.random.rand(len(grid_cluster.trainable_nodes)))
    print(output_state.shape)

Running the circuit
-------------------

To run the circuit, we can simply call ``simulator`` with the circuit as an argument.
If the measurement angle of a node is fixed (i.e. the measurement object :obj:`Ment` is
not trainable), you will not need to specify the measurement angle in the call to the
simulator.

.. ipython:: python

    num_angles = len(grid_cluster.trainable_nodes)
    output_state = simulator(np.random.rand(num_angles))
    print(output_state.shape)

Different inputs
----------------

If you want to run the circuit with a particular input state, you can specify it with the
keyword argument ``input_state``. 

.. ipython:: python

    random_state = mp.utils.generate_haar_random_states(3)
    simulator.reset(input_state = random_state)
    output_state = simulator(np.zeros(num_angles))
    print(output_state.shape)

Observable costs
----------------

The JAX tensor-network backend can evaluate Pauli-sum observables directly,
which is useful for variational algorithms.

.. ipython:: python

    wire = mp.templates.linear_cluster(5)
    observable = mp.Observable({"Z": 1.0})
    simulator = mp.PatternSimulator(wire, backend='jax-tn')
    value, gradient = simulator.expectation_and_grad(
        np.zeros(len(wire.trainable_nodes)),
        observable,
    )
    print(value)

Exact autodiff is the default when ``shots=None``. To emulate finite
measurement statistics while keeping the MBQC state preparation exact, pass a
shot count to the same observable API:

.. ipython:: python

    exact_value = simulator.expectation(
        np.zeros(len(wire.trainable_nodes)),
        observable,
    )
    shot_value = simulator.expectation(
        np.zeros(len(wire.trainable_nodes)),
        observable,
        shots=200,
        seed=7,
    )
    print(exact_value, shot_value)

The VQE helper uses this split too: ``gradient_method="auto"`` selects JAX
autodiff for exact ``jax-tn`` objectives and parameter-shift when ``shots`` is
set. You can still request ``"parameter-shift"``, ``"fd"``, or ``"jax"``
explicitly.

.. _clifford-t-long-wire:

Clifford+T rotations on a long wire
-----------------------------------

The ``jax-tn`` backend is especially useful for one-dimensional clusters. A
long 1D cluster can represent a compiled single-qubit rotation using only
discrete XY-plane measurement bases. In the standard one-way convention, an
XY measurement at angle ``alpha`` implements a ``J(alpha)`` step, so Clifford+T
circuits can be lowered to a chain whose fixed measurement angles are only
``0``, ``pi/2``, and ``pi/4``.

The optional ``pygridsynth`` package can synthesize a near-``T`` Z-rotation into
Clifford+T gates. The fallback string below is the sequence returned for this
example with tolerance ``1e-4``. Install it with ``pip install "mentpy[synthesis]"``
when you want to compile new angles. The compiler is asked for ``-target_theta``
because the positive measurement pairs below implement the adjoint ``S`` and
``T`` phases in this forced-outcome convention.

.. ipython:: python

    target_theta = np.pi / 4 + 1e-3
    epsilon = "1e-4"
    fallback_gates = (
        "HTHTSHTSHTHTSHTHTHTSHTSHTSHTSHTHTHTSHTSHTHTSHTHTHTHTHTHTHT"
        "SHTHTHTSHTHTSHTHTSHTSHTSHTHTSHTSHTSHTSHTHTHTSHTSHTHTSHWWWWW"
    )

    try:
        import mpmath
        from pygridsynth.gridsynth import gridsynth_gates
        gates = gridsynth_gates(
            theta=-(mpmath.mp.pi / 4 + mpmath.mpf("0.001")),
            epsilon=mpmath.mpf(epsilon),
        )
    except ImportError:
        gates = fallback_gates

    def clifford_t_to_1d_angles(gates):
        angles = []
        for gate in gates:
            if gate == "H":
                angles.append(0.0)
            elif gate == "S":
                angles.extend([np.pi / 2, 0.0])
            elif gate == "T":
                angles.extend([np.pi / 4, 0.0])
            elif gate == "X":
                angles.extend([0.0, np.pi / 2, 0.0, np.pi / 2, 0.0, 0.0])
            elif gate == "W":
                pass
            else:
                raise ValueError(f"Unsupported gate {gate!r}")
        return np.array(angles)

    angles = clifford_t_to_1d_angles(gates)
    wire = mp.templates.linear_cluster(len(angles) + 1)
    for node, angle in zip(wire.outputc, angles):
        wire[node] = mp.Ment(angle, "XY")

    print(f"Clifford+T gates: {len(gates)}")
    print(f"T gates: {gates.count('T')}")
    print(f"1D cluster nodes: {wire.graph.number_of_nodes()}")
    print(sorted({round(float(angle / np.pi), 2) for angle in angles}))

The circuit above has no trainable angles. It is a fixed, long measurement
pattern whose bases are only ``X``, ``Y``, and ``(X+Y)/sqrt(2)``. We can verify
the effective one-qubit unitary by simulating the two computational-basis inputs
with the tensor-network backend:

.. ipython:: python

    def simulated_unitary(circuit):
        columns = []
        for state in [np.array([1.0, 0.0]), np.array([0.0, 1.0])]:
            output = mp.PatternSimulator(
                circuit,
                input_state=state,
                backend="jax-tn",
            ).run([], output_form="sv")
            columns.append(np.asarray(output))
        return np.column_stack(columns)

    def rz(theta):
        return np.diag([np.exp(-0.5j * theta), np.exp(0.5j * theta)])

    def operator_error_up_to_phase(actual, target):
        phase = np.vdot(actual, target)
        phase = phase / abs(phase)
        return np.linalg.norm(phase * actual - target, ord=2)

    unitary = simulated_unitary(wire)
    error = operator_error_up_to_phase(unitary, rz(target_theta))
    print(error)

Scaling a one-dimensional tensor-network contraction
----------------------------------------------------

The tensor-network backend does not build the full ``2**n`` statevector for all
measured qubits. In a one-dimensional cluster with one input and one output, the
output state is still a single-qubit object even if the measured wire is long.
That is the kind of low-treewidth pattern where ``jax-tn`` can go far beyond a
dense statevector simulator.

This executed example keeps the docs build small while still using the same
code path as the long compiled rotation above:

.. ipython:: python

    large_n = 101
    large_wire = mp.templates.linear_cluster(large_n)
    fixed_angles = np.resize(
        [0.0, np.pi / 4, 0.0, np.pi / 2],
        len(large_wire.outputc),
    )

    for node, angle in zip(large_wire.outputc, fixed_angles):
        large_wire[node] = mp.Ment(float(angle), "XY")

    large_simulator = mp.PatternSimulator(
        large_wire,
        backend="jax-tn",
        output_form="sv",
    )
    large_state = large_simulator.run([])

    print(large_wire.graph.number_of_nodes())
    print(large_state.shape)
    print(np.linalg.norm(np.asarray(large_state)))

For a local benchmark, increase ``n_nodes`` in the same pattern. The following
block is intentionally not executed during the docs build:

.. code-block:: python

   import time
   import numpy as np
   import mentpy as mp

   n_nodes = 10_001
   wire = mp.templates.linear_cluster(n_nodes)
   angles = np.resize([0.0, np.pi / 4, 0.0, np.pi / 2], len(wire.outputc))
   for node, angle in zip(wire.outputc, angles):
       wire[node] = mp.Ment(float(angle), "XY")

   simulator = mp.PatternSimulator(wire, backend="jax-tn", output_form="sv")
   start = time.perf_counter()
   state = simulator.run([])
   elapsed = time.perf_counter() - start

   print(f"nodes: {wire.graph.number_of_nodes()}")
   print(f"output shape: {state.shape}")
   print(f"time: {elapsed:.3f}s")

For high-output patterns such as a full 100-qubit GHZ state, the output
representation itself is the limiting object: a dense statevector has
``2**100`` amplitudes. Use ``jax-tn`` for large measured resources with compact
outputs or observable costs, and keep dense output states small.

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
    simulator = mp.PatternSimulator(grid_cluster)
    print(simulator)

The default ``backend="rust"`` supports native tensor contraction, sampled
trajectories, and gradients of postselected patterns, including XY, XZ, YZ,
XYZ and Pauli measurements. ``backend="numpy"`` is a reference implementation.
``backend="jax-tn"`` remains available for code using JAX transformations.

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

Pass the measurement angles in ``trainable_nodes`` order. The simulator returns
a density matrix by default. Pass ``output_form="sv"`` for the statevector of
the selected branch.
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

The Rust tensor-network backend can evaluate Pauli-sum observables directly,
which is useful for variational algorithms.

.. ipython:: python

    wire = mp.templates.linear_cluster(5)
    observable = mp.Observable({"Z": 1.0})
    simulator = mp.PatternSimulator(wire, backend='rust')
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

Use ``expectation_and_grad`` for exact observable objectives. For finite-shot
objectives, differentiate a cost function with
``mp.gradients.get_gradient(cost, angles, method="parameter-shift")`` instead.
See :doc:`../tutorials/intro-mb-vqe` for an ansatz, a molecular Hamiltonian, and
an explicit optimization loop.

The Rust backend differentiates the normalized branch, so it includes changes
in postselection probability. Parameters follow ``trainable_nodes`` order;
XYZ measurements use a pair of angles and return a pair of derivatives.
With ``force0=True``, ``sample`` draws output shots from the selected branch.
With ``force0=False``, the Rust backend uses a lazy statevector trajectory and
resamples intermediate measurements for every output shot, applying available
flow corrections. Exact expectations in this mode describe one sampled branch,
not an ensemble average. Native gradients require ``force0=True``. The JAX
compatibility backend only supports postselected tensor contraction.

Native contractions represent default plus-state inputs implicitly and split
high-degree copy tensors into rank-three factors. ``max_elements`` limits the
size of an individual planned tensor; it is not a bound on total memory used
by the gradient tape. Returning a statevector still requires exponential
memory in the number of quantum outputs.

Pauli propagation
-----------------

When you need an expectation rather than the entire output state, use
``backend="pauli"`` for causal-flow patterns with X, Y or XY measurements.
The backend follows the flow paths to obtain logical rotations and CZ gates,
then propagates the observable backward as a sparse Pauli sum. Propagation and
reverse-mode gradients run in Rust; packed Pauli words and Clifford operations
come from `PPVM <https://github.com/QuEraComputing/ppvm>`_. These features are
included in the standard installation.

.. ipython:: python

    pattern = mp.templates.grid_cluster(2, 4)
    simulator = mp.PatternSimulator(pattern, backend="pauli")
    angles = np.full(len(pattern.trainable_nodes), 0.3)
    observable = mp.Observable({"ZZ": 0.7, "XY": -0.2})
    value, gradient = simulator.expectation_and_grad(angles, observable)
    print(value, gradient.shape)

This avoids an exponential output-state allocation and is useful for wide
patterns whose observables remain sparse. Non-Clifford rotations can still
produce exponentially many terms. ``max_terms`` limits the expansion and
``max_tape`` limits gradient storage; exceeding either raises an error without
truncating the result. ``peak_terms`` reports the largest expansion of the last
call. Gradients preserve terms whose value is zero but whose derivative is
nonzero, including at Clifford angles.

The current implementation supports up to 1024 logical wires, unmeasured
outputs, and arbitrary normalized input vectors in input-node order. Default
plus inputs and additional plus-state path roots stay implicit. Patterns with
only generalized flow or Pauli flow are not yet supported by this backend.
There is no approximate truncation mode yet.

With ``force0=False``, exact expectations describe a sampled corrected
trajectory. Finite-shot expectations sample the Pauli eigenvalue and resample
adaptive trajectories for each shot. Their distribution agrees with state
simulation; the same seed need not produce the same bit sequence across
different algorithms. Native gradients require ``force0=True``.

``run``, ``probabilities`` and joint ``sample`` calls use the Rust state
simulator, with its usual output-memory costs. Use ``expectation`` or
``expectation_and_grad`` to obtain the sparse-propagation benefits. Run
``python -m benchmarks.pauli`` to compare the two algorithms on your machine.

On Linux with an NVIDIA GPU, select ``device="cuda:0"`` to use cuPauliProp
through MentPy's native Rust interface. The standard Linux installation includes
cuPauliProp and the CUDA runtime; a compatible CUDA 13 driver is required. A
CUDA toolkit and a separate MentPy extra are not needed. CPU installations do
not load these libraries or initialize a GPU.

.. code-block:: python

    simulator = mp.PatternSimulator(pattern, backend="pauli", device="cuda:0")
    value, gradient = simulator.expectation_and_grad(angles, observable)

GPU expectations and reverse-mode differentiation preserve the same branch and
ordering conventions. Default plus-state readout stays on GPU. For an explicit
entangled input vector, its final Pauli expectations are computed in Rust on CPU
using the supplied vector, and their cotangents seed GPU backpropagation. This
avoids expanding the input into a density operator. State/probability/joint-shot
methods still use the CPU Rust state simulator.

``max_memory`` limits MentPy's owned GPU buffers, including the saved expansions
and workspace. It excludes the CUDA context, library internals and host metadata.
The default is 256 MiB. Exceeding this limit or ``max_terms`` raises an error;
the calculation does not silently truncate or switch devices. CPU gradients use
``max_tape`` instead. ``CUDA_VISIBLE_DEVICES`` is respected.

GPU launch and allocation costs dominate small patterns. Use
``python -m benchmarks.cupauli --device cuda:0`` to measure first-call latency,
warm runtime, and term growth against CPU propagation. The benchmark checks
expectations and derivatives before reporting timings.

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

Output probabilities and shots
------------------------------

``probabilities`` returns exact output probabilities. ``sample`` returns a
``Samples`` object with output bitstrings, counts, shot count, measurement basis,
and empirical frequencies. Output columns follow ``quantum_output_nodes``.

.. ipython:: python

    wire = mp.templates.linear_cluster(5)
    sample_sim = mp.PatternSimulator(wire, backend="jax-tn")
    angles = np.zeros(len(wire.trainable_nodes))
    sample_sim.probabilities(angles)
    samples = sample_sim.sample(angles, shots=1024, seed=7)
    samples.counts
    x_samples = sample_sim.sample(angles, shots=1024, basis="X", seed=7)
    x_samples.expectation

Measurement branches
--------------------

Both backends default to ``force0=True``, which selects outcome zero at every
intermediate measurement. A zero-probability branch raises an error. JAX raises
a runtime error when compiled execution reaches an impossible branch. MentPy
uses the precision configured by the application. Set ``JAX_ENABLE_X64=1`` before
starting Python when the calculation needs double precision.

Use ``backend="numpy", force0=False`` to sample intermediate outcomes. The
simulator applies the pattern's available flow corrections and evaluates
controlled measurements against the recorded outcomes. Each ``run()``
returns a single trajectory and exposes its outcomes in ``sim.outcomes``.
``probabilities()`` and exact ``expectation()`` refer to the state of that
trajectory. ``sample()`` and ``expectation(..., shots=...)`` draw an independent
trajectory for each shot in this mode.

The legacy backend names ``numpy-sv``, ``numpy-dm``, ``trajectory``, and
``pennylane`` select the unified NumPy simulator. Use
:doc:`../tutorials/import-export-circuits` for native PennyLane execution or
other framework exports.

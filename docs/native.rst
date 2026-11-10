Native execution
================

The standard installation includes the Rust extension and its Python interfaces.
Rust code lives in ``mentpy/_native``; users continue to build patterns with
``MBQCircuit`` and execute them with ``PatternSimulator``. JAX remains available
for existing code and arbitrary JAX objectives, but the default simulator's
``expectation_and_grad`` does not require tracing a JAX computation.

Implemented paths
-----------------

.. list-table::
   :header-rows: 1
   :widths: 25 45 30

   * - Workload
     - Execution
     - Limits
   * - Flow
     - Causal traversal, generalized flow, algebraic Pauli flow, and packed GF(2)
       solves in Rust. Generalized flow reuses elimination within each layer.
     - Graph labels and public objects are assembled in Python.
   * - Tensor simulation and gradients
     - Rust contraction planning, complex GEMM, normalization, observable action,
       and iterative reverse-mode differentiation.
     - Postselected branches; cost depends on the contraction intermediates and
       output size. Zero-probability branches raise.
   * - Sampled trajectories
     - Rust lazy graph preparation, measurements, corrections, output sampling,
       and batched shots.
     - Adaptive Python expressions are resolved between native quantum steps.
       Trajectories currently run on CPU.
   * - Pauli expectations and gradients
     - Packed PPVM words and Clifford operations, sparse Rust propagation, and
       coefficient adjoints. Linux wheels can load cuPauliProp at runtime.
     - Causal-flow X/Y/XY patterns, at most 1024 logical wires on CPU. Exact
       propagation raises on term/tape/memory limits; it never silently truncates.
   * - Lie closure and local-Clifford equivalence
     - Packed symplectic commutators, deduplication, GF(2) reduction, and
       certificate search in Rust.
     - Lie bases can grow exponentially. LC correspondence follows node labels,
       without searching over permutations; disconnected components are solved
       independently.

Backend choice
--------------

``backend="rust"`` is the default. It contracts the fixed zero-outcome branch
when ``force0=True`` and samples corrected measurement trajectories when
``force0=False``. Controlled angles use the recorded outcomes. Output order
follows the circuit's quantum output nodes. A fixed explicit seed repeats a
call, while a constructor seed initializes an advancing random stream.

``backend="pauli"`` is useful when an observable stays sparse under backward
propagation. It offers exact or finite-shot expectations and native analytic
gradients for the fixed branch. For full states, probabilities, and joint output
shots it delegates to the Rust state simulator; those methods do not inherit
sparse-observable scaling. Generalized/Pauli-flow *solving* is native, but
lowering those broader patterns to Pauli propagation remains future work.

NumPy remains an independent reference and compatibility backend. JAX remains
useful when composing existing JAX calculations. Rust gradients differentiate
MentPy's supported expectation queries; they do not automatically differentiate
arbitrary Python functions. An optimization loop can combine returned gradients
using the chain rule.

GPU availability
----------------

Linux x86-64 and ARM64 installs include the cuPauliProp 0.5 and CUDA 13 runtime
packages. ``backend="pauli", device="cuda:0"`` loads them only when requested.
A CUDA 13-compatible NVIDIA driver is required, but a toolkit is not needed.
Small sparse problems can be faster on CPU because GPU setup and transfers cost
more than the arithmetic.

The Candle CUDA tensor engine is also implemented and tested, but currently
requires the source build described in :doc:`getting-started`. Ordinary wheels
contain the CPU tensor engine and runtime-loaded GPU Pauli integration. They do
not yet provide the Candle CUDA tensor engine. Native sampled trajectories
remain CPU-only.

Validation and benchmarks
-------------------------

Tests compare tiny patterns with dense simulation and direct circuit results,
check normalized gradients against finite differences, cover all measurement
planes, and exercise impossible branches, adaptive outcomes, output ordering,
and repeated shots. Pauli derivative tests retain paths whose coefficient is
zero at Clifford angles. LC tests enumerate every pair of four-vertex graphs
and verify returned symplectic certificates.

The native wheels are built and smoke-tested on Linux x86-64/ARM64, Windows x86-64,
and macOS Intel/ARM. GPU tests are opt-in because hosted CI runners lack NVIDIA
hardware. The cuPauliProp and Candle paths have been exercised on an RTX PRO
6000 Blackwell GPU. Use ``MENTPY_TEST_CUPAULI=cuda:0`` for cuPauliProp tests and
``MENTPY_TEST_DEVICE=cuda:0`` for the tensor tests in a CUDA-enabled build.

Reproducible benchmarks live in ``benchmarks``. They report actual timings for
the chosen graph and hardware; Rust or GPU execution is not a universal speedup.
For example, wider Pauli expansions can favor the GPU while tensor contraction
can outperform either Pauli engine when the expansion becomes dense.

Resource states and further work
--------------------------------

The current quantum preparation model is still a qubit graph state with explicit
input states. General MPS resources, higher-dimensional sites, and local Kraus
instruments are not implemented. An AKLT wire requires that extension; a qubit
CZ graph should not be labeled an AKLT state. The proposed first resource
extension is a spin-1 AKLT MPS with measurement probabilities, conditional states,
and logical boundary readout. Two-dimensional AKLT and approximate propagation
need separate correctness and error-budget milestones.

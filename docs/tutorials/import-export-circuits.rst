Importing and Exporting Circuits
================================

.. meta::
   :description: Import and export circuits between MentPy and PennyLane, CUDA-Q, Cirq, and Qiskit
   :keywords: MBQC, interop, PennyLane, CUDA-Q, Cirq, Qiskit, circuit conversion

.. admonition:: Note
   :class: note

   Install the optional interop packages with ``pip install "mentpy[interop]"``.
   CUDA-Q is a separate optional extra on supported Python 3.11+ platforms:
   Linux or Apple silicon macOS for CPU simulation, with GPU acceleration on
   supported Linux systems: ``pip install "mentpy[cudaq]"``.

MentPy imports gate-model circuits by decomposing the supported operations into
MBQC Pauli-rotation templates. The current import subset is intentionally
explicit: single-qubit Pauli rotations, the common fixed Pauli phase gates, and
``H``, ``CZ``, and ``CNOT``/``CX``.

Wire order matters when moving between frameworks. Pass ``wire_order`` or
``qubit_order`` whenever the source framework has named wires or qubit objects.

Import circuits into MBQC
-------------------------

PennyLane operations or tapes can be imported directly:

.. code-block:: python

   import pennylane as qml
   import mentpy as mp

   pennylane_ops = [
       qml.Hadamard(wires=0),
       qml.CNOT(wires=[0, 1]),
       qml.RZ(0.2, wires=1),
   ]

   mbqc_circuit = mp.utils.from_pennylane(pennylane_ops, wire_order=[0, 1])

Cirq circuits use Cirq qubit objects for the order:

.. code-block:: python

   import cirq
   import mentpy as mp

   qubits = cirq.LineQubit.range(2)
   cirq_circuit = cirq.Circuit(
       cirq.H(qubits[0]),
       cirq.CNOT(qubits[0], qubits[1]),
       cirq.rz(0.2)(qubits[1]),
   )

   mbqc_circuit = mp.utils.from_cirq(cirq_circuit, qubit_order=qubits)

Qiskit ``QuantumCircuit`` objects use circuit qubit indices or qubit objects:

.. code-block:: python

   from qiskit import QuantumCircuit
   import mentpy as mp

   qiskit_circuit = QuantumCircuit(2)
   qiskit_circuit.h(0)
   qiskit_circuit.cx(0, 1)
   qiskit_circuit.rz(0.2, 1)

   mbqc_circuit = mp.utils.from_qiskit(qiskit_circuit, qubit_order=[0, 1])

CUDA-Q kernels are imported through CUDA-Q's OpenQASM 2 translation path, then
through the same supported gate subset:

.. code-block:: python

   import cudaq
   import mentpy as mp

   @cudaq.kernel
   def kernel(theta: float):
       q = cudaq.qvector(2)
       h(q[0])
       x.ctrl(q[0], q[1])
       rz(theta, q[1])

   mbqc_circuit = mp.utils.from_cudaq(kernel, n_qubits=2, parameters=[0.2])

Symbolic gate parameters in imported PennyLane, Cirq, or Qiskit circuits become
trainable MentPy measurement angles.

Export MBQC circuits
--------------------

An MBQC circuit can be exported to each supported backend:

.. code-block:: python

   import mentpy as mp

   mbqc_circuit = mp.templates.linear_cluster(5)

   pennylane_qnode = mp.utils.to_pennylane(mbqc_circuit)
   cirq_circuit = mp.utils.to_cirq(mbqc_circuit)
   qiskit_circuit = mp.utils.to_qiskit(mbqc_circuit)
   cudaq_kernel = mp.utils.to_cudaq(mbqc_circuit)

Cirq, Qiskit, and CUDA-Q exports preserve MBQC adaptivity with dynamic-circuit
constructs. Trainable measurement angles are exported as backend parameters by
default:

.. code-block:: python

   qiskit_circuit = mp.utils.to_qiskit(
       mbqc_circuit,
       parameter_prefix="theta",
   )

   fixed_qiskit_circuit = mp.utils.to_qiskit(
       mbqc_circuit,
       parameters=[0.0] * len(mbqc_circuit.trainable_nodes),
   )

Compile circuits with PyZX
--------------------------

Install the optional compiler dependency with ``pip install "mentpy[compiler]"``.
``mp.compile`` accepts QASM strings, PyZX circuits, and PyZX graphs. The first
compiler backend converts circuits to ZX diagrams, runs a PyZX reduction, and
returns both graph statistics and the reduced graph. Set ``extract=True`` when
you also want PyZX to extract an optimized circuit.

.. code-block:: python

   import mentpy as mp

   qasm = """OPENQASM 2.0;
   include "qelib1.inc";
   qreg q[1];
   h q[0];
   h q[0];
   """

   compiled = mp.compile(qasm, extract=True)
   print(compiled.before)
   print(compiled.after)

   reduced_graph = compiled.reduced
   reduced_qasm = compiled.circuit.to_qasm()

The current compiler API is deliberately small. It exposes PyZX-backed
``full_reduce``, ``teleport_reduce``, and ``clifford_simp`` reductions without
requiring PyZX at import time.

Exchange observables
--------------------

The same interop helpers also move Pauli Hamiltonians between MentPy and the
backend-native observable types:

.. code-block:: python

   import cirq
   import pennylane as qml
   from qiskit.quantum_info import SparsePauliOp
   import mentpy as mp

   hamiltonian = mp.Observable({"ZI": 1.0, "XX": -0.4}, constant=-1.2)

   pennylane_h = mp.utils.to_pennylane(hamiltonian, wire_order=[0, 1])
   cirq_h = mp.utils.to_cirq(hamiltonian, qubit_order=cirq.LineQubit.range(2))
   qiskit_h = mp.utils.to_qiskit(hamiltonian)

   roundtrip = mp.utils.from_qiskit(
       SparsePauliOp(["ZI", "XX", "II"], coeffs=[1.0, -0.4, -1.2])
   )

CUDA-Q spin operators are supported when CUDA-Q is installed:

.. code-block:: python

   cudaq_h = mp.utils.to_cudaq(hamiltonian, wire_order=[0, 1])
   mentpy_h = mp.utils.from_cudaq(cudaq_h, n_qubits=2)

Supported conversion rules
--------------------------

``from_pennylane`` supports ``PauliRot``, ``RX``, ``RY``, ``RZ``, ``Hadamard``,
``CZ``, and ``CNOT`` operations, plus PennyLane Hamiltonians and Pauli
operators.

``from_cirq`` supports ``PauliStringPhasor``, one-qubit ``X/Y/ZPowGate``
rotations, ``H``, ``CZ``, and ``CNOT``, plus ``PauliString`` and ``PauliSum``.

``from_qiskit`` supports ``QuantumCircuit`` objects with ``rx``, ``ry``,
``rz``, ``x``, ``y``, ``z``, ``s``, ``sdg``, ``t``, ``tdg``, ``sx``, ``sxdg``,
``h``, ``cz``, and ``cx``/``cnot`` gates, plus ``SparsePauliOp`` and ``Pauli``.
Barriers and terminal measurements are skipped during unitary circuit import.

``from_cudaq`` supports CUDA-Q spin operators. Kernel import requires CUDA-Q's
OpenQASM 2 translation API and the same imported gate subset used by Cirq.

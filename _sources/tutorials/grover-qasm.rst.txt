Grover search from OpenQASM
===========================

This example searches for ``11111`` among 32 bitstrings. We define the gate
circuit in Qiskit, export OpenQASM 2, and import it as an MBQC pattern. The same
steps work with PennyLane; an example follows below.

Install the optional dependencies first:

.. code-block:: bash

   pip install "mentpy[compiler,interop]"

The complete example is available in :download:`grover.py <../../examples/grover.py>`.

Define and export the circuit
-----------------------------

The oracle changes the sign of ``|11111>``. Four Grover iterations give a
success probability of about 0.99918. We leave the output qubits unmeasured;
MentPy will measure them after executing the MBQC pattern.

.. ipython:: python

   from qiskit import QuantumCircuit, transpile, qasm2
   from qiskit.quantum_info import Statevector
   n = 5
   circuit = QuantumCircuit(n)
   circuit.h(range(n));

   def oracle():
       circuit.h(n - 1)
       circuit.mcx(list(range(n - 1)), n - 1)
       circuit.h(n - 1)
       return None

   for _ in range(4):
       oracle()
       circuit.h(range(n))
       circuit.x(range(n))
       oracle()
       circuit.x(range(n))
       circuit.h(range(n))

   circuit = transpile(circuit, basis_gates=["h", "rz", "cx"], optimization_level=1)
   qasm = qasm2.dumps(circuit)
   pattern = mp.from_qasm(qasm)
   len(pattern), len(pattern.trainable_nodes)

``from_qasm`` returns an :class:`mp.MBQCircuit` with fixed measurement angles,
input and output nodes, a measurement order, and causal flow. It also accepts a
path to a QASM file. Gate phases are represented by XY measurement angles;
Hadamards use one-bit teleportation and entanglers use graph edges.

Inspect the graph
-----------------

The graph contains thousands of measured nodes but only five quantum outputs.
The following layout puts each logical wire on a row. A complete overview and
the first 50 nodes show how teleportation chains and entangling edges fit
together.

.. ipython:: python

   successors = {node: pattern.flow(node) for node in pattern.outputc}
   rows = {}
   for wire, node in enumerate(pattern.input_nodes):
       while node is not None:
           rows[node] = -wire
           node = successors.get(node)

   columns = nx.utils.UnionFind(pattern.graph.nodes)
   for u, v in pattern.graph.edges:
       if successors.get(u) != v and successors.get(v) != u:
           columns.union(u, v)

   dependencies = nx.DiGraph()
   dependencies.add_nodes_from(columns[node] for node in pattern.graph)
   dependencies.add_edges_from((columns[u], columns[v]) for u, v in successors.items())
   order = list(nx.topological_sort(dependencies))
   layers = {}
   for column in order:
       layers[column] = 1 + max((layers[p] for p in dependencies.predecessors(column)), default=-1)

   positions = {node: (layers[columns[node]], rows[node]) for node in pattern.graph}
   fig, axes = plt.subplots(2, 1, figsize=(12, 5))
   nx.draw(pattern.graph, pos=positions, ax=axes[0], node_size=2, width=0.2, with_labels=False)
   axes[0].set_title("Complete MBQC graph")
   nodes = list(pattern.graph.nodes)[:50]
   nx.draw(pattern.graph.subgraph(nodes), pos=positions, ax=axes[1], node_size=65, font_size=6, with_labels=True)
   axes[1].set_title("First 50 nodes")
   @savefig grover-mbqc.png width=100%
   plt.tight_layout()

Execute and collect statistics
------------------------------

MentPy uses ``|+>`` inputs by default. To reproduce Qiskit's ``|00000>`` input,
we supply it explicitly. The ``trajectory`` backend prepares graph qubits as
needed and removes each measured qubit. Its memory depends on the active
frontier rather than the total graph size. It supports XY patterns with causal
flow, including this compiled circuit.

.. ipython:: python

   initial = np.zeros(2**n)
   initial[0] = 1
   sim = mp.PatternSimulator(pattern, input_state=initial, backend="trajectory")
   probabilities = sim.probabilities()
   samples = sim.sample(shots=4096, seed=7)
   probabilities[-1], samples.counts["11111"], samples.shots
   reference = Statevector.from_instruction(circuit).reverse_qargs().probabilities()
   assert np.allclose(probabilities, reference, atol=1e-8)
   samples.probabilities["11111"]

Columns and Pauli strings follow ``pattern.quantum_output_nodes``. The first
output is the leftmost bit, corresponding to wire zero here. Qiskit displays
wire zero on the right, so the reference above reverses its tensor axes. The
marked state is symmetric, but this distinction matters for other bitstrings.

``samples.bitstrings`` has shape ``(4096, 5)``. ``samples.counts`` records how
many times each bitstring occurred, and ``samples.probabilities`` contains the
observed frequencies. The seed makes output sampling reproducible.

We can also measure a Pauli string or evaluate a weighted observable:

.. ipython:: python

   observable = mp.Observable({"ZIIII": 0.5, "IIIIZ": 0.5})
   sim.expectation([], observable)
   sim.expectation([], observable, shots=4096, seed=7)
   x_samples = sim.sample(shots=4096, basis="XXXXX", seed=7)
   x_samples.expectation, x_samples.standard_error

Each noncommuting Pauli term in ``Observable`` uses a separate set of shots.
A sample result with a product basis records joint outcomes in that basis.
Finite shots add output measurement noise; the state preparation itself uses
exact complex amplitudes. To sample intermediate MBQC measurements as well,
construct the trajectory simulator with ``force0=False, seed=7``. It applies
flow corrections, so the corrected output distribution is unchanged.

Export from PennyLane
---------------------

PennyLane can export the same search using ``qml.to_openqasm``. Use a tape
without terminal measurements and ``measure_all=False`` so that MentPy receives
a unitary circuit. Older PennyLane versions provide the export method on the
tape; recent versions use ``qml.to_openqasm``. Top-level global phases are
ignored because they do not affect measurement statistics.

.. ipython:: python

   import pennylane as qml
   with qml.tape.QuantumTape() as tape:
       for wire in range(n):
           qml.Hadamard(wire)
       for _ in range(4):
           qml.Hadamard(n - 1)
           qml.MultiControlledX(wires=range(n))
           qml.Hadamard(n - 1)
           qml.GroverOperator(wires=range(n))

   pl_qasm = (
       tape.to_openqasm(measure_all=False)
       if hasattr(tape, "to_openqasm")
       else qml.to_openqasm(tape, measure_all=False)
   )

   pl_pattern = mp.from_qasm(pl_qasm)
   pl_sim = mp.PatternSimulator(pl_pattern, input_state=initial, backend="trajectory")
   assert np.allclose(pl_sim.probabilities(), reference, atol=1e-8)
   pl_sim.sample(shots=4096, seed=7).counts["11111"]

Optional ZX simplification
--------------------------

``mp.compile(qasm)`` applies PyZX's ``full_reduce`` strategy, extracts a gate
circuit, and translates it to an executable pattern in ``result.pattern``.
``result.before`` and ``result.after`` contain ZX graph statistics, while
``result.circuit`` is the extracted PyZX circuit. Reduction can change the
size of the resulting pattern in either direction; a smaller ZX diagram does
not necessarily give a smaller teleportation pattern.

.. ipython:: python

   small_qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\nh q[0];\nh q[0];\n'
   result = mp.compile(small_qasm)
   result.before["input_gates"], result.after["extracted_gates"]

``mp.from_qasm(qasm, optimize=True)`` returns only the optimized MBQC pattern.
``mp.compile(qasm, to_mbqc=False)`` retains just the ZX reduction result; add
``extract=True`` to request an extracted circuit without a pattern.

The importer accepts unitary OpenQASM 2 programs whose gates PyZX can decompose
into Hadamards, X/Z phases, CNOTs and CZs. Measurements, resets, classical
conditions and unbound parameters are rejected. Bind parameters before export.
This is ZX simplification followed by gate-to-pattern translation; it does not
translate arbitrary ZX diagrams directly to measurement patterns.

References
----------

* `Qiskit OpenQASM export <https://quantum.cloud.ibm.com/docs/en/guides/interoperate-qiskit-qasm2>`_
* `PennyLane OpenQASM export <https://docs.pennylane.ai/en/stable/code/api/pennylane.to_openqasm.html>`_
* `PyZX circuit simplification <https://pyzx.readthedocs.io/en/latest/notebooks/simplify.html>`_

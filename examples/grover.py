# Copyright 2026 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Search for |11111> with Qiskit/PennyLane, QASM, and MBQC.

Run from an editable install with ``python examples/grover.py``.
"""

import numpy as np
import mentpy as mp


def qiskit_grover(n=5, iterations=4):
    """Return an unmeasured Grover circuit in a portable QASM gate set."""
    from qiskit import QuantumCircuit, transpile

    circuit = QuantumCircuit(n)
    circuit.h(range(n))

    def oracle():
        circuit.h(n - 1)
        circuit.mcx(list(range(n - 1)), n - 1)
        circuit.h(n - 1)

    for _ in range(iterations):
        oracle()
        circuit.h(range(n))
        circuit.x(range(n))
        oracle()
        circuit.x(range(n))
        circuit.h(range(n))
    return transpile(circuit, basis_gates=["h", "rz", "cx"], optimization_level=1)


def pennylane_grover(n=5, iterations=4):
    """Return the same Grover search as a PennyLane QuantumScript."""
    import pennylane as qml

    with qml.tape.QuantumTape() as tape:
        for wire in range(n):
            qml.Hadamard(wire)
        for _ in range(iterations):
            qml.Hadamard(n - 1)
            qml.MultiControlledX(wires=range(n))
            qml.Hadamard(n - 1)
            qml.GroverOperator(wires=range(n))
    return tape


def to_qasm(tape):
    """Export a PennyLane tape across supported PennyLane versions."""
    import pennylane as qml

    if hasattr(tape, "to_openqasm"):
        return tape.to_openqasm(measure_all=False)
    return qml.to_openqasm(tape, measure_all=False)


def simulate(qasm, n=5):
    """Compile QASM and return the graph, exact probabilities, and output shots."""
    pattern = mp.from_qasm(qasm)
    initial = np.zeros(2**n)
    initial[0] = 1
    sim = mp.PatternSimulator(pattern, input_state=initial, backend="trajectory")
    probabilities = sim.probabilities()
    samples = sim.sample(shots=4096, seed=7)
    return pattern, probabilities, samples


if __name__ == "__main__":
    from qiskit import qasm2
    import pennylane as qml

    for name, qasm in [
        ("Qiskit", qasm2.dumps(qiskit_grover())),
        ("PennyLane", to_qasm(pennylane_grover())),
    ]:
        pattern, probabilities, samples = simulate(qasm)
        print(name, len(pattern), probabilities[-1], samples.counts)

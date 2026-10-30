# Copyright 2026 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Independent gate-circuit checks for executable MBQC compilation."""

import numpy as np
import pytest
import mentpy as mp

qiskit = pytest.importorskip("qiskit")
pytest.importorskip("pyzx")
from qiskit import QuantumCircuit, qasm2
from qiskit.quantum_info import Statevector


def density(state):
    return np.outer(state, state.conj())


@pytest.mark.parametrize("optimize", [False, True])
def test_qasm_matches_qiskit_for_entangled_input(optimize):
    rng = np.random.default_rng(19)
    circuit = QuantumCircuit(3)
    circuit.h(1)
    circuit.rx(0.37, 0)
    circuit.ry(-0.53, 2)
    circuit.rz(0.71, 1)
    circuit.cx(2, 0)
    circuit.cz(0, 1)
    circuit.cz(0, 1)
    circuit.swap(1, 2)
    circuit.x(0)
    circuit.sdg(1)
    circuit.t(2)
    initial = rng.normal(size=8) + 1j * rng.normal(size=8)
    initial /= np.linalg.norm(initial)
    expected = Statevector(initial).reverse_qargs().evolve(circuit).reverse_qargs().data
    pattern = mp.from_qasm(qasm2.dumps(circuit), optimize=optimize)
    assert not pattern.trainable_nodes
    for backend in ["trajectory", "jax-tn"]:
        sim = mp.PatternSimulator(pattern, input_state=initial, backend=backend)
        np.testing.assert_allclose(sim.run([]), density(expected), atol=1e-9)
    sim = mp.PatternSimulator(
        pattern, input_state=initial, backend="trajectory", force0=False, seed=3
    )
    outcomes = set()
    for _ in range(5):
        np.testing.assert_allclose(sim.run([]), density(expected), atol=1e-9)
        outcomes.update(sim.outcomes.values())
    assert outcomes == {0, 1}


def test_paths_identity_idle_wires_and_compile_result(tmp_path):
    circuit = QuantumCircuit(3)
    circuit.h(2)
    path = tmp_path / "circuit.qasm"
    path.write_text(qasm2.dumps(circuit))
    for pattern in [mp.from_qasm(path), mp.compile(path).pattern]:
        sim = mp.PatternSimulator(
            pattern, input_state=np.eye(1, 8, 0).ravel(), backend="trajectory"
        )
        np.testing.assert_allclose(
            sim.probabilities(), [0.5, 0.5, 0, 0, 0, 0, 0, 0], atol=1e-9
        )
    pattern = mp.from_qasm(qasm2.dumps(QuantumCircuit(2)))
    sim = mp.PatternSimulator(
        pattern, input_state=np.array([1, 0, 0, 0]), backend="trajectory"
    )
    np.testing.assert_allclose(sim.probabilities(), [1, 0, 0, 0])


@pytest.mark.parametrize(
    "command", ["measure q[0] -> c[0];", "reset q[0];", "if(c==1) x q[0];"]
)
def test_reject_nonunitary_qasm(command):
    qasm = 'OPENQASM 2.0;\ninclude "qelib1.inc";\nqreg q[1];\ncreg c[1];\n' + command
    with pytest.raises(
        (ValueError, TypeError),
        match="(?i)(unsupported|unitary|measurement|reset|condition)",
    ):
        mp.from_qasm(qasm)


def test_five_qubit_grover_from_both_frontends():
    import runpy
    from pathlib import Path

    example = runpy.run_path(str(Path(__file__).parents[2] / "examples" / "grover.py"))
    qiskit_grover, pennylane_grover, simulate = (
        example[name] for name in ("qiskit_grover", "pennylane_grover", "simulate")
    )
    import pennylane as qml

    circuit = qiskit_grover()
    reference = Statevector.from_instruction(circuit).reverse_qargs().probabilities()
    for qasm in [
        qasm2.dumps(circuit),
        example["to_qasm"](pennylane_grover()),
    ]:
        pattern, probabilities, samples = simulate(qasm)
        np.testing.assert_allclose(probabilities, reference, atol=1e-8)
        assert probabilities[-1] > 0.99
        assert samples.counts["11111"] > 4000
        assert samples.shots == 4096
        assert len(pattern) > 100


@pytest.mark.parametrize(
    "reduction", ["full_reduce", "teleport_reduce", "clifford_simp"]
)
def test_zx_reductions_preserve_the_executed_circuit(reduction):
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.rz(0.42, 1)
    circuit.t(0)
    result = mp.compile(qasm2.dumps(circuit), reduction=reduction)
    sim = mp.PatternSimulator(
        result.pattern, input_state=np.array([1, 0, 0, 0]), backend="trajectory"
    )
    reference = Statevector.from_instruction(circuit).reverse_qargs().data
    np.testing.assert_allclose(sim.run([]), density(reference), atol=1e-9)
    assert result.before["input_t_gates"] == 1

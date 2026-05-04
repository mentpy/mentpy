"""Tests for optional interoperability helpers."""

import sys
import types

import numpy as np
import pytest

import mentpy as mp


def _density(state):
    return np.outer(state, np.conj(state))


def _assert_state_equal_up_to_global_phase(actual, expected, atol=1e-6):
    actual = np.asarray(actual, dtype=complex)
    expected = np.asarray(expected, dtype=complex)
    actual = actual / np.linalg.norm(actual)
    expected = expected / np.linalg.norm(expected)
    assert np.allclose(_density(actual), _density(expected), atol=atol)


def test_pennylane_hamiltonian_to_observable():
    qml = pytest.importorskip("pennylane")

    hamiltonian = qml.Hamiltonian(
        [1.0, -0.5],
        [qml.PauliZ(0), qml.PauliX(1) @ qml.PauliY(0)],
    )

    obs = mp.utils.from_pennylane(hamiltonian, wire_order=[0, 1])

    assert obs.terms == {"ZI": 1.0, "YX": -0.5}


def test_observable_to_pennylane_roundtrip():
    qml = pytest.importorskip("pennylane")

    obs = mp.Observable({"ZI": 1.0, "YX": -0.5}, constant=0.25)
    hamiltonian = mp.utils.to_pennylane(obs, wire_order=[0, 1])

    assert isinstance(hamiltonian, qml.Hamiltonian)

    roundtrip = mp.utils.from_pennylane(hamiltonian, wire_order=[0, 1])
    assert np.allclose(roundtrip.constant, obs.constant)
    assert roundtrip.terms == obs.terms


def test_from_pennylane_single_pauli():
    qml = pytest.importorskip("pennylane")

    obs = mp.utils.from_pennylane(qml.PauliX(1), wire_order=[0, 1])

    assert obs.terms == {"IX": 1.0}


def test_from_pennylane_paulirot_to_mbqcircuit():
    qml = pytest.importorskip("pennylane")

    circuit = mp.utils.from_pennylane(
        qml.PauliRot(0.25, "XZ", wires=[0, 1]),
        wire_order=[0, 1],
    )
    xy_angles = [
        ment.angle
        for ment in circuit.measurements.values()
        if ment is not None and ment.plane == "XY"
    ]

    assert isinstance(circuit, mp.MBQCircuit)
    assert len(circuit.input_nodes) == 2
    assert len(circuit.output_nodes) == 2
    assert np.isclose(xy_angles[0], -0.25)


def test_from_pennylane_axis_rotations_to_mbqcircuit():
    qml = pytest.importorskip("pennylane")

    circuit = mp.utils.from_pennylane(
        [qml.RX(0.1, wires=0), qml.RY(0.2, wires=1), qml.RZ(0.3, wires=0)],
        wire_order=[0, 1],
    )
    xy_angles = [
        ment.angle
        for ment in circuit.measurements.values()
        if ment is not None and ment.plane == "XY"
    ]

    assert isinstance(circuit, mp.MBQCircuit)
    assert len(circuit.input_nodes) == 2
    assert len(circuit.output_nodes) == 2
    assert np.allclose(sorted(xy_angles), [-0.3, -0.1, 0.2])


def test_from_pennylane_h_cnot_to_mbqcircuit_unitary():
    qml = pytest.importorskip("pennylane")
    pytest.importorskip("jax")

    circuit = mp.utils.from_pennylane(
        [qml.Hadamard(wires=0), qml.CNOT(wires=[0, 1])],
        wire_order=[0, 1],
    )
    input_state = np.array([0.2 + 0.1j, -0.3j, 0.4, 0.5 - 0.2j])
    input_state = input_state / np.linalg.norm(input_state)
    output_state = mp.PatternSimulator(
        circuit, input_state=input_state, backend="jax-tn"
    ).run([], output_form="sv")

    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    identity = np.eye(2)
    cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    expected = cnot @ np.kron(h, identity) @ input_state

    assert isinstance(circuit, mp.MBQCircuit)
    assert len(circuit.input_nodes) == 2
    assert len(circuit.output_nodes) == 2
    _assert_state_equal_up_to_global_phase(output_state, expected)


def test_mbqcircuit_to_pennylane_qnode():
    pytest.importorskip("pennylane")

    circuit = mp.templates.linear_cluster(3)
    qnode = mp.utils.to_pennylane(circuit)
    input_state = np.array([1, 1]) / np.sqrt(2)
    density = qnode(np.zeros(len(circuit.outputc)), st=input_state)

    assert callable(qnode)
    assert density.shape == (2, 2)


def test_from_cirq_pauli_string_phasor_to_mbqcircuit():
    cirq = pytest.importorskip("cirq")

    qubits = cirq.LineQubit.range(2)
    pauli_string = cirq.PauliString({qubits[0]: cirq.X, qubits[1]: cirq.Z})
    circuit = mp.utils.from_cirq(pauli_string**0.25, qubit_order=qubits)
    xy_angles = [
        ment.angle
        for ment in circuit.measurements.values()
        if ment is not None and ment.plane == "XY"
    ]

    assert isinstance(circuit, mp.MBQCircuit)
    assert len(circuit.input_nodes) == 2
    assert len(circuit.output_nodes) == 2
    assert np.isclose(xy_angles[0], -np.pi / 4)


def test_from_cirq_axis_rotations_to_mbqcircuit():
    cirq = pytest.importorskip("cirq")

    qubits = cirq.LineQubit.range(2)
    cirq_circuit = cirq.Circuit(
        cirq.rx(0.1)(qubits[0]),
        cirq.ry(0.2)(qubits[1]),
        cirq.rz(0.3)(qubits[0]),
    )
    circuit = mp.utils.from_cirq(cirq_circuit, qubit_order=qubits)
    xy_angles = [
        ment.angle
        for ment in circuit.measurements.values()
        if ment is not None and ment.plane == "XY"
    ]

    assert isinstance(circuit, mp.MBQCircuit)
    assert len(circuit.input_nodes) == 2
    assert len(circuit.output_nodes) == 2
    assert np.allclose(sorted(xy_angles), [-0.3, -0.1, 0.2])


def test_from_cirq_h_cnot_to_mbqcircuit_unitary():
    cirq = pytest.importorskip("cirq")
    pytest.importorskip("jax")

    qubits = cirq.LineQubit.range(2)
    circuit = mp.utils.from_cirq(
        cirq.Circuit(cirq.H(qubits[0]), cirq.CNOT(qubits[0], qubits[1])),
        qubit_order=qubits,
    )
    input_state = np.array([0.1 - 0.3j, 0.25, -0.4j, 0.7 + 0.2j])
    input_state = input_state / np.linalg.norm(input_state)
    output_state = mp.PatternSimulator(
        circuit, input_state=input_state, backend="jax-tn"
    ).run([], output_form="sv")

    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    identity = np.eye(2)
    cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    expected = cnot @ np.kron(h, identity) @ input_state

    assert isinstance(circuit, mp.MBQCircuit)
    assert len(circuit.input_nodes) == 2
    assert len(circuit.output_nodes) == 2
    _assert_state_equal_up_to_global_phase(output_state, expected)


def test_mbqcircuit_to_cirq_dynamic_circuit():
    cirq = pytest.importorskip("cirq")

    circuit = mp.templates.linear_cluster(3)
    cirq_circuit = mp.utils.to_cirq(circuit)
    operations = list(cirq_circuit.all_operations())

    assert isinstance(cirq_circuit, cirq.Circuit)
    assert sum(isinstance(op.gate, cirq.MeasurementGate) for op in operations) == 2
    assert any(cirq.is_parameterized(op) for op in operations)
    assert any(getattr(op, "classical_controls", None) for op in operations)


def test_qiskit_sparse_pauli_roundtrip():
    pytest.importorskip("qiskit")
    from qiskit.quantum_info import SparsePauliOp

    sparse_pauli = SparsePauliOp(["ZI", "YX", "II"], coeffs=[1.0, -0.5, 0.25])
    obs = mp.utils.from_qiskit(sparse_pauli)

    assert np.allclose(obs.constant, 0.25)
    assert obs.terms == {"ZI": 1.0, "YX": -0.5}

    roundtrip = mp.utils.from_qiskit(mp.utils.to_qiskit(obs))
    assert np.allclose(roundtrip.constant, obs.constant)
    assert roundtrip.terms == obs.terms


def test_from_qiskit_axis_rotations_to_mbqcircuit():
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit

    qiskit_circuit = QuantumCircuit(2)
    qiskit_circuit.rx(0.1, 0)
    qiskit_circuit.ry(0.2, 1)
    qiskit_circuit.rz(0.3, 0)

    circuit = mp.utils.from_qiskit(qiskit_circuit, qubit_order=[0, 1])
    xy_angles = [
        ment.angle
        for ment in circuit.measurements.values()
        if ment is not None and ment.plane == "XY"
    ]

    assert isinstance(circuit, mp.MBQCircuit)
    assert len(circuit.input_nodes) == 2
    assert len(circuit.output_nodes) == 2
    assert np.allclose(sorted(xy_angles), [-0.3, -0.1, 0.2])


def test_from_qiskit_h_cx_to_mbqcircuit_unitary():
    pytest.importorskip("qiskit")
    pytest.importorskip("jax")
    from qiskit import QuantumCircuit

    qiskit_circuit = QuantumCircuit(2)
    qiskit_circuit.h(0)
    qiskit_circuit.cx(0, 1)

    circuit = mp.utils.from_qiskit(qiskit_circuit, qubit_order=[0, 1])
    input_state = np.array([0.1 - 0.3j, 0.25, -0.4j, 0.7 + 0.2j])
    input_state = input_state / np.linalg.norm(input_state)
    output_state = mp.PatternSimulator(
        circuit, input_state=input_state, backend="jax-tn"
    ).run([], output_form="sv")

    h = np.array([[1, 1], [1, -1]]) / np.sqrt(2)
    identity = np.eye(2)
    cnot = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]])
    expected = cnot @ np.kron(h, identity) @ input_state

    assert isinstance(circuit, mp.MBQCircuit)
    assert len(circuit.input_nodes) == 2
    assert len(circuit.output_nodes) == 2
    _assert_state_equal_up_to_global_phase(output_state, expected)


def test_mbqcircuit_to_qiskit_dynamic_circuit():
    pytest.importorskip("qiskit")
    from qiskit import QuantumCircuit

    circuit = mp.templates.linear_cluster(3)
    qiskit_circuit = mp.utils.to_qiskit(circuit)
    names = [instruction.operation.name for instruction in qiskit_circuit.data]

    assert isinstance(qiskit_circuit, QuantumCircuit)
    assert names.count("measure") == 2
    assert any(name == "if_else" for name in names)
    assert any(
        str(parameter).startswith("theta") for parameter in qiskit_circuit.parameters
    )


def test_openfermion_roundtrip():
    openfermion = pytest.importorskip("openfermion")

    qubit_operator = (
        openfermion.QubitOperator("", 0.25)
        + openfermion.QubitOperator("Z0", 1.0)
        + openfermion.QubitOperator("Y0 X1", -0.5)
    )

    obs = mp.utils.from_openfermion(qubit_operator, n_qubits=2)

    assert np.allclose(obs.constant, 0.25)
    assert obs.terms == {"ZI": 1.0, "YX": -0.5}

    roundtrip = mp.utils.to_openfermion(obs)
    roundtrip.compress()
    qubit_operator.compress()

    assert roundtrip.terms == qubit_operator.terms


class _FakeCudaQPauli:
    def __init__(self, name):
        self.name = name


class _FakeCudaQElement:
    def __init__(self, target, pauli):
        self.target = target
        self._pauli = pauli

    def as_pauli(self):
        return _FakeCudaQPauli(self._pauli)


class _FakeCudaQSpinTerm:
    def __init__(self, coefficient=1.0, ops=None):
        self._coefficient = coefficient
        self._ops = {} if ops is None else dict(ops)

    @property
    def degrees(self):
        return sorted(self._ops)

    def __iter__(self):
        return iter(
            _FakeCudaQElement(target, pauli)
            for target, pauli in sorted(self._ops.items())
        )

    def __mul__(self, other):
        if isinstance(other, (int, float, complex)):
            return _FakeCudaQSpinTerm(self._coefficient * other, self._ops)
        ops = dict(self._ops)
        ops.update(other._ops)
        return _FakeCudaQSpinTerm(self._coefficient * other._coefficient, ops)

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        if isinstance(other, _FakeCudaQSpinSum):
            return _FakeCudaQSpinSum([self, *other._terms])
        return _FakeCudaQSpinSum([self, other])

    def evaluate_coefficient(self, parameters=None):
        return self._coefficient

    def is_identity(self):
        return len(self._ops) == 0


class _FakeCudaQSpinSum:
    def __init__(self, terms):
        self._terms = terms

    @property
    def degrees(self):
        degrees = []
        for term in self._terms:
            for degree in term.degrees:
                if degree not in degrees:
                    degrees.append(degree)
        return sorted(degrees)

    def __iter__(self):
        return iter(self._terms)

    def __add__(self, other):
        if isinstance(other, _FakeCudaQSpinSum):
            return _FakeCudaQSpinSum([*self._terms, *other._terms])
        return _FakeCudaQSpinSum([*self._terms, other])


class _FakeCudaQForEachTermContainer:
    def __init__(self, terms):
        self._terms = terms

    def for_each_term(self, callback):
        for term in self._terms:
            callback(term)


class _FakeCudaQForEachPauliTerm:
    def __init__(self, coefficient=1.0, ops=None):
        self._coefficient = coefficient
        self._ops = {} if ops is None else dict(ops)

    @property
    def degrees(self):
        return sorted(self._ops)

    def evaluate_coefficient(self, parameters=None):
        return self._coefficient

    def for_each_pauli(self, callback):
        for target, pauli in sorted(self._ops.items()):
            callback(_FakeCudaQPauli(pauli), target)


class _FakeCudaQWordTerm:
    def __init__(self, word, coefficient=1.0):
        self._word = word
        self._coefficient = coefficient

    def get_pauli_word(self, pad_identities):
        return self._word

    def get_coefficient(self):
        return self._coefficient


class _FakeCudaQSpin:
    @staticmethod
    def x(target):
        return _FakeCudaQSpinTerm(ops={target: "X"})

    @staticmethod
    def y(target):
        return _FakeCudaQSpinTerm(ops={target: "Y"})

    @staticmethod
    def z(target):
        return _FakeCudaQSpinTerm(ops={target: "Z"})

    @staticmethod
    def i(target=None):
        return _FakeCudaQSpinTerm()


class _FakeCudaQParameter:
    def __init__(self, expression):
        self.expression = expression

    def __neg__(self):
        return _FakeCudaQParameter(f"-({self.expression})")

    def __repr__(self):
        return self.expression


class _FakeCudaQParameterList:
    def __getitem__(self, index):
        return _FakeCudaQParameter(f"theta[{index}]")


class _FakeCudaQKernel:
    def __init__(self):
        self.operations = []

    def qalloc(self, n_qubits):
        self.operations.append(("qalloc", n_qubits))
        return [f"q{i}" for i in range(n_qubits)]

    def h(self, target):
        self.operations.append(("h", target))

    def rz(self, angle, target):
        self.operations.append(("rz", angle, target))

    def x(self, target):
        self.operations.append(("x", target))

    def z(self, target):
        self.operations.append(("z", target))

    def cz(self, left, right):
        self.operations.append(("cz", left, right))

    def mz(self, target, key=None):
        measurement = f"m:{target}:{key}"
        self.operations.append(("mz", target, key))
        return measurement

    def c_if(self, measurement, then_function):
        self.operations.append(("c_if", measurement))
        then_function()


def _fake_cudaq_module(qasm=None):
    cudaq = types.ModuleType("cudaq")
    cudaq.spin = _FakeCudaQSpin

    def make_kernel(*arg_types):
        kernel = _FakeCudaQKernel()
        if arg_types:
            return kernel, _FakeCudaQParameterList()
        return kernel

    cudaq.make_kernel = make_kernel
    if qasm is not None:
        cudaq.translate = lambda _kernel, *args, **kwargs: qasm
    return cudaq


def test_cudaq_spin_operator_roundtrip(monkeypatch):
    monkeypatch.setitem(sys.modules, "cudaq", _fake_cudaq_module())

    obs = mp.Observable({"ZI": 1.0, "YX": -0.5}, constant=0.25)
    spin_op = mp.utils.to_cudaq(obs, wire_order=[0, 1])
    roundtrip = mp.utils.from_cudaq(spin_op, n_qubits=2)

    assert np.allclose(roundtrip.constant, obs.constant)
    assert roundtrip.terms == obs.terms


def test_from_cudaq_fake_for_each_term_container(monkeypatch):
    monkeypatch.setitem(sys.modules, "cudaq", _fake_cudaq_module())
    spin_op = _FakeCudaQForEachTermContainer(
        [
            _FakeCudaQForEachPauliTerm(2.0, {1: "X"}),
            _FakeCudaQForEachPauliTerm(0.5),
        ]
    )

    obs = mp.utils.from_cudaq(spin_op, n_qubits=2)

    assert np.allclose(obs.constant, 0.5)
    assert obs.terms == {"IX": 2.0}
    assert obs.n_qubits == 2


def test_from_cudaq_fake_get_pauli_word_term(monkeypatch):
    monkeypatch.setitem(sys.modules, "cudaq", _fake_cudaq_module())

    obs = mp.utils.from_cudaq(_FakeCudaQWordTerm("XIZ", 3.0))

    assert obs.constant == 0.0
    assert obs.terms == {"XIZ": 3.0}
    assert obs.n_qubits == 3


def test_from_cudaq_fake_rejects_terms_outside_wire_order(monkeypatch):
    monkeypatch.setitem(sys.modules, "cudaq", _fake_cudaq_module())
    spin_op = _FakeCudaQSpinTerm(ops={2: "Z"})

    with pytest.raises(ValueError, match="not in wire_order"):
        mp.utils.from_cudaq(spin_op, wire_order=[0, 1])


def test_to_cudaq_fake_requires_spin_helpers(monkeypatch):
    cudaq = types.ModuleType("cudaq")
    monkeypatch.setitem(sys.modules, "cudaq", cudaq)

    with pytest.raises(ImportError, match="spin helpers"):
        mp.utils.to_cudaq(mp.Observable.identity(1))


def test_to_cudaq_fake_empty_observable_uses_identity(monkeypatch):
    monkeypatch.setitem(sys.modules, "cudaq", _fake_cudaq_module())

    spin_op = mp.utils.to_cudaq(mp.Observable.identity(2, coeff=0.0), wire_order=[0, 1])
    obs = mp.utils.from_cudaq(spin_op, n_qubits=2)

    assert obs.constant == 0.0
    assert obs.terms == {}
    assert obs.n_qubits == 2


def test_real_cudaq_spin_operator_roundtrip():
    cudaq = pytest.importorskip("cudaq")
    if not hasattr(cudaq, "spin"):
        pytest.skip("CUDA-Q spin API is not available.")

    obs = mp.Observable({"ZI": 1.0, "YX": -0.5}, constant=0.25)
    spin_op = mp.utils.to_cudaq(obs, wire_order=[0, 1])
    roundtrip = mp.utils.from_cudaq(spin_op, n_qubits=2)

    assert np.allclose(roundtrip.constant, obs.constant)
    assert roundtrip.terms == obs.terms


def test_mbqcircuit_to_cudaq_kernel(monkeypatch):
    monkeypatch.setitem(sys.modules, "cudaq", _fake_cudaq_module())

    circuit = mp.templates.linear_cluster(3)
    kernel = mp.utils.to_cudaq(circuit)
    operations = kernel.operations

    assert operations[0] == ("qalloc", circuit.graph.number_of_nodes())
    assert any(op[0] == "cz" for op in operations)
    assert any(op[0] == "rz" for op in operations)
    assert any(op[0] == "mz" for op in operations)
    assert any(op[0] == "c_if" for op in operations)
    assert any(op[0] in {"x", "z"} for op in operations)


def test_mbqcircuit_to_cudaq_validates_parameter_count(monkeypatch):
    monkeypatch.setitem(sys.modules, "cudaq", _fake_cudaq_module())
    circuit = mp.templates.linear_cluster(3)

    with pytest.raises(ValueError, match="parameters length"):
        mp.utils.to_cudaq(circuit, parameters=[])


def test_cudaq_kernel_openqasm_import(monkeypatch):
    qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    h q[0];
    cx q[0], q[1];
    """
    monkeypatch.setitem(sys.modules, "cudaq", _fake_cudaq_module(qasm=qasm))

    circuit = mp.utils.from_cudaq(object())

    assert isinstance(circuit, mp.MBQCircuit)
    assert len(circuit.input_nodes) == 2
    assert len(circuit.output_nodes) == 2

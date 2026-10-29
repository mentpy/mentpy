# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Optional interoperability helpers."""

import numpy as np

from mentpy.operators import Observable

__all__ = [
    "from_pennylane",
    "to_pennylane",
    "from_openfermion",
    "to_openfermion",
    "from_cirq",
    "to_cirq",
    "from_qiskit",
    "to_qiskit",
    "from_cudaq",
    "to_cudaq",
]


_PL_PAULI_NAMES = {
    "PauliX": "X",
    "PauliY": "Y",
    "PauliZ": "Z",
    "Identity": "I",
}


def from_pennylane(obj, wire_order=None):
    """Convert a PennyLane Hamiltonian/Pauli expression to an Observable.

    PennyLane ``PauliRot``/``RX``/``RY``/``RZ`` operations and common Clifford
    gates (``Hadamard``, ``CZ``, ``CNOT``), or circuits/tapes composed only of
    those operations, are converted to MBQC Pauli-rotation templates.
    """
    qml = _require("pennylane", "PennyLane")

    rotations = _pennylane_pauli_rotations(obj, wire_order)
    if rotations is not None:
        return _pauli_rotations_to_mbqc(rotations)

    try:
        coeffs, ops = obj.terms()
    except Exception:
        coeffs, ops = None, None

    if coeffs is None and _is_pennylane_operator(obj):
        coeffs, ops = [1.0], [obj]
    elif coeffs is None:
        raise NotImplementedError(
            "from_pennylane currently supports PennyLane Hamiltonians and "
            "Pauli operators, plus PauliRot/RX/RY/RZ/H/CZ/CNOT circuit "
            "conversion."
        )

    wire_order = _resolve_wire_order(wire_order, ops)
    terms = {}
    constant = 0.0
    for coeff, op in zip(coeffs, ops):
        pauli = _pennylane_op_to_pauli(op, wire_order)
        if set(pauli) == {"I"}:
            constant += coeff
        else:
            terms[pauli] = terms.get(pauli, 0.0) + coeff

    observable = Observable(terms, constant=constant)
    if observable.n_qubits is None:
        observable.n_qubits = len(wire_order)
    return observable


def to_pennylane(obj, wire_order=None, **kwargs):
    """Convert MentPy observables or MBQC circuits to PennyLane objects."""
    qml = _require("pennylane", "PennyLane")

    if _is_mbqcircuit(obj):
        from mentpy.simulators.pennylane_simulator import mbqcircuit_to_circuit

        return mbqcircuit_to_circuit(obj, **kwargs)

    observable = _as_observable(obj)
    wire_order = _default_wire_order(observable, wire_order)

    coeffs = []
    ops = []
    for pauli, coeff in observable.terms.items():
        coeffs.append(coeff)
        ops.append(_pauli_to_pennylane_op(pauli, wire_order, qml))

    if observable.constant != 0 or not coeffs:
        coeffs.append(observable.constant)
        ops.append(qml.Identity(wire_order[0] if wire_order else 0))

    return qml.Hamiltonian(coeffs, ops)


def from_openfermion(qubit_operator, n_qubits=None, wire_order=None):
    """Convert an OpenFermion ``QubitOperator`` to an Observable."""
    _require("openfermion", "OpenFermion")

    if wire_order is None:
        if n_qubits is None:
            max_wire = -1
            for term in qubit_operator.terms:
                for wire, _pauli in term:
                    max_wire = max(max_wire, wire)
            n_qubits = max_wire + 1
        wire_order = list(range(n_qubits))
    else:
        wire_order = list(wire_order)

    wire_to_index = {wire: index for index, wire in enumerate(wire_order)}
    terms = {}
    constant = 0.0

    for term, coeff in qubit_operator.terms.items():
        if term == ():
            constant += coeff
            continue
        pauli = ["I"] * len(wire_order)
        for wire, op in term:
            pauli[wire_to_index[wire]] = op
        pauli = "".join(pauli)
        terms[pauli] = terms.get(pauli, 0.0) + coeff

    observable = Observable(terms, constant=constant)
    if observable.n_qubits is None:
        observable.n_qubits = len(wire_order)
    return observable


def to_openfermion(observable, wire_order=None):
    """Convert an :class:`mentpy.Observable` to OpenFermion ``QubitOperator``."""
    openfermion = _require("openfermion", "OpenFermion")
    observable = _as_observable(observable)
    wire_order = _default_wire_order(observable, wire_order)

    out = openfermion.QubitOperator((), observable.constant)
    for pauli, coeff in observable.terms.items():
        term = tuple((wire, op) for wire, op in zip(wire_order, pauli) if op != "I")
        out += openfermion.QubitOperator(term, coeff)
    return out


def from_cirq(obj, qubit_order=None):
    """Convert Cirq Pauli objects to MentPy observables or MBQC circuits.

    Cirq ``PauliStringPhasor``, one-qubit ``X/Y/ZPowGate`` operations, and
    common Clifford gates (``H``, ``CZ``, ``CNOT``), or circuits composed only
    of those operations, are converted to MBQC Pauli-rotation templates.
    """
    cirq = _require("cirq", "Cirq")
    rotations = _cirq_pauli_rotations(cirq, obj, qubit_order)
    if rotations is not None:
        return _pauli_rotations_to_mbqc(rotations)

    if isinstance(obj, cirq.PauliString):
        items = [(obj, obj.coefficient)]
    elif isinstance(obj, cirq.PauliSum):
        items = [(term, term.coefficient) for term in obj]
    else:
        raise NotImplementedError(
            "from_cirq currently supports Cirq PauliString/PauliSum objects "
            "and circuits composed of Pauli rotations, H, CZ, and CNOT."
        )

    if qubit_order is None:
        qubits = []
        for term, _coeff in items:
            for qubit in term.qubits:
                if qubit not in qubits:
                    qubits.append(qubit)
        qubit_order = sorted(qubits)
    qubit_order = list(qubit_order)
    index = {qubit: i for i, qubit in enumerate(qubit_order)}

    terms = {}
    constant = 0.0
    for term, coeff in items:
        pauli = ["I"] * len(qubit_order)
        for qubit, gate in term.items():
            pauli[index[qubit]] = str(gate)
        pauli = "".join(pauli)
        if set(pauli) == {"I"}:
            constant += coeff
        else:
            terms[pauli] = terms.get(pauli, 0.0) + coeff

    observable = Observable(terms, constant=constant)
    if observable.n_qubits is None:
        observable.n_qubits = len(qubit_order)
    return observable


def to_cirq(obj, qubit_order=None, **kwargs):
    """Convert MentPy observables or MBQC circuits to Cirq objects."""
    cirq = _require("cirq", "Cirq")

    if _is_mbqcircuit(obj):
        return _mbqcircuit_to_cirq(cirq, obj, qubit_order=qubit_order, **kwargs)

    observable = _as_observable(obj)
    qubit_order = _default_cirq_qubits(cirq, observable, qubit_order)

    out = cirq.PauliSum()
    if observable.constant != 0:
        out += observable.constant * cirq.PauliString()
    for pauli, coeff in observable.terms.items():
        factors = {}
        for qubit, op in zip(qubit_order, pauli):
            if op == "X":
                factors[qubit] = cirq.X
            elif op == "Y":
                factors[qubit] = cirq.Y
            elif op == "Z":
                factors[qubit] = cirq.Z
        out += coeff * cirq.PauliString(factors)
    return out


def from_qiskit(obj, qubit_order=None):
    """Convert Qiskit objects to MentPy observables or MBQC circuits.

    Qiskit ``QuantumCircuit`` objects composed of ``rx/ry/rz`` rotations,
    ``x/y/z/s/sdg/t/tdg/sx/sxdg``, ``h``, ``cz``, and ``cx``/``cnot`` gates are
    converted to MBQC Pauli-rotation templates. ``SparsePauliOp`` and ``Pauli``
    objects are converted to :class:`Observable` objects.
    """
    qiskit = _require("qiskit", "Qiskit")

    rotations = _qiskit_pauli_rotations(qiskit, obj, qubit_order)
    if rotations is not None:
        return _pauli_rotations_to_mbqc(rotations)

    SparsePauliOp, Pauli = _qiskit_pauli_types()
    if isinstance(obj, Pauli):
        obj = SparsePauliOp(obj)
    if not isinstance(obj, SparsePauliOp):
        raise NotImplementedError(
            "from_qiskit currently supports Qiskit SparsePauliOp/Pauli objects "
            "and QuantumCircuit objects composed of Pauli rotations, H, CZ, "
            "and CX/CNOT."
        )

    terms = {}
    constant = 0.0
    for label, coeff in _qiskit_sparse_pauli_items(obj):
        label = str(label).upper()
        if set(label) == {"I"}:
            constant += coeff
        else:
            terms[label] = terms.get(label, 0.0) + coeff

    observable = Observable(terms, constant=constant)
    if observable.n_qubits is None:
        observable.n_qubits = int(obj.num_qubits)
    return observable


def to_qiskit(obj, qubit_order=None, parameters=None, parameter_prefix="theta"):
    """Convert MentPy observables or MBQC circuits to Qiskit objects.

    Observables become ``qiskit.quantum_info.SparsePauliOp`` instances. MBQC
    circuits become dynamic ``QuantumCircuit`` objects with graph-state
    preparation, adaptive measurements, and classically controlled corrections.
    """
    _require("qiskit", "Qiskit")
    if _is_mbqcircuit(obj):
        return _mbqcircuit_to_qiskit(
            obj,
            qubit_order=qubit_order,
            parameters=parameters,
            parameter_prefix=parameter_prefix,
        )

    SparsePauliOp, _Pauli = _qiskit_pauli_types()
    observable = _as_observable(obj)
    n_qubits = observable.n_qubits
    if n_qubits is None:
        if qubit_order is None:
            raise ValueError(
                "qubit_order is required when exporting an empty Observable."
            )
        n_qubits = len(qubit_order)
    elif qubit_order is not None and len(qubit_order) != n_qubits:
        raise ValueError("qubit_order length must match observable.n_qubits.")

    labels = []
    coeffs = []
    if observable.constant != 0 or not observable.terms:
        labels.append("I" * n_qubits)
        coeffs.append(observable.constant)
    for pauli, coeff in observable.terms.items():
        labels.append(pauli)
        coeffs.append(coeff)

    return SparsePauliOp(labels, coeffs)


def from_cudaq(obj, n_qubits=None, wire_order=None, parameters=None):
    """Convert CUDA-Q spin operators or simple kernels to MentPy objects.

    Spin operators/terms become :class:`Observable` objects. CUDA-Q kernels are
    translated through OpenQASM 2.0 and then imported through the Cirq circuit
    converter, so this path supports the gate subset accepted by
    :func:`from_cirq`.
    """
    cudaq = _require("cudaq", "CUDA-Q")

    try:
        terms = _cudaq_spin_terms(obj)
    except NotImplementedError as spin_error:
        return _cudaq_kernel_to_mbqc(
            cudaq,
            obj,
            n_qubits=n_qubits,
            wire_order=wire_order,
            parameters=parameters,
            spin_error=spin_error,
        )

    if wire_order is None:
        wire_order = _cudaq_wire_order(obj, terms, n_qubits)
    else:
        wire_order = list(wire_order)
    if n_qubits is not None and len(wire_order) != n_qubits:
        raise ValueError("wire_order length must match n_qubits.")

    wire_to_index = {wire: index for index, wire in enumerate(wire_order)}
    observable_terms = {}
    constant = 0.0
    parameters = {} if parameters is None else parameters

    for term in terms:
        coeff = _cudaq_term_coefficient(term, parameters)
        pauli = ["I"] * len(wire_order)

        for target, op in _cudaq_term_ops(term, len(wire_order)):
            if target not in wire_to_index:
                raise ValueError(
                    f"CUDA-Q term targets wire {target}, which is not in wire_order."
                )
            pauli[wire_to_index[target]] = op

        pauli = "".join(pauli)
        if set(pauli) == {"I"}:
            constant += coeff
        else:
            observable_terms[pauli] = observable_terms.get(pauli, 0.0) + coeff

    observable = Observable(observable_terms, constant=constant)
    if observable.n_qubits is None:
        observable.n_qubits = len(wire_order)
    return observable


def to_cudaq(obj, wire_order=None, parameters=None):
    """Convert MentPy observables or MBQC circuits to CUDA-Q objects.

    Observables are converted to CUDA-Q spin operators. MBQC circuits are
    lowered to a CUDA-Q kernel with graph preparation, adaptive measurements,
    and classically controlled flow corrections. Trainable measurement angles
    become a single list argument unless concrete ``parameters`` are supplied.
    """
    cudaq = _require("cudaq", "CUDA-Q")
    if _is_mbqcircuit(obj):
        return _mbqcircuit_to_cudaq(
            cudaq, obj, qubit_order=wire_order, parameters=parameters
        )

    observable = obj
    observable = _as_observable(observable)
    wire_order = _default_wire_order(observable, wire_order)
    spin = getattr(cudaq, "spin", None)
    if spin is None:
        raise ImportError("CUDA-Q spin helpers are not available as cudaq.spin.")

    out = None
    if observable.constant != 0:
        out = observable.constant * _cudaq_identity(spin, wire_order)

    for pauli, coeff in observable.terms.items():
        term = None
        for wire, op in zip(wire_order, pauli):
            if op == "I":
                continue
            factor = {
                "X": spin.x,
                "Y": spin.y,
                "Z": spin.z,
            }[
                op
            ](wire)
            term = factor if term is None else term * factor
        if term is None:
            term = _cudaq_identity(spin, wire_order)
        term = coeff * term
        out = term if out is None else out + term

    if out is None:
        out = 0.0 * _cudaq_identity(spin, wire_order)
    return out


def _require(module_name, label):
    try:
        return __import__(module_name)
    except ImportError as exc:
        raise ImportError(
            f"{label} is required for this conversion. Install the matching extra "
            f"or package before calling this function."
        ) from exc


def _as_observable(obj):
    if not isinstance(obj, Observable):
        raise TypeError(f"Expected Observable, got {type(obj)}.")
    return obj


def _is_mbqcircuit(obj):
    return (
        hasattr(obj, "graph")
        and hasattr(obj, "measurements")
        and hasattr(obj, "input_nodes")
        and hasattr(obj, "output_nodes")
    )


def _is_pennylane_operator(obj):
    return hasattr(obj, "wires") and hasattr(obj, "name")


def _resolve_wire_order(wire_order, ops):
    if wire_order is not None:
        return list(wire_order)

    wires = []
    for op in ops:
        for wire in list(op.wires):
            if wire not in wires:
                wires.append(wire)
    try:
        return sorted(wires)
    except TypeError:
        return wires


def _pennylane_op_to_pauli(op, wire_order):
    pauli = ["I"] * len(wire_order)
    wire_to_index = {wire: index for index, wire in enumerate(wire_order)}

    for factor in _pennylane_factors(op):
        name = factor.name
        if name not in _PL_PAULI_NAMES:
            raise ValueError(f"Unsupported PennyLane operator {factor}.")
        if name == "Identity":
            continue
        if len(factor.wires) != 1:
            raise ValueError(f"Unsupported multi-wire operator {factor}.")
        pauli[wire_to_index[list(factor.wires)[0]]] = _PL_PAULI_NAMES[name]

    return "".join(pauli)


def _pennylane_factors(op):
    if hasattr(op, "operands") and op.operands is not None:
        factors = []
        for operand in op.operands:
            factors.extend(_pennylane_factors(operand))
        return factors
    return [op]


def _pennylane_pauli_rotations(obj, wire_order):
    operations = _pennylane_operations(obj)
    if operations is None:
        return None
    if len(operations) == 0:
        raise ValueError("Cannot convert an empty PennyLane circuit.")
    if not all(_is_supported_pennylane_gate(op) for op in operations):
        return None

    if wire_order is None:
        wire_order = _resolve_wire_order(None, operations)
    else:
        wire_order = list(wire_order)

    rotations = []
    for op in operations:
        rotations.extend(_pennylane_gate_rotations(op, wire_order))
    return rotations


def _is_supported_pennylane_gate(op):
    name = getattr(op, "name", None)
    if name == "PauliRot":
        return True
    if name in {"RX", "RY", "RZ", "Hadamard"}:
        return len(op.wires) == 1
    return name in {"CZ", "CNOT"} and len(op.wires) == 2


def _pennylane_gate_rotations(op, wire_order):
    name = getattr(op, "name", None)
    wires = list(op.wires)

    if name == "PauliRot":
        pauli = _embed_pauli_word(
            op.hyperparameters["pauli_word"],
            wires,
            wire_order,
        )
        # PennyLane rotations are exp(-i theta P / 2). MentPy's Pauli-rotation
        # template has a sign convention that depends on whether the Pauli
        # word contains Y factors.
        return [
            (
                pauli,
                _template_angle_for_minus_rotation(
                    pauli, _maybe_float(op.parameters[0])
                ),
            )
        ]

    if name in {"RX", "RY", "RZ"}:
        pauli = _embed_pauli_word(name[-1], wires, wire_order)
        return [
            (
                pauli,
                _template_angle_for_minus_rotation(
                    pauli, _maybe_float(op.parameters[0])
                ),
            )
        ]

    if name == "Hadamard":
        return _hadamard_rotations(wires[0], wire_order)

    if name == "CZ":
        return _cz_rotations(wires[0], wires[1], wire_order)

    if name == "CNOT":
        control, target = wires
        return [
            *_hadamard_rotations(target, wire_order),
            *_cz_rotations(control, target, wire_order),
            *_hadamard_rotations(target, wire_order),
        ]

    raise ValueError(f"Unsupported PennyLane operation {op}.")


def _pennylane_operations(obj):
    if _is_supported_pennylane_gate(obj):
        return [obj]
    if hasattr(obj, "operations"):
        return list(obj.operations)
    if isinstance(obj, (list, tuple)) and all(hasattr(op, "name") for op in obj):
        return list(obj)
    return None


def _cirq_pauli_rotations(cirq, obj, qubit_order):
    if isinstance(obj, cirq.Circuit):
        operations = list(obj.all_operations())
    elif isinstance(obj, cirq.PauliStringPhasor):
        operations = [obj]
    elif _is_supported_cirq_gate(cirq, obj):
        operations = [obj]
    elif isinstance(obj, (list, tuple)) and all(
        _is_supported_cirq_gate(cirq, op) for op in obj
    ):
        operations = list(obj)
    else:
        return None

    if len(operations) == 0:
        raise ValueError("Cannot convert an empty Cirq circuit.")
    if not all(_is_supported_cirq_gate(cirq, op) for op in operations):
        return None

    if qubit_order is None:
        qubits = []
        for op in operations:
            for qubit in op.qubits:
                if qubit not in qubits:
                    qubits.append(qubit)
        qubit_order = sorted(qubits)
    else:
        qubit_order = list(qubit_order)

    rotations = []
    for op in operations:
        rotations.extend(_cirq_gate_rotations(cirq, op, qubit_order))
    return rotations


def _is_supported_cirq_gate(cirq, op):
    if isinstance(op, cirq.PauliStringPhasor):
        return True
    if not hasattr(op, "gate"):
        return False

    qubits = getattr(op, "qubits", ())
    if len(qubits) == 1:
        if _cirq_pow_gate_pauli(cirq, op.gate) is not None:
            return True
        return isinstance(op.gate, cirq.HPowGate) and _is_exponent_one(op.gate)
    if len(qubits) == 2:
        return isinstance(op.gate, (cirq.CZPowGate, cirq.CNotPowGate)) and (
            _is_exponent_one(op.gate)
        )
    return False


def _cirq_gate_rotations(cirq, op, qubit_order):
    if isinstance(op, cirq.PauliStringPhasor):
        pauli = ["I"] * len(qubit_order)
        qubit_to_index = {qubit: i for i, qubit in enumerate(qubit_order)}
        for qubit, gate in op.pauli_string.items():
            pauli[qubit_to_index[qubit]] = str(gate).upper()
        pauli = "".join(pauli)
        # Cirq PauliStringPhasor is equivalent up to global phase to
        # exp(i*pi*(exponent_pos - exponent_neg)*P/2).
        angle = np.pi * (op.exponent_pos - op.exponent_neg)
        return [
            (
                pauli,
                _template_angle_for_plus_rotation(pauli, _maybe_float(angle)),
            )
        ]

    pauli = _cirq_pow_gate_pauli(cirq, op.gate)
    if pauli is not None:
        embedded = _embed_pauli_word(pauli, [op.qubits[0]], qubit_order)
        # Cirq XPow/YPow/ZPow gates are exp(-i*pi*exponent*P/2), up to global
        # phase, including cirq.rx/ry/rz helpers.
        return [
            (
                embedded,
                _template_angle_for_minus_rotation(
                    embedded, _maybe_float(np.pi * op.gate.exponent)
                ),
            )
        ]

    if isinstance(op.gate, cirq.HPowGate):
        return _hadamard_rotations(op.qubits[0], qubit_order)

    if isinstance(op.gate, cirq.CZPowGate):
        return _cz_rotations(op.qubits[0], op.qubits[1], qubit_order)

    if isinstance(op.gate, cirq.CNotPowGate):
        control, target = op.qubits
        return [
            *_hadamard_rotations(target, qubit_order),
            *_cz_rotations(control, target, qubit_order),
            *_hadamard_rotations(target, qubit_order),
        ]

    raise ValueError(f"Unsupported Cirq operation {op}.")


def _cirq_pow_gate_pauli(cirq, gate):
    if isinstance(gate, cirq.XPowGate):
        return "X"
    if isinstance(gate, cirq.YPowGate):
        return "Y"
    if isinstance(gate, cirq.ZPowGate):
        return "Z"
    return None


def _is_exponent_one(gate):
    exponent = _maybe_float(getattr(gate, "exponent", None))
    return exponent is not None and np.isclose(np.mod(exponent, 2.0), 1.0)


def _qiskit_pauli_types():
    try:
        from qiskit.quantum_info import Pauli, SparsePauliOp
    except ImportError as exc:
        raise ImportError(
            "Qiskit quantum_info is required for this conversion. Install "
            "qiskit before calling this function."
        ) from exc
    return SparsePauliOp, Pauli


def _qiskit_sparse_pauli_items(sparse_pauli):
    if hasattr(sparse_pauli, "to_list"):
        return sparse_pauli.to_list()
    if hasattr(sparse_pauli, "label_iter"):
        return list(sparse_pauli.label_iter())
    return [
        (pauli.to_label(), coeff)
        for pauli, coeff in zip(sparse_pauli.paulis, sparse_pauli.coeffs)
    ]


def _qiskit_pauli_rotations(qiskit, obj, qubit_order):
    try:
        from qiskit import QuantumCircuit
    except ImportError:
        QuantumCircuit = ()

    if isinstance(obj, QuantumCircuit):
        operations = _qiskit_circuit_operations(obj)
        qubit_order = _qiskit_resolve_qubit_order(obj, qubit_order)
    elif _is_qiskit_instruction_like(obj):
        operation, qubits = _qiskit_instruction_parts(obj)
        operations = [(operation, qubits)]
        if qubit_order is None:
            qubit_order = list(qubits)
        else:
            qubit_order = list(qubit_order)
    else:
        return None

    if len(operations) == 0:
        raise ValueError("Cannot convert an empty Qiskit circuit.")
    if not all(_is_supported_qiskit_gate(op, qubits) for op, qubits in operations):
        return None

    rotations = []
    for op, qubits in operations:
        rotations.extend(_qiskit_gate_rotations(op, qubits, qubit_order))
    return rotations


def _qiskit_resolve_qubit_order(circuit, qubit_order):
    if qubit_order is None:
        return list(circuit.qubits)

    resolved = []
    for qubit in qubit_order:
        if isinstance(qubit, int):
            resolved.append(circuit.qubits[qubit])
        else:
            resolved.append(qubit)

    circuit_qubits = set(circuit.qubits)
    missing = [qubit for qubit in resolved if qubit not in circuit_qubits]
    if missing:
        raise ValueError(f"qubit_order contains qubits not in the circuit: {missing}.")
    return resolved


def _qiskit_circuit_operations(circuit):
    operations = []
    for instruction in circuit.data:
        operation, qubits = _qiskit_instruction_parts(instruction)
        name = _qiskit_operation_name(operation)
        if name in {"barrier", "delay", "measure"}:
            continue
        operations.append((operation, qubits))
    return operations


def _is_qiskit_instruction_like(obj):
    if not hasattr(obj, "qubits"):
        return False
    return hasattr(obj, "operation") or hasattr(obj, "name")


def _qiskit_instruction_parts(instruction):
    if hasattr(instruction, "operation") and hasattr(instruction, "qubits"):
        return instruction.operation, list(instruction.qubits)
    if hasattr(instruction, "name") and hasattr(instruction, "qubits"):
        return instruction, list(instruction.qubits)
    try:
        operation, qubits, _clbits = instruction
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Unsupported Qiskit instruction representation {instruction}."
        ) from exc
    return operation, list(qubits)


def _is_supported_qiskit_gate(operation, qubits):
    if _qiskit_operation_condition(operation) is not None:
        return False

    name = _qiskit_operation_name(operation)
    if len(qubits) == 1:
        if name in {"rx", "ry", "rz"}:
            return len(getattr(operation, "params", ())) == 1
        return name in {"x", "y", "z", "s", "sdg", "t", "tdg", "sx", "sxdg", "h"}
    if len(qubits) == 2:
        return name in {"cz", "cx", "cnot"}
    return False


def _qiskit_gate_rotations(operation, qubits, qubit_order):
    name = _qiskit_operation_name(operation)

    if name in {"rx", "ry", "rz"}:
        pauli = _embed_pauli_word(name[-1].upper(), [qubits[0]], qubit_order)
        return [
            (
                pauli,
                _template_angle_for_minus_rotation(
                    pauli, _maybe_float(operation.params[0])
                ),
            )
        ]

    fixed_rotation = _qiskit_fixed_axis_angle(name)
    if fixed_rotation is not None:
        axis, angle = fixed_rotation
        pauli = _embed_pauli_word(axis, [qubits[0]], qubit_order)
        return [(pauli, _template_angle_for_minus_rotation(pauli, angle))]

    if name == "h":
        return _hadamard_rotations(qubits[0], qubit_order)

    if name == "cz":
        return _cz_rotations(qubits[0], qubits[1], qubit_order)

    if name in {"cx", "cnot"}:
        control, target = qubits
        return [
            *_hadamard_rotations(target, qubit_order),
            *_cz_rotations(control, target, qubit_order),
            *_hadamard_rotations(target, qubit_order),
        ]

    raise ValueError(f"Unsupported Qiskit operation {operation}.")


def _qiskit_fixed_axis_angle(name):
    return {
        "x": ("X", np.pi),
        "y": ("Y", np.pi),
        "z": ("Z", np.pi),
        "s": ("Z", np.pi / 2),
        "sdg": ("Z", -np.pi / 2),
        "t": ("Z", np.pi / 4),
        "tdg": ("Z", -np.pi / 4),
        "sx": ("X", np.pi / 2),
        "sxdg": ("X", -np.pi / 2),
    }.get(name)


def _qiskit_operation_name(operation):
    return str(getattr(operation, "name", "")).lower()


def _qiskit_operation_condition(operation):
    for attr in ("condition", "condition_expr"):
        condition = getattr(operation, attr, None)
        if condition is not None:
            return condition
    return None


def _hadamard_rotations(wire, wire_order):
    return [
        (_embed_pauli_word("X", [wire], wire_order), np.pi / 2),
        (_embed_pauli_word("X", [wire], wire_order), np.pi / 2),
        (_embed_pauli_word("Y", [wire], wire_order), -np.pi / 2),
    ]


def _cz_rotations(left, right, wire_order):
    return [
        (_embed_pauli_word("Z", [left], wire_order), -np.pi / 2),
        (_embed_pauli_word("Z", [right], wire_order), -np.pi / 2),
        (_embed_pauli_word("ZZ", [left, right], wire_order), np.pi / 2),
    ]


def _embed_pauli_word(pauli_word, wires, wire_order):
    pauli = ["I"] * len(wire_order)
    wire_to_index = {wire: index for index, wire in enumerate(wire_order)}
    for wire, op in zip(wires, pauli_word):
        pauli[wire_to_index[wire]] = op
    return "".join(pauli)


def _template_angle_for_minus_rotation(pauli, angle):
    """Angle for a target ``exp(-i angle P/2)`` operation."""
    if angle is None:
        return None
    return angle if "Y" in pauli else -angle


def _template_angle_for_plus_rotation(pauli, angle):
    """Angle for a target ``exp(+i angle P/2)`` operation."""
    if angle is None:
        return None
    return -angle if "Y" in pauli else angle


def _maybe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _pauli_rotations_to_mbqc(rotations):
    from mentpy.mbqc import hstack
    from mentpy.mbqc.templates import from_pauli
    from mentpy.operators import Ment, PauliOp

    circuits = []
    for pauli, angle in rotations:
        circuit = from_pauli(PauliOp(pauli))
        if angle is not None:
            for node in list(circuit.trainable_nodes):
                circuit[node] = Ment(angle, "XY")
        circuits.append(circuit)

    return (
        circuits[0] if len(circuits) == 1 else hstack(circuits, initialize_flow=False)
    )


def _pauli_to_pennylane_op(pauli, wire_order, qml):
    op = None
    for wire, char in zip(wire_order, pauli):
        if char == "I":
            continue
        factor = {
            "X": qml.PauliX,
            "Y": qml.PauliY,
            "Z": qml.PauliZ,
        }[
            char
        ](wire)
        op = factor if op is None else op @ factor
    return op if op is not None else qml.Identity(wire_order[0] if wire_order else 0)


def _default_wire_order(observable, wire_order):
    if wire_order is not None:
        return list(wire_order)
    if observable.n_qubits is None:
        raise ValueError("wire_order is required for an empty Observable.")
    return list(range(observable.n_qubits))


def _default_cirq_qubits(cirq, observable, qubit_order):
    if qubit_order is not None:
        return list(qubit_order)
    if observable.n_qubits is None:
        raise ValueError("qubit_order is required for an empty Observable.")
    return cirq.LineQubit.range(observable.n_qubits)


def _mbqcircuit_to_cirq(cirq, mbqcircuit, qubit_order=None, parameter_prefix="theta"):
    """Lower an MBQC pattern to a dynamic Cirq circuit."""
    import sympy

    from mentpy.mbqc.measurement_angles import MeasurementAngleResolver

    nodes = list(mbqcircuit.graph.nodes())
    if qubit_order is None:
        qubits = dict(zip(nodes, cirq.LineQubit.range(len(nodes))))
    elif isinstance(qubit_order, dict):
        qubits = dict(qubit_order)
    else:
        qubit_order = list(qubit_order)
        if len(qubit_order) != len(nodes):
            raise ValueError("qubit_order must provide one qubit per MBQC node.")
        qubits = dict(zip(nodes, qubit_order))

    out = cirq.Circuit()
    for node in mbqcircuit.inputc:
        out.append(cirq.H(qubits[node]))
    for u, v in mbqcircuit.graph.edges:
        out.append(cirq.CZ(qubits[u], qubits[v]))

    resolver = MeasurementAngleResolver(mbqcircuit)
    outcomes = resolver.zero_outcomes()
    order_index = {
        node: index for index, node in enumerate(mbqcircuit.measurement_order)
    }
    measurement_nodes = [
        node
        for node in mbqcircuit.measurement_order
        if mbqcircuit.measurements[node] is not None
    ]

    for node in measurement_nodes:
        command = resolver.resolve(
            node, np.zeros(len(mbqcircuit.trainable_nodes)), outcomes
        )
        if command.plane not in {"X", "Y", "XY"}:
            raise ValueError(
                f"Cirq export currently supports X/Y/XY measurements, got {command.plane}."
            )
        if command.trainable:
            angle = sympy.Symbol(f"{parameter_prefix}{command.trainable_index}")
        elif command.plane == "X":
            angle = 0.0
        elif command.plane == "Y":
            angle = np.pi / 2
        else:
            angle = command.angle

        out.append(cirq.rz(-angle)(qubits[node]))
        out.append(cirq.H(qubits[node]))
        key = f"m{node}"
        out.append(cirq.measure(qubits[node], key=key))

        correction = mbqcircuit.flow.correction_op(node).matrix[0]
        n_nodes = mbqcircuit.graph.number_of_nodes()
        x_targets = np.flatnonzero(correction[:n_nodes])
        z_targets = np.flatnonzero(correction[n_nodes:])
        for target in x_targets:
            if _is_future_correction_target(int(target), node, order_index):
                out.append(cirq.X(qubits[int(target)]).with_classical_controls(key))
        for target in z_targets:
            if _is_future_correction_target(int(target), node, order_index):
                out.append(cirq.Z(qubits[int(target)]).with_classical_controls(key))

    return out


def _mbqcircuit_to_qiskit(
    mbqcircuit, qubit_order=None, parameters=None, parameter_prefix="theta"
):
    """Lower an MBQC pattern to a dynamic Qiskit circuit."""
    from qiskit import QuantumCircuit
    from qiskit.circuit import ParameterVector

    from mentpy.mbqc.measurement_angles import MeasurementAngleResolver

    nodes = list(mbqcircuit.graph.nodes())
    node_to_qubit = _qiskit_mbqc_qubit_indices(nodes, qubit_order)

    measurement_nodes = [
        node
        for node in mbqcircuit.measurement_order
        if mbqcircuit.measurements[node] is not None
    ]
    classical_bit = {node: index for index, node in enumerate(measurement_nodes)}
    out = QuantumCircuit(len(nodes), len(measurement_nodes))

    if parameters is None:
        angle_values = ParameterVector(
            parameter_prefix, len(mbqcircuit.trainable_nodes)
        )
    else:
        angle_values = list(parameters)
        if len(angle_values) != len(mbqcircuit.trainable_nodes):
            raise ValueError(
                "parameters length must match the number of trainable nodes."
            )

    for node in mbqcircuit.inputc:
        out.h(node_to_qubit[node])
    for u, v in mbqcircuit.graph.edges:
        out.cz(node_to_qubit[u], node_to_qubit[v])

    resolver = MeasurementAngleResolver(mbqcircuit)
    outcomes = resolver.zero_outcomes()
    order_index = {
        node: index for index, node in enumerate(mbqcircuit.measurement_order)
    }

    for node in measurement_nodes:
        command = resolver.resolve(node, angle_values, outcomes, xy=True)
        out.rz(-command.angle, node_to_qubit[node])
        out.h(node_to_qubit[node])
        out.measure(node_to_qubit[node], classical_bit[node])

        correction = mbqcircuit.flow.correction_op(node).matrix[0]
        n_nodes = mbqcircuit.graph.number_of_nodes()
        x_targets = np.flatnonzero(correction[:n_nodes])
        z_targets = np.flatnonzero(correction[n_nodes:])
        for target in x_targets:
            target = int(target)
            if _is_future_correction_target(target, node, order_index):
                _qiskit_conditionally_apply(
                    out, "x", node_to_qubit[target], classical_bit[node]
                )
        for target in z_targets:
            target = int(target)
            if _is_future_correction_target(target, node, order_index):
                _qiskit_conditionally_apply(
                    out, "z", node_to_qubit[target], classical_bit[node]
                )

    return out


def _qiskit_mbqc_qubit_indices(nodes, qubit_order):
    if qubit_order is None:
        ordered_nodes = list(nodes)
    else:
        ordered_nodes = list(qubit_order)
        if len(ordered_nodes) != len(nodes) or set(ordered_nodes) != set(nodes):
            raise ValueError("qubit_order must list each MBQC node exactly once.")
    return {node: index for index, node in enumerate(ordered_nodes)}


def _qiskit_conditionally_apply(circuit, gate_name, qubit, clbit_index):
    clbit = circuit.clbits[clbit_index]
    if hasattr(circuit, "if_test"):
        with circuit.if_test((clbit, 1)):
            getattr(circuit, gate_name)(qubit)
        return

    instruction_set = getattr(circuit, gate_name)(qubit)
    if hasattr(instruction_set, "c_if"):
        return instruction_set.c_if(clbit, 1)

    raise NotImplementedError(
        "Qiskit export requires QuantumCircuit.if_test or InstructionSet.c_if "
        "support for adaptive MBQC corrections."
    )


def _is_future_correction_target(target, measured_node, order_index):
    return (
        target != measured_node
        and order_index.get(target, np.inf) > order_index[measured_node]
    )


def _mbqcircuit_to_cudaq(cudaq, mbqcircuit, qubit_order=None, parameters=None):
    """Lower an MBQC pattern to a CUDA-Q dynamic kernel."""
    from mentpy.mbqc.measurement_angles import MeasurementAngleResolver

    if not hasattr(cudaq, "make_kernel"):
        raise ImportError("CUDA-Q make_kernel API is required for MBQC export.")

    if parameters is None and len(mbqcircuit.trainable_nodes) > 0:
        made = cudaq.make_kernel(list)
        kernel, angle_values = made[0], made[1]
    else:
        kernel = cudaq.make_kernel()
        if parameters is None:
            angle_values = []
        else:
            angle_values = list(parameters)
            if len(angle_values) != len(mbqcircuit.trainable_nodes):
                raise ValueError(
                    "parameters length must match the number of trainable nodes."
                )

    nodes = list(mbqcircuit.graph.nodes())
    if qubit_order is not None and len(qubit_order) != len(nodes):
        raise ValueError("wire_order must provide one CUDA-Q qubit per MBQC node.")

    qubits = kernel.qalloc(len(nodes))
    qubit_by_node = {node: qubits[index] for index, node in enumerate(nodes)}

    for node in mbqcircuit.inputc:
        _cudaq_apply_kernel_gate(kernel, "h", qubit_by_node[node])
    for u, v in mbqcircuit.graph.edges:
        _cudaq_apply_two_qubit_gate(kernel, "cz", qubit_by_node[u], qubit_by_node[v])

    resolver = MeasurementAngleResolver(mbqcircuit)
    outcomes = resolver.zero_outcomes()
    order_index = {
        node: index for index, node in enumerate(mbqcircuit.measurement_order)
    }
    measurement_nodes = [
        node
        for node in mbqcircuit.measurement_order
        if mbqcircuit.measurements[node] is not None
    ]

    for node in measurement_nodes:
        command = resolver.resolve(node, angle_values, outcomes, xy=True)
        angle = command.angle
        _cudaq_apply_kernel_gate(kernel, "rz", -angle, qubit_by_node[node])
        _cudaq_apply_kernel_gate(kernel, "h", qubit_by_node[node])
        measurement = _cudaq_measure_z(kernel, qubit_by_node[node], f"m{node}")

        correction = mbqcircuit.flow.correction_op(node).matrix[0]
        n_nodes = mbqcircuit.graph.number_of_nodes()
        x_targets = np.flatnonzero(correction[:n_nodes])
        z_targets = np.flatnonzero(correction[n_nodes:])
        for target in x_targets:
            target = int(target)
            if _is_future_correction_target(target, node, order_index):
                _cudaq_conditionally_apply(
                    kernel, measurement, "x", qubit_by_node[target]
                )
        for target in z_targets:
            target = int(target)
            if _is_future_correction_target(target, node, order_index):
                _cudaq_conditionally_apply(
                    kernel, measurement, "z", qubit_by_node[target]
                )

    return kernel


def _cudaq_apply_kernel_gate(kernel, name, *args):
    gate = getattr(kernel, name, None)
    if gate is None:
        raise NotImplementedError(f"CUDA-Q Kernel API does not expose {name}.")
    return gate(*args)


def _cudaq_apply_two_qubit_gate(kernel, name, left, right):
    gate = getattr(kernel, name, None)
    if gate is not None:
        return gate(left, right)

    fallback = {
        "cz": "z",
        "cx": "x",
    }.get(name)
    if fallback is not None:
        controlled_gate = getattr(kernel, fallback, None)
        if controlled_gate is not None and hasattr(controlled_gate, "ctrl"):
            return controlled_gate.ctrl(left, right)

    raise NotImplementedError(f"CUDA-Q Kernel API does not expose {name}.")


def _cudaq_measure_z(kernel, qubit, key):
    try:
        return kernel.mz(qubit, key)
    except TypeError:
        return kernel.mz(qubit)


def _cudaq_conditionally_apply(kernel, measurement, gate_name, qubit):
    if not hasattr(kernel, "c_if"):
        raise NotImplementedError(
            "CUDA-Q Kernel API c_if support is required for adaptive MBQC export."
        )

    def then_function():
        _cudaq_apply_kernel_gate(kernel, gate_name, qubit)

    return kernel.c_if(measurement, then_function)


def _cudaq_kernel_to_mbqc(
    cudaq, kernel, n_qubits=None, wire_order=None, parameters=None, spin_error=None
):
    if not hasattr(cudaq, "translate"):
        if spin_error is not None:
            raise spin_error
        raise NotImplementedError(
            "CUDA-Q kernel import requires cudaq.translate(..., format='openqasm2')."
        )

    args = parameters if isinstance(parameters, (list, tuple)) else []
    try:
        qasm = cudaq.translate(kernel, *args, format="openqasm2")
    except TypeError:
        qasm = cudaq.translate("openqasm2", kernel, *args)

    cirq = _require("cirq", "Cirq")
    try:
        from cirq.contrib.qasm_import import circuit_from_qasm
    except ImportError as exc:
        raise ImportError(
            "CUDA-Q kernel import requires Cirq's QASM importer. Install "
            "mentpy[interop] so the 'ply' dependency is available."
        ) from exc

    cirq_circuit = circuit_from_qasm(qasm)
    cirq_circuit = _cirq_without_measurements(cirq, cirq_circuit)

    if wire_order is None:
        qubits = sorted(cirq_circuit.all_qubits())
        if n_qubits is not None and len(qubits) < n_qubits:
            qubits = list(cirq.LineQubit.range(n_qubits))
        wire_order = qubits

    return from_cirq(cirq_circuit, qubit_order=wire_order)


def _cirq_without_measurements(cirq, circuit):
    return cirq.Circuit(
        op
        for op in circuit.all_operations()
        if not isinstance(op.gate, cirq.MeasurementGate)
    )


def _cudaq_spin_terms(obj):
    if _is_cudaq_spin_term(obj):
        return [obj]
    try:
        return [term.copy() if hasattr(term, "copy") else term for term in obj]
    except TypeError:
        pass
    if hasattr(obj, "for_each_term"):
        terms = []
        obj.for_each_term(
            lambda term: terms.append(term.copy() if hasattr(term, "copy") else term)
        )
        return terms
    raise NotImplementedError(
        "from_cudaq currently supports CUDA-Q spin operators and spin terms."
    )


def _is_cudaq_spin_term(obj):
    return hasattr(obj, "evaluate_coefficient") or hasattr(obj, "get_pauli_word")


def _cudaq_wire_order(obj, terms, n_qubits):
    if n_qubits is not None:
        return list(range(n_qubits))

    degrees = []
    for source in (obj, *terms):
        for degree in getattr(source, "degrees", []) or []:
            if degree not in degrees:
                degrees.append(degree)
    if degrees:
        return list(range(max(degrees) + 1))

    max_degree = -1
    for term in terms:
        for target, _op in _cudaq_term_ops(term, 0):
            max_degree = max(max_degree, target)
    return list(range(max_degree + 1)) if max_degree >= 0 else [0]


def _cudaq_term_coefficient(term, parameters):
    if hasattr(term, "evaluate_coefficient"):
        try:
            coefficient = term.evaluate_coefficient(parameters)
        except TypeError:
            coefficient = term.evaluate_coefficient()
        if coefficient is not None:
            return coefficient
    if hasattr(term, "get_coefficient"):
        return term.get_coefficient()
    return 1.0


def _cudaq_term_ops(term, pad_identities):
    try:
        return [
            (element.target, _cudaq_pauli_char(element.as_pauli()))
            for element in term
            if _cudaq_pauli_char(element.as_pauli()) != "I"
        ]
    except TypeError:
        if hasattr(term, "for_each_pauli"):
            ops = []

            def append_op(pauli, target):
                op = _cudaq_pauli_char(pauli)
                if op != "I":
                    ops.append((target, op))

            term.for_each_pauli(append_op)
            return ops

    if hasattr(term, "get_pauli_word"):
        word = term.get_pauli_word(pad_identities)
        return [(index, op) for index, op in enumerate(str(word).upper()) if op != "I"]

    raise NotImplementedError("Unsupported CUDA-Q spin term representation.")


def _cudaq_pauli_char(pauli):
    name = getattr(pauli, "name", str(pauli)).upper()
    if "." in name:
        name = name.rsplit(".", 1)[-1]
    if name in {"X", "Y", "Z", "I"}:
        return name
    raise ValueError(f"Unsupported CUDA-Q Pauli operator {pauli}.")


def _cudaq_identity(spin, wire_order):
    if hasattr(spin, "i"):
        try:
            return spin.i()
        except TypeError:
            return spin.i(wire_order[0] if wire_order else 0)
    raise ImportError("CUDA-Q spin.i identity helper is not available.")

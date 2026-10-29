# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Optional PyZX-backed circuit compilation."""

from dataclasses import dataclass
import importlib
import os
import re
from pathlib import Path
from typing import Any, Dict

__all__ = ["CompileResult", "compile", "from_qasm"]


@dataclass(frozen=True)
class CompileResult:
    """Result returned by :func:`compile`.

    Attributes
    ----------
    backend:
        Compiler backend used for the reduction.
    reduction:
        Reduction strategy applied by the backend.
    input_type:
        Type of input accepted by the compiler.
    before:
        Structural graph statistics before reduction.
    after:
        Structural graph statistics after reduction.
    reduced:
        Reduced backend object. For the first PyZX-backed version this is a
        PyZX graph reduced in place by PyZX.
    circuit:
        Optional extracted PyZX circuit. This is populated when ``extract=True``
        and PyZX can extract a circuit from the reduced diagram.
    pattern:
        Executable MBQCircuit, unless ``to_mbqc=False``.

    Group
    -----
    compiler
    """

    backend: str
    reduction: str
    input_type: str
    before: Dict[str, Any]
    after: Dict[str, Any]
    reduced: Any
    circuit: Any = None
    pattern: Any = None


def compile(
    obj: Any,
    *,
    backend: str = "pyzx",
    reduction: str = "full_reduce",
    extract: bool = False,
    copy: bool = True,
    quiet: bool = True,
    to_mbqc: bool = True,
):
    """Compile a QASM string or PyZX object using an optional backend.

    PyZX is imported lazily so importing :mod:`mentpy` does not require the
    optional dependency. The reduced object in the returned result is the PyZX
    graph produced from the input and simplified with the selected PyZX
    reduction. By default, ``result.pattern`` is an executable MBQCircuit and
    ``result.circuit`` is the extracted gate circuit. Set ``to_mbqc=False`` to
    skip pattern construction, and ``extract=True`` to still extract a circuit.

    Group
    -----
    compiler
    """

    if backend != "pyzx":
        raise ValueError("Only the 'pyzx' compiler backend is currently supported.")

    zx = _require_pyzx()
    graph, input_type, source_circuit = _as_pyzx_graph(zx, obj, copy_graph=copy)
    before = _graph_stats(graph)
    before.update(_circuit_stats(source_circuit, prefix="input_"))

    graph = _reduce_graph(zx, graph, reduction=reduction, quiet=quiet)
    after = _graph_stats(graph)
    circuit = _extract_circuit(zx, graph, quiet=quiet) if extract or to_mbqc else None
    after.update(_circuit_stats(circuit, prefix="extracted_"))

    return CompileResult(
        backend="pyzx",
        reduction=reduction,
        input_type=input_type,
        before=before,
        after=after,
        reduced=graph,
        circuit=circuit,
        pattern=_circuit_to_mbqc(circuit) if to_mbqc else None,
    )


def from_qasm(source, *, optimize=False):
    """Import a unitary OpenQASM 2 circuit as an executable MBQC pattern.

    ``source`` is QASM text or a path. Wire zero is the leftmost tensor axis;
    supply ``|0>`` inputs explicitly when matching a gate-circuit simulator.
    Set ``optimize=True`` for PyZX full reduction and circuit extraction before
    translating to the one-bit teleportation gate set. Measurements, resets,
    classical conditions and unbound parameters are rejected.

    Parameters
    ----------
    source : str or pathlib.Path
        OpenQASM 2 source text or a circuit file path.
    optimize : bool
        Apply ZX simplification before translating the gate circuit.

    Returns
    -------
    MBQCircuit
        Fixed-angle pattern with input/output nodes and causal flow.

    Group
    -----
    compiler
    """
    zx = _require_pyzx()
    graph, _, circuit = _as_pyzx_graph(zx, source, copy_graph=True)
    if optimize:
        graph = _reduce_graph(zx, graph, reduction="full_reduce", quiet=True)
        circuit = _extract_circuit(zx, graph, quiet=True)
    return _circuit_to_mbqc(circuit)


def _circuit_to_mbqc(circuit):
    """Translate H, Z/X phases and entanglers to a causal-flow pattern.

    A measured wire segment implements H Rz(-angle). Two such segments
    implement Rz; surrounding it with H implements Rx. Entangling edges are
    placed between fresh wire ends so repeated CZs never alias graph edges.
    """
    import numpy as np
    from mentpy.mbqc import GraphState, MBQCircuit
    from mentpy.operators import Ment

    graph = GraphState()
    inputs = list(range(circuit.qubits))
    graph.add_nodes_from(inputs)
    wires = inputs.copy()
    measurements = {}
    order = []
    successors = {}

    def teleport(wire, angle=0.0):
        old = wires[wire]
        new = len(graph)
        graph.add_edge(old, new)
        measurements[old] = Ment(float(angle), "XY")
        successors[old] = new
        order.append(old)
        wires[wire] = new

    def phase(wire, angle):
        teleport(wire, -angle)
        teleport(wire)

    def cz(left, right):
        graph.add_edge(wires[left], wires[right])
        # Identity teleportations separate this entangler from later gates.
        for wire in (left, right):
            teleport(wire)
            teleport(wire)

    for gate in circuit.to_basic_gates().gates:
        name = gate.name
        if name == "HAD":
            teleport(gate.target)
        elif name in {"ZPhase", "Z", "S", "T"}:
            phase(gate.target, float(gate.phase) * np.pi)
        elif name in {"XPhase", "NOT", "SX"}:
            teleport(gate.target)
            phase(gate.target, float(gate.phase) * np.pi)
            teleport(gate.target)
        elif name == "CZ":
            cz(gate.control, gate.target)
        elif name == "CNOT":
            teleport(gate.target)
            cz(gate.control, gate.target)
            teleport(gate.target)
        else:
            raise ValueError(f"Unsupported unitary QASM gate: {name}.")

    measurements.update({node: None for node in wires})
    pattern = MBQCircuit(
        graph, inputs, wires, measurements=measurements, initialize_flow=False
    )
    pattern._measurement_order = order + wires
    # The construction supplies a causal flow; no expensive search is needed.
    flow = pattern.flow
    flow.func = successors.get
    indices = {node: i for i, node in enumerate(order + wires)}
    flow.partial_order = lambda u, v: indices[u] < indices[v]
    flow.flow_initialized = True
    flow.name = "cFlow"
    flow.depth = len(order)
    flow.layers = [[node] for node in order] + [wires.copy()]
    # Preserve wire order even when the last gate touched a lower wire.
    pattern._quantum_output_nodes = wires.copy()
    return pattern


def _require_pyzx():
    try:
        return importlib.import_module("pyzx")
    except ModuleNotFoundError as exc:
        if exc.name != "pyzx":
            raise
        raise ImportError(
            "PyZX is required for mentpy.compile. Install it with "
            "`pip install pyzx` or the `mentpy[compiler]` extra."
        ) from exc


def _as_pyzx_graph(zx, obj, *, copy_graph):
    circuit_type = getattr(zx, "Circuit", None)

    if isinstance(obj, (str, os.PathLike)):
        is_path = isinstance(obj, os.PathLike) or (
            "OPENQASM" not in obj and os.path.exists(obj)
        )
        source = Path(obj).read_text() if is_path else obj
        source = re.sub(r"//[^\n]*", "", source)
        if not re.match(r"\s*OPENQASM\s+2\.0\s*;", source):
            raise ValueError("Only unitary OpenQASM 2.0 programs are supported.")
        if re.search(r"\b(measure|reset|if)\b", source):
            raise ValueError(
                "Only unitary QASM is supported; remove measurements, resets and conditions."
            )

        # PennyLane emits top-level gphase statements. They have no effect on
        # MBQC output probabilities or expectations. Never drop a phase inside
        # a custom gate, where later controls could make it observable.
        def drop_phase(match):
            prefix = source[: match.start()]
            if prefix.count("{") != prefix.count("}"):
                raise ValueError("Unsupported global phase inside a custom gate.")
            return ""

        source = re.sub(r"\bgphase\s*\([^;]*?\)[^;]*;", drop_phase, source)
        # PyZX expects the descriptor and include on separate lines.
        source = re.sub(r";\s*", ";\n", source)
        circuit = circuit_type.from_qasm(source)
        return circuit.to_graph(), "path" if is_path else "qasm", circuit

    if circuit_type is not None and isinstance(obj, circuit_type):
        return obj.to_graph(), "pyzx.Circuit", obj

    if _is_pyzx_graph(obj):
        return _copy_graph(obj) if copy_graph else obj, "pyzx.Graph", None

    raise TypeError(
        "Expected a QASM string, PyZX Circuit, or PyZX Graph; "
        f"got {type(obj).__name__}."
    )


def _is_pyzx_graph(obj):
    return all(
        hasattr(obj, name)
        for name in ("num_vertices", "num_edges", "vertices", "edges")
    )


def _copy_graph(graph):
    copy_method = getattr(graph, "copy", None)
    if copy_method is None:
        raise TypeError("PyZX graph inputs must provide copy() or use copy=False.")
    return copy_method()


def _reduce_graph(zx, graph, *, reduction, quiet):
    reducers = {
        "full_reduce": "full_reduce",
        "teleport_reduce": "teleport_reduce",
        "clifford_simp": "clifford_simp",
    }
    if reduction not in reducers:
        raise ValueError(
            "Unsupported PyZX reduction. Expected one of "
            f"{sorted(reducers)}, got {reduction!r}."
        )

    reducer = getattr(zx, reducers[reduction], None)
    if reducer is None:
        reducer = getattr(importlib.import_module("pyzx.simplify"), reducers[reduction])
    result = (
        reducer(graph)
        if reduction == "teleport_reduce"
        else reducer(graph, quiet=quiet)
    )
    return result if _is_pyzx_graph(result) else graph


def _extract_circuit(zx, graph, *, quiet):
    extractor = getattr(zx, "extract_circuit", None)
    if extractor is None:
        extractor = importlib.import_module("pyzx.extract").extract_circuit
    graph_copy = _copy_graph(graph)
    # teleport_reduce preserves circuit-like structure, rather than producing
    # the graph-like form required by the general extractor.
    if not zx.is_graph_like(graph_copy):
        zx.to_graph_like(graph_copy)
    try:
        return extractor(graph_copy, quiet=quiet)
    except TypeError:
        return extractor(graph_copy)


def _graph_stats(graph):
    return {
        "vertices": _read_count(graph, "num_vertices"),
        "edges": _read_count(graph, "num_edges"),
        "inputs": _read_count(graph, "num_inputs"),
        "outputs": _read_count(graph, "num_outputs"),
    }


def _circuit_stats(circuit, *, prefix):
    if circuit is None:
        return {}
    gates = list(getattr(circuit, "gates", []))
    stats = {f"{prefix}gates": len(gates), f"{prefix}t_gates": _count_t_gates(gates)}
    two_qubit = 0
    for gate in gates:
        qubits = getattr(gate, "qubits", None)
        if qubits is not None and len(qubits) == 2:
            two_qubit += 1
            continue
        target = getattr(gate, "target", None)
        control = getattr(gate, "control", None)
        if target is not None and control is not None:
            two_qubit += 1
    stats[f"{prefix}two_qubit_gates"] = two_qubit
    return stats


def _count_t_gates(gates):
    total = 0
    for gate in gates:
        name = gate.__class__.__name__.lower()
        if name in {"t", "tdg", "tgate"}:
            total += 1
            continue
        phase = getattr(gate, "phase", None)
        if getattr(phase, "denominator", None) == 4 and "phase" in name:
            total += 1
    return total


def _read_count(obj, name):
    value = getattr(obj, name, None)
    if value is None:
        return None
    if callable(value):
        return value()
    return value

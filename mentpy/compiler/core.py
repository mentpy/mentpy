# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Optional PyZX-backed circuit compilation."""

from dataclasses import dataclass
import importlib
import os
from typing import Any, Dict

__all__ = ["CompileResult", "compile"]


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


def compile(
    obj: Any,
    *,
    backend: str = "pyzx",
    reduction: str = "full_reduce",
    extract: bool = False,
    copy: bool = True,
    quiet: bool = True,
):
    """Compile a QASM string or PyZX object using an optional backend.

    PyZX is imported lazily so importing :mod:`mentpy` does not require the
    optional dependency. The reduced object in the returned result is the PyZX
    graph produced from the input and simplified with the selected PyZX
    reduction. Set ``extract=True`` to also ask PyZX for an equivalent circuit.

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

    _reduce_graph(zx, graph, reduction=reduction, quiet=quiet)
    after = _graph_stats(graph)
    circuit = _extract_circuit(zx, graph, quiet=quiet) if extract else None
    after.update(_circuit_stats(circuit, prefix="extracted_"))

    return CompileResult(
        backend="pyzx",
        reduction=reduction,
        input_type=input_type,
        before=before,
        after=after,
        reduced=graph,
        circuit=circuit,
    )


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

    if isinstance(obj, os.PathLike):
        if circuit_type is None or not hasattr(circuit_type, "load"):
            raise RuntimeError("The installed PyZX version cannot load circuit files.")
        circuit = circuit_type.load(os.fspath(obj))
        return circuit.to_graph(), "path", circuit

    if isinstance(obj, str):
        if circuit_type is None or not hasattr(circuit_type, "from_qasm"):
            raise RuntimeError("The installed PyZX version cannot parse QASM strings.")
        if os.path.exists(obj):
            circuit = circuit_type.load(obj)
            return circuit.to_graph(), "path", circuit
        circuit = circuit_type.from_qasm(obj)
        return circuit.to_graph(), "qasm", circuit

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
    try:
        return reducer(graph, quiet=quiet)
    except TypeError:
        return reducer(graph)


def _extract_circuit(zx, graph, *, quiet):
    extractor = getattr(zx, "extract_circuit", None)
    if extractor is None:
        extractor = importlib.import_module("pyzx.extract").extract_circuit
    graph_copy = _copy_graph(graph)
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
        if phase in {1, -1} and "phase" in name:
            total += 1
    return total


def _read_count(obj, name):
    value = getattr(obj, name, None)
    if value is None:
        return None
    if callable(value):
        return value()
    return value

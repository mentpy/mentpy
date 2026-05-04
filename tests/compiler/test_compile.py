# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Tests for optional compiler integrations."""

import pytest

import mentpy as mp
import mentpy.compiler.core as compiler_core

QASM_HH = """OPENQASM 2.0;
include "qelib1.inc";
qreg q[1];
h q[0];
h q[0];
"""


def test_compile_missing_pyzx_has_actionable_error(monkeypatch):
    real_import_module = compiler_core.importlib.import_module

    def fake_import_module(name, *args, **kwargs):
        if name == "pyzx":
            raise ModuleNotFoundError("No module named 'pyzx'", name="pyzx")
        return real_import_module(name, *args, **kwargs)

    monkeypatch.setattr(compiler_core.importlib, "import_module", fake_import_module)

    with pytest.raises(ImportError, match="mentpy\\[compiler\\]"):
        mp.compile(QASM_HH)


def test_compile_reduces_qasm_and_extracts_circuit():
    pytest.importorskip("pyzx")

    result = mp.compile(QASM_HH, extract=True)

    assert result.backend == "pyzx"
    assert result.reduction == "full_reduce"
    assert result.input_type == "qasm"
    assert result.circuit is not None
    assert result.after["vertices"] <= result.before["vertices"]
    assert result.after["edges"] <= result.before["edges"]
    assert result.before["input_gates"] == 2
    assert result.after["extracted_gates"] == 0


def test_compile_accepts_pyzx_graph_without_mutating_by_default():
    zx = pytest.importorskip("pyzx")
    graph = zx.Circuit.from_qasm(QASM_HH).to_graph()
    original_vertices = graph.num_vertices()

    result = mp.compile(graph)

    assert result.input_type == "pyzx.Graph"
    assert graph.num_vertices() == original_vertices
    assert result.after["vertices"] <= original_vertices


def test_compile_rejects_unknown_reduction():
    pytest.importorskip("pyzx")

    with pytest.raises(ValueError, match="Unsupported PyZX reduction"):
        mp.compile(QASM_HH, reduction="not-a-pass")


def test_compile_rejects_unknown_backend():
    with pytest.raises(ValueError, match="'pyzx'"):
        mp.compile(QASM_HH, backend="other")

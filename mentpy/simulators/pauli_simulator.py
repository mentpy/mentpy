# Copyright 2026 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Sparse Pauli propagation for causal-flow MBQC patterns."""

from numbers import Integral
from importlib.metadata import distribution, PackageNotFoundError
import sys

import numpy as np

from mentpy import _rust
from mentpy.mbqc.measurement_angles import MeasurementAngleResolver
from mentpy.operators import Observable
from .base_simulator import BaseSimulator
from .results import normalize_output_form, validate_shots
from .rust_simulator import RustSimulator

__all__ = ["PauliSimulator"]


class PauliSimulator(BaseSimulator):
    """Propagate observables through causal-flow paths in Rust.

    Notes
    -----
    Expectations and gradients use sparse Pauli sums, without constructing a
    resource or output state. Clifford propagation uses PPVM's packed words;
    a native reverse-mode tape differentiates rotation coefficients. Arbitrary
    XY angles can cause exponential term growth. Exact calculations raise at
    the memory limits; no terms are approximately discarded.

    Intermediate outcomes and output ordering follow the other backends.
    ``force0=False`` evaluates a corrected trajectory, resampling it for each
    shot when estimating an observable. ``run``, ``probabilities`` and ``sample``
    delegate to the Rust state simulator and retain its dense-output costs.
    A finite-shot expectation samples the Pauli eigenvalue directly, so seeded
    bit sequences can differ from state-based sampling with the same law.

    Group
    -----
    simulators
    """

    _statevector_output = True

    def __init__(
        self,
        mbqcircuit,
        input_state=None,
        *,
        force0=True,
        seed=None,
        max_terms=100000,
        max_tape=5000000,
        max_elements=2**26,
        schedule=None,
        output_form="dm",
        device="cpu",
        max_memory=2**28,
    ):
        """Create a sparse expectation simulator.

        Parameters
        ----------
        mbqcircuit : MBQCircuit
            Causal-flow pattern with X, Y or XY measurements, unmeasured
            outputs and at most 1024 logical wires. Other patterns raise an
            error instead of silently selecting a different algorithm.
        input_state : array-like, optional
            Normalized vector in input-node order. Default plus inputs and
            additional plus-state path roots are represented implicitly.
        force0 : bool
            Postselect zero outcomes, or sample corrected trajectories.
        seed : int, optional
            Seed for intermediate and output sampling.
        max_terms : int
            Limit on distinct Pauli terms in each expansion, including
            zero-valued terms needed by differentiation.
        max_tape : int
            CPU limit on scalar nodes in the reverse-mode tape. Each node stores
            at most two parent edges, in addition to an adjoint during backprop.
        max_elements : int
            Tensor-size limit for dense-output methods delegated to Rust.
        schedule : sequence, optional
            Measurement schedule respecting causal flow.
        output_form : str
            Default ``run`` form, "dm" or "sv".
        device : str
            "cpu" or "cuda:N". CUDA uses cuPauliProp 0.5 on Linux with an
            NVIDIA driver supporting CUDA 13. Libraries are included in the
            standard Linux installation and loaded only for GPU expectations.
        max_memory : int
            Byte limit for GPU buffers owned by this simulator, including
            saved expansions and workspace. Library/context allocations and
            host metadata are outside this limit. Default 256 MiB.
        """
        super().__init__(mbqcircuit)
        for name, value in (("max_terms", max_terms), ("max_tape", max_tape)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{name} must be a positive integer.")
        self.max_terms = int(max_terms)
        self.max_tape = int(max_tape)
        self.max_elements = max_elements
        self.device = device
        gpu = None
        if device != "cpu":
            if (
                not isinstance(device, str)
                or not device.startswith("cuda:")
                or not device[5:].isdigit()
            ):
                raise ValueError('device must be "cpu" or "cuda:N".')
            if (
                isinstance(max_memory, bool)
                or not isinstance(max_memory, Integral)
                or max_memory < 1
            ):
                raise ValueError(
                    "max_memory must be a positive integer number of bytes."
                )
            if sys.platform != "linux":
                raise ValueError("cuPauliProp requires Linux and an NVIDIA GPU.")
            try:
                library = distribution("cupauliprop-cu13").locate_file(
                    "cuquantum/lib/libcupauliprop.so.0"
                )
                runtime = distribution("nvidia-cuda-runtime").locate_file(
                    "nvidia/cu13/lib/libcudart.so.13"
                )
            except PackageNotFoundError as exc:
                raise ImportError(
                    "CUDA libraries are missing; reinstall mentpy to restore its Linux dependencies."
                ) from exc
            gpu = (str(library), str(runtime), int(device[5:]), int(max_memory))
        self.peak_terms = 0
        self.force0 = force0
        self.rng = np.random.default_rng(seed)
        self._default_output_form = normalize_output_form(output_form)
        self.resolver = MeasurementAngleResolver(mbqcircuit, schedule)
        self.resolver.validate_dependencies()
        operations, output_wires = self._extract()
        self._circuit = _rust.PauliCircuit(
            len(output_wires),
            len(mbqcircuit.input_nodes),
            output_wires,
            operations,
            len(self.resolver.measurement_nodes),
            self.max_terms,
            self.max_tape,
            gpu=gpu,
        )
        self._adaptive = any(
            self.resolver._dependencies(node)
            for node in self.resolver.measurement_nodes
        )
        self._fixed = None
        if not self._adaptive:
            values, indices, _ = self._resolve(
                np.zeros(len(mbqcircuit.trainable_nodes)), None
            )
            self._fixed = np.asarray(values)
            self._dest = np.asarray([i for i, p in enumerate(indices) if p is not None])
            self._source = np.asarray([p for p in indices if p is not None], dtype=int)
            self._dest = self._dest.astype(int)
        self.reset(input_state)

    def _extract(self):
        """Translate flow paths to logical J and CZ operations once."""
        pattern = self.mbqcircuit
        flow = pattern.flow
        flow.initialize_flow()
        if flow.name != "cFlow" or flow.func is None:
            # A graph with no measurements already is a graph-state circuit.
            if self.resolver.measurement_nodes:
                raise ValueError("The pauli backend requires causal flow.")
        if set(pattern.quantum_output_nodes) != set(pattern.output_nodes):
            raise ValueError("The pauli backend requires unmeasured output nodes.")
        self.resolver._flow_corrections()
        successors = {node: flow(node) for node in self.resolver.measurement_nodes}
        targets = set(successors.values())
        roots = list(pattern.input_nodes) + [
            node
            for node in pattern.graph
            if node not in targets and node not in pattern.input_nodes
        ]
        active = {node: wire for wire, node in enumerate(roots)}
        operations, edges = [], set()
        for i, node in enumerate(self.resolver.measurement_nodes):
            target = successors[node]
            if (
                node not in active
                or target in active
                or target not in pattern.graph[node]
            ):
                raise ValueError(
                    "The pauli backend requires disjoint causal-flow paths."
                )
            wire = active[node]
            for neighbor in pattern.graph[node]:
                edge = frozenset((node, neighbor))
                if edge in edges:
                    continue
                edges.add(edge)
                if neighbor != target:
                    if neighbor not in active:
                        raise ValueError("Measurement order must respect causal flow.")
                    operations.append((1, wire, active[neighbor]))
            operations.append((0, wire, i))
            active[target] = active.pop(node)
        if set(active) != set(pattern.quantum_output_nodes):
            raise ValueError(
                "The pauli backend requires paths ending at quantum outputs."
            )
        for u, v in pattern.graph.edges:
            if frozenset((u, v)) not in edges:
                operations.append((1, active[u], active[v]))
        return operations, [active[node] for node in pattern.quantum_output_nodes]

    def reset(self, input_state=None):
        """Clear outcomes and optionally replace the normalized input vector."""
        if input_state is not None:
            state = np.asarray(input_state, dtype=complex)
            if state.shape != (
                2 ** len(self.mbqcircuit.input_nodes),
            ) or not np.isclose(np.linalg.norm(state), 1):
                raise ValueError(
                    "input_state must be a normalized vector in input-node order."
                )
            self.input_state = state.copy()
        self.outcomes = {}
        self.peak_terms = 0

    def _resolve(self, angles, rng):
        values, indices, outcomes = [], [], {}
        for node in self.resolver.measurement_nodes:
            command = self.resolver.resolve(node, angles, outcomes, xy=True)
            if command.plane not in {"X", "Y", "XY"}:
                raise ValueError("The pauli backend supports X, Y and XY measurements.")
            values.append(float(command.angle))
            indices.append(command.trainable_index if command.trainable else None)
            outcomes[node] = 0 if rng is None else int(rng.random() >= 0.5)
        return values, indices, outcomes

    def _angles(self, angles, rng):
        self.resolver.validate_parameters(angles)
        if self._fixed is None:
            values, indices, self.outcomes = self._resolve(angles, rng)
            dest = [i for i, p in enumerate(indices) if p is not None]
            source = [p for p in indices if p is not None]
        else:
            values = self._fixed.copy()
            dest, source = self._dest, self._source
            values[dest] = np.asarray(angles, dtype=float)[source]
            measured = self.resolver.measurement_nodes
            outcomes = (
                np.zeros(len(measured), dtype=int)
                if rng is None
                else (rng.random(len(measured)) >= 0.5).astype(int)
            )
            self.outcomes = dict(zip(measured, map(int, outcomes)))
        return values, dest, source

    def _evaluate(self, values, observable, gradient=False, active=None):
        inputs = (
            {}
            if self.input_state is None
            else {"input_re": self.input_state.real, "input_im": self.input_state.imag}
        )
        value, derivative, peak = self._circuit.expectation(
            values,
            [(word, float(np.real(c))) for word, c in observable.terms.items()],
            constant=float(np.real(observable.constant)),
            gradient=gradient,
            active=None if active is None else active.tolist(),
            **inputs,
        )
        self.peak_terms = max(self.peak_terms, peak)
        return value, derivative

    def expectation(
        self,
        angles,
        observable,
        shots=None,
        seed=None,
        *,
        input_state=None,
        output_form=None,
    ):
        """Evaluate a Pauli sum exactly, or sample each term's eigenvalue."""
        if not isinstance(observable, Observable):
            observable = Observable(observable)
        observable._validate_qubits(len(self.mbqcircuit.quantum_output_nodes))
        if shots is not None:
            shots = validate_shots(shots)
        self.reset(input_state)
        rng = self.rng if seed is None else np.random.default_rng(seed)
        if shots is None:
            values, _, _ = self._angles(angles, None if self.force0 else rng)
            return self._evaluate(values, observable)[0]
        value = np.real(observable.constant)
        if self.force0:
            values, _, _ = self._angles(angles, None)
        else:
            self.resolver.validate_parameters(angles)
        for word, coefficient in observable.terms.items():
            term = Observable(word)
            if self.force0:
                mean = self._evaluate(values, term)[0]
                estimate = (
                    2 * rng.binomial(shots, np.clip((1 + mean) / 2, 0, 1)) / shots - 1
                )
            else:
                total = 0
                for _ in range(shots):
                    values, _, _ = self._angles(angles, rng)
                    mean = self._evaluate(values, term)[0]
                    total += 1 if rng.random() < np.clip((1 + mean) / 2, 0, 1) else -1
                estimate = total / shots
            value += np.real(coefficient) * estimate
        return float(value)

    def expectation_and_grad(self, angles, observable, *, input_state=None, shots=None):
        """Return an exact expectation and its native reverse-mode gradient."""
        if shots is not None or not self.force0:
            raise ValueError(
                "Pauli autodiff requires an exact force0=True expectation."
            )
        if not isinstance(observable, Observable):
            observable = Observable(observable)
        observable._validate_qubits(len(self.mbqcircuit.quantum_output_nodes))
        self.reset(input_state)
        values, dest, source = self._angles(angles, None)
        active = np.zeros(len(values), dtype=bool)
        active[dest] = True
        value, derivative = self._evaluate(
            values, observable, gradient=bool(len(angles)), active=active
        )
        result = np.zeros(len(angles))
        np.add.at(result, source, np.asarray(derivative)[dest])
        return value, result

    def measure(self, angle, **kwargs):
        """Use :meth:`run` to execute a complete pattern."""
        raise NotImplementedError("Use run() to execute a complete pattern.")

    def _dense(self, method, angles, **kwargs):
        self.reset(kwargs.pop("input_state", None))
        # Validate the supported measurement set before selecting the dense
        # output representation. Validation must not advance the user's RNG.
        self._angles(angles, None)
        simulator = RustSimulator(
            self.mbqcircuit,
            self.input_state,
            force0=self.force0,
            seed=self.rng,
            schedule=self.resolver.schedule,
            output_form=self._default_output_form,
            max_elements=self.max_elements,
        )
        result = getattr(simulator, method)(angles, **kwargs)
        self.outcomes = simulator.outcomes
        return result

    def run(self, angles=(), output_form=None, **kwargs):
        """Return a dense state using the Rust state simulator."""
        return self._dense("run", angles, output_form=output_form, **kwargs)

    def probabilities(self, angles=(), basis=None, **kwargs):
        """Return dense output probabilities using the Rust state simulator."""
        return self._dense("probabilities", angles, basis=basis, **kwargs)

    def sample(self, angles=(), *, shots=1024, basis=None, seed=None, **kwargs):
        """Return joint output shots using the Rust state simulator."""
        return self._dense(
            "sample", angles, shots=shots, basis=basis, seed=seed, **kwargs
        )

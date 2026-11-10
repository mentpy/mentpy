# Copyright 2026 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""NumPy simulation with lazy graph preparation."""

import numpy as np

from mentpy.mbqc.measurement_angles import MeasurementAngleResolver, xy_plane_angle
from mentpy.operators import Ment
from .base_simulator import BaseSimulator, _window_size
from .results import normalize_output_form

__all__ = ["NumpySimulator"]


class NumpySimulator(BaseSimulator):
    """Simulate an MBQC pattern by keeping only active qubits in memory.

    With ``force0=True``, postselect zero at each measurement. With
    ``force0=False``, sample each measurement and apply available flow
    corrections. Without flow, the sampled state follows the pattern's
    explicit measurement dependencies. Either mode can return a statevector
    or its density matrix. ``outcomes`` records the most recent run.

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
        schedule=None,
        window_size=None,
        output_form="dm",
    ):
        super().__init__(mbqcircuit, input_state)
        self.force0 = force0
        self._default_output_form = normalize_output_form(output_form)
        self.rng = np.random.default_rng(seed)
        self.resolver = MeasurementAngleResolver(mbqcircuit, schedule)
        self.resolver.validate_dependencies()
        if window_size is not None and (window_size != 1 or schedule is not None):
            _window_size(
                mbqcircuit,
                self.resolver.schedule,
                self.resolver.measurement_nodes,
                window_size,
            )
        self.corrections = {} if force0 else self.resolver._flow_corrections()
        if input_state is None:
            dimension = 2 ** len(mbqcircuit.input_nodes)
            input_state = np.ones(dimension) / np.sqrt(dimension)
        self.reset(input_state)

    def reset(self, input_state=None):
        if input_state is not None:
            state = np.asarray(input_state, dtype=complex)
            if state.shape != (
                2 ** len(self.mbqcircuit.input_nodes),
            ) or not np.isclose(np.linalg.norm(state), 1):
                raise ValueError(
                    "input_state must be a normalized vector in input-node order."
                )
            self.input_state = state.copy()
        self._outcomes = {}

    def measure(self, angle, **kwargs):
        raise NotImplementedError("Use run() to execute a complete pattern.")

    def run(self, angles=(), output_form=None, *, input_state=None, seed=None):
        form = (
            self._default_output_form
            if output_form is None
            else normalize_output_form(output_form)
        )
        rng = self.rng if seed is None else np.random.default_rng(seed)
        self.resolver.validate_parameters(angles)
        self.reset(input_state)
        graph = self.mbqcircuit.graph
        active = list(self.mbqcircuit.input_nodes)
        state = self.input_state.reshape((2,) * len(active)).copy()
        prepared = set(active)
        applied_edges = set()
        outcomes = {}
        self.max_frontier = len(active)

        def entangle(edges):
            for u, v in edges:
                key = frozenset((u, v))
                if key not in applied_edges and u in active and v in active:
                    index = [slice(None)] * len(active)
                    index[active.index(u)] = index[active.index(v)] = 1
                    state[tuple(index)] *= -1
                    applied_edges.add(key)

        def prepare(node):
            nonlocal state
            if node not in prepared:
                state = np.tensordot(state, np.ones(2) / np.sqrt(2), axes=0)
                active.append(node)
                prepared.add(node)
                self.max_frontier = max(self.max_frontier, len(active))
            entangle((node, neighbor) for neighbor in graph.neighbors(node))

        def correct(node, pauli):
            nonlocal state
            prepare(node)
            if pauli == "x":
                # X must follow every incident CZ. Z commutes with pending CZs.
                for neighbor in graph.neighbors(node):
                    if neighbor not in prepared:
                        prepare(neighbor)
                state = np.flip(state, axis=active.index(node))
            else:
                index = [slice(None)] * len(active)
                index[active.index(node)] = 1
                state[tuple(index)] *= -1

        entangle(graph.edges)
        for node in self.resolver.measurement_nodes:
            prepare(node)
            for neighbor in graph.neighbors(node):
                if neighbor not in prepared:
                    prepare(neighbor)
            cmd = self.resolver.resolve(node, angles, outcomes)
            axis = active.index(node)
            if cmd.plane in {"X", "Y", "XY"}:
                angle = xy_plane_angle(cmd.plane, cmd.angle)
                zero, one = np.moveaxis(state, axis, 0)
                one = np.exp(-1j * angle) * one
                first = (zero + one) / np.sqrt(2)
                second = None if self.force0 else (zero - one) / np.sqrt(2)
            else:
                _, vectors = np.linalg.eigh(Ment(cmd.angle, cmd.plane).matrix())
                first = np.tensordot(vectors[:, 1].conj(), state, axes=(0, axis))
                second = (
                    None
                    if self.force0
                    else np.tensordot(vectors[:, 0].conj(), state, axes=(0, axis))
                )
            probability = np.vdot(first, first).real
            outcome = 0
            if second is not None:
                other = np.vdot(second, second).real
                outcome = int(rng.random() >= probability / (probability + other))
                if outcome:
                    first, probability = second, other
            if not np.isfinite(probability) or probability <= 1e-28:
                raise ValueError(
                    f"Selected measurement branch has zero probability at node {node}."
                )
            state = first / np.sqrt(probability)
            active.remove(node)
            outcomes[node] = outcome
            if outcome:
                for target, pauli in self.corrections.get(node, ()):
                    correct(target, pauli)

        output_order = self.mbqcircuit.quantum_output_nodes
        for node in output_order:
            prepare(node)
        state = np.transpose(
            state, [active.index(node) for node in output_order]
        ).reshape(-1)
        self.outcomes = outcomes
        return state if form == "sv" else np.outer(state, state.conj())

# Copyright 2026 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""MBQC trajectories with lazy graph preparation and an active state frontier."""

import numpy as np

from .base_simulator import BaseSimulator
from mentpy.mbqc.measurement_angles import MeasurementAngleResolver

__all__ = ["TrajectorySimulator"]


class TrajectorySimulator(BaseSimulator):
    """Simulate XY patterns with causal flow, removing qubits as they are measured.

    Memory scales as 2**frontier_width, rather than 2**graph_size. With
    ``force0=False``, intermediate measurements are sampled and flow corrections
    are applied. With ``force0=True``, the deterministic zero branch is used.
    Outputs and supplied input-state axes follow the circuit's node lists.

    Group
    -----
    simulators
    """

    _statevector_output = True

    def __init__(self, mbqcircuit, input_state=None, *, force0=True, seed=None):
        super().__init__(mbqcircuit, input_state)
        mbqcircuit.flow.initialize_flow()
        if mbqcircuit.flow.name != "cFlow":
            raise ValueError("The trajectory backend requires causal flow (cFlow).")
        if mbqcircuit.classical_output_nodes:
            raise ValueError(
                "Use quantum outputs and sample() for output measurements."
            )
        self.force0 = force0
        self.rng = np.random.default_rng(seed)
        self.resolver = MeasurementAngleResolver(mbqcircuit)
        positions = {node: i for i, node in enumerate(self.resolver.schedule)}
        for node in self.resolver.measurement_nodes:
            target = mbqcircuit.flow(node)
            future = {target, *mbqcircuit.graph.neighbors(target)} - {node}
            if any(positions[v] <= positions[node] for v in future):
                raise ValueError("Measurement order must respect causal flow.")
        if input_state is None:
            input_state = np.ones(2 ** len(mbqcircuit.input_nodes)) / np.sqrt(
                2 ** len(mbqcircuit.input_nodes)
            )
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
        raise NotImplementedError("Use run() to execute a complete trajectory.")

    def run(self, angles=(), output_form="dm", *, input_state=None):
        self.resolver.validate_parameters(angles)
        self.reset(input_state)
        graph = self.mbqcircuit.graph
        active = list(self.mbqcircuit.input_nodes)
        state = self.input_state.reshape((2,) * len(active)).copy()
        prepared = set(active)
        applied_edges = set()
        self.max_frontier = len(active)

        def entangle(edges=None):
            nonlocal state
            for u, v in (graph.edges if edges is None else edges):
                key = frozenset((u, v))
                if key not in applied_edges and u in active and v in active:
                    shape = [1] * len(active)
                    shape[active.index(u)] = shape[active.index(v)] = 2
                    phase = np.ones(shape)
                    index = [slice(None)] * len(active)
                    index[active.index(u)] = index[active.index(v)] = 1
                    phase[tuple(index)] = -1
                    state = state * phase
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
            # Preparing neighbors first ensures corrections follow all CZs.
            prepare(node)
            for neighbor in graph.neighbors(node):
                if neighbor not in prepared:
                    prepare(neighbor)
            axis = active.index(node)
            if pauli == "X":
                state = np.flip(state, axis=axis)
            else:
                shape = [1] * len(active)
                shape[axis] = 2
                state = state * np.array([1, -1]).reshape(shape)

        entangle()
        for node in self.resolver.measurement_nodes:
            prepare(node)
            for neighbor in graph.neighbors(node):
                if neighbor not in prepared:
                    prepare(neighbor)
            angle = self.resolver.angle(node, angles, self.outcomes, xy=True)
            axis = active.index(node)
            zero, one = np.take(state, 0, axis=axis), np.take(state, 1, axis=axis)
            branches = [
                (zero + sign * np.exp(-1j * angle) * one) / np.sqrt(2)
                for sign in (1, -1)
            ]
            probabilities = np.array([np.vdot(b, b).real for b in branches])
            outcome = (
                0
                if self.force0
                else int(self.rng.choice(2, p=probabilities / probabilities.sum()))
            )
            if probabilities[outcome] <= 1e-28:
                raise ValueError("Selected measurement branch has zero probability.")
            state = branches[outcome] / np.sqrt(probabilities[outcome])
            active.remove(node)
            self.outcomes[node] = outcome
            if outcome:
                successor = self.mbqcircuit.flow(node)
                correct(successor, "X")
                for target in graph.neighbors(successor):
                    if target != node and target not in self.outcomes:
                        correct(target, "Z")
        for node in self.mbqcircuit.quantum_output_nodes:
            prepare(node)
        output_order = self.mbqcircuit.quantum_output_nodes
        state = np.transpose(
            state, [active.index(node) for node in output_order]
        ).reshape(-1)
        if output_form in {"sv", "statevector"}:
            return state
        if output_form in {"dm", "densitymatrix"}:
            return np.outer(state, state.conj())
        raise ValueError(f"Unsupported output form: {output_form}.")

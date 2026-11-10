# Copyright 2026 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Native tensor-network simulation and reverse-mode differentiation."""

import numpy as np

from mentpy import _rust
from mentpy.mbqc.measurement_angles import MeasurementAngleResolver
from mentpy.operators import Observable
from .base_simulator import BaseSimulator, _window_size
from .results import Samples, normalize_output_form, validate_shots

__all__ = ["RustSimulator"]


def _basis(plane, angle):
    """Map a measurement plane to azimuth and latitude."""
    if plane == "XY":
        return angle, 0.0
    if plane == "XZ":
        return 0.0, angle
    if plane == "YZ":
        return np.pi / 2, angle
    if plane == "XYZ":
        return tuple(angle)
    return {"X": (0.0, 0.0), "Y": (np.pi / 2, 0.0), "Z": (0.0, np.pi / 2)}[plane]


class RustSimulator(BaseSimulator):
    """Simulate graph-state patterns and differentiate expectations in Rust.

    Notes
    -----
    Contraction planning, tensor operations and reverse-mode differentiation
    run in Rust. CPU contractions use complex GEMM kernels and an iterative
    reverse-mode tape; CUDA contractions use Candle. The output vector has ``2**n`` entries for
    ``n`` quantum outputs. :meth:`expectation_and_grad` differentiates the
    normalized selected branch, including its normalization probability.
    Sampled execution uses a lazy active-state simulation on CPU and applies
    available flow corrections after each intermediate outcome. ``run`` and
    exact expectations describe one trajectory; ``sample`` resamples the
    intermediate measurements for each output shot.

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
        output_form="dm",
        schedule=None,
        max_elements=2**26,
        device="cpu",
        seed=None,
        window_size=None,
    ):
        """Create a native tensor-network simulator.

        Parameters
        ----------
        mbqcircuit : MBQCircuit
            Pattern to simulate. All measurement planes are supported.
        input_state : array-like, optional
            Normalized vector in input-node order. The default ``|+>`` product
            state is represented without allocating an exponential input array.
        force0 : bool
            Postselect zero at each measurement, or sample outcomes and apply
            available flow corrections when False.
        output_form : str
            Default output form, "dm" or "sv". Both describe the same branch.
        max_elements : int
            Maximum number of elements in a planned tensor or active trajectory
            state. This does not bound total memory used by differentiation.
        device : str
            "cpu" or "cuda:N". CUDA requires an extension built with CUDA support
            and currently supports postselected tensor contraction only.
        seed : int, optional
            Seed for output-shot sampling when a call does not supply its own seed.
        """
        super().__init__(mbqcircuit)
        self.rng = np.random.default_rng(seed)
        if not force0 and device != "cpu":
            raise NotImplementedError(
                "Sampled native trajectories currently require device='cpu'."
            )
        self.force0 = force0
        self._default_output_form = normalize_output_form(output_form)
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
        self._adaptive = any(
            self.resolver._dependencies(node)
            for node in self.resolver.measurement_nodes
        )
        self._forced_outcomes = dict.fromkeys(self.resolver.measurement_nodes, 0)
        self._measurement_info = [
            self.resolver.resolve(
                node, np.zeros(len(mbqcircuit.trainable_nodes)), self._forced_outcomes
            )
            for node in self.resolver.measurement_nodes
        ]
        initial = [0.0] * len(mbqcircuit.trainable_nodes)
        for command in self._measurement_info:
            if command.trainable and command.plane == "XYZ":
                initial[command.trainable_index] = (0.0, 0.0)
        values, self._indices = self._resolve_angles(initial)
        self._angle_template = np.asarray(values, dtype=float)
        self._scalar_dest = np.array(
            [i for i, index in enumerate(self._indices) if isinstance(index, int)],
            dtype=int,
        )
        self._scalar_source = np.array(
            [self._indices[i] for i in self._scalar_dest], dtype=int
        )
        self._pair_dest = [
            (i, index)
            for i, index in enumerate(self._indices)
            if isinstance(index, tuple)
        ]
        self._max_elements = max_elements
        self.device = device
        self._network = None
        self.reset(input_state)

    def _build_network(self):
        nodes = {node: i for i, node in enumerate(self.mbqcircuit.graph)}
        self._network_explicit = self.input_state is not None
        specification = (
            len(nodes),
            [(nodes[u], nodes[v]) for u, v in self.mbqcircuit.graph.edges],
            [nodes[n] for n in self.mbqcircuit.input_nodes],
            [nodes[n] for n in self.mbqcircuit.quantum_output_nodes],
            [nodes[n] for n in self.resolver.measurement_nodes],
        )
        if not self.force0:
            self._network = _rust.Frontier(
                *specification,
                [
                    [
                        (nodes[target], int(pauli == "z"))
                        for target, pauli in self.corrections.get(node, ())
                    ]
                    for node in self.resolver.measurement_nodes
                ],
                max_elements=self._max_elements,
            )
            if self.input_state is not None:
                self._network.set_input(self.input_state.real, self.input_state.imag)
            return
        self._network = _rust.TensorNetwork(
            *specification,
            explicit_input=self._network_explicit,
            max_elements=self._max_elements,
            device_name=self.device,
        )

    @property
    def peak_elements(self):
        """Largest planned tensor or active state in the last sampled trajectory."""
        return (
            self._network.peak_elements
            if self.force0
            else 2**self._network.max_frontier
        )

    @property
    def max_frontier(self):
        """Largest active qubit count in the last sampled trajectory."""
        return None if self.force0 else self._network.max_frontier

    def reset(self, input_state=None):
        """Clear outcomes, optionally replacing the normalized input vector."""
        if input_state is not None:
            state = np.asarray(input_state, dtype=complex)
            if state.shape != (
                2 ** len(self.mbqcircuit.input_nodes),
            ) or not np.isclose(np.linalg.norm(state), 1):
                raise ValueError(
                    "input_state must be a normalized vector in input-node order."
                )
            self.input_state = state.copy()
        if self._network is None or self._network_explicit != (
            self.input_state is not None
        ):
            self._build_network()
        elif not self.force0:
            if self.input_state is not None:
                self._network.set_input(self.input_state.real, self.input_state.imag)
            else:
                self._network.set_input()
        self.outcomes = {}

    def _input(self):
        if self.input_state is None:
            return {}
        return {"input_re": self.input_state.real, "input_im": self.input_state.imag}

    def _resolve_angles(self, angles):
        values, indices = [], []
        for command in self._measurement_info:
            plane = command.plane
            index = command.trainable_index if command.trainable else None
            angle = command.angle if index is None else angles[index]
            if plane == "XY":
                pair, mapping = (angle, 0), (index, None)
            elif plane == "XZ":
                pair, mapping = (0, angle), (None, index)
            elif plane == "YZ":
                pair, mapping = (np.pi / 2, angle), (None, index)
            elif plane == "XYZ":
                pair, mapping = angle, (
                    ((index, 0), (index, 1)) if index is not None else (None, None)
                )
            else:
                pair = {"X": (0, 0), "Y": (np.pi / 2, 0), "Z": (0, np.pi / 2)}[plane]
                mapping = (None, None)
            values.extend(pair)
            indices.extend(mapping)
        return values, indices

    def _angles(self, angles):
        self.resolver.validate_parameters(angles)
        values = self._angle_template.copy()
        if self._pair_dest:
            for dest, source in zip(self._scalar_dest, self._scalar_source):
                values[dest] = angles[source]
            for dest, (source, component) in self._pair_dest:
                values[dest] = angles[source][component]
        else:
            if len(self._scalar_dest):
                try:
                    selected = np.asarray(angles, dtype=float)[self._scalar_source]
                except ValueError:
                    selected = [angles[i] for i in self._scalar_source]
                values[self._scalar_dest] = selected
        return values

    def measure(self, angle, **kwargs):
        """Use :meth:`run` to contract a complete pattern."""
        raise NotImplementedError("Use run() to execute a complete pattern.")

    def run(self, angles=(), output_form=None, *, input_state=None, seed=None):
        """Return the normalized selected branch in quantum-output-node order."""
        form = (
            self._default_output_form
            if output_form is None
            else normalize_output_form(output_form)
        )
        self.reset(input_state)
        if self.force0:
            values = self._angles(angles)
            re, im = self._network.run(values, **self._input())
            self.outcomes = self._forced_outcomes.copy()
        else:
            self.resolver.validate_parameters(angles)
            rng = self.rng if seed is None else np.random.default_rng(seed)
            draws = rng.random(len(self.resolver.measurement_nodes))
            if self._adaptive:
                outcomes = {}
                for node, draw in zip(self.resolver.measurement_nodes, draws):
                    command = self.resolver.resolve(node, angles, outcomes)
                    outcomes[node] = self._network.step(
                        *_basis(command.plane, command.angle), draw
                    )
                re, im = self._network.finish()
            else:
                re, im = self._network.run(self._angles(angles), draws)
                outcomes = dict(
                    zip(self.resolver.measurement_nodes, self._network.outcomes)
                )
            self.outcomes = outcomes
        state = np.asarray(re) + 1j * np.asarray(im)
        return state if form == "sv" else np.outer(state, state.conj())

    def _evaluate(self, angles, observable, gradient, input_state=None):
        if not isinstance(observable, Observable):
            observable = Observable(observable)
        observable._validate_qubits(len(self.mbqcircuit.quantum_output_nodes))
        self.reset(input_state)
        values = self._angles(angles)
        value, derivative = self._network.expectation(
            values,
            [(word, float(np.real(coeff))) for word, coeff in observable.terms.items()],
            constant=float(np.real(observable.constant)),
            gradient=gradient,
            **self._input(),
        )
        self.outcomes = self._forced_outcomes.copy()
        if not gradient:
            return value
        if not self._pair_dest:
            result = np.zeros(len(angles))
            np.add.at(
                result, self._scalar_source, np.asarray(derivative)[self._scalar_dest]
            )
            return value, result
        # Match trainable-node order, retaining zeros for conditionally inactive
        # parameters. XYZ measurements use two components per parameter.
        result = [np.zeros_like(angle, dtype=float) for angle in angles]
        for index, entry in zip(self._indices, derivative):
            if isinstance(index, tuple):
                result[index[0]][index[1]] += entry
            elif index is not None:
                result[index] += entry
        try:
            result = np.asarray(result, dtype=float)
        except ValueError:
            result = tuple(result)
        return value, result

    def expectation(self, angles, observable, shots=None, seed=None, **kwargs):
        """Evaluate an expectation, optionally estimating each term from shots."""
        if shots is not None or not self.force0:
            return super().expectation(angles, observable, shots, seed, **kwargs)
        kwargs.pop("output_form", None)
        return self._evaluate(angles, observable, False, **kwargs)

    def sample(self, angles=(), *, shots=1024, basis=None, seed=None, **kwargs):
        """Draw joint output shots, resampling intermediate outcomes when requested."""
        if self.force0 or self._adaptive:
            return super().sample(angles, shots=shots, basis=basis, seed=seed, **kwargs)
        shots = validate_shots(shots)
        nodes = tuple(self.mbqcircuit.quantum_output_nodes)
        basis = "Z" * len(nodes) if basis is None else basis.upper()
        if len(basis) != len(nodes) or not set(basis) <= set("IXYZ"):
            raise ValueError("basis must contain one I, X, Y or Z per output.")
        kwargs.pop("output_form", None)
        self.reset(kwargs.pop("input_state", None))
        if kwargs:
            raise TypeError(f"Unexpected sampling options: {', '.join(kwargs)}")
        rng = self.rng if seed is None else np.random.default_rng(seed)
        values = self._angles(angles)
        width = len(self.resolver.measurement_nodes) + 1
        bits = np.empty((shots, len(nodes)), dtype=np.int8)
        # Bound random-number storage independently of the requested shot count.
        batch = max(1, min(256, 2**18 // width))
        for start in range(0, shots, batch):
            size = min(batch, shots - start)
            draws = rng.random(size * width)
            data = self._network.sample(values, draws, basis)
            bits[start : start + size] = np.frombuffer(data, dtype=np.uint8).reshape(
                size, len(nodes)
            )
        self.outcomes = dict(
            zip(self.resolver.measurement_nodes, self._network.outcomes)
        )
        return Samples(bits, nodes, basis)

    def expectation_and_grad(self, angles, observable, *, input_state=None, shots=None):
        """Return an exact expectation and its native reverse-mode gradient.

        Gradients follow ``mbqcircuit.trainable_nodes`` order. Each XYZ entry
        has two components, matching the pair of input angles for that node.
        Shot estimates cannot be differentiated by this method.
        """
        if shots is not None:
            raise ValueError("Shot-sampled expectations do not support autodiff.")
        if not self.force0:
            raise ValueError(
                "Autodiff requires a fixed postselected branch (force0=True)."
            )
        return self._evaluate(angles, observable, True, input_state)

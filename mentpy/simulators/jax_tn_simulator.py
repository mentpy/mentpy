# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""A tensor network simulator using JAX for MBQC circuits with autodiff support."""

import jax
import jax.numpy as jnp
import numpy as np
import opt_einsum as oe

# Enable 64-bit precision in JAX (needed for matching numpy-sv accuracy)
jax.config.update("jax_enable_x64", True)

from mentpy.operators import Observable
from mentpy.mbqc.mbqcircuit import MBQCircuit
from mentpy.mbqc.measurement_angles import MeasurementAngleResolver
from mentpy.simulators.base_simulator import BaseSimulator
from mentpy.simulators.jax_utils import (
    build_graph_state_tensors,
    measurement_vector,
)

__all__ = ["JaxTNSimulator"]


class JaxTNSimulator(BaseSimulator):
    """A tensor network simulator using JAX for MBQC circuits.

    This simulator represents graph states as tensor networks and contracts the
    full forced-outcome measurement pattern with JAX arrays. For graphs with low
    treewidth (e.g., linear clusters), this is exponentially more efficient than
    full state vector simulation.

    Supports JAX autodiff for gradient computation through the simulation.

    Parameters
    ----------
    mbqcircuit : MBQCircuit
        The MBQC circuit to simulate.
    input_state : np.ndarray, optional
        The input state in ``mbqcircuit.input_nodes`` order. Defaults to |+>^n.
    force0 : bool
        If True (default), force measurement outcome 0 (deterministic mode).
    output_form : str
        Default output form: "dm" for density matrix, "sv" for state vector.

    Group
    -----
    simulators
    """

    def __init__(
        self,
        mbqcircuit: MBQCircuit,
        input_state: np.ndarray = None,
        **kwargs,
    ) -> None:
        super().__init__(mbqcircuit, input_state)

        self.force0 = kwargs.pop("force0", True)
        self._default_output_form = kwargs.pop("output_form", "dm")

        if not self.force0:
            raise NotImplementedError(
                "JAX TN simulator currently only supports force0=True."
            )

        self._angle_resolver = MeasurementAngleResolver(mbqcircuit)
        self._schedule = self._angle_resolver.schedule
        self._schedule_measure = self._angle_resolver.measurement_nodes
        self._forced_outcomes = self._angle_resolver.zero_outcomes()

        # Build static measurement info (node indices, angles, trainability)
        self._measurement_info = []
        for node in self._schedule_measure:
            command = self._angle_resolver.resolve(
                node, np.zeros(len(mbqcircuit.trainable_nodes)), self._forced_outcomes
            )
            if command.plane not in ["X", "Y", "XY"]:
                raise ValueError(
                    f"Node {node} has plane {command.plane}, "
                    "but only XY plane is supported by JAX TN simulator."
                )
            is_trainable = command.trainable
            trainable_idx = (
                -1 if command.trainable_index is None else command.trainable_index
            )
            fixed_angle = None if is_trainable else command.angle
            self._measurement_info.append(
                (
                    node,
                    is_trainable,
                    trainable_idx,
                    fixed_angle,
                    command.plane,
                )
            )

        # Store graph structure for building TN
        self._all_nodes = list(mbqcircuit.graph.nodes)
        self._input_nodes = list(mbqcircuit.input_nodes)
        self._output_nodes = list(mbqcircuit.output_nodes)
        self._quantum_output_nodes = list(mbqcircuit.quantum_output_nodes)

        if input_state is None:
            input_state = self._default_input_state()
        self._set_input_state_tensor(input_state)

        # Build initial tensor network for the legacy step-by-step measure API.
        self._init_tensors, self._init_bonds, self._init_node_axes = (
            build_graph_state_tensors(mbqcircuit.graph, self._all_nodes)
        )

        self._tn_tensors, self._tn_bonds, self._tn_node_axes = (
            build_graph_state_tensors(
                mbqcircuit.graph, self._all_nodes, input_nodes=self._input_nodes
            )
        )
        self._build_contractor()

        # Current state for step-by-step measurement
        self._current_step = 0
        self._tensors = dict(self._init_tensors)
        self._bonds = list(self._init_bonds)
        self._node_axes = {k: dict(v) for k, v in self._init_node_axes.items()}

    def _default_input_state(self):
        """Return the default ``|+>`` input state in circuit input-node order."""
        state = np.array([1.0 + 0.0j])
        plus = np.array([1.0, 1.0], dtype=np.complex128) / np.sqrt(2)
        for _ in self._input_nodes:
            state = np.kron(state, plus)
        return state

    def _set_input_state_tensor(self, input_state):
        """Store an input state tensor with axes ordered as ``input_nodes``."""
        n_inputs = len(self._input_nodes)
        expected_size = 2**n_inputs
        input_state = np.asarray(input_state, dtype=np.complex128)
        if input_state.size != expected_size:
            raise ValueError(
                f"Input state has size {input_state.size}, expected {expected_size} "
                f"for {n_inputs} input qubits."
            )
        shape = (2,) * n_inputs
        self.input_state = input_state.reshape(expected_size)
        self._input_state_tensor = jnp.asarray(input_state.reshape(shape))

    def _build_contractor(self):
        """Prebuild a tensor-network contraction expression for this circuit."""
        label = 0
        self._physical_labels = {}
        for node in self._all_nodes:
            self._physical_labels[node] = label
            label += 1

        edge_labels = {}
        for edge in self.mbqcircuit.graph.edges:
            edge_key = tuple(sorted(edge))
            edge_labels[edge_key] = label
            label += 1

        self._operand_specs = []
        expression_args = []

        for node in self._all_nodes:
            tensor = self._tn_tensors[node]
            indices = [self._physical_labels[node]]
            indices.extend(
                edge_labels[tuple(sorted((node, neighbor)))]
                for neighbor in sorted(self.mbqcircuit.graph.neighbors(node))
            )
            self._operand_specs.append(("constant", tensor))
            expression_args.extend([tensor.shape, indices])

        if self._input_nodes:
            input_indices = [self._physical_labels[node] for node in self._input_nodes]
            input_shape = (2,) * len(self._input_nodes)
            self._operand_specs.append(("input", None))
            expression_args.extend([input_shape, input_indices])

        for info_index, (node, *_rest) in enumerate(self._measurement_info):
            self._operand_specs.append(("measurement", info_index))
            expression_args.extend([(2,), [self._physical_labels[node]]])

        output_indices = [
            self._physical_labels[node] for node in self._quantum_output_nodes
        ]
        self._contract_expression = oe.contract_expression(
            *expression_args, output_indices, optimize="auto"
        )

    def measure(self, angle: float, **kwargs):
        """Perform a single measurement step.

        Parameters
        ----------
        angle : float
            The measurement angle.

        Returns
        -------
        outcome : int
            The measurement outcome (always 0 with force0=True).
        """
        raise NotImplementedError(
            "JAX TN simulator contracts full forced-outcome patterns via run(); "
            "step-by-step measurement is not supported."
        )

    def run(self, angles, output_form=None, **kwargs):
        """Run the full simulation.

        Parameters
        ----------
        angles : array-like
            Measurement angles for trainable nodes.
        output_form : str, optional
            "dm" for density matrix, "sv" for state vector.

        Returns
        -------
        jnp.ndarray
            The output state as density matrix or state vector.
        """
        if kwargs.get("input_state") is not None:
            self.reset(input_state=kwargs.get("input_state"))

        if output_form is None:
            output_form = self._default_output_form

        angles = jnp.asarray(angles, dtype=jnp.float64)

        if len(angles) != len(self.mbqcircuit.trainable_nodes):
            raise ValueError(
                f"Number of angles ({len(angles)}) does not match "
                f"number of trainable nodes ({len(self.mbqcircuit.trainable_nodes)})."
            )

        # Use pure forward pass for autodiff compatibility
        result = self._forward(angles, output_form)

        # Reset internal state after run
        self.reset()

        return result

    def _forward(self, angles, output_form="dm"):
        """Pure forward pass through the tensor network.

        This method is structured to be compatible with JAX transformations.
        It does not mutate self - all state is local.

        Parameters
        ----------
        angles : jnp.ndarray
            Trainable measurement angles.
        output_form : str
            "dm" or "sv".

        Returns
        -------
        jnp.ndarray
            Output state.
        """
        operands = []
        measurement_vectors = []
        for (
            node,
            is_trainable,
            trainable_idx,
            fixed_angle,
            plane,
        ) in self._measurement_info:
            if is_trainable:
                angle = angles[trainable_idx]
            else:
                angle = jnp.asarray(fixed_angle, dtype=jnp.float64)
            if plane == "X":
                angle = jnp.asarray(0.0, dtype=jnp.float64)
            elif plane == "Y":
                angle = jnp.asarray(np.pi / 2, dtype=jnp.float64)
            measurement_vectors.append(measurement_vector(angle))

        for kind, value in self._operand_specs:
            if kind == "constant":
                operands.append(value)
            elif kind == "input":
                operands.append(self._input_state_tensor)
            elif kind == "measurement":
                operands.append(measurement_vectors[value])

        output_qubits = self._quantum_output_nodes
        output_tensor = self._contract_expression(*operands, backend="jax")
        sv = output_tensor.reshape(2 ** len(output_qubits))
        # Normalize (measurement projections leave the state unnormalized)
        sv = sv / jnp.linalg.norm(sv)

        if output_form.lower() in ("dm", "densitymatrix"):
            return jnp.outer(sv, jnp.conj(sv))
        elif output_form.lower() in ("sv", "statevector"):
            return sv
        else:
            raise ValueError(f"Output form {output_form} is not supported.")

    def reset(self, input_state=None):
        """Reset the simulator to the initial state.

        Parameters
        ----------
        input_state : np.ndarray, optional
            New input state (currently unused - only |+>^n supported).
        """
        if input_state is not None:
            self._set_input_state_tensor(input_state)

        self._current_step = 0
        self._tensors = dict(self._init_tensors)
        self._bonds = list(self._init_bonds)
        self._node_axes = {k: dict(v) for k, v in self._init_node_axes.items()}
        self._outcomes = {}

    def run_and_grad(self, angles, cost_fn, **kwargs):
        """Run the simulation and compute gradients via JAX autodiff.

        Parameters
        ----------
        angles : array-like
            Measurement angles for trainable nodes.
        cost_fn : callable
            A function that takes the output density matrix/state vector
            and returns a scalar cost.

        Returns
        -------
        cost_value : float
            The cost function value.
        gradient : jnp.ndarray
            Gradient of the cost w.r.t. angles.
        """
        output_form = kwargs.get("output_form", self._default_output_form)

        def full_cost(angles_):
            result = self._forward(angles_, output_form)
            return jnp.real(cost_fn(result))

        angles = jnp.asarray(angles, dtype=jnp.float64)
        cost_value, gradient = jax.value_and_grad(full_cost)(angles)
        return cost_value, gradient

    def expectation(self, angles, observable, shots=None, seed=None, **kwargs):
        """Evaluate an observable expectation value.

        Parameters
        ----------
        angles : array-like
            Measurement angles for trainable nodes.
        observable : Observable or compatible constructor input
            Weighted Pauli observable acting on the quantum output wires.

        Returns
        -------
        jnp.ndarray
            Scalar expectation value.
        """
        if not isinstance(observable, Observable):
            observable = Observable(observable)
        state = self._forward(jnp.asarray(angles, dtype=jnp.float64), "sv")
        if shots is not None:
            return observable.sample_expectation(np.asarray(state), shots, seed=seed)
        return observable.expectation(state)

    def expectation_and_grad(self, angles, observable, **kwargs):
        """Evaluate an observable expectation and its gradient."""
        if kwargs.get("shots") is not None:
            raise ValueError(
                "Shot-sampled expectations are stochastic and not compatible with "
                "JAX autodiff. Use parameter-shift or finite-difference gradients."
            )
        if not isinstance(observable, Observable):
            observable = Observable(observable)

        def cost(angles_):
            return self.expectation(angles_, observable, **kwargs)

        angles = jnp.asarray(angles, dtype=jnp.float64)
        return jax.value_and_grad(cost)(angles)

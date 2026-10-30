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
        The input state in ``mbqcircuit.input_nodes`` order. Defaults to ``|+>^n``.
    force0 : bool
        If True (default), force measurement outcome 0 (deterministic mode).
    output_form : str
        Default output form: "dm" for density matrix, "sv" for state vector.

    Group
    -----
    simulators
    """

    _statevector_output = True

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

        self._build_contractor()
        self._gradient_cache = {}

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
        if input_state.shape != (expected_size,) or not np.isclose(
            np.linalg.norm(input_state), 1
        ):
            raise ValueError(
                f"Input state must be normalized with shape ({expected_size},), got {input_state.shape} "
                f"for {n_inputs} input qubits."
            )
        shape = (2,) * n_inputs
        self.input_state = input_state.reshape(expected_size)
        self._input_state_tensor = jnp.asarray(input_state.reshape(shape))

    def _build_contractor(self):
        """Factor graph-state copy constraints into tensors of rank at most three."""
        next_label = iter(
            range(
                10 * (len(self._all_nodes) + self.mbqcircuit.graph.number_of_edges())
                + 1
            )
        )
        physical = {node: next(next_label) for node in self._all_nodes}
        legs = {node: [physical[node]] for node in self._all_nodes}
        self._operand_specs = []
        expression_args = []

        def add(kind, value, shape, indices):
            self._operand_specs.append((kind, value))
            expression_args.extend([shape, indices])

        cz = jnp.asarray([[1, 1], [1, -1]], dtype=jnp.complex128)
        copy3 = np.zeros((2, 2, 2), dtype=complex)
        copy3[0, 0, 0] = copy3[1, 1, 1] = 1
        copy3 = jnp.asarray(copy3)
        for u, v in self.mbqcircuit.graph.edges:
            a, b = next(next_label), next(next_label)
            legs[u].append(a)
            legs[v].append(b)
            add("constant", cz, (2, 2), [a, b])
        for node in self._all_nodes:
            indices = legs[node]
            if len(indices) == 1:
                add("constant", jnp.ones(2, dtype=jnp.complex128), (2,), indices)
            else:
                # Equality of arbitrarily many legs is a chain of COPY tensors.
                head = indices[0]
                for leg in indices[1:-1]:
                    tail = next(next_label)
                    add("constant", copy3, (2, 2, 2), [head, leg, tail])
                    head = tail
                add(
                    "constant",
                    jnp.eye(2, dtype=jnp.complex128),
                    (2, 2),
                    [head, indices[-1]],
                )
        if self._input_nodes:
            add(
                "input",
                None,
                (2,) * len(self._input_nodes),
                [physical[n] for n in self._input_nodes],
            )
        for index, (node, *_rest) in enumerate(self._measurement_info):
            add("measurement", index, (2,), [physical[node]])
        outputs = [physical[node] for node in self._quantum_output_nodes]
        self._contract_expression = oe.contract_expression(
            *expression_args, outputs, optimize="auto"
        )

    def _contract(self, operands):
        """Rescale intermediate tensors to avoid long-pattern over/underflow."""
        tensors = list(operands)
        for indices, _, equation, *_ in self._contract_expression.contraction_list:
            values = [tensors.pop(i) for i in indices]
            result = oe.contract(equation, *values, backend="jax", optimize="auto")
            scale = jax.lax.stop_gradient(jnp.max(jnp.abs(result)))
            result = result / jnp.where(scale > 0, scale, 1)
            tensors.append(result)
        return tensors[0]

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

    def _forward(self, angles, output_form="dm", input_tensor=None):
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
                operands.append(
                    self._input_state_tensor if input_tensor is None else input_tensor
                )
            elif kind == "measurement":
                operands.append(measurement_vectors[value])

        output_qubits = self._quantum_output_nodes
        output_tensor = self._contract(operands)
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
            New normalized input state in input-node order.
        """
        if input_state is not None:
            self._set_input_state_tensor(input_state)

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
        self._angle_resolver.validate_parameters(angles)
        output_form = kwargs.get("output_form", self._default_output_form)

        def full_cost(angles_):
            result = self._forward(angles_, output_form)
            return jnp.real(cost_fn(result))

        angles = jnp.asarray(angles, dtype=jnp.float64)
        cost_value, gradient = jax.value_and_grad(full_cost)(angles)
        return cost_value, gradient

    def probabilities(self, angles=(), basis=None, **kwargs):
        """Return differentiable output probabilities in a Pauli product basis."""
        n = len(self._quantum_output_nodes)
        basis = "Z" * n if basis is None else basis.upper()
        if len(basis) != n or not set(basis) <= set("IXYZ"):
            raise ValueError("basis must contain one I, X, Y or Z per output.")
        state = self.run(angles, output_form="sv", **kwargs).reshape((2,) * n)
        h = jnp.asarray([[1, 1], [1, -1]]) / jnp.sqrt(2)
        for axis, char in enumerate(basis):
            if char in "XY":
                rotation = h if char == "X" else h @ jnp.diag(jnp.asarray([1, -1j]))
                state = jnp.moveaxis(
                    jnp.tensordot(rotation, state, axes=(1, axis)), 0, axis
                )
        return jnp.abs(state.reshape(-1)) ** 2

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
        state = self.run(angles, output_form="sv", **kwargs)
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

        self._angle_resolver.validate_parameters(angles)
        if kwargs.get("input_state") is not None:
            self.reset(kwargs["input_state"])
        key = (
            tuple(observable.terms.items()),
            observable.constant,
            observable.n_qubits,
        )
        if key not in self._gradient_cache:
            if len(self._gradient_cache) >= 8:
                self._gradient_cache.clear()
            frozen_observable = observable.copy()

            def cost(parameters, state):
                return frozen_observable.expectation(
                    self._forward(parameters, "sv", input_tensor=state)
                )

            self._gradient_cache[key] = jax.jit(jax.value_and_grad(cost, argnums=0))
        angles = jnp.asarray(angles, dtype=jnp.float64)
        return self._gradient_cache[key](angles, self._input_state_tensor)

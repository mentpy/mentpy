# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Measurement-angle resolution utilities."""

from dataclasses import dataclass
from itertools import product
from typing import Any, Optional, Sequence

import numpy as np

__all__ = ["MeasurementCommand", "MeasurementAngleResolver", "xy_plane_angle"]


@dataclass(frozen=True)
class MeasurementCommand:
    """Resolved measurement information for one MBQC node.

    Group
    -----
    mbqc
    """

    node: Any
    plane: str
    angle: Any
    trainable: bool
    trainable_index: Optional[int] = None


class MeasurementAngleResolver:
    """Resolve fixed, trainable, and outcome-controlled measurement angles.

    Parameters
    ----------
    mbqcircuit : MBQCircuit
        Circuit whose measurement metadata should be resolved.
    schedule : sequence, optional
        Measurement order. If omitted, ``mbqcircuit.measurement_order`` is used.

    Group
    -----
    mbqc
    """

    def __init__(self, mbqcircuit, schedule: Optional[Sequence] = None):
        self.mbqcircuit = mbqcircuit
        self.schedule = self._resolve_schedule(schedule)
        self.trainable_index = {
            node: index for index, node in enumerate(mbqcircuit.trainable_nodes)
        }
        self.measurement_nodes = self._measurement_nodes()

    def validate_parameters(self, angles):
        """Validate that an angle vector matches the circuit trainable nodes."""
        if len(angles) != len(self.mbqcircuit.trainable_nodes):
            raise ValueError(
                f"Number of angles ({len(angles)}) does not match number of "
                f"trainable nodes ({len(self.mbqcircuit.trainable_nodes)})."
            )

    def resolve(self, node, angles, outcomes=None, xy=False):
        """Resolve one node to a :class:`MeasurementCommand`.

        Set ``xy=True`` to map fixed X/Y-plane measurements to their equivalent
        XY-plane angles, which is useful for backends implemented through
        ``RZ(-theta); H`` projections.
        """
        outcomes = outcomes or {}
        ment = self.mbqcircuit[node]
        if ment is None:
            raise ValueError(f"Node {node} is not measured.")

        plane = self._resolve_attr(ment, "plane", outcomes)
        angle = self._resolve_attr(ment, "angle", outcomes)
        trainable_index = self.trainable_index.get(node)
        trainable = trainable_index is not None and angle is None

        if angle is None:
            if trainable_index is None:
                raise ValueError(
                    f"Node {node} has no fixed angle and is not trainable."
                )
            angle = angles[trainable_index]

        if xy:
            angle = xy_plane_angle(plane, angle)

        return MeasurementCommand(
            node=node,
            plane=plane,
            angle=angle,
            trainable=trainable,
            trainable_index=trainable_index,
        )

    def angle(self, node, angles, outcomes=None, xy=False):
        """Resolve and return only the measurement angle for a node."""
        return self.resolve(node, angles, outcomes=outcomes, xy=xy).angle

    def resolve_many(self, angles, outcomes=None, nodes=None, xy=False):
        """Resolve multiple measurement nodes in schedule order."""
        self.validate_parameters(angles)
        nodes = self.measurement_nodes if nodes is None else nodes
        return [self.resolve(node, angles, outcomes=outcomes, xy=xy) for node in nodes]

    def validate_dependencies(self):
        """Check that conditions only reference already measured nodes."""
        if len(self.schedule) != len(self.mbqcircuit.graph) or set(
            self.schedule
        ) != set(self.mbqcircuit.graph):
            raise ValueError("Schedule must contain each graph node exactly once.")
        measured = set()
        for node in self.measurement_nodes:
            dependencies = self._dependencies(node)
            if not dependencies <= measured:
                raise ValueError(
                    f"Measurement at node {node!r} depends on outcomes that are not yet available."
                )
            measured.add(node)

    def _flow_corrections(self):
        """Return the X and Z corrections for future nodes in the schedule."""
        flow = self.mbqcircuit.flow
        flow.initialize_flow()
        if flow.func is None:
            return {}
        order = {node: i for i, node in enumerate(self.schedule)}
        outputs = set(self.mbqcircuit.output_nodes)
        corrections = {}
        for node in self.measurement_nodes:
            if node not in outputs:
                x, z = flow._corrections(node)
                if flow.name in {"cFlow", "gFlow"} and any(
                    order[target] < order[node] for target in x | z
                ):
                    raise ValueError("Measurement order must respect flow corrections.")
                corrections[node] = tuple(
                    (target, pauli)
                    for pauli, targets in (("x", x), ("z", z))
                    for target in sorted(targets, key=order.__getitem__)
                    if order[target] > order[node]
                )
        return corrections

    def _dependencies(self, node):
        condition = getattr(self.mbqcircuit[node], "condition", None)
        return set(getattr(condition, "cond_nodes", ()))

    def _xy_rotations(self, node, angles):
        """Yield the RZ angle and its earlier measurement controls.

        Expand a Boolean condition into products of measured bits. This
        uses only controls on outcome one, including for negated predicates.
        """
        measurement = self.mbqcircuit[node]
        condition = getattr(measurement, "condition", None)
        if condition is None:
            yield -self.angle(node, angles, xy=True), ()
            return
        dependencies = self._dependencies(node)
        controls = [
            candidate
            for candidate in self.measurement_nodes
            if candidate in dependencies
        ]
        if not controls:
            yield -self.angle(node, angles, {}, xy=True), ()
            return
        # Boolean Möbius transform. The coefficient for a bit mask controls
        # a rotation on precisely the measurements selected by that mask.
        width = len(controls)
        coefficients = [
            int(bool(condition(dict(zip(controls, reversed(bits))))))
            for bits in product((0, 1), repeat=width)
        ]
        false_angle = self.angle(node, angles, dict.fromkeys(controls, 0), xy=True)
        true_outcomes = next(
            (bits for bits, value in enumerate(coefficients) if value), None
        )
        if true_outcomes is None:
            yield -false_angle, ()
            return
        true_angle = self.angle(
            node,
            angles,
            {control: (true_outcomes >> i) & 1 for i, control in enumerate(controls)},
            xy=True,
        )
        if coefficients[0]:
            # Use any false assignment when the condition is true at zero.
            false_outcomes = next(
                (bits for bits, value in enumerate(coefficients) if not value), None
            )
            yield -true_angle, ()
            if false_outcomes is None:
                return
            false_angle = self.angle(
                node,
                angles,
                {
                    control: (false_outcomes >> i) & 1
                    for i, control in enumerate(controls)
                },
                xy=True,
            )
        else:
            yield -false_angle, ()
        for i in range(width):
            for mask in range(1 << width):
                if mask & (1 << i):
                    coefficients[mask] -= coefficients[mask ^ (1 << i)]
        for mask, coefficient in enumerate(coefficients[1:], start=1):
            if coefficient:
                selected = tuple(
                    control for i, control in enumerate(controls) if mask & (1 << i)
                )
                # Keep separate rotations so backends need only negation of
                # symbolic angles, including CUDA-Q kernel arguments.
                for _ in range(abs(coefficient)):
                    yield (-true_angle if coefficient > 0 else true_angle), selected
                    yield (false_angle if coefficient > 0 else -false_angle), selected

    def zero_outcomes(self):
        """Return deterministic force-zero outcomes for every circuit node."""
        return {node: 0 for node in self.mbqcircuit.graph.nodes}

    def _resolve_schedule(self, schedule):
        if schedule is not None:
            return list(schedule)
        order = getattr(self.mbqcircuit, "measurement_order", None)
        return list(self.mbqcircuit.graph.nodes if order is None else order)

    def _measurement_nodes(self):
        scheduled = [
            node for node in self.schedule if self.mbqcircuit[node] is not None
        ]
        scheduled_set = set(scheduled)
        return scheduled + [
            node
            for node in self.mbqcircuit.graph.nodes
            if node not in scheduled_set and self.mbqcircuit[node] is not None
        ]

    @staticmethod
    def _resolve_attr(ment, attr, outcomes):
        value = getattr(ment, attr)
        if callable(value):
            return value(outcomes)
        return value


def xy_plane_angle(plane, angle):
    """Return the XY-plane angle equivalent for supported X/Y/XY measurements."""
    if plane == "X":
        return 0.0
    if plane == "Y":
        return np.pi / 2
    if plane == "XY":
        return angle
    raise ValueError(f"Plane {plane} is not supported as an XY-plane measurement.")

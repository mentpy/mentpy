# Copyright 2023 Luis Mantilla
#
# Licensed under the Apache License, Version 2.0.
# See <http://www.apache.org/licenses/LICENSE-2.0> for details.
"""Measurement-angle resolution utilities."""

from dataclasses import dataclass
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

    def zero_outcomes(self):
        """Return deterministic force-zero outcomes for every circuit node."""
        return {node: 0 for node in self.mbqcircuit.graph.nodes}

    def _resolve_schedule(self, schedule):
        if schedule is not None:
            return list(schedule)
        try:
            return list(self.mbqcircuit.measurement_order)
        except Exception:
            return list(self.mbqcircuit.graph.nodes)

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

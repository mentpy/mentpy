"""Tests for MBQC measurement-angle resolution."""

from types import SimpleNamespace

import networkx as nx
import numpy as np
import pytest

import mentpy as mp
from mentpy.mbqc.measurement_angles import MeasurementAngleResolver, xy_plane_angle


class _FakeCircuit:
    def __init__(self, measurements, trainable_nodes=(), measurement_order=None):
        self.measurements = dict(measurements)
        self.trainable_nodes = list(trainable_nodes)
        self.graph = nx.Graph()
        self.graph.add_nodes_from(self.measurements)
        if measurement_order is not None:
            self.measurement_order = list(measurement_order)

    def __getitem__(self, node):
        return self.measurements[node]


def _ment(angle, plane="XY"):
    return SimpleNamespace(angle=angle, plane=plane)


def test_resolver_maps_fixed_xy_planes():
    gs = mp.templates.linear_cluster(5)
    gs[0] = mp.Ment("X")
    gs[1] = mp.Ment("Y")
    resolver = mp.MeasurementAngleResolver(gs)
    angles = np.zeros(len(gs.trainable_nodes))

    assert resolver.angle(0, angles, xy=True) == 0.0
    assert np.allclose(resolver.angle(1, angles, xy=True), np.pi / 2)


def test_resolver_selects_controlled_trainable_branch():
    gs = mp.templates.linear_cluster(5)
    gs[1] = mp.ControlMent(
        True,
        true_angle=None,
        true_plane="XY",
        false_angle=0,
        false_plane="X",
    )
    resolver = mp.MeasurementAngleResolver(gs)
    angles = np.arange(len(gs.trainable_nodes), dtype=float) + 0.25

    command = resolver.resolve(1, angles, outcomes={})

    assert command.trainable is True
    assert command.trainable_index == gs.trainable_nodes.index(1)
    assert command.angle == angles[command.trainable_index]
    assert command.plane == "XY"


def test_resolver_selects_controlled_fixed_branch():
    gs = mp.templates.linear_cluster(5)
    gs[1] = mp.ControlMent(
        False,
        true_angle=None,
        true_plane="XY",
        false_angle=0,
        false_plane="X",
    )
    resolver = mp.MeasurementAngleResolver(gs)
    angles = np.arange(len(gs.trainable_nodes), dtype=float) + 0.25

    command = resolver.resolve(1, angles, outcomes={})

    assert command.trainable is False
    assert command.trainable_index == gs.trainable_nodes.index(1)
    assert command.angle == 0
    assert command.plane == "X"


def test_resolver_orders_measurements_and_zero_outcomes():
    circuit = _FakeCircuit(
        {
            0: _ment(0.1),
            1: None,
            2: _ment(None),
            3: _ment(0.3),
        },
        trainable_nodes=[2],
        measurement_order=[2, 0],
    )

    resolver = MeasurementAngleResolver(circuit)
    commands = resolver.resolve_many([0.7])

    assert resolver.measurement_nodes == [2, 0, 3]
    assert resolver.zero_outcomes() == {0: 0, 1: 0, 2: 0, 3: 0}
    assert [command.node for command in commands] == [2, 0, 3]
    assert commands[0].trainable is True
    assert commands[0].trainable_index == 0
    assert commands[0].angle == 0.7


def test_resolver_falls_back_to_graph_order_without_schedule():
    circuit = _FakeCircuit({0: _ment(0.1), 1: None, 2: _ment(0.2)})

    resolver = MeasurementAngleResolver(circuit)

    assert resolver.schedule == [0, 1, 2]
    assert resolver.measurement_nodes == [0, 2]


def test_resolver_validates_parameters_and_measurement_presence():
    circuit = _FakeCircuit({0: None, 1: _ment(None)}, trainable_nodes=[1])
    resolver = MeasurementAngleResolver(circuit)

    with pytest.raises(ValueError, match="Number of angles"):
        resolver.resolve_many([])

    with pytest.raises(ValueError, match="not measured"):
        resolver.resolve(0, [0.25])


def test_resolver_rejects_unresolved_nontrainable_angle():
    circuit = _FakeCircuit({0: _ment(None)}, trainable_nodes=[])
    resolver = MeasurementAngleResolver(circuit)

    with pytest.raises(ValueError, match="no fixed angle"):
        resolver.resolve(0, [])


def test_resolver_uses_outcomes_for_callable_metadata():
    ment = SimpleNamespace(
        plane=lambda outcomes: "XY" if outcomes["branch"] else "X",
        angle=lambda outcomes: outcomes["angle"],
    )
    circuit = _FakeCircuit({"node": ment})
    resolver = MeasurementAngleResolver(circuit)

    command = resolver.resolve(
        "node", [], outcomes={"branch": True, "angle": 0.75}
    )

    assert command.node == "node"
    assert command.plane == "XY"
    assert command.angle == 0.75
    assert command.trainable is False


def test_xy_plane_angle_rejects_unsupported_planes():
    with pytest.raises(ValueError, match="not supported"):
        xy_plane_angle("Z", 0.0)


def test_numpy_sv_uses_resolver_for_fixed_controlment():
    controlled = mp.templates.linear_cluster(3)
    controlled[0] = mp.ControlMent(
        True,
        true_angle=0.0,
        true_plane="XY",
        false_angle=0,
        false_plane="X",
    )
    plain = mp.templates.linear_cluster(3)
    plain[0] = mp.Ment(0.0, "XY")

    angles = np.zeros(len(controlled.trainable_nodes))
    controlled_dm = mp.PatternSimulator(controlled, backend="numpy-sv").run(angles)
    plain_dm = mp.PatternSimulator(plain, backend="numpy-sv").run(angles)

    assert np.allclose(controlled_dm, plain_dm)
